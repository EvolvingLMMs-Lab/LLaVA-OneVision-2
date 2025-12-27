#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
# Usage
python s1_get_tokenlens_v4-sft.py --config ./configs/s1_config_MMR_sft_780k.yaml
"""

import os
import json
import orjson
import threading
import logging
import psutil
import tempfile
import queue
import yaml
import argparse
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from heapq import merge
from PIL import Image
from jinja2 import Template
from transformers import AutoProcessor
from transformers import BitsAndBytesConfig
from qwen_vl_utils import fetch_image
from queue import Empty
import multiprocessing
from multiprocessing import Pool, Manager, Value

# Declare a global, cross-process counter (defined in the main module to be inherited by child processes).
global_total_counter = None

# ✅ Parse command-line arguments
parser = argparse.ArgumentParser(description="Token Length Processor")
parser.add_argument("--config", type=str, default="config.yaml", help="Path to config.yaml")
parser.add_argument("--log-level", type=str, default=None,
                    choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                    help="Override log level from config")
args = parser.parse_args()

# ✅ Load configuration file
CONFIG_PATH = Path(args.config)
if not CONFIG_PATH.exists():
    raise FileNotFoundError(f"Configuration file does not exist: {CONFIG_PATH}")
with open(CONFIG_PATH, 'r', encoding='utf-8') as f:
    cfg = yaml.safe_load(f)

# ✅ Read parameters from configuration file, override existing constants
MAX_TOKEN_LEN = cfg['sample']['max_len']
task_type = cfg['sample']['task_type']
DEL_ONE_TOKEN = cfg['sample']['del_one_token']

DEFAULT_DIRECTORY = Path(cfg['data']['directory'])
OUTPUT_FILE = Path(cfg['data']['output_base'])
TOKEN_INFO_FILE = Path(cfg['data']['output_token'])
CKPT_DIR = cfg['model']['checkpoint']
MIN_PIXELS = cfg['image']['min_pixels']
MAX_PIXELS = cfg['image']['max_pixels']
image_resolution = cfg['image']['baidu_resolution']
TIME_OUT = cfg['processing']['time_out']
# 归并参数（仅两级：stage0 → stage1）
STAGE1_CHUNK = cfg['processing']['stage1_merge_chunk']
chunk_size = cfg['processing']['chunk_size']
n_workers = cfg['processing']['n_workers']
MIN_WORKERS = cfg['processing']['min_workers']
MAX_WORKERS = cfg['processing']['max_workers']
use_shm = cfg['logging']['use_shm']
log_level = cfg['logging']['level']
log_file = cfg['logging']['file']
if args.log_level:
    log_level = args.log_level.upper()

# ✅ Configure logging - detailed record of data flow and merge process
file_handler = logging.FileHandler(
    log_file,
    delay=True,
    encoding='utf-8'
)
stream_handler = logging.StreamHandler()

logging.basicConfig(
    level=log_level,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[file_handler, stream_handler]
)
logger = logging.getLogger(__name__)

EXTENSIONS = (".json", ".jpg")


temp_dir = '/dev/shm' if use_shm else None  # None 表示使用系统默认临时目录

def count_lines(file_path):
    """ Count valid lines in file (non-empty and contain delimiter)"""
    if not os.path.exists(file_path) or os.path.getsize(file_path) == 0:
        return 0
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return sum(1 for line in f if line.strip() and ':' in line.strip())
    except Exception as e:
        logger.error(f"❌ Error counting lines for {file_path}: {str(e)}")
        return 0

def find_paired_files(directory):
    directory = Path(directory)
    files = os.listdir(directory)
    json_set = {f[:-5] for f in files if f.lower().endswith('.json')}
    img_set  = {f[:-4] for f in files if f.lower().endswith(('.jpg', '.jpeg'))}
    paired = json_set & img_set
    logger.info(f"Found {len(paired)} file pairs.")
    return paired

def find_valid_files(fname_json, rel_img_path):
    from s1_mr_sft_data_proc_indcoding import split_json_file
    valid_names = split_json_file(
                    fname_json, 
                    rel_img_path,
                    chunk_dim=2000,
                    m=8
    )
    return valid_names

def find_valid_json(directory):
    directory = Path(directory)
    files = os.listdir(directory)
    json_set = {f[:-5] for f in files if f.lower().endswith('.json')}
    logger.info(f"Found {len(json_set)} JSON files.")
    return json_set    

def write_base_names_to_file(base_names, output_file):
    """ Write paired file names to output file"""
    try:
        content = "\n".join(sorted(base_names)) + "\n"
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(content)
        logger.info(f"ℹ️ Wrote {len(base_names)} paired filenames to {output_file}")
    except Exception as e:
        logger.error(f"❌ Error writing to {output_file}: {str(e)}")
        raise


def read_lines_in_chunks(file_path, chunk_size):
    """ Read file content in chunks, each chunk contains up to chunk_size lines"""
    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"{file_path} does not exist.")
    
    with open(file_path, 'r', encoding='utf-8') as f:
        while True:
            chunk = [line.strip() for _, line in zip(range(chunk_size), f) if line.strip()]
            if not chunk:
                break
            logger.info(f"ℹ️ Read data chunk containing {len(chunk)} samples.")
            yield chunk


# ✅ Precompile template for efficiency
"""
Todo:
    1) put into .yaml 
    2) Add support for user-defined processing functions beyond "jinja2+processor" 
"""
if task_type=="pretrain":
    CAP_TEMPLATE = Template("<|vision_start|><|image_pad|><|vision_end|>{{ captions[0].content }}<|im_end|>")
elif task_type=="sft":
    chat_template  = """{% set image_count = namespace(value=0) %}{% set video_count = namespace(value=0) %}{% for message in messages %}{% if loop.first and message['role'] != 'system' %}<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n{% endif %}<|im_start|>{{ message['role'] }}\n{{ message['content'] | replace('<image>', '<|vision_start|><|image_pad|><|vision_end|>') }}<|im_end|>\n{% endfor %}{% if add_generation_prompt %}<|im_start|>assistant\n{% endif %}"""
    CAP_TEMPLATE = Template(chat_template)
    pass

def process_sample(json_path, img_path, processor):
    """ Process a single sample and return a tuple (token_len, file name)"""
    try:
        if not Path(json_path).exists():
            raise FileNotFoundError(f"❌ JSON file does not exist: {json_path}")
        # if not Path(img_path).exists():
        #     raise FileNotFoundError(f"❌ Image file does not exist: {img_path}")

        # Read and render JSON content
        with open(json_path, 'r', encoding='utf-8') as f:
            json_data = json.load(f)
        # with open(json_path, 'rb') as f:
        #     json_data = orjson.loads(f.read())
        if task_type=="pretrain":
            txt_input = CAP_TEMPLATE.render(captions=json_data['captions'])
        elif task_type=="sft":
            # txt_input = CAP_TEMPLATE.render(json_data)
            txt_input = CAP_TEMPLATE.render(json_data,tokenize=False, add_generation_prompt=False)
        if img_path=="_____.jpg":
            img_input = None
        else:
            def baidu_img_proc(image, image_resolution):
                image = Image.open(image)
                if max(image.width, image.height) > image_resolution:
                    resize_factor = image_resolution / max(image.width, image.height)
                    width, height = int(image.width * resize_factor), int(image.height * resize_factor)
                    image = image.resize((width, height), resample=Image.NEAREST)

                return image

            if image_resolution:
                img_path = baidu_img_proc(img_path, image_resolution)
                
            
            img_input = fetch_image({
                'type': 'image',
                'image': img_path,
                "min_pixels": MIN_PIXELS,
                "max_pixels": MAX_PIXELS,
            })
        # print(img_input)
        # Calculate token number
        base_name = Path(json_path).stem
        inputs = processor(
            text=[txt_input],
            images=img_input,
            videos=None,
            padding=True,
            return_tensors="pt",
        )
        # print(inputs["input_ids"])
        # print(inputs["input_ids"].shape)
        return (inputs["input_ids"].shape[1], base_name)

    except Exception as e:
        return (None, f"❌ Failed to process sample [{Path(json_path).stem}]: {str(e)}")


def get_adaptive_workers(min_workers=20, max_workers=96):
    """Dynamically adjust the number of threads based on system load"""
    try:
        cpu_usage = psutil.cpu_percent(interval=0.5)
        mem_usage = psutil.virtual_memory().percent
        if cpu_usage > 80 or mem_usage > 85:
            adjusted = max(min_workers, max_workers // 2)
            logger.info(f"High system load, adjusting thread count to {adjusted} (CPU: {cpu_usage}%, Memory: {mem_usage}%)")
            return adjusted
        return max_workers
    except Exception as e:
        logger.warning(f"System load check failed, falling back to {max_workers} threads: {str(e)}")
        return max_workers

gt_maxlen=0
def merge_files_by_token(input_files, output_file, max_token=MAX_TOKEN_LEN):
    """Merge multiple sorted files by token_len, filter out lines > max_token, return (output_path, line_count)"""
    if not input_files:
        logger.warning("⚠️ No files to merge")
        return (None, 0)

    # Validate input files and count total lines
    valid_files = []
    total_lines = 0
    for f in input_files:
        line_count = count_lines(f)
        if line_count > 0:
            valid_files.append(f)
            total_lines += line_count
            logger.debug(f"ℹ️ Merging file {os.path.basename(f)} with {line_count} entries.")
        else:
            logger.warning(f"⚠️ Skipping empty or invalid file: {os.path.basename(f)}")

    if not valid_files:
        return (None, 0)

    # Define a sorting key (sorted by the token_len integer)
    def sort_key(line):
        # _, token_str = line.strip().split(':', 1)
        token_str = line.strip().split(':')[-1]
        return int(token_str)

    try:
        with open(output_file, 'w', encoding='utf-8') as out_f:
            # Create iterator for all files.
            iterators = []
            file_handles = []
            for fpath in valid_files:
                try:
                    fh = open(fpath, 'r', encoding='utf-8')
                    file_handles.append(fh)
                    iterators.append(((sort_key(line), line) for line in fh))
                except Exception as e:
                    logger.error(f"❌ 打开文件 {os.path.basename(fpath)} 失败: {str(e)}")

            # Merge sort and write, filtering out lines with token count > max_token (other conditions can be added later)
            filtered_max_len = 0
            for _, line in merge(*iterators, key=lambda x: x[0]):
                token_str = line.strip().split(':')[-1]
                if int(token_str) <= max_token:
                    out_f.write(line)
                else:
                    logger.warning(f"⚠️ Token length: {token_str} > {max_token}: filtered out!")
                    filtered_max_len+=1
                    gt_maxlen

            # Close all file handles
            for fh in file_handles:
                try:
                    fh.close()
                except Exception as e:
                    logger.warning(f"⚠️ 关闭文件 {fh.name} 失败: {str(e)}")

        # Verify output file integrity
        output_lines = count_lines(output_file)+filtered_max_len
        if output_lines != total_lines:   # Filter out lines with token count > max_token
            logger.error(f"❌ Merge data loss! {total_lines} lines in, {output_lines} lines out. Deleted bad file.")
            if os.path.exists(output_file):
                os.remove(output_file)
            return (None, 0)
        else:
            logger.info(f"✅ 📊 Merge successful. Input: {total_lines} lines, Output: {output_lines-filtered_max_len} lines (token ≤ {max_token}).")

        return (output_file, output_lines-filtered_max_len)
    except Exception as e:
        logger.error(f"❌ File merge failed: {str(e)}")
        if os.path.exists(output_file):
            try:
                os.remove(output_file)
            except Exception as e:
                logger.warning(f"⚠️ Failed to delete the corrupted file {output_file}: {str(e)}")
        return (None, 0)


def stage1_merger(input_queue, chunk_size, stage1_files, stop_event):
    """
    Fixed version of stage1 merging threads
    - Ensure all stage0 files are merged, including the last batch with fewer than 10 files
    - Resolve thread timeout and data loss issues
    """
    buffer = []
    batch_counter = 0
    logger.info(f"💡 Stage1 merge thread started. Merging every {chunk_size} stage0 files.")

    try:
        # Loop condition: the queue has files, or the buffer has files, or no stop signal is received.
        while (not input_queue.empty()) or buffer or (not stop_event.is_set()):
            # Fetch files from the queue (with timeout to prevent permanent blocking)
            if not input_queue.empty():
                try:
                    file_path = input_queue.get(timeout=1)  # Timeout after 1 second to avoid permanent blocking.
                    buffer.append(file_path)
                    input_queue.task_done()
                    logger.debug(f"ℹ️ Stage1 received file {os.path.basename(file_path)}, buffer: {len(buffer)}/{chunk_size}")

                    # If the buffer has enough files, execute the merge
                    if len(buffer) >= chunk_size:
                        batch_counter += 1
                        merged_file = tempfile.NamedTemporaryFile(
                            mode='w', delete=False,
                            prefix=f"stage1_batch{batch_counter:03d}_",
                            encoding='utf-8',
                            dir=temp_dir
                        ).name
                        
                        # 执行合并
                        merged_path, line_count = merge_files_by_token(buffer, merged_file)
                        if merged_path and line_count > 0:
                            stage1_files.append(merged_path)
                            logger.info(f"📊 Stage1 batch {batch_counter} done: {os.path.basename(merged_path)}  ({line_count} lines, {len(buffer)} files merged).")
                        else:
                            logger.warning(f"⚠️ Skipping Stage1 batch {batch_counter} due to merge failure.")

                        # Clear the buffer after successful merge
                        buffer = []
                except Empty:
                    continue  # Continue the loop if the queue is empty.
                except Exception as e:
                    logger.error(f"❌ Stage1 error while processing file: {str(e)}", exc_info=True)
            else:
                # If the queue is empty, check if we need to force merge remaining files.
                if buffer and stop_event.is_set():
                    # If the stop signal is received and the buffer has files, force merge.
                    batch_counter += 1
                    merged_file = tempfile.NamedTemporaryFile(
                        mode='w', delete=False,
                        prefix=f"stage1_remaining_batch{batch_counter:03d}_",
                        encoding='utf-8',
                        dir=temp_dir
                    ).name
                    
                    merged_path, line_count = merge_files_by_token(buffer, merged_file)
                    if merged_path and line_count > 0:
                        stage1_files.append(merged_path)
                        logger.info(f"📊 Stage1 remaining files merged: {os.path.basename(merged_path)} with {line_count} entries from {len(buffer)} files.")
                    else:
                        logger.warning(f"⚠️ Skipping Stage1 remaining batch due to merge failure.")
                    buffer = []
                else:
                    # Sleep briefly to reduce CPU usage
                    threading.Event().wait(0.5)

        # Final check: Ensure the buffer is empty (to prevent omissions)
        if buffer:
            logger.error(f"❌ Stage1 thread exited with {len(buffer)} files in buffer unprocessed! Data loss may occur.")

    except Exception as e:
        logger.error(f"❌ Stage1 thread exception exit: {str(e)}", exc_info=True)
    finally:
        logger.info(f"📊 Stage1 thread exit, {len(stage1_files)} files generated.")

# Processing function for each process (responsible for handling a large chunk)
def process_chunk(args):
    """
    Processing logic for each process: handles a large chunk of data, with parallel processing using multiple threads.
    
    Args:
        args: A tuple containing chunk data, processor configuration, and queues for inter-process communication.
    """
    # Get the global counter from the global variable, not from the arguments
    global global_total_counter
    
    chunk_idx, chunk, ckpt_dir, min_pixels, max_pixels, stage0_queue = args
    processor = None
    processed_count = 0  # Record the number of valid samples processed by the current process
    
    try:
        # Each process initializes its own processor (processors cannot be shared between processes)
        # quant_config = BitsAndBytesConfig(load_in_4bit=True)
        processor = AutoProcessor.from_pretrained(
            ckpt_dir,
            min_pixels=min_pixels,
            max_pixels=max_pixels,
            trust_remote_code=True,
            use_fast=False
        )
        # Generate the list of file paths for the current chunk
        full_paths = []
        for fn in chunk:
            cur_json = str(DEFAULT_DIRECTORY / f"{fn}.json")
            # logger.info(f"👉 Process {multiprocessing.current_process().name} json file: {cur_json}.....{type(cur_json)}")
            if f"{fn}.json".startswith("__img--output_"):
                cur_img = "_____.jpg"
                # cur_img = str(DEFAULT_DIRECTORY / f"{cur_img}")
            else:     
                with open(cur_json, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    cur_img = data['images'][0]
                    cur_img = str(DEFAULT_DIRECTORY / f"{cur_img}")
            full_paths.append(cur_json)
            full_paths.append(cur_img)
            # print(f"--------------cur_json:{cur_json}, cur_img:{cur_img}-------------------")
            

        n_samples = len(chunk)
        logger.info(f"👉 Process {multiprocessing.current_process().name} starts processing chunk {chunk_idx} with {n_samples} samples")
        
        # Process each sample in the chunk using a thread pool (reuse threads)
        n_workers = get_adaptive_workers(min_workers=MIN_WORKERS, max_workers=MAX_WORKERS)  # Reduce the number of workers per process
        chunk_results = []
        with ThreadPoolExecutor(
            max_workers=n_workers,
            thread_name_prefix=f"proc-{multiprocessing.current_process().pid}-thread"
        ) as executor:
            tasks = [
                executor.submit(
                    process_sample,
                    full_paths[idx*2],
                    full_paths[idx*2+1],
                    processor
                ) for idx in range(n_samples)
            ]
            
            # Collect the results of each thread task
            for future in as_completed(tasks):
                try:
                    token_len, name = future.result()
                    if DEL_ONE_TOKEN:
                        token_len += 1
                    if token_len is not None:
                        chunk_results.append((token_len, name))
                        processed_count += 1  # Record the number of valid samples processed
                    else:
                        logger.warning(name)
                except Exception as e:
                    logger.error(f"❌ Process {multiprocessing.current_process().name} thread task error: {str(e)}")
        
        # Write the results to the stage0 file and put it into the cross-process queue
        if chunk_results:
            chunk_results_sorted = sorted(chunk_results, key=lambda x: x[0])
            with tempfile.NamedTemporaryFile(
                mode='w+', delete=False,
                prefix=f"stage0_chunk{chunk_idx:03d}_",
                encoding='utf-8',
                dir=temp_dir  
            ) as f:
                stage0_file = f.name
                for token_len, name in chunk_results_sorted:
                    f.write(f"{name}:{token_len}\n")
            
            line_count = count_lines(stage0_file)
            stage0_queue.put(stage0_file)

            proc_status = "🟢" if processed_count==n_samples else "🟡"
            logger.info(f"{proc_status} 进程 {multiprocessing.current_process().name} 完成块 {chunk_idx}，有效样本 {processed_count}/{n_samples}")
            
            # 【Key】Accumulate total data volume across processes (using Value atomic operations)
            with global_total_counter.get_lock():
                global_total_counter.value += processed_count
                
            return stage0_file  # Return the path of the generated stage0 file for subsequent cleanup
        
    except Exception as e:
        logger.error(f"❌ Process {multiprocessing.current_process().name} failed: {str(e)}")
    finally:
        if processor:
            del processor
    return None


###
def main():
    global global_total_counter  # Reference the global counter
    processor = None   # Model processor instance
    stage0_files = []  # Record all stage0 files (for verification and cleanup)
    stage1_files = []  # Record all stage1 files (for final merging)

    try:

        logger.info(f"💡 --------------Start the data processing flow--------------")
        
        # 1. Find paired files and write to a temporary file (samples where the JSON and JPG file names are the same)
        # base_names = find_paired_files(DEFAULT_DIRECTORY)    # DEFAULT_DIRECTORY is the location for storing raw data (JPG and JSON files)
        base_names = find_valid_json(DEFAULT_DIRECTORY)
        total_original = len(base_names)  # Total number of original samples
        logger.info(f"👉 Found {total_original} pairs of original sample files")
        if total_original == 0:
            logger.warning("⚠️ No original samples found, exiting the program")
            return
        # Write the paired file names to a file for subsequent chunk reading
        write_base_names_to_file(base_names, OUTPUT_FILE)
        
        # 2. Initialize the cross-process queue (for passing stage0 file paths to the merging thread)
        manager = Manager()  # Process-sharing queue requires Manager
        stage0_queue = manager.Queue()
        stop_event = manager.Event()  # Cross-process stop signal

        # Cross-process counter for counting the total number of processed samples (initial value: 0)
        global_total_counter = Value('i', 0)  # 'i' indicates the integer type, used for inter-process sharing.

        # 3. Start the stage1 merging thread (daemon thread)
        stage1_thread = threading.Thread(
            target=stage1_merger,
            args=(stage0_queue, STAGE1_CHUNK, stage1_files, stop_event),
            daemon=True
        )
        stage1_thread.start()
        logger.info("💡 stage1 merging thread has started")

        # 4. Process data and generate stage0 files (each chunk is processed and sorted individually)
        # n_workers = 96 #get_adaptive_workers()

        # 4.1 Read all data chunks (for distribution to multiple processes)
        # chunk_size = chunk_size  # The size of each large chunk processed by each process (adjust based on memory)
        all_chunks = list(read_lines_in_chunks(OUTPUT_FILE, chunk_size))
        total_chunks = len(all_chunks)
        n_processes = min(multiprocessing.cpu_count(), total_chunks)
        logger.info(f"👉 Split into {total_chunks} chunks, launching {n_processes} processes.")

        # 4.2 Prepare process pool arguments (including model configuration, queue, etc.)
        process_args = [
            (
                idx + 1,  # chunk index
                chunk,    # chunk data
                CKPT_DIR, # model path
                MIN_PIXELS,
                MAX_PIXELS,
                stage0_queue,  # cross-process queue for stage0 files
            ) for idx, chunk in enumerate(all_chunks)
        ]
        
        # 4.3 Launch process pool (number of processes recommended to set to 1~2 times the number of CPU cores)
        with Pool(processes=n_processes) as process_pool:
            # Process all large chunks in parallel.
            # stage0_files = process_pool.map(process_chunk, process_args)
            result = process_pool.map_async(process_chunk, process_args)
            try:
                stage0_files = result.get(timeout=TIME_OUT)  # Set timeout for process completion (in seconds)
            except multiprocessing.TimeoutError:
                logger.error("❌ Some processes timed out, force termination")
                process_pool.terminate()
        
        # Filter out empty results.
        stage0_files = [f for f in stage0_files if f is not None]
        logger.info(f"✅ All processes completed, generated {len(stage0_files)} stage0 files.")  
        # Statistics
        total_processed = global_total_counter.value  # Directly get from global variable  # Total processed samples
        logger.info(f"👉 Total original samples: {total_original}, Valid processed samples: {total_processed}")

        # Data integrity check
        if total_processed != total_original:
            logger.warning(f"❌ Data integrity check failed! Original {total_original} samples, processed {total_processed} samples, difference {total_original - total_processed} samples.")
        else:
            logger.info("✅ Data integrity check passed, all samples were processed successfully.")
        
        # 5. Wait for all stage0 files to be processed (ensure all files are merged)
        # Wait for all stage0 files to be processed in the queue.
        logger.info("🔄 Waiting for stage0 queue to process all files...")
        stage0_queue.join()  # Block until all stage0 files are consumed
        logger.info("💡 All stage0 files have been processed and merged.")

        # Send a stop signal to the stage1 threads to force processing of remaining files.
        logger.info("💡 Sending stop signal to stage1 thread to force processing remaining files...")
        stop_event.set()

        timeout_counter = 0
        while stage1_thread.is_alive() and timeout_counter < 60:
            logger.debug(f"🔄 Waiting for stage1 thread to complete ({timeout_counter}/60 seconds)...")
            threading.Event().wait(1)  # Wait for 1 second to retry
            timeout_counter += 1
        
        if stage1_thread.is_alive():
            logger.warning("⚠️ Stage1 thread did not exit on timeout. Anomaly suspected (force-merge of remaining files was attempted).")
        else:
            logger.info("💡 Stage1 thread has exited normally.")

        # Verify that the number of stage1 files matches (1 stage1 file is merged from every 10 stage0 files; batches with fewer than 10 stage0 files also count as 1 stage1 file)
        expected_stage1_count = (len(stage0_files) + STAGE1_CHUNK - 1) // STAGE1_CHUNK
        if len(stage1_files) != expected_stage1_count:
            logger.warning(f"⚠️ ℹ️  Stage1 file count mismatch! Expected {expected_stage1_count} files, but got {len(stage1_files)} files.")
        else:
            logger.info(f"✅ Stage1 file count verification passed: {len(stage1_files)} files.")

        # 6. Finally, merge all stage1 files into token_info_1.txt.
        if not stage1_files:
            logger.warning("⚠️ No stage1 files were generated. Please check if the intermediate processing steps encountered any errors.")
            return

        # Count the total data volume of stage1 files.
        stage1_total = sum(count_lines(f) for f in stage1_files)
        logger.info(f"ℹ️ Starting final merge: {len(stage1_files)} stage1 files, total records: {stage1_total}.")

        # Merge into the final file.
        final_path, final_lines = merge_files_by_token(stage1_files, TOKEN_INFO_FILE)

        if final_path and final_lines > 0:
            logger.info(f"✅ Final result file generated: {TOKEN_INFO_FILE}, total records: {final_lines}.")
            # Verify the total data volume.
            if final_lines != total_processed:
                logger.error(f"❌ Data volume mismatch! Processed {total_processed} records, but final file contains {final_lines} records.")
            else:
                logger.info("✅💡 Data volume verification passed, all records have been successfully written to the final file.")
        else:
            logger.error("❌ Final file merge failed.")

        # Verify the final file again after merge.
        if os.path.exists(TOKEN_INFO_FILE):
            final_count = count_lines(TOKEN_INFO_FILE)
            logger.info(f"ℹ️ Final result file contains {final_count} records.")
            if final_count != total_processed:
                logger.error(f"❌ Final file data incomplete! Processed {total_processed} records, but final file contains {final_count} records.")
            else:
                logger.info("✅ Final file data integrity verification passed.")

    except Exception as e:
        logger.error(f"❌ Critical failure in main process: {str(e)}", exc_info=True)
    finally:
        # Clean up resources.
        if processor:
            del processor

        # Ensure the stop signal is triggered.
        stop_event.set()

        if stage1_thread and stage1_thread.is_alive():
            stage1_thread.join(timeout=2)        
        
        # Wait for the final file to be written.
        threading.Event().wait(2)

        # Clean up temporary files (keep the final file).
        all_temp_files = stage0_files + stage1_files
        for fpath in all_temp_files:
            if fpath != str(TOKEN_INFO_FILE) and os.path.exists(fpath):
                try:
                    os.remove(fpath)
                    logger.debug(f"✅ Temporary file cleaned: {os.path.basename(fpath)}")
                except Exception as e:
                    logger.warning(f"❌ Failed to clean temporary file {os.path.basename(fpath)}: {str(e)}")

        logger.info("✅ Program execution completed.")


if __name__ == "__main__":
    main()
