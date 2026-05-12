#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
# Usage
python s1_get_tokenlens_v4-sft.py --config ./configs/s1_config_BMR_sft_780k.yaml
"""

import argparse
import json
import logging
import multiprocessing
import os
import tempfile
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from heapq import merge
from multiprocessing import Manager, Pool, Value
from pathlib import Path
from queue import Empty

import psutil
import yaml
from jinja2 import Template
from PIL import Image
from qwen_vl_utils import fetch_image

from transformers import AutoProcessor


# Global cross-process counter. It is defined in the main module so child processes can inherit it.
global_total_counter = None

# Parse command-line arguments.
parser = argparse.ArgumentParser(description="Token Length Processor")
parser.add_argument("--config", type=str, default="config.yaml", help="Path to config.yaml")
parser.add_argument(
    "--log-level",
    type=str,
    default=None,
    choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    help="Override log level from config",
)
args = parser.parse_args()

# Load configuration.
CONFIG_PATH = Path(args.config)
if not CONFIG_PATH.exists():
    raise FileNotFoundError(f"Configuration file does not exist: {CONFIG_PATH}")
with open(CONFIG_PATH, "r", encoding="utf-8") as f:
    cfg = yaml.safe_load(f)

# Read parameters from config and override the original constants.
MAX_TOKEN_LEN = cfg["sample"]["max_len"]
task_type = cfg["sample"]["task_type"]
DEL_ONE_TOKEN = cfg["sample"]["del_one_token"]

DEFAULT_DIRECTORY = Path(cfg["data"]["directory"])
OUTPUT_FILE = Path(cfg["data"]["output_base"])
TOKEN_INFO_FILE = Path(cfg["data"]["output_token"])
CKPT_DIR = cfg["model"]["checkpoint"]
MIN_PIXELS = cfg["image"]["min_pixels"]
MAX_PIXELS = cfg["image"]["max_pixels"]
image_resolution = cfg["image"]["baidu_resolution"]
TIME_OUT = cfg["processing"]["time_out"]
# Merge settings. This legacy flow uses only two levels: stage0 -> stage1.
STAGE1_CHUNK = cfg["processing"]["stage1_merge_chunk"]
chunk_size = cfg["processing"]["chunk_size"]
n_workers = cfg["processing"]["n_workers"]
MIN_WORKERS = cfg["processing"]["min_workers"]
MAX_WORKERS = cfg["processing"]["max_workers"]
use_shm = cfg["logging"]["use_shm"]
log_level = cfg["logging"]["level"]
log_file = cfg["logging"]["file"]
if args.log_level:
    log_level = args.log_level.upper()

# Logging configuration for detailed data-flow and merge tracing.
file_handler = logging.FileHandler(log_file, delay=True, encoding="utf-8")
stream_handler = logging.StreamHandler()

logging.basicConfig(
    level=log_level, format="%(asctime)s - %(levelname)s - %(message)s", handlers=[file_handler, stream_handler]
)
logger = logging.getLogger(__name__)

EXTENSIONS = (".json", ".jpg")


temp_dir = "/dev/shm" if use_shm else None  # None means using the system default temporary directory.


def count_lines(file_path):
    """Count valid lines that are non-empty and contain the separator."""
    if not os.path.exists(file_path) or os.path.getsize(file_path) == 0:
        return 0
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            return sum(1 for line in f if line.strip() and ":" in line.strip())
    except Exception as e:
        logger.error(f"❌ Failed to count lines in {file_path}: {str(e)}")
        return 0


def find_paired_files(directory):
    directory = Path(directory)
    files = os.listdir(directory)
    json_set = {f[:-5] for f in files if f.lower().endswith(".json")}
    img_set = {f[:-4] for f in files if f.lower().endswith((".jpg", ".jpeg"))}
    paired = json_set & img_set
    logger.info(f"Found {len(paired)} matched file pairs")
    return paired


def find_valid_files(fname_json, rel_img_path):
    from s1_mr_sft_data_proc_indcoding import split_json_file

    valid_names = split_json_file(fname_json, rel_img_path, chunk_dim=2000, m=8)
    return valid_names


def find_valid_json(directory):
    directory = Path(directory)
    files = os.listdir(directory)
    json_set = {f[:-5] for f in files if f.lower().endswith(".json")}
    logger.info(f"Found {len(json_set)} JSON files")
    return json_set


def write_base_names_to_file(base_names, output_file):
    """Write paired base names to a file."""
    try:
        content = "\n".join(sorted(base_names)) + "\n"
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(content)
        logger.info(f"ℹ️ Wrote {len(base_names)} paired base names to {output_file}")
    except Exception as e:
        logger.error(f"❌ Failed to write {output_file}: {str(e)}")
        raise


def read_lines_in_chunks(file_path, chunk_size):
    """Read file contents in chunks."""
    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"{file_path} does not exist")

    with open(file_path, "r", encoding="utf-8") as f:
        while True:
            chunk = [line.strip() for _, line in zip(range(chunk_size), f) if line.strip()]
            if not chunk:
                break
            logger.info(f"ℹ️ Read a data chunk with {len(chunk)} samples")
            yield chunk


# Precompile templates.
"""
Todo:
    1) Move this into the YAML config.
    2) Add support for non-"jinja2 + processor" custom handlers.
"""
if task_type == "pretrain":
    CAP_TEMPLATE = Template("<|vision_start|><|image_pad|><|vision_end|>{{ captions[0].content }}<|im_end|>")
elif task_type == "sft":
    chat_template = """{% set image_count = namespace(value=0) %}{% set video_count = namespace(value=0) %}{% for message in messages %}{% if loop.first and message['role'] != 'system' %}<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n{% endif %}<|im_start|>{{ message['role'] }}\n{{ message['content'] | replace('<image>', '<|vision_start|><|image_pad|><|vision_end|>') }}<|im_end|>\n{% endfor %}{% if add_generation_prompt %}<|im_start|>assistant\n{% endif %}"""
    CAP_TEMPLATE = Template(chat_template)
    pass


def process_sample(json_path, img_path, processor):
    """Process one sample and return (token_len, file_name)."""
    try:
        if not Path(json_path).exists():
            raise FileNotFoundError(f"❌ JSON file does not exist: {json_path}")
        # if not Path(img_path).exists():
        #     raise FileNotFoundError(f"❌ Image file does not exist: {img_path}")

        # Read and render JSON content.
        with open(json_path, "r", encoding="utf-8") as f:
            json_data = json.load(f)
        # with open(json_path, 'rb') as f:
        #     json_data = orjson.loads(f.read())
        if task_type == "pretrain":
            txt_input = CAP_TEMPLATE.render(captions=json_data["captions"])
        elif task_type == "sft":
            # txt_input = CAP_TEMPLATE.render(json_data)
            txt_input = CAP_TEMPLATE.render(json_data, tokenize=False, add_generation_prompt=False)
        if img_path == "_____.jpg":
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

            img_input = fetch_image(
                {
                    "type": "image",
                    "image": img_path,
                    "min_pixels": MIN_PIXELS,
                    "max_pixels": MAX_PIXELS,
                }
            )
        # print(img_input)
        # Compute token length.
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
        return (None, f"❌ Processing failed [{Path(json_path).stem}]: {str(e)}")


def get_adaptive_workers(min_workers=20, max_workers=96):
    """Adjust thread count according to system load."""
    try:
        cpu_usage = psutil.cpu_percent(interval=0.5)
        mem_usage = psutil.virtual_memory().percent
        if cpu_usage > 80 or mem_usage > 85:
            adjusted = max(min_workers, max_workers // 2)
            logger.info(
                f"System load is high; adjusted thread count to {adjusted} (CPU: {cpu_usage}%, memory: {mem_usage}%)"
            )
            return adjusted
        return max_workers
    except Exception as e:
        logger.warning(f"Failed to get system load; using default thread count {max_workers}: {str(e)}")
        return max_workers


gt_maxlen = 0


def merge_files_by_token(input_files, output_file, max_token=MAX_TOKEN_LEN):
    """Merge sorted files by token_len, filter rows above max_token, and return (output_path, row_count)."""
    if not input_files:
        logger.warning("⚠️ No files to merge")
        return (None, 0)

    # Validate input files and count total records.
    valid_files = []
    total_lines = 0
    for f in input_files:
        line_count = count_lines(f)
        if line_count > 0:
            valid_files.append(f)
            total_lines += line_count
            logger.debug(f"ℹ️ Input file {os.path.basename(f)} contains {line_count} records")
        else:
            logger.warning(f"⚠️ File {os.path.basename(f)} is empty or invalid; skipping")

    if not valid_files:
        return (None, 0)

    # Define sort key by integer token_len.
    def sort_key(line):
        token_str = line.strip().split(":")[-1]
        return int(token_str)

    try:
        with open(output_file, "w", encoding="utf-8") as out_f:
            # Create iterators for all files.
            iterators = []
            file_handles = []
            for fpath in valid_files:
                try:
                    fh = open(fpath, "r", encoding="utf-8")
                    file_handles.append(fh)
                    iterators.append((sort_key(line), line) for line in fh)
                except Exception as e:
                    logger.error(f"❌ Failed to open file {os.path.basename(fpath)}: {str(e)}")

            # Merge-sort and write.
            # for _, line in merge(*iterators, key=lambda x: x[0]):
            #     out_f.write(line)
            # Merge-sort and write while filtering rows above max_token. More filters can be added later.
            filtered_max_len = 0
            for _, line in merge(*iterators, key=lambda x: x[0]):
                _, token_str = line.strip().split(":", 1)
                if int(token_str) <= max_token:  # Keep only rows within the configured maximum length.
                    out_f.write(line)
                else:
                    logger.warning(f"⚠️ Token length {token_str} > {max_token}: filtered out")
                    filtered_max_len += 1
                    gt_maxlen

            # Close all file handles.
            for fh in file_handles:
                try:
                    fh.close()
                except Exception as e:
                    logger.warning(f"⚠️ Failed to close file {fh.name}: {str(e)}")

        # Validate output data completeness.
        output_lines = count_lines(output_file) + filtered_max_len
        if output_lines != total_lines:  # Rows filtered by length are counted back for validation.
            logger.error(
                f"❌ Merge data mismatch: input {total_lines}, output {output_lines}; removed invalid output file"
            )
            if os.path.exists(output_file):
                os.remove(output_file)
            return (None, 0)
        else:
            logger.info(
                f"✅ 📊 Merge succeeded: input {total_lines}, output {output_lines - filtered_max_len} rows with token <= {max_token}"
            )

        return (output_file, output_lines - filtered_max_len)
    except Exception as e:
        logger.error(f"❌ Failed to merge files: {str(e)}")
        if os.path.exists(output_file):
            try:
                os.remove(output_file)
            except Exception as e:
                logger.warning(f"⚠️ Failed to delete invalid output file {output_file}: {str(e)}")
        return (None, 0)


def stage1_merger(input_queue, chunk_size, stage1_files, stop_event):
    """
    Robust stage1 merge thread.
    - Ensures every stage0 file is merged, including the final partial batch.
    - Avoids thread timeouts and data loss.
    """
    buffer = []
    batch_counter = 0
    logger.info(f"💡 Stage1 merge thread started; merging every {chunk_size} stage0 files")

    try:
        # Continue while the queue has files, the buffer has files, or the stop signal has not arrived.
        while (not input_queue.empty()) or buffer or (not stop_event.is_set()):
            # Pull a file from the queue with a timeout to avoid permanent blocking.
            if not input_queue.empty():
                try:
                    file_path = input_queue.get(timeout=1)  # One-second timeout avoids permanent blocking.
                    buffer.append(file_path)
                    input_queue.task_done()
                    logger.debug(
                        f"ℹ️ Stage1 received {os.path.basename(file_path)}; buffer: {len(buffer)}/{chunk_size}"
                    )

                    # Merge once the buffer reaches the target batch size.
                    if len(buffer) >= chunk_size:
                        batch_counter += 1
                        merged_file = tempfile.NamedTemporaryFile(
                            mode="w",
                            delete=False,
                            prefix=f"stage1_batch{batch_counter:03d}_",
                            encoding="utf-8",
                            dir=temp_dir,
                        ).name

                        # Execute merge.
                        merged_path, line_count = merge_files_by_token(buffer, merged_file)
                        if merged_path and line_count > 0:
                            stage1_files.append(merged_path)
                            logger.info(
                                f"📊 Stage1 batch {batch_counter} complete: {os.path.basename(merged_path)}, {line_count} rows from {len(buffer)} files"
                            )
                        else:
                            logger.warning(f"⚠️ Stage1 batch {batch_counter} merge failed; skipping this batch")

                        # Clear buffer after a batch attempt.
                        buffer = []
                except Empty:
                    continue  # Continue when the queue is empty.
                except Exception as e:
                    logger.error(f"❌ Stage1 file processing error: {str(e)}", exc_info=True)
            else:
                # If the queue is empty, check whether remaining buffered files must be force-merged.
                if buffer and stop_event.is_set():
                    # Stop signal received and the buffer still has files; force a final merge.
                    batch_counter += 1
                    merged_file = tempfile.NamedTemporaryFile(
                        mode="w",
                        delete=False,
                        prefix=f"stage1_remaining_batch{batch_counter:03d}_",
                        encoding="utf-8",
                        dir=temp_dir,
                    ).name

                    merged_path, line_count = merge_files_by_token(buffer, merged_file)
                    if merged_path and line_count > 0:
                        stage1_files.append(merged_path)
                        logger.info(
                            f"📊 Stage1 remaining-file merge complete: {os.path.basename(merged_path)}, {line_count} rows from {len(buffer)} files"
                        )
                    else:
                        logger.warning("❌ Stage1 remaining-file merge failed; data may be missing")
                    buffer = []
                else:
                    # Sleep briefly to reduce CPU usage.
                    threading.Event().wait(0.5)

        # Final check: the buffer should be empty.
        if buffer:
            logger.error(f"❌ Stage1 thread exited with {len(buffer)} unprocessed buffered files; data was lost")

    except Exception as e:
        logger.error(f"❌ Stage1 thread exited unexpectedly: {str(e)}", exc_info=True)
    finally:
        logger.info(f"📊 Stage1 thread exited; generated {len(stage1_files)} files")


# Per-process function that handles one large chunk.
def process_chunk(args):
    """
    Process one large chunk in a single process, using threads internally.

    Args:
        args: Tuple containing chunk data, processor configuration, queues, and related settings.
    """
    # Use the global counter instead of passing it as an argument.
    global global_total_counter

    chunk_idx, chunk, ckpt_dir, min_pixels, max_pixels, stage0_queue = args
    processor = None
    processed_count = 0  # Number of valid samples processed by this process.

    try:
        # Initialize one processor per process because processor instances cannot be shared across processes.
        # quant_config = BitsAndBytesConfig(load_in_4bit=True)
        processor = AutoProcessor.from_pretrained(
            ckpt_dir, min_pixels=min_pixels, max_pixels=max_pixels, trust_remote_code=True, use_fast=False
        )
        # Build the file path list for the current chunk.
        full_paths = []
        for fn in chunk:
            cur_json = str(DEFAULT_DIRECTORY / f"{fn}.json")
            # logger.info(f"👉 Process {multiprocessing.current_process().name} JSON file: {cur_json}.....{type(cur_json)}")
            if f"{fn}.json".startswith("__img--output_"):
                cur_img = "_____.jpg"
                # cur_img = str(DEFAULT_DIRECTORY / f"{cur_img}")
            else:
                with open(cur_json, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    cur_img = data["images"][0]
                    cur_img = str(DEFAULT_DIRECTORY / f"{cur_img}")
            full_paths.append(cur_json)
            full_paths.append(cur_img)
            # print(f"--------------cur_json:{cur_json}, cur_img:{cur_img}-------------------")

        n_samples = len(chunk)
        logger.info(
            f"👉 Process {multiprocessing.current_process().name} started chunk {chunk_idx} with {n_samples} samples"
        )

        # Create an in-process thread pool and reuse threads.
        n_workers = get_adaptive_workers(
            min_workers=MIN_WORKERS, max_workers=MAX_WORKERS
        )  # The per-process thread count can be reduced when needed.
        chunk_results = []
        with ThreadPoolExecutor(
            max_workers=n_workers, thread_name_prefix=f"proc-{multiprocessing.current_process().pid}-thread"
        ) as executor:
            tasks = [
                executor.submit(process_sample, full_paths[idx * 2], full_paths[idx * 2 + 1], processor)
                for idx in range(n_samples)
            ]

            # Collect thread task results.
            for future in as_completed(tasks):
                try:
                    token_len, name = future.result()
                    if DEL_ONE_TOKEN:
                        token_len += 1
                    if token_len is not None:
                        chunk_results.append((token_len, name))
                        processed_count += 1  # Count valid samples.
                    else:
                        logger.warning(name)
                except Exception as e:
                    logger.error(f"❌ In-process task error: {str(e)}")

        # Write a stage0 file and put it into the cross-process queue.
        if chunk_results:
            chunk_results_sorted = sorted(chunk_results, key=lambda x: x[0])
            with tempfile.NamedTemporaryFile(
                mode="w+", delete=False, prefix=f"stage0_chunk{chunk_idx:03d}_", encoding="utf-8", dir=temp_dir
            ) as f:
                stage0_file = f.name
                for token_len, name in chunk_results_sorted:
                    f.write(f"{name}:{token_len}\n")

            stage0_queue.put(stage0_file)  # Put into the cross-process queue.
            # logger.info(f"Process {multiprocessing.current_process().name} completed chunk {chunk_idx}, generated {line_count} rows")
            # logger.info(f"Process {multiprocessing.current_process().name} completed chunk {chunk_idx}, valid samples {processed_count}/{n_samples}")
            proc_status = "🟢" if processed_count == n_samples else "🟡"
            logger.info(
                f"{proc_status} Process {multiprocessing.current_process().name} completed chunk {chunk_idx}; valid samples {processed_count}/{n_samples}"
            )

            # Atomically accumulate the total sample count across processes.
            with global_total_counter.get_lock():
                global_total_counter.value += processed_count

            return stage0_file  # Return generated file path for later cleanup.

    except Exception as e:
        logger.error(f"❌ Process {multiprocessing.current_process().name} failed: {str(e)}")
    finally:
        if processor:
            del processor
    return None


###
def main():
    global global_total_counter  # Reference the global counter.
    processor = None  # Model processor instance.
    stage0_files = []  # All stage0 files, used for validation and cleanup.
    stage1_files = []  # All stage1 files, used for final merge.

    try:
        logger.info("💡 --------------Starting data processing pipeline--------------")

        # 1. Find source sample names and write them to a temporary file.
        # base_names = find_paired_files(DEFAULT_DIRECTORY)    # DEFAULT_DIRECTORY stores original jpg/json files.
        base_names = find_valid_json(DEFAULT_DIRECTORY)
        total_original = len(base_names)  # Total source samples.
        logger.info(f"👉 Found {total_original} source sample files")
        if total_original == 0:
            logger.warning("⚠️ No source samples found; exiting")
            return
        # Write source sample names for later chunked reads.
        write_base_names_to_file(base_names, OUTPUT_FILE)

        # 2. Initialize the cross-process queue used to pass stage0 paths to the merge thread.
        manager = Manager()  # Manager is required for cross-process shared queues.
        stage0_queue = manager.Queue()
        stop_event = manager.Event()  # Cross-process stop signal.

        # Cross-process counter for total processed samples.
        global_total_counter = Value("i", 0)  # 'i' means integer.

        # 3. Start the stage1 merge thread as a daemon.
        stage1_thread = threading.Thread(
            target=stage1_merger, args=(stage0_queue, STAGE1_CHUNK, stage1_files, stop_event), daemon=True
        )
        stage1_thread.start()
        logger.info("💡 Stage1 merge thread started")

        # 4. Process data and generate sorted stage0 files for each chunk.
        # n_workers = 96 #get_adaptive_workers()

        # 4.1 Read all chunks before assigning them to processes.
        # chunk_size controls the large chunk size handled by each process.
        all_chunks = list(read_lines_in_chunks(OUTPUT_FILE, chunk_size))
        total_chunks = len(all_chunks)
        n_processes = min(multiprocessing.cpu_count(), total_chunks)
        logger.info(f"👉 Split into {total_chunks} chunks; starting {n_processes} worker processes")

        # 4.2 Prepare process-pool arguments.
        process_args = [
            (
                idx + 1,  # Chunk index.
                chunk,  # Chunk data.
                CKPT_DIR,  # Model path.
                MIN_PIXELS,
                MAX_PIXELS,
                stage0_queue,  # Cross-process queue.
            )
            for idx, chunk in enumerate(all_chunks)
        ]

        # 4.3 Start the process pool.
        with Pool(processes=n_processes) as process_pool:
            # Process all large chunks in parallel.
            # stage0_files = process_pool.map(process_chunk, process_args)
            result = process_pool.map_async(process_chunk, process_args)
            try:
                stage0_files = result.get(timeout=TIME_OUT)  # Timeout setting.
            except multiprocessing.TimeoutError:
                logger.error("❌ Some worker processes timed out; terminating the pool")
                process_pool.terminate()

        # Filter empty results.
        stage0_files = [f for f in stage0_files if f is not None]
        logger.info(f"✅ All worker processes completed; generated {len(stage0_files)} stage0 files")
        # Count processed data.
        total_processed = global_total_counter.value  # Read total processed samples from the global counter.
        logger.info(f"👉 Source samples: {total_original}, valid processed samples: {total_processed}")

        # Validate data completeness.
        if total_processed != total_original:
            logger.warning(
                f"❌ Data is incomplete: source {total_original}, processed {total_processed}, missing {total_original - total_processed}"
            )
        else:
            logger.info("✅ Data completeness check passed; every sample was processed")

        # 5. Wait for all generated files to be merged.
        # Wait until every stage0 queue item has been consumed.
        logger.info("🔄 Waiting for the stage0 queue to finish...")
        stage0_queue.join()  # Block until all stage0 files have been consumed.
        logger.info("💡 All stage0 queue files have been processed")

        # Send a stop signal so the stage1 thread force-merges remaining files.
        logger.info("💡 Notifying stage1 thread to stop and process remaining files...")
        stop_event.set()

        # Wait up to 60 seconds so large file merges can finish.
        timeout_counter = 0
        while stage1_thread.is_alive() and timeout_counter < 60:
            logger.debug(f"🔄 Waiting for stage1 thread completion ({timeout_counter}/60 seconds)")
            threading.Event().wait(1)  # Wait one second before retrying.
            timeout_counter += 1

        if stage1_thread.is_alive():
            logger.warning(
                "⚠️ Stage1 thread did not exit before timeout; remaining files were still force-merge attempted"
            )
        else:
            logger.info("💡 Stage1 thread exited normally")

        # Validate stage1 file count. Each STAGE1_CHUNK stage0 files produce one stage1 file; partial batches count too.
        expected_stage1_count = (len(stage0_files) + STAGE1_CHUNK - 1) // STAGE1_CHUNK
        if len(stage1_files) != expected_stage1_count:
            logger.warning(
                f"⚠️ ℹ️ Stage1 file count mismatch: expected {expected_stage1_count}, got {len(stage1_files)}"
            )
        else:
            logger.info(f"✅ Stage1 file count check passed: {len(stage1_files)} files")

        # 6. Merge all stage1 files into the final token-info file.
        if not stage1_files:
            logger.warning("⚠️ No stage1 files were generated; check intermediate processing")
            return

        # Count total stage1 rows.
        stage1_total = sum(count_lines(f) for f in stage1_files)
        logger.info(f"ℹ️ Starting final merge: {len(stage1_files)} stage1 files, {stage1_total} total rows")

        # Merge into the final file.
        final_path, final_lines = merge_files_by_token(stage1_files, TOKEN_INFO_FILE)

        if final_path and final_lines > 0:
            logger.info(f"✅ Final result file generated: {TOKEN_INFO_FILE}, containing {final_lines} rows")
            # Validate total row count.
            if final_lines != total_processed:
                logger.error(f"❌ Row count mismatch: processed {total_processed}, final file {final_lines}")
            else:
                logger.info("✅💡 Row count check passed; all data was written to the final file")
        else:
            logger.error("❌ Final file merge failed")

        # Verify again after final merge.
        if os.path.exists(TOKEN_INFO_FILE):
            final_count = count_lines(TOKEN_INFO_FILE)
            logger.info(f"ℹ️ Final result file contains {final_count} rows")
            if final_count != total_processed:
                logger.error(f"❌ Final file is incomplete: processed {total_processed}, final file {final_count}")
            else:
                logger.info("✅ Final file completeness check passed")

    except Exception as e:
        logger.error(f"❌ Main pipeline error: {str(e)}", exc_info=True)
    finally:
        # Clean up resources.
        if processor:
            del processor

        # Ensure the stop signal is set.
        stop_event.set()

        if stage1_thread and stage1_thread.is_alive():
            stage1_thread.join(timeout=2)

        # Wait for final file writes to settle.
        threading.Event().wait(2)

        # Clean temporary files while preserving the final output.
        all_temp_files = stage0_files + stage1_files
        for fpath in all_temp_files:
            if fpath != str(TOKEN_INFO_FILE) and os.path.exists(fpath):
                try:
                    os.remove(fpath)
                    logger.debug(f"Cleaned temporary file: {os.path.basename(fpath)}")
                except Exception as e:
                    logger.warning(f"Failed to clean temporary file {os.path.basename(fpath)}: {str(e)}")

        logger.info("Program finished")


if __name__ == "__main__":
    main()
