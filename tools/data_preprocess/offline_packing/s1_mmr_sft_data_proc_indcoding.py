import json
import os
import shutil
import logging
from concurrent.futures import ThreadPoolExecutor
from multiprocessing import Manager, cpu_count, Process
from tqdm import tqdm

# 1）Assign unique numeric IDs to the __img--output for QA data, separate from the numbering for VQA data.

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger()

# ---------- tool ----------
def extract_filename_without_ext(image_path: str) -> str:
    return os.path.splitext(os.path.basename(image_path))[0]



# ---------------------------- patch 1 ----------------------------
# feat: Add a thread-safe counter for duplicate names
from collections import defaultdict
import re
import threading

def _unique_filename(name: str, name_counter, name_lock) -> str:
    base, ext = os.path.splitext(name)
    with name_lock:
        # Use .get() to avoid KeyError.
        cnt = name_counter.get(name, 0)
        name_counter[name] = cnt + 1
        if cnt == 0:
            return name
        return f"{base}_{cnt}{ext}"

# -----------------------------------------------------------------



# ---------- processing single data item ----------
def _process_single_item(args):
    """
    Thread-level: Process a single data item.
    Parameters are packed into a tuple for easy submission to ThreadPoolExecutor.
    """
    # item, base_dir, output_dir, rel_img_path, no_img_indices = args
    (item, base_dir, output_dir, rel_img_path, no_img_indices,
     name_counter, name_lock) = args   # patch 6

    # ---------- Organize the original image paths. ----------
    original_image_paths = []
    if item.get("images"):
        original_image_paths = item["images"] if isinstance(item["images"], list) else [item["images"]]
    else:
        item["images"] = []

    if rel_img_path:
        original_image_paths = [
            os.path.normpath(os.path.join(base_dir, rel_img_path, p))
            for p in original_image_paths
        ]
    else:
        original_image_paths = [
            os.path.normpath(os.path.join(base_dir, p))
            for p in original_image_paths
        ]

    # ---------- This script renames all images to a consistent format and then copies them to the output folder. ----------
    new_image_basenames = []
    for src_path in original_image_paths:
        if not os.path.exists(src_path):
            logger.warning(f"IMG not found: {src_path}")
            continue
        old_name = os.path.basename(src_path)
        # new_name = _unique_filename(old_name)      
        new_name = _unique_filename(old_name, name_counter, name_lock)
        new_image_basenames.append(new_name)

        dst_path = os.path.join(output_dir, new_name)
        try:
            shutil.copy2(src_path, dst_path)
        except Exception as e:
            logger.error(f"Image copy failed: {src_path} -> {dst_path} | {e}")

    # ---------- Update the JSON with new image basenames ----------
    item["images"] = new_image_basenames


    #--------------patch 001----------
    # ✨ New：If no images exist, return None directly.
    if original_image_paths and not new_image_basenames:
        logger.info(f"Skip item with no valid images: {item.get('id', item['_orig_index'])}")
        return None
    #--------------patch 001 end----------
    
    # ---------- Generate a filename for the JSON file ----------
    if new_image_basenames:
        json_name_root = os.path.splitext(new_image_basenames[0])[0]
    else:
        idx_in_no_img = no_img_indices.index(item['_orig_index'])
        json_name_root = f"__img--output_{idx_in_no_img:08d}"

    # json_name = _unique_filename(json_name_root + ".json")
    json_name = _unique_filename(json_name_root + ".json", name_counter, name_lock)
    json_path = os.path.join(output_dir, json_name)
    try:
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(item, f, indent=4, ensure_ascii=False)
    except Exception as e:
        logger.error(f"JSON write failed: {json_path} | {e}")

    return os.path.splitext(json_name)[0]

# ---------- Process-level ----------
def _worker_process(job_queue, result_list, base_dir, output_dir,
                    rel_img_path, m, no_img_indices,
                    name_counter, name_lock):   # <-- patch4
    while True:
        try:
            chunk = job_queue.get_nowait()
        except:
            break

        logger.info(f"[PID {os.getpid()}] Processing chunk with {len(chunk)} items.")
        # Construct the argument list
        arg_list = [(item, base_dir, output_dir, rel_img_path, no_img_indices, name_counter, name_lock)
                    for item in chunk]

        valid_names = []
        with ThreadPoolExecutor(max_workers=m) as pool:
            for fut in tqdm(pool.map(_process_single_item, arg_list),
                            total=len(arg_list),
                            desc=f"PID-{os.getpid()}",
                            leave=False):
                if fut is not None:          # ✨ Filter out None items. patch 002
                    valid_names.append(fut)
        result_list.extend(valid_names)

# ---------- Main entry ----------
def split_json_file(fin_name, rel_img_path=None, *, chunk_dim=1000, m=8):
    # read json
    try:
        with open(fin_name, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except Exception as e:
        logger.error(f"JSON read failed: {e}")
        return set()

    if not isinstance(data, list):
        logger.error("Expected JSON root to be an array")
        return set()

    # Log original indices & gather indices missing images
    for i, item in enumerate(data):
        item['_orig_index'] = i
    no_img_indices = [i for i, item in enumerate(data) if not item.get("images")]

    # Prepare directories
    base_dir = os.path.dirname(os.path.abspath(fin_name))
    output_dir = os.path.join(base_dir, "split_json_files")
    # output_dir = os.path.join(base_dir, "split_json_fs")
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir, exist_ok=True)

    # Split data into chunks
    total = len(data)
    num_chunks = (total + chunk_dim - 1) // chunk_dim
    chunks = [data[i * chunk_dim:(i + 1) * chunk_dim] for i in range(num_chunks)]

    max_workers = min(num_chunks, cpu_count())
    logger.info(f"[JOB] Total: {total} | Chunks: {num_chunks} | Procs: {max_workers} | Threads/Proc: {m}")


    with Manager() as manager:
        job_queue = manager.Queue()
        for c in chunks:
            job_queue.put(c)

        result_list = manager.list()
        name_counter = manager.dict()           # <-- new patch2
        name_lock    = manager.Lock()           # <-- new patch2

        processes = [
            Process(target=_worker_process,
                    args=(job_queue, result_list, base_dir,
                          output_dir, rel_img_path, m, no_img_indices,
                          name_counter, name_lock))   # <-- new patch3
            for _ in range(max_workers)
        ]
        for p in processes:
            p.start()
        for p in processes:
            p.join()

        all_valid_names = set(result_list)

    logger.info(f"[JOB] All {total} items processed. Valid JSON files: {len(all_valid_names)}")
    return all_valid_names

# ---------- script ----------
if __name__ == "__main__":
    # f_json = "/vlm/data/llava_next_500/sampled_data.json"
    f_json = "/data_1/llava_next_raw_full/megatron_format_780k.json"
    rel_img = "images"
    res = split_json_file(
        f_json,
        rel_img,
        chunk_dim=2000,
        m=8
    )
    print(f"Generated {len(res)} files.")