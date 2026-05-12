import json
import logging
import os
import shutil
from concurrent.futures import ThreadPoolExecutor
from multiprocessing import Manager, Process, cpu_count

from tqdm import tqdm


# Assign independent numbering to __img--output samples.

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger()


# ---------- Utilities ----------
def extract_filename_without_ext(image_path: str) -> str:
    return os.path.splitext(os.path.basename(image_path))[0]


# ---------------------------- patch 1 ----------------------------
# Thread-safe duplicate-name counter.


def _unique_filename(name: str, name_counter, name_lock) -> str:
    base, ext = os.path.splitext(name)
    with name_lock:
        # Use get() to avoid KeyError.
        cnt = name_counter.get(name, 0)
        name_counter[name] = cnt + 1
        if cnt == 0:
            return name
        return f"{base}_{cnt}{ext}"


# -----------------------------------------------------------------


# ---------- Single-item processing ----------
def _process_single_item(args):
    """
    Thread-level worker for one sample.
    Arguments are packed into a tuple for ThreadPoolExecutor.
    """
    # item, base_dir, output_dir, rel_img_path, no_img_indices = args
    (item, base_dir, output_dir, rel_img_path, no_img_indices, name_counter, name_lock) = args  # patch 6

    # ---------- Normalize original image paths ----------
    original_image_paths = []
    if item.get("images"):
        original_image_paths = item["images"] if isinstance(item["images"], list) else [item["images"]]
    else:
        item["images"] = []

    if rel_img_path:
        original_image_paths = [
            os.path.normpath(os.path.join(base_dir, rel_img_path, p)) for p in original_image_paths
        ]
    else:
        original_image_paths = [os.path.normpath(os.path.join(base_dir, p)) for p in original_image_paths]

    # ---------- Rename and copy images consistently ----------
    new_image_basenames = []
    for src_path in original_image_paths:
        if not os.path.exists(src_path):
            logger.warning(f"Image does not exist: {src_path}")
            continue
        old_name = os.path.basename(src_path)
        # new_name = _unique_filename(old_name)          # May rename duplicates.
        new_name = _unique_filename(old_name, name_counter, name_lock)
        new_image_basenames.append(new_name)

        dst_path = os.path.join(output_dir, new_name)
        try:
            shutil.copy2(src_path, dst_path)
        except Exception as e:
            logger.error(f"Failed to copy image: {src_path} -> {dst_path} | {e}")

    # Keep the JSON images field in sync.
    item["images"] = new_image_basenames

    # --------------patch 001----------
    # Return None when none of the referenced images exists.
    if original_image_paths and not new_image_basenames:
        logger.info(f"Skipping item without valid images: {item.get('id', item['_orig_index'])}")
        return None
    # --------------patch 001 end----------

    # ---------- Generate JSON filename ----------
    if new_image_basenames:
        json_name_root = os.path.splitext(new_image_basenames[0])[0]
    else:
        idx_in_no_img = no_img_indices.index(item["_orig_index"])
        json_name_root = f"__img--output_{idx_in_no_img:08d}"

    # json_name = _unique_filename(json_name_root + ".json")
    json_name = _unique_filename(json_name_root + ".json", name_counter, name_lock)
    json_path = os.path.join(output_dir, json_name)
    try:
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(item, f, indent=4, ensure_ascii=False)
    except Exception as e:
        logger.error(f"Failed to write JSON: {json_path} | {e}")

    return os.path.splitext(json_name)[0]


# ---------- Process-level worker ----------
def _worker_process(
    job_queue, result_list, base_dir, output_dir, rel_img_path, m, no_img_indices, name_counter, name_lock
):  # <-- patch4
    while True:
        try:
            chunk = job_queue.get_nowait()
        except Exception:
            break

        logger.info(f"Process {os.getpid()} is handling a chunk with {len(chunk)} items")
        # Build worker arguments.
        arg_list = [
            (item, base_dir, output_dir, rel_img_path, no_img_indices, name_counter, name_lock) for item in chunk
        ]

        valid_names = []
        with ThreadPoolExecutor(max_workers=m) as pool:
            for fut in tqdm(
                pool.map(_process_single_item, arg_list), total=len(arg_list), desc=f"PID-{os.getpid()}", leave=False
            ):
                if fut is not None:  # Filter out skipped items.
                    valid_names.append(fut)
        result_list.extend(valid_names)


# ---------- Main entry ----------
def split_json_file(fin_name, rel_img_path=None, *, chunk_dim=1000, m=8):
    # Read source data.
    try:
        with open(fin_name, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        logger.error(f"Failed to read JSON: {e}")
        return set()

    if not isinstance(data, list):
        logger.error("The JSON root node is not an array")
        return set()

    # Attach original indices and collect image-free samples.
    for i, item in enumerate(data):
        item["_orig_index"] = i
    no_img_indices = [i for i, item in enumerate(data) if not item.get("images")]

    # Prepare directories.
    base_dir = os.path.dirname(os.path.abspath(fin_name))
    output_dir = os.path.join(base_dir, "split_json_files")
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir, exist_ok=True)

    # Split data into chunks.
    total = len(data)
    num_chunks = (total + chunk_dim - 1) // chunk_dim
    chunks = [data[i * chunk_dim : (i + 1) * chunk_dim] for i in range(num_chunks)]

    max_workers = min(num_chunks, cpu_count())
    logger.info(
        f"Total {total} items split into {num_chunks} chunks; starting {max_workers} processes with {m} threads each"
    )

    with Manager() as manager:
        job_queue = manager.Queue()
        for c in chunks:
            job_queue.put(c)

        result_list = manager.list()
        name_counter = manager.dict()  # Shared duplicate-name counter.
        name_lock = manager.Lock()  # Shared lock for duplicate-name updates.

        processes = [
            Process(
                target=_worker_process,
                args=(
                    job_queue,
                    result_list,
                    base_dir,
                    output_dir,
                    rel_img_path,
                    m,
                    no_img_indices,
                    name_counter,
                    name_lock,
                ),
            )
            for _ in range(max_workers)
        ]
        for p in processes:
            p.start()
        for p in processes:
            p.join()

        all_valid_names = set(result_list)

    logger.info("All processing complete")
    return all_valid_names


# ---------- Script entry ----------
if __name__ == "__main__":
    # f_json = "/vlm/data/llava_next_500/sampled_data.json"
    f_json = "/data_1/llava_next_raw_full/megatron_format_780k.json"
    rel_img = "images"
    res = split_json_file(f_json, "images", chunk_dim=2000, m=8)
    print(f"Generated {len(res)} files")
