import json
import os
import shutil
import logging
from concurrent.futures import ThreadPoolExecutor
from multiprocessing import Manager, cpu_count, Process
from tqdm import tqdm

# 1）__img--output 独立进行编号

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger()

# ---------- 工具 ----------
def extract_filename_without_ext(image_path: str) -> str:
    return os.path.splitext(os.path.basename(image_path))[0]



# ---------------------------- patch 1 ----------------------------
# 新增：线程安全重名计数器
from collections import defaultdict
import re
import threading

def _unique_filename(name: str, name_counter, name_lock) -> str:
    base, ext = os.path.splitext(name)
    with name_lock:
        # 用 get 避免 KeyError
        cnt = name_counter.get(name, 0)
        name_counter[name] = cnt + 1
        if cnt == 0:
            return name
        return f"{base}_{cnt}{ext}"

# -----------------------------------------------------------------



# ---------- 单元素处理 ----------
def _process_single_item(args):
    """
    线程级：处理单条数据
    参数打包成元组，便于 ThreadPoolExecutor
    """
    # item, base_dir, output_dir, rel_img_path, no_img_indices = args
    (item, base_dir, output_dir, rel_img_path, no_img_indices,
     name_counter, name_lock) = args   # patch 6

    # ---------- 整理原始图片路径 ----------
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

    # ---------- 统一重命名并拷贝图片 ----------
    new_image_basenames = []
    for src_path in original_image_paths:
        if not os.path.exists(src_path):
            logger.warning(f"图片不存在：{src_path}")
            continue
        old_name = os.path.basename(src_path)
        # new_name = _unique_filename(old_name)          # 可能改名
        new_name = _unique_filename(old_name, name_counter, name_lock)
        new_image_basenames.append(new_name)

        dst_path = os.path.join(output_dir, new_name)
        try:
            shutil.copy2(src_path, dst_path)
        except Exception as e:
            logger.error(f"拷贝图片失败: {src_path} -> {dst_path} | {e}")

    # 同步更新 JSON 里的 images
    item["images"] = new_image_basenames


    #--------------patch 001----------
    # ✨ 新增：所有图片都不存在，直接返回 None
    if original_image_paths and not new_image_basenames:
        logger.info(f"跳过无有效图片的元素：{item.get('id', item['_orig_index'])}")
        return None
    #--------------patch 001 end----------
    
    # ---------- 生成 json 文件名 ----------
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
        logger.error(f"写 JSON 失败: {json_path} | {e}")

    return os.path.splitext(json_name)[0]

# ---------- 进程级 ----------
def _worker_process(job_queue, result_list, base_dir, output_dir,
                    rel_img_path, m, no_img_indices,
                    name_counter, name_lock):   # <-- patch4
    while True:
        try:
            chunk = job_queue.get_nowait()
        except Exception:
            break

        logger.info(f"进程 {os.getpid()} 处理 chunk（{len(chunk)} 条）")
        # 构造参数列表
        arg_list = [(item, base_dir, output_dir, rel_img_path, no_img_indices, name_counter, name_lock)
                    for item in chunk]

        valid_names = []
        with ThreadPoolExecutor(max_workers=m) as pool:
            for fut in tqdm(pool.map(_process_single_item, arg_list),
                            total=len(arg_list),
                            desc=f"PID-{os.getpid()}",
                            leave=False):
                if fut is not None:          # ✨ 过滤掉 None。patch 002
                    valid_names.append(fut)
        result_list.extend(valid_names)

# ---------- 主入口 ----------
def split_json_file(fin_name, rel_img_path=None, *, chunk_dim=1000, m=8):
    # 读数据
    try:
        with open(fin_name, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except Exception as e:
        logger.error(f"读取 JSON 失败: {e}")
        return set()

    if not isinstance(data, list):
        logger.error("JSON 根节点不是数组")
        return set()

    # 打原始索引 & 收集无图索引
    for i, item in enumerate(data):
        item['_orig_index'] = i
    no_img_indices = [i for i, item in enumerate(data) if not item.get("images")]

    # 目录准备
    base_dir = os.path.dirname(os.path.abspath(fin_name))
    output_dir = os.path.join(base_dir, "split_json_files")
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir, exist_ok=True)

    # 分块
    total = len(data)
    num_chunks = (total + chunk_dim - 1) // chunk_dim
    chunks = [data[i * chunk_dim:(i + 1) * chunk_dim] for i in range(num_chunks)]

    max_workers = min(num_chunks, cpu_count())
    logger.info(f"共 {total} 条，切成 {num_chunks} 块，启动 {max_workers} 进程，每进程 {m} 线程")

    with Manager() as manager:
        job_queue = manager.Queue()
        for c in chunks:
            job_queue.put(c)

        result_list = manager.list()
        name_counter = manager.dict()           # <-- 新增 patch2
        name_lock    = manager.Lock()           # <-- 新增 patch2

        processes = [
            Process(target=_worker_process,
                    args=(job_queue, result_list, base_dir,
                          output_dir, rel_img_path, m, no_img_indices,
                          name_counter, name_lock))   # <-- 新增 patch3
            for _ in range(max_workers)
        ]
        for p in processes:
            p.start()
        for p in processes:
            p.join()

        all_valid_names = set(result_list)

    logger.info("全部处理完成")
    return all_valid_names

# ---------- 脚本 ----------
if __name__ == "__main__":
    # f_json = "/vlm/data/llava_next_500/sampled_data.json"
    f_json = "/data_1/llava_next_raw_full/megatron_format_780k.json"
    rel_img = "images"
    res = split_json_file(
        f_json,
        "images",
        chunk_dim=2000,
        m=8
    )
    print(f"共生成 {len(res)} 个文件")
