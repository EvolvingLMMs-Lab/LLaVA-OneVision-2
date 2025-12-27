# ### 所有代码放到一起，只运行这一块就可以
# Step1: 
# python -u s2_prepare_rawsamples-emova.py 2>&1 | tee s2_proc.log
# python -u s2_prepare_rawsamples-llava_vqa.py 2>&1 | tee s2_proc_llava.log
# python -u s2_prepare_rawsamples-vqa_1000k.py 2>&1 | tee ./logs/s2_proc_vqa_1000k.log
# python -u s2_prepare_rawsamples-vqa_1000k-16k.py 2>&1 | tee ./logs/s2_proc_vqa_1000k-16k.log
# python -u s2_prepare_rawsamples-vqa_5500k-16k.py 2>&1 | tee ./logs/s2_proc_vqa_5500k-16k.log
# python -u s2_prepare_rawsamples-vqa_5500k-16k-fast.py 2>&1 | tee ./logs/s2_proc_vqa_5500k-16k-fast.log
# python -u s2_prepare_rawsamples-vqa_pretrain_5M-8k-fast.py 2>&1 | tee ./logs/s2_proc_vqa_pretrain_5M-8k-fast.log

# python -u s2_prepare_rawsamples-mr_sft_780k-8k-fast.py 2>&1 | tee ./logs/s2_prepare_rawsamples-mr_sft_780k-8k-fast.log

import bisect
import os
import re
import json
import sys
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

# ### 参数配置
# target_directory = "/workspace/test/packing"   # 最终数据存放的位置

current_file = Path(__file__).resolve()
target_directory = current_file.parent
newDir = "raw_packing_data_mr_sft_780k-8k-fast"                   # 转 webdataset 之前数据的存放位置 (jpg, json)
SRC_DIR_IMGS = "/data_1/llava_next_raw_full/split_json_files"   # 原始 img  数据的存放位置
SRC_DIR_JSONS = "/data_1/llava_next_raw_full/split_json_files"   # 原始 json 数据的存放位置
SRC_DST_EXTENSIONS = ("jpg", "json")
f_toklens_originalsample = os.path.join(target_directory, "token_info_MR_sft_780k_8k.txt")
PACKED_LENGTH = 8192
dst_dir = os.path.join(target_directory,newDir)
MAX_WORKERS = 96  # 线程池大小（根据CPU核心数和IO性能调整）


"""
task_type 的设置：
    sft：VQA 格式的 pretrain
    pretrain：caption 格式的 pretrain
    bmr：混合数据集多轮对话格式的 sft
"""
task_type = "bmr"



f_TEST=False     # test 示例输出（仅做测试用：生成少量数据）
n_packed_samples=400  # 测试用，输出几条打包后的数据

# PROMPTS = # Creating a list of the provided English prompts
PROMPTS = [
    "What about this picture?",
    "Please provide a vivid description of the image.",
    "Please Depict the image in words."
    "Could you please transcribe thr image into a descriptive paragraph?"
    "What is the content of this figure?",
    "What do you see here?",
    "Tell me about this image.",
    "What's going on in this artwork?",
    "What is depicted in this painting?",
    "What is the subject matter here?",
    "What can you make out in this picture?",
    "What's the main thing shown in this image?",
    "What's the gist of this artwork?",
    "What's the essence of this figure?",
    "What's the general idea here?",
    "What does this image show?",
    "What's the core element in this painting?",
    "What's the overview of this scene?",
    "What's the primary focus of this artwork?",
    "What's the fundamental subject matter?",
    "What's the general view presented?",
    "What's the main impression given by this picture?",
    "What's the central theme shown?",
    "What's the overall presentation here?",
    "What's the key element you notice?",
    "What's the fundamental concept in this image?",
    "What's the overall content?",
    "What's the main thing you get from this?",
    "What's the general subject?",
    "What's the core idea conveyed?",
    "What's the basic representation?",
    "What's the main point of this figure?"
]

import random

def find_long_file_pairs(directory, length_threshold=62):
    """
    找出长文件（img,json）对中的图像文件，返回带有图像扩展名的完整文件名
    
    参数:
        directory: 要检查的目录路径
        length_threshold: 文件名长度阈值，默认62
        
    返回:
        符合条件的图像文件名（带扩展名）列表
    """
    import os
    from collections import defaultdict
    # 存储所有文件的文件名部分及其对应的完整文件名
    file_parts = defaultdict(list)
    image_extensions = ('.jpg', '.jpeg', '.png', '.gif', '.bmp')
    
    try:
        # 遍历目录中的所有文件，按文件名部分分组
        for filename in os.listdir(directory):
            name_part, ext = os.path.splitext(filename)
            ext = ext.lower()
            # 只关注图片和json文件
            if ext in ('.json',) + image_extensions:
                file_parts[name_part].append(filename)
                
        # 找出符合条件的图像文件
        long_image_files = []
        for name_part, filenames in file_parts.items():
            # 检查文件名长度和文件对完整性
            if (len(name_part) > length_threshold and 
                any(f.endswith('.json') for f in filenames) and 
                any(f.lower().endswith(image_extensions) for f in filenames)):
                
                # 只添加图像文件
                for filename in filenames:
                    if filename.lower().endswith(image_extensions):
                        long_image_files.append(filename)
                
        return long_image_files
        
    except FileNotFoundError:
        # print(f"错误：目录 '{directory}' 不存在")
        return []
    except PermissionError:
        # print(f"错误：没有访问目录 '{directory}' 的权限")
        return []
    except Exception as e:
        # print(f"处理目录时发生错误：{str(e)}")
        return []


# res_long_img_names = find_long_file_pairs(SRC_DIR_JSONS)

def filter_filenames(filenames, prefix, exclude_suffix):
    """
    筛选出以指定前缀开头且不以指定后缀结尾的文件名
    
    参数:
        filenames: 文件名列表
        prefix: 文件名需要包含的前缀（如"james-tissot"）
        exclude_suffix: 需要排除的文件后缀（如"json"）
        
    返回:
        符合条件的文件名列表
    """
    # 转义前缀中的特殊字符，确保正则匹配正确
    escaped_prefix = re.escape(prefix)
    # 构建正则表达式模式
    pattern = rf'^{escaped_prefix}(?!.*\.{exclude_suffix}$).*$'
    
    # 编译正则表达式
    regex = re.compile(pattern)
    
    # 筛选符合条件的文件名
    return [filename for filename in filenames if regex.match(filename)]

def get_random_prompts(prompts, n):
    if n > len(prompts):
        # 允许重复
        return random.choices(prompts, k=n)0
    else:
        # 不允许重复
        return random.sample(prompts, n)

# 全局变量 - 用元组存储（不可变，效率更高）
BASE_NAMES = []  # 初始化为空元组，后续会被替换 (所有在原始数据集中的 sample 名称， 已经按照 tokens 长度排序)

def search_for_fit(numbers: List[int], capacity: int) -> int:
    """Finds the index of largest number that fits into the knapsack with the given capacity."""
    index = bisect.bisect(numbers, capacity)
    return -1 if index == 0 else (index - 1)

def greedy_knapsack(numbers: List[int], capacity: int) -> Tuple[List[List[int]], List[List[int]]]:
    r"""Implement efficient greedy algorithm with binary search for the knapsack problem.
    参数
    ----
    numbers : List[int]
        物品大小列表，可以为任意顺序（这里是升序输入进来的）
    capacity : int
        背包容量

    返回
    ----
    Tuple[List[List[int]], List[List[int]]]
        第一个列表：每个背包里的物品大小
        第二个列表：每个背包里物品对应的原始下标
    
    """
    # 保存原始索引，与输入的numbers一一对应
    indexed_numbers = [(val, idx) for idx, val in enumerate(numbers)]
    # 由于输入已排序，直接使用即可（保持与原逻辑一致的处理方式）
    knapsacks = []
    index_knapsacks = []
    iii = int(0)
    while indexed_numbers:
        current_knapsack = []
        current_indices = []
        remaining_capacity = capacity

        while True:
            # 提取当前数值列表用于查找（保持升序）
            current_values = [val for val, idx in indexed_numbers]
            index = search_for_fit(current_values, remaining_capacity)
            if index == -1:
                break  # 没有可放入当前背包的物品了

            # 取出找到的物品及其原始索引
            val, idx = indexed_numbers.pop(index)
            remaining_capacity -= val
            current_knapsack.append(val)
            current_indices.append(idx)

        if iii%1000==0:
            print(f"---------第{iii}个pack----------")
            print(f"{current_knapsack}--->{sum(current_knapsack)}")
            print(current_indices)
            print(f"\n")
        iii+=1
        knapsacks.append(tuple(current_knapsack))
        index_knapsacks.append(tuple(current_indices))

    return tuple(knapsacks), tuple(index_knapsacks)   # 使用了 tuple

def extract_content(json_file):
    try:
        # 打开并加载JSON文件
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)

        if task_type=="sft":
            try:
                user_content = next(msg["content"] for msg in data["messages"] if msg["role"] == "assistant")
                return user_content
            except Exception as e:
                pass
        # 提取content内容
        # 假设captions数组至少有一个元素
        elif task_type=="pretrain":
            if data.get('captions') and len(data['captions']) > 0:
                return data['captions'][0].get('content', "")
            else:
                assert 0, "未找到有效的caption内容"
                # return "未找到有效的caption内容"
            
    except FileNotFoundError:
        return f"错误：文件 {json_file} 不存在"
    except json.JSONDecodeError:
        return f"错误：文件 {json_file} 不是有效的JSON格式"
    except Exception as e:
        return f"提取过程中发生错误：{str(e)}"

def extract_prompt(json_file):
    try:
        # 打开并加载JSON文件
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # 提取助手回复
        assistant_content = next(msg["content"] for msg in data["messages"] if msg["role"] == "user")
        return assistant_content
        
        # # 提取图片路径（可选）
        # image_path = data["images"][0] if data["images"] else None
            
    except FileNotFoundError:
        return f"错误：文件 {json_file} 不存在"
    except json.JSONDecodeError:
        return f"错误：文件 {json_file} 不是有效的JSON格式"
    except Exception as e:
        return f"提取过程中发生错误：{str(e)}"    

def extract_img_prompt_content(json_file: str) -> Tuple[List[str], List[str], List[str]]:
    try:
        # 打开并加载 JSON 文件
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # 1)images
        imgs = data.get("images", [])
        if not imgs:
            images = []
        else:
            images = [os.path.join(SRC_DIR_IMGS,imgs[0])]

        messages = data.get("messages", [])

        assistant_contents = [
            msg["content"]
            for msg in messages
            if isinstance(msg, dict) and msg.get("role") == "assistant" and "content" in msg
        ]

        user_contents = [
            msg["content"]
            for msg in messages
            if isinstance(msg, dict) and msg.get("role") == "user" and "content" in msg
        ]

        return images, user_contents, assistant_contents
        
    except FileNotFoundError:
        return f"错误：文件 {json_file} 不存在"
    except json.JSONDecodeError:
        return f"错误：文件 {json_file} 不是有效的JSON格式"
    except Exception as e:
        return f"提取过程中发生错误：{str(e)}"

def prepare_dirs(target_dir, new_dir):
    os.chdir(target_dir)
    print(f"--------change to directory {target_dir}--------")
    # 创建新目录
    if not os.path.exists(new_dir):
        os.makedirs(new_dir)
        print(f"Directory '{new_dir}' created.")
    else:
        print(f"Directory '{new_dir}' already exists.")


def dataset_tokinfo_generator(f_name):
    """
    数据集token信息生成器，逐行读取并解析文件内容
    
    参数:
        f_name (str): 包含token信息的文件路径
        
    生成:
        tuple: (base_name, token_len) - 解析后的基础文件名和token长度
    """
    try:
        with open(f_name, 'r', encoding='utf-8') as f:
            for line in f:
                # 跳过空行
                stripped_line = line.strip()
                if not stripped_line:
                    continue
                    
                # 按冒号分割并验证格式
                parts = stripped_line.split(':')
                if len(parts) == 2:
                    base_name = parts[0].strip()
                    token_len_str = parts[1].strip()
                    
                    try:
                        token_len = int(token_len_str)
                        yield (base_name, token_len)
                    except ValueError:
                        print(
                            f"警告: 无法将 '{token_len_str}' 转换为整数，已跳过该行",
                            file=sys.stderr
                        )
                        continue
                        
    except FileNotFoundError:
        print(f"错误: 文件 '{f_name}' 不存在", file=sys.stderr)
        return
    except Exception as e:
        print(f"处理文件时发生错误: {str(e)}", file=sys.stderr)
        return


class TokenInfoReader:
    """
    Token信息读取器
    
    支持分批读取、全量读取和断点续读功能，适用于处理包含token信息的文本文件。
    文件格式要求: 每行一条记录，格式为 "base_name: token_len"
    """
    
    def __init__(self, f_name):
        """
        初始化读取器
        
        参数:
            f_name (str): 包含token信息的文件路径
        """
        self.f_name = f_name
        self.generator = dataset_tokinfo_generator(f_name)
        self._current_position = 0  # 记录已读取的记录数量

    def read(self, count=None):
        """
        读取记录
        
        参数:
            count (int, optional): 要读取的记录数量，默认为None（读取全部剩余记录）
            
        返回:
            tuple: (base_names列表, token_lens列表, 实际读取数量)
        """
        base_names = []
        token_lens = []
        read_count = 0
        
        # 循环读取直到达到指定数量或文件结束
        while True:
            # 检查是否已达到指定读取数量
            if count is not None and read_count >= count:
                break
                
            try:
                # 从生成器获取下一条记录
                base_name, token_len = next(self.generator)
                base_names.append(base_name)
                token_lens.append(token_len)
                read_count += 1
                self._current_position += 1
                
            except StopIteration:
                # 已读取到文件末尾
                break
        
        return base_names, token_lens, read_count
    
    def get_current_position(self):
        """
        获取当前读取位置
        
        返回:
            int: 已读取的记录总数
        """
        return self._current_position


def process_knapsack(s1, idx_knapsack, dst_dir):
    """
    处理单个 packing 数据
    
    参数:
        s1: 当前处理组的索引
        idx_knapsack: 背包中包含的索引列表
        dst_dir: 目标目录路径
    """
    # global BASE_NAMES
    
    packed_imgs, packed_caps = [], []   # 单个 packed-sample 的构成
    
    # 获取基础文件名
    # packed_b_names = (BASE_NAMES[idx] for idx in idx_knapsack)
    packed_b_names = (idx["name"] for idx in idx_knapsack)
    
    # 构建源文件信息
    if task_type == "pretrain":
        packed_info = (
            (os.path.join(SRC_DIR_IMGS, f"{b_name}.{SRC_DST_EXTENSIONS[0]}"),
             extract_content(os.path.join(SRC_DIR_JSONS, f"{b_name}.{SRC_DST_EXTENSIONS[1]}")))
            for b_name in packed_b_names
        )
    elif task_type == "sft":
        packed_info = (
            (os.path.join(SRC_DIR_IMGS, f"{b_name}.{SRC_DST_EXTENSIONS[0]}"),
             extract_content(os.path.join(SRC_DIR_JSONS, f"{b_name}.{SRC_DST_EXTENSIONS[1]}")),
             extract_prompt(os.path.join(SRC_DIR_JSONS, f"{b_name}.{SRC_DST_EXTENSIONS[1]}")))
            for b_name in packed_b_names
        ) 
    elif task_type == "bmr":
        packed_info = (
            extract_img_prompt_content(os.path.join(SRC_DIR_JSONS, f"{b_name}.{SRC_DST_EXTENSIONS[1]}"))
            for b_name in packed_b_names
        ) 
        
    # 目标JSON文件路径
    json_dst = os.path.join(dst_dir, f"ps_{s1:08d}.{SRC_DST_EXTENSIONS[1]}")
    
    # 处理每张图片和对应的描述
    if task_type=="pretrain":
        for s2, (img_src, cap_src) in enumerate(packed_info):
            # 目标图片路径
            img_name_dst = f"ps_{s1:08d}.img{s2:03d}.{SRC_DST_EXTENSIONS[0]}"
            # img_name_dst = f"img{s2:03d}.{SRC_DST_EXTENSIONS[0]}"    # 看后面具体需求决定使用哪一个
            img_dst = os.path.join(dst_dir, img_name_dst)
            
            # 收集信息
            # packed_imgs.append(img_name_dst)
            packed_imgs.append(f"img{s2:03d}.{SRC_DST_EXTENSIONS[0]}")
            packed_caps.append(cap_src)
            
            # 复制图片
            shutil.copyfile(img_src, img_dst)
        # 此处也可以调用大模型来生成 提问(对于 纯 captioning 数据)
        selected_prompts = get_random_prompts(PROMPTS, len(packed_imgs))
    elif task_type=="sft":
        selected_prompts = []
        for s2, (img_src, cap_src, prompt_src) in enumerate(packed_info):
            # 目标图片路径
            img_name_dst = f"ps_{s1:08d}.img{s2:03d}.{SRC_DST_EXTENSIONS[0]}"
            # img_name_dst = f"img{s2:03d}.{SRC_DST_EXTENSIONS[0]}"    # 看后面具体需求决定使用哪一个
            img_dst = os.path.join(dst_dir, img_name_dst)
            
            # 收集信息
            # packed_imgs.append(img_name_dst)
            packed_imgs.append(f"img{s2:03d}.{SRC_DST_EXTENSIONS[0]}")
            packed_caps.append(cap_src)
            
            # 复制图片
            shutil.copyfile(img_src, img_dst)

            # prompts
            selected_prompts.append(prompt_src)
        pass
    elif task_type=="bmr":
        selected_prompts = []
        for s2, (img_src, prompt_src, cap_src) in enumerate(packed_info):
            if not img_src:
                packed_imgs.append([])
            else:
                # 目标图片路径
                name, ext = os.path.splitext(img_src[0])
                img_name_dst = f"ps_{s1:08d}.img{s2:03d}{ext}"
                img_dst = os.path.join(dst_dir, img_name_dst)
                
                # 复制图片
                shutil.copyfile(img_src[0], img_dst)
    
                # 收集 image 信息
                packed_imgs.append([f"img{s2:03d}{ext}"])
                # cnt_imgs += 1
                
            # 收集其它信息
            packed_caps.append(cap_src)
            selected_prompts.append(prompt_src)        
        pass
        
    # 生成JSON文件
    json_data = {
        "images": packed_imgs,
        "captions": packed_caps,
        "prompts": selected_prompts
    }
    # print(packed_imgs)
    
    try:
        with open(json_dst, 'w', encoding='utf-8') as f:
            json.dump(json_data, f, indent=4, ensure_ascii=False)
            # json.dump(json_data, f)
    except Exception as e:
        print(f"线程 {threading.current_thread().name} 生成JSON文件 {json_dst} 失败: {str(e)}")
    return s1


if __name__ == "__main__":
    ## 1. 创建工作目录
    print("Step1-----------------已创建工作环境-----------------Start")
    prepare_dirs(target_directory, newDir)
    print("Step1-----------------已创建工作环境-----------------Stop\n\n")
    
    ## 2. 获取原始数据集信息（没有处理之前）
    # 可以用于构建多个 pool，分块 packing（read的参数决定 packing cache size）
    print("Step2-----------------读取原ds的 tokenlen 信息-----------------Start")
    info_reader = TokenInfoReader(f_toklens_originalsample)
    base_names, token_lens, n_count = info_reader.read()
    
    # global BASE_NAMES
    BASE_NAMES=tuple(base_names)
    print(f"已读取{n_count}条数据")
    # print(BASE_NAMES)
    print("Step2-----------------读取原ds的 tokenlen 信息-----------------Stop\n\n")
    
    # 3. packing分组
    #调用 packing-group 进行分组
    print("Step3-----------------packing 分组-----------------Start")
    # knapsacks, idx_knapsacks= greedy_knapsack(token_lens, PACKED_LENGTH)
    # print(idx_knapsacks[10])
    # print(knapsacks[10])
    import pickle
    def load_bin_boxes(file_path: str):
        """
        加载单步装箱结果
        """
        with open(file_path, 'rb') as f:
            bin_boxes = pickle.load(f)
        print(f"已加载装箱结果: {file_path}")
        return bin_boxes

    # bin_boxs = load_bin_boxes("./s2_ckpt/bins_boxs_8k.pkl")
    bin_boxs = load_bin_boxes("./s2_ckpt/bins_boxs_mr_sft_8k.pkl")
    
    # total_knapsacks = len(idx_knapsacks)
    total_knapsacks = len(bin_boxs)
    
    print(f"原始数据----{n_count}----条，packing后变为----{total_knapsacks}----条")
    print("Step3-----------------packing 分组-----------------Stop\n\n")

    print("Step4----------------- 开始构建新数据集 -----------------Start")
    print(f"开始处理 {total_knapsacks} 组数据，使用 {MAX_WORKERS} 个线程")

    #4. 使用线程池处理所有pack
    with ThreadPoolExecutor(max_workers=MAX_WORKERS, thread_name_prefix="PackThread") as executor:
        # 提交所有任务
        if f_TEST:
            futures = {
                executor.submit(process_knapsack, s1, idx_knapsack, dst_dir): s1
                for s1, idx_knapsack in enumerate(bin_boxs[0:n_packed_samples])
            }
        else:
            futures = {
                executor.submit(process_knapsack, s1, idx_knapsack, dst_dir): s1
                for s1, idx_knapsack in enumerate(bin_boxs)
            }

        # tqdm 自动跟踪完成数
        from tqdm import tqdm
        tty = open(os.devnull, 'w') if os.name == 'nt' else open('/dev/tty', 'w')
        for future in tqdm(as_completed(futures),
                           total=len(futures),
                           desc="Packing progress",
                           unit="pack",
                           file=tty
                          ):
            try:
                future.result()
            except Exception as e:
                s1 = futures[future]
                print(f"\n处理第 {s1} 组数据时发生错误: {e}")

    print("Step4-----------------Sccessful！！！！---- 构建新数据集成功 -----------------Stop")
