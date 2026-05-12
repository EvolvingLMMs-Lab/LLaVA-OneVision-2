# Legacy one-file conversion script.
# Step1:
# python -u s2_prepare_rawsamples-emova.py 2>&1 | tee s2_proc.log
# python -u s2_prepare_rawsamples-llava_vqa.py 2>&1 | tee s2_proc_llava.log
# python -u s2_prepare_rawsamples-vqa_1000k.py 2>&1 | tee ./logs/s2_proc_vqa_1000k.log
# python -u s2_prepare_rawsamples-vqa_1000k-16k.py 2>&1 | tee ./logs/s2_proc_vqa_1000k-16k.log
# python -u s2_prepare_rawsamples-vqa_5500k-16k.py 2>&1 | tee ./logs/s2_proc_vqa_5500k-16k.log
# python -u s2_prepare_rawsamples-vqa_5500k-16k-fast.py 2>&1 | tee ./logs/s2_proc_vqa_5500k-16k-fast.log
# python -u s2_prepare_rawsamples-vqa_pretrain_5M-8k-fast.py 2>&1 | tee ./logs/s2_proc_vqa_pretrain_5M-8k-fast.log

# python -u s2_prepare_rawsamples-bmr_sft_780k-8k-fast.py 2>&1 | tee ./logs/s2_prepare_rawsamples-bmr_sft_780k-8k-fast.log

import bisect
import json
import os
import random
import re
import shutil
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path


# Configuration.
# target_directory = "/workspace/test/packing"   # Final output location.

current_file = Path(__file__).resolve()
target_directory = current_file.parent
newDir = "raw_packing_data_mr_sft_780k-8k-fast"  # Output directory before WebDataset conversion.
SRC_DIR_IMGS = "/data_1/llava_next_raw_full/split_json_files"  # Source image directory.
SRC_DIR_JSONS = "/data_1/llava_next_raw_full/split_json_files"  # Source JSON directory.
SRC_DST_EXTENSIONS = ("jpg", "json")
f_toklens_originalsample = os.path.join(target_directory, "token_info_MR_sft_780k_8k.txt")
PACKED_LENGTH = 8192
dst_dir = os.path.join(target_directory, newDir)
MAX_WORKERS = 96  # Thread pool size; tune according to CPU cores and I/O performance.


"""
task_type settings:
    sft: VQA-style pretraining format.
    pretrain: caption-style pretraining format.
    bmr: mixed multi-turn SFT format.
"""
task_type = "bmr"


f_TEST = False  # Test output mode; only generate a small number of samples.
n_packed_samples = 400  # Number of packed samples to generate in test mode.

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
    "What's the main point of this figure?",
]


def find_long_file_pairs(directory, length_threshold=62):
    """
    Find image files in long-name image/JSON pairs.

    Args:
        directory: Directory to inspect.
        length_threshold: Filename length threshold. Defaults to 62.

    Returns:
        Image filenames with extensions that satisfy the criteria.
    """
    import os
    from collections import defaultdict

    # Store filename stems and their corresponding complete filenames.
    file_parts = defaultdict(list)
    image_extensions = (".jpg", ".jpeg", ".png", ".gif", ".bmp")

    try:
        # Group files by filename stem.
        for filename in os.listdir(directory):
            name_part, ext = os.path.splitext(filename)
            ext = ext.lower()
            # Only inspect image and JSON files.
            if ext in (".json",) + image_extensions:
                file_parts[name_part].append(filename)

        # Find qualifying image files.
        long_image_files = []
        for name_part, filenames in file_parts.items():
            # Check filename length and file-pair completeness.
            if (
                len(name_part) > length_threshold
                and any(f.endswith(".json") for f in filenames)
                and any(f.lower().endswith(image_extensions) for f in filenames)
            ):
                # Only append image files.
                for filename in filenames:
                    if filename.lower().endswith(image_extensions):
                        long_image_files.append(filename)

        return long_image_files

    except FileNotFoundError:
        # print(f"Error: directory '{directory}' does not exist")
        return []
    except PermissionError:
        # print(f"Error: no permission to access directory '{directory}'")
        return []
    except Exception:
        # print(f"Error while processing directory: {str(e)}")
        return []


# res_long_img_names = find_long_file_pairs(SRC_DIR_JSONS)


def filter_filenames(filenames, prefix, exclude_suffix):
    """
    Filter filenames that start with the given prefix and do not end with the excluded suffix.

    Args:
        filenames: List of filenames.
        prefix: Required filename prefix, for example "james-tissot".
        exclude_suffix: File suffix to exclude, for example "json".

    Returns:
        Filtered filenames.
    """
    # Escape special characters in the prefix so the regex matches correctly.
    escaped_prefix = re.escape(prefix)
    # Build regex pattern.
    pattern = rf"^{escaped_prefix}(?!.*\.{exclude_suffix}$).*$"

    # Compile regex.
    regex = re.compile(pattern)

    # Return matching filenames.
    return [filename for filename in filenames if regex.match(filename)]


def get_random_prompts(prompts, n):
    if n > len(prompts):
        # Allow duplicates.
        return random.choices(prompts, k=n)
    else:
        # Do not allow duplicates.
        return random.sample(prompts, n)


# Global base-name list. It is replaced later with token-length-sorted source sample names.
BASE_NAMES = []


def search_for_fit(numbers: list[int], capacity: int) -> int:
    """Finds the index of largest number that fits into the knapsack with the given capacity."""
    index = bisect.bisect(numbers, capacity)
    return -1 if index == 0 else (index - 1)


def greedy_knapsack(numbers: list[int], capacity: int) -> tuple[list[list[int]], list[list[int]]]:
    r"""Implement efficient greedy algorithm with binary search for the knapsack problem.
    Args
    ----
    numbers : List[int]
        Item sizes. The current call path passes them in ascending order.
    capacity : int
        Knapsack capacity.

    Returns
    ----
    Tuple[List[List[int]], List[List[int]]]
        First list: item sizes in each knapsack.
        Second list: original item indices in each knapsack.

    """
    # Preserve original indices for each input number.
    indexed_numbers = [(val, idx) for idx, val in enumerate(numbers)]
    # The input is already sorted in this flow, so keep the original behavior.
    knapsacks = []
    index_knapsacks = []
    iii = 0
    while indexed_numbers:
        current_knapsack = []
        current_indices = []
        remaining_capacity = capacity

        while True:
            # Extract current values for lookup while preserving ascending order.
            current_values = [val for val, idx in indexed_numbers]
            index = search_for_fit(current_values, remaining_capacity)
            if index == -1:
                break  # No item can fit into the current knapsack.

            # Remove the selected item and its original index.
            val, idx = indexed_numbers.pop(index)
            remaining_capacity -= val
            current_knapsack.append(val)
            current_indices.append(idx)

        if iii % 1000 == 0:
            print(f"---------pack {iii}----------")
            print(f"{current_knapsack}--->{sum(current_knapsack)}")
            print(current_indices)
            print("\n")
        iii += 1
        knapsacks.append(tuple(current_knapsack))
        index_knapsacks.append(tuple(current_indices))

    return tuple(knapsacks), tuple(index_knapsacks)


def extract_content(json_file):
    try:
        # Open and load JSON content.
        with open(json_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        if task_type == "sft":
            try:
                user_content = next(msg["content"] for msg in data["messages"] if msg["role"] == "assistant")
                return user_content
            except Exception:
                pass
        # Extract content. Assumes the captions array contains at least one element.
        elif task_type == "pretrain":
            if data.get("captions") and len(data["captions"]) > 0:
                return data["captions"][0].get("content", "")
            else:
                assert 0, "No valid caption content found"
                # return "No valid caption content found"

    except FileNotFoundError:
        return f"Error: file {json_file} does not exist"
    except json.JSONDecodeError:
        return f"Error: file {json_file} is not valid JSON"
    except Exception as e:
        return f"Error during extraction: {str(e)}"


def extract_prompt(json_file):
    try:
        # Open and load JSON content.
        with open(json_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        # Extract the user prompt.
        assistant_content = next(msg["content"] for msg in data["messages"] if msg["role"] == "user")
        return assistant_content

        # # Extract image path if needed.
        # image_path = data["images"][0] if data["images"] else None

    except FileNotFoundError:
        return f"Error: file {json_file} does not exist"
    except json.JSONDecodeError:
        return f"Error: file {json_file} is not valid JSON"
    except Exception as e:
        return f"Error during extraction: {str(e)}"


def extract_img_prompt_content(json_file: str) -> tuple[list[str], list[str], list[str]]:
    try:
        # Open and load JSON content.
        with open(json_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        # 1)images
        imgs = data.get("images", [])
        if not imgs:
            images = []
        else:
            images = [os.path.join(SRC_DIR_IMGS, imgs[0])]

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
        return f"Error: file {json_file} does not exist"
    except json.JSONDecodeError:
        return f"Error: file {json_file} is not valid JSON"
    except Exception as e:
        return f"Error during extraction: {str(e)}"


def prepare_dirs(target_dir, new_dir):
    os.chdir(target_dir)
    print(f"--------change to directory {target_dir}--------")
    # Create the output directory.
    if not os.path.exists(new_dir):
        os.makedirs(new_dir)
        print(f"Directory '{new_dir}' created.")
    else:
        print(f"Directory '{new_dir}' already exists.")


def dataset_tokinfo_generator(f_name):
    """
    Generate dataset token information by reading and parsing a file line by line.

    Args:
        f_name (str): Path to the token-info file.

    Yields:
        tuple: (base_name, token_len) parsed from each valid line.
    """
    try:
        with open(f_name, "r", encoding="utf-8") as f:
            for line in f:
                # Skip empty lines.
                stripped_line = line.strip()
                if not stripped_line:
                    continue

                # Split by colon and validate format.
                parts = stripped_line.split(":")
                if len(parts) == 2:
                    base_name = parts[0].strip()
                    token_len_str = parts[1].strip()

                    try:
                        token_len = int(token_len_str)
                        yield (base_name, token_len)
                    except ValueError:
                        print(
                            f"Warning: could not convert '{token_len_str}' to integer; skipped this line",
                            file=sys.stderr,
                        )
                        continue

    except FileNotFoundError:
        print(f"Error: file '{f_name}' does not exist", file=sys.stderr)
        return
    except Exception as e:
        print(f"Error while processing file: {str(e)}", file=sys.stderr)
        return


class TokenInfoReader:
    """
    Token-info reader.

    Supports batched reads, full reads, and resumable reads for token-info text files.
    Required line format: "base_name: token_len".
    """

    def __init__(self, f_name):
        """
        Initialize the reader.

        Args:
            f_name (str): Path to the token-info file.
        """
        self.f_name = f_name
        self.generator = dataset_tokinfo_generator(f_name)
        self._current_position = 0  # Number of records already read.

    def read(self, count=None):
        """
        Read records.

        Args:
            count (int, optional): Number of records to read. None means read all remaining records.

        Returns:
            tuple: (base_names, token_lens, actual_read_count).
        """
        base_names = []
        token_lens = []
        read_count = 0

        # Read until the target count is reached or the file ends.
        while True:
            # Check whether the requested count has been reached.
            if count is not None and read_count >= count:
                break

            try:
                # Fetch the next record from the generator.
                base_name, token_len = next(self.generator)
                base_names.append(base_name)
                token_lens.append(token_len)
                read_count += 1
                self._current_position += 1

            except StopIteration:
                # End of file reached.
                break

        return base_names, token_lens, read_count

    def get_current_position(self):
        """
        Get the current read position.

        Returns:
            int: Total number of records already read.
        """
        return self._current_position


def process_knapsack(s1, idx_knapsack, dst_dir):
    """
    Process one packed group.

    Args:
        s1: Current group index.
        idx_knapsack: Indices included in the knapsack.
        dst_dir: Target directory path.
    """
    # global BASE_NAMES

    packed_imgs, packed_caps = [], []  # Contents of one packed sample.

    # Fetch base names.
    # packed_b_names = (BASE_NAMES[idx] for idx in idx_knapsack)
    packed_b_names = (idx["name"] for idx in idx_knapsack)

    # Build source file information.
    if task_type == "pretrain":
        packed_info = (
            (
                os.path.join(SRC_DIR_IMGS, f"{b_name}.{SRC_DST_EXTENSIONS[0]}"),
                extract_content(os.path.join(SRC_DIR_JSONS, f"{b_name}.{SRC_DST_EXTENSIONS[1]}")),
            )
            for b_name in packed_b_names
        )
    elif task_type == "sft":
        packed_info = (
            (
                os.path.join(SRC_DIR_IMGS, f"{b_name}.{SRC_DST_EXTENSIONS[0]}"),
                extract_content(os.path.join(SRC_DIR_JSONS, f"{b_name}.{SRC_DST_EXTENSIONS[1]}")),
                extract_prompt(os.path.join(SRC_DIR_JSONS, f"{b_name}.{SRC_DST_EXTENSIONS[1]}")),
            )
            for b_name in packed_b_names
        )
    elif task_type == "bmr":
        packed_info = (
            extract_img_prompt_content(os.path.join(SRC_DIR_JSONS, f"{b_name}.{SRC_DST_EXTENSIONS[1]}"))
            for b_name in packed_b_names
        )

    # Target JSON file path.
    json_dst = os.path.join(dst_dir, f"ps_{s1:08d}.{SRC_DST_EXTENSIONS[1]}")

    # Process each image and corresponding caption/answer.
    if task_type == "pretrain":
        for s2, (img_src, cap_src) in enumerate(packed_info):
            # Target image path.
            img_name_dst = f"ps_{s1:08d}.img{s2:03d}.{SRC_DST_EXTENSIONS[0]}"
            # img_name_dst = f"img{s2:03d}.{SRC_DST_EXTENSIONS[0]}"    # Choose this form if downstream requires it.
            img_dst = os.path.join(dst_dir, img_name_dst)

            # Collect metadata.
            # packed_imgs.append(img_name_dst)
            packed_imgs.append(f"img{s2:03d}.{SRC_DST_EXTENSIONS[0]}")
            packed_caps.append(cap_src)

            # Copy image.
            shutil.copyfile(img_src, img_dst)
        # A model can optionally be called here to generate prompts for pure captioning data.
        selected_prompts = get_random_prompts(PROMPTS, len(packed_imgs))
    elif task_type == "sft":
        selected_prompts = []
        for s2, (img_src, cap_src, prompt_src) in enumerate(packed_info):
            # Target image path.
            img_name_dst = f"ps_{s1:08d}.img{s2:03d}.{SRC_DST_EXTENSIONS[0]}"
            # img_name_dst = f"img{s2:03d}.{SRC_DST_EXTENSIONS[0]}"    # Choose this form if downstream requires it.
            img_dst = os.path.join(dst_dir, img_name_dst)

            # Collect metadata.
            # packed_imgs.append(img_name_dst)
            packed_imgs.append(f"img{s2:03d}.{SRC_DST_EXTENSIONS[0]}")
            packed_caps.append(cap_src)

            # Copy image.
            shutil.copyfile(img_src, img_dst)

            # prompts
            selected_prompts.append(prompt_src)
        pass
    elif task_type == "bmr":
        selected_prompts = []
        for s2, (img_src, prompt_src, cap_src) in enumerate(packed_info):
            if not img_src:
                packed_imgs.append([])
            else:
                # Target image path.
                name, ext = os.path.splitext(img_src[0])
                img_name_dst = f"ps_{s1:08d}.img{s2:03d}{ext}"
                img_dst = os.path.join(dst_dir, img_name_dst)

                # Copy image.
                shutil.copyfile(img_src[0], img_dst)

                # Collect image metadata.
                packed_imgs.append([f"img{s2:03d}{ext}"])
                # cnt_imgs += 1

            # Collect other metadata.
            packed_caps.append(cap_src)
            selected_prompts.append(prompt_src)
        pass

    # Generate JSON file.
    json_data = {"images": packed_imgs, "captions": packed_caps, "prompts": selected_prompts}
    # print(packed_imgs)

    try:
        with open(json_dst, "w", encoding="utf-8") as f:
            json.dump(json_data, f, indent=4, ensure_ascii=False)
            # json.dump(json_data, f)
    except Exception as e:
        print(f"Thread {threading.current_thread().name} failed to generate JSON file {json_dst}: {str(e)}")
    return s1


if __name__ == "__main__":
    # 1. Create working directory.
    print("Step1-----------------Create working environment-----------------Start")
    prepare_dirs(target_directory, newDir)
    print("Step1-----------------Create working environment-----------------Stop\n\n")

    # 2. Read source dataset token-length information before packing.
    # This can be used to build multiple pools for chunked packing; read() controls the packing cache size.
    print("Step2-----------------Read source dataset token-length info-----------------Start")
    info_reader = TokenInfoReader(f_toklens_originalsample)
    base_names, token_lens, n_count = info_reader.read()

    # global BASE_NAMES
    BASE_NAMES = tuple(base_names)
    print(f"Read {n_count} records")
    # print(BASE_NAMES)
    print("Step2-----------------Read source dataset token-length info-----------------Stop\n\n")

    # 3. Load packing groups.
    # Packing groups are produced by the notebook or current bin packer.
    print("Step3-----------------Load packing groups-----------------Start")
    # knapsacks, idx_knapsacks= greedy_knapsack(token_lens, PACKED_LENGTH)
    # print(idx_knapsacks[10])
    # print(knapsacks[10])
    import pickle

    def load_bin_boxes(file_path: str):
        """
        Load single-step bin-packing results.
        """
        with open(file_path, "rb") as f:
            bin_boxes = pickle.load(f)
        print(f"Loaded bin-packing results: {file_path}")
        return bin_boxes

    # bin_boxs = load_bin_boxes("./s2_ckpt/bins_boxs_8k.pkl")
    bin_boxs = load_bin_boxes("./s2_ckpt/bins_boxs_mr_sft_8k.pkl")

    # total_knapsacks = len(idx_knapsacks)
    total_knapsacks = len(bin_boxs)

    print(f"Source records: {n_count}; packed groups: {total_knapsacks}")
    print("Step3-----------------Load packing groups-----------------Stop\n\n")

    print("Step4-----------------Build packed dataset-----------------Start")
    print(f"Processing {total_knapsacks} groups with {MAX_WORKERS} threads")

    # 4. Process all packs with a thread pool.
    with ThreadPoolExecutor(max_workers=MAX_WORKERS, thread_name_prefix="PackThread") as executor:
        # Submit all tasks.
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

        # tqdm tracks completed tasks automatically.
        from tqdm import tqdm

        tty = open(os.devnull, "w") if os.name == "nt" else open("/dev/tty", "w")
        for future in tqdm(as_completed(futures), total=len(futures), desc="Packing progress", unit="pack", file=tty):
            try:
                future.result()
            except Exception as e:
                s1 = futures[future]
                print(f"\nError while processing group {s1}: {e}")

    print("Step4-----------------Successful----Packed dataset created-----------------Stop")
