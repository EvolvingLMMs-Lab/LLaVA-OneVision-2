#!/usr/bin/env python3
"""
把目录级多图 captioning 数据打包成 WebDataset
目录结构：
raw_packing_data/
├── ps_00000000.img000.jpg
├── ps_00000000.img001.jpg
├── ps_00000000.json

...

JSON format(pretrain):
{
  "images": ["img000.jpg", "img001.jpg", ...],
  "prompt": ["描述", "what about", ""],
  "captions": ["stri", "str2", "str3"]
}
一条 json + 对应若干 jpg = 1 条 tar 记录
"""

"""

JSON format (multi-round + blend data for sft)：
{
  "images":   [["img000.jpg"], ["img001.jpg"], []],
  "prompt":   [["描述这幅图"], ["what about this fig"], ["How are you?", "I am fine too."]],
  "captions": [["stri"], ["str2"], ["I am fine and you?", "Have a nice day"]]
}
一条 json + 对应若干 jpg(可以为0) = 1 条 tar 记录
"""

# #####-----------------------------------------######
# #####-----------------------------------------######
# #####-----------------------------------------######

import argparse
import uuid
import json
import os
import yaml
import webdataset as wds
from tqdm import tqdm
from pathlib import Path
from megatron.energon.epathlib import EPath
from megatron.energon.flavors import BaseWebdatasetFactory
from megatron.energon.flavors.webdataset import MAIN_FOLDER_NAME
from megatron.energon.flavors.webdataset.prepare import WebdatasetPreparator
from megatron.energon.flavors.webdataset.structs import ShardInfo, WebdatasetInfo, WebdatasetSplits


def sample_loader_template(media: str=None):
    """Returns a template for a sample_loader.py file."""
    return "\n".join([
        "def sample_loader(sample: dict) -> dict:",
        "    messages=[]",
        "    for message in sample['json']:",
        "        assert message['role'] in ['system','user','assistant']",
        "        messages.append(dict(",
        "            role=message['role'],",
        "            content=message['content']",
        "        ))",
        "    return dict(",
        "        __key__=sample['__key__'],",
        "        __restore_key__=sample['__restore_key__'],",
        "        video=sample.get('mp4'),",
        "        image=sample.get('jpg')," if media == 'mix' else "",
        "        messages=messages,",
        "    )",
        "def part_filter(part: str) -> bool:",
        "    return True",
    ])

# ## ZXW   

def sample_loader_template_caption(media=None):
    """适配整条多图 captioning 的 loader"""
    return "\n".join([
        "def sample_loader(sample: dict) -> dict:",
        "    data = sample['json']",
        "    images = [sample.get(f'img{i}.jpg') for i in range(len(data['images']))]",
        "    captions = data['captions'] ",
        "    prompts = data['prompts']",
        "    return dict(",
        "        __key__=sample['__key__'],",
        "        __restore_key__=sample['__restore_key__'],",
        "        captions=captions,",
        "        prompts=prompts,",
        "        images=images,",
        "    )",
        "def part_filter(part: str) -> bool:",
        "    return True",
    ])
# 生成 1 条数据
def stream_samples_caption(src_dir: str):
    for json_path in Path(src_dir).glob("*.json"):
        sample_id = json_path.stem            # ps_00000000
        with json_path.open("r", encoding="utf-8") as f:
            raw = json.load(f)
        yield {
            "id": sample_id,
            "images": raw["images"],           # [img000.jpg, ...]
            "prompts": raw.get("prompts", []),  # [str, ...]
            "captions": raw["captions"]        # [str, ...]
        }

def construct_sample_caption(args, entry):
    """整条样本打包"""
    sample = {"__key__": entry["id"]}
    for idx, img_name in enumerate(entry["images"]):
        img_path = os.path.join(args.image_dir, f"{entry["id"]}.{img_name}")
        with open(img_path, "rb") as f:
            sample[f"img{idx}.jpg"] = f.read()

    payload = {
        "prompts": entry["prompts"],
        "captions": entry["captions"],
        "images": entry["images"]
    }
    sample["json"] = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    return sample

def construct_bmr_sample(args, entry):
    "multi-round & blend data package"
    sample = {"__key__": entry["id"]}
    for idx, img_name in enumerate(entry["images"]):
        # print(img_name)
        img_path = os.path.join(args.image_dir, f"{entry["id"]}.{img_name[0]}") if img_name else None
        if img_name:
            with open(img_path, "rb") as f:
                sample[f"img{idx}.jpg"] = f.read()
    payload = {
        "prompts": entry["prompts"],
        "captions": entry["captions"],
        "images": entry["images"]
    }

    sample["json"] = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    return sample

# ## ZXW

def construct_sample(args, vision, path, entry):
    """ construct webdataset sample """
    # 断言vision的值只能是'image'或'video'
    assert vision == 'image' or vision == 'video'
    # 根据vision的值，确定directory的路径
    directory = args.image_dir if vision == 'image' else args.video_dir

    # 打开vision_file文件
    with open(os.path.join(directory, path), "rb") as vision_file:
        # 读取vision_file文件的内容
        vision_data = vision_file.read()
    # 构造sample字典
    sample = {
        "__key__": entry.get('id', path).replace('.', '_'),
        # 根据vision的值，确定sample的键
        "jpg" if vision == 'image' else 'mp4': vision_data,
        # 将entry[args.columns_messages]转换为json格式，并编码为utf-8
        "json": json.dumps(entry[args.columns_messages]).encode("utf-8"),
    }
    # 返回sample字典
    return sample


def convert_to_wds(args):
    """ Convert dataset to wds format """
    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)
    
    tar = os.path.join(args.output_dir, 'pretrain-%06d.tar')
    if args.mode == "caption_pack":
        # 新模式 1
        with wds.ShardWriter(tar, maxcount=args.maxcount, maxsize=args.maxsize) as sink:
            for entry in tqdm(stream_samples_caption(args.json_file)):
                sample=construct_sample_caption(args, entry)
                # print(sample.keys())
                sink.write(sample)
                # break
                # sink.write(construct_sample_caption(args.image_dir, entry))
                
        write_config(EPath(args.output_dir).absolute(), args.media,
                     template_func=sample_loader_template_caption,
                     class_name="PackedCaptioningSample")
    elif args.mode == "bmr_pack":
        # 新模式 2
        with wds.ShardWriter(tar, maxcount=args.maxcount, maxsize=args.maxsize) as sink:
            for entry in tqdm(stream_samples_caption(args.json_file)):
                sample=construct_bmr_sample(args, entry)
                # print(sample.keys())
                sink.write(sample)
                # break
                # sink.write(construct_sample_caption(args.image_dir, entry))
                
        write_config(EPath(args.output_dir).absolute(), args.media,
                     template_func=sample_loader_template_caption,
                     class_name="PackedCaptioningSample")
        pass
    print(f"Dataset successfully converted to wds")



def write_config(path: EPath, media=None, template_func=None, class_name=None):
    (path / MAIN_FOLDER_NAME).mkdir(exist_ok=True)
    all_tars = list(path.glob("**/*.tar")) + list(path.glob("**/*.tgz"))
    all_tars = [str(p.relative_to(path)) for p in sorted(all_tars)]

    if class_name is None:
        class_name = "MultiMixQASample" if media == 'mix' else "MultiVidQASample"
    dataset_definition = {
        "sample_type": {
            "__module__": "aiak_training_llm.data.multimodal",
            "__class__": class_name,
        },
        "part_filter": "sample_loader.py:part_filter",
        "sample_loader": "sample_loader.py:sample_loader"
    }

    with (path / MAIN_FOLDER_NAME / "dataset.yaml").open("w") as f:
        yaml.dump(dataset_definition, f, sort_keys=False)

    tpl = (template_func or sample_loader_template)(media)
    with (path / MAIN_FOLDER_NAME / "sample_loader.py").open("w") as f:
        f.write(tpl)

    BaseWebdatasetFactory.prepare_dataset(
        path,
        all_tars,
        split_parts_ratio=[("train", 1.0), ("val", 0), ("test", 0)],
        tar_index_only=False,
        workers=32,
    )


def _add_arguments(parser: argparse.ArgumentParser):
    """Add arguments"""
    group = parser.add_argument_group(title='wds')
    group.add_argument('--output_dir', type=str, required=True, help='Output directory')
    group.add_argument('--json_file', type=str, required=True,
                       help='目录（多图 captioning）或单文件（旧格式）')
    group.add_argument('--image_dir', type=str, required=False, help='Image directory')
    group.add_argument('--video_dir', type=str, required=False, help='Video directory')
    group.add_argument('--maxcount', type=int, default=10000, help='Number of samples per shard')
    group.add_argument('--maxsize', type=int, default=3000000000, help='Maximum size of each shard')
    group.add_argument('--media', type=str, choices=["mix", "image", "video"], default="image", help='Media type')
    group.add_argument('--columns_messages', type=str, default="messages", help='Column name for messages')
    # 新增模式选择
    group.add_argument('--mode', type=str,
                       choices=["chat", "caption_pack", "bmr_pack"],
                       default="chat",
                       help="chat=旧格式(单图对话); caption_pack=新格式(整条多图caption)")
    return parser


def parse_args():
    """arguments"""
    parser = argparse.ArgumentParser()
    _add_arguments(parser)
    args = parser.parse_args()

    return args


def main():
    """main function"""
    args = parse_args()
    convert_to_wds(args)


if __name__ == '__main__':
    main()


