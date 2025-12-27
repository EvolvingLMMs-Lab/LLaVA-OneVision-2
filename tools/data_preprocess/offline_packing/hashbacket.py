import os
import sys
import time
import logging
import numpy as np
from pathlib import Path
from itertools import islice
from tqdm import tqdm
from collections import defaultdict
from typing import List, Tuple, Dict, Optional, Union, Set
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
import queue
import bisect

class HashBucketProcessor:
    """哈希桶处理器，用于处理大型数据文件并进行高效装箱"""
    
    DTYPE_SAMPLE_INFO = np.dtype([
        ("w", np.uint16),       # 用于存储 ViT 部分的权重（可以是 ViT 部分的像素数或者 ViT 部分的处理能力）
        ("l", np.uint16),       # 用于存储 llm 部分的曲种（ LLM 输入部分的 tokens 数亩）
        ("name", "U64")        # sample‘s name
    ])

    def __init__(self, file_path: Union[str, Path], logger: Optional[logging.Logger] = None):
        self.file_path = Path(file_path)
        if not self.file_path.exists():
            raise FileNotFoundError(f"文件不存在: {file_path}")
            
        self.hash_buckets = defaultdict(lambda: np.array([], dtype=self.DTYPE_SAMPLE_INFO))
        self.total_lines = 0
        self.hb2_keys = []   # 可以除以哪些 2 的幂次
        self._logger = logger or self._setup_default_logger()

    @staticmethod
    def _setup_default_logger() -> logging.Logger:
        """设置默认日志器"""
        logger = logging.getLogger(__name__)
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            logger.setLevel(logging.INFO)
        return logger
    
    def estimate_memory_usage(self) -> int:
        """估算当前哈希桶内存占用"""
        total_size = sys.getsizeof(self.hash_buckets)
        for key, arr in self.hash_buckets.items():
            total_size += sys.getsizeof(key) + arr.nbytes
        return total_size
    
    def _count_file_lines(self) -> int:
        """计算文件总行数，使用更高效的方法"""
        try:
            with self.file_path.open('rb') as f:
                return sum(1 for _ in f)
        except Exception as e:
            self._logger.warning(f"快速计数失败，使用标准方法: {e}")
            with self.file_path.open('r', encoding='utf-8') as f:
                return sum(1 for _ in f)

    def _parse_line(self, line: str) -> Optional[Tuple[int, int, str]]:
        """解析单行数据，返回 (w, l, name) 或 None"""
        line = line.strip()
        if ':' not in line:
            return None
            
        try:
            name, key_str = line.split(':', 1)
            key = int(key_str)
            if 0 <= key <= 65535:
                return (0, key, name)
        except (ValueError, IndexError):
            pass
        return None
        
    def _update_buckets(self, parsed_data: List[Tuple[int, int, str]]) -> None:
        """更新哈希桶"""
        data_array = np.array(parsed_data, dtype=self.DTYPE_SAMPLE_INFO)
        unique_l_values = np.unique(data_array['l'])

        for l_val in unique_l_values:
            mask = data_array['l'] == l_val
            chunk = data_array[mask]
            
            if l_val in self.hash_buckets:
                self.hash_buckets[l_val] = np.concatenate([self.hash_buckets[l_val], chunk])
            else:
                self.hash_buckets[l_val] = chunk
                
    def build_buckets(self, chunk_size: int = 100000) -> None:
        """构建哈希桶"""
        self.total_lines = self._count_file_lines()
        self._logger.info(f"开始处理文件，总行数: {self.total_lines}")
        
        with self.file_path.open('r', encoding='utf-8') as file:
            with tqdm(total=self.total_lines, unit='行', desc='构建哈希桶') as pbar:
                while True:
                    lines = list(islice(file, chunk_size))
                    if not lines:
                        break

                    pbar.update(len(lines))
                    
                    # 并行解析数据
                    parsed_data = []
                    for line in lines:
                        parsed = self._parse_line(line)
                        if parsed:
                            parsed_data.append(parsed)

                    if parsed_data:
                        self._update_buckets(parsed_data)                                

    @staticmethod
    def factors_of_two(a: int, C: int) -> List[Tuple[int, int]]:
        """返回所有满足 b * 2^n = a 且 b > C 的 (b, n) 对"""
        if a < 0 or C < 0:
            raise ValueError("a 必须为正整数，C 必须为非负整数")
        res = []
        n = 0
        b = a
        while b > C:
            res.append((b, n))
            if b & 1:
                break
            b >>= 1
            n += 1
        return res

    def find_items(self, capacity: int) -> defaultdict[np.ndarray]:
        """从哈希桶中查找符合条件的项目"""

        if not self.hash_buckets:
            self._logger.warning("哈希桶为空，请先构建哈希桶")
            return
            
        for key, value in self.hash_buckets.items():
            if not isinstance(value, np.ndarray) or value.dtype != self.DTYPE_SAMPLE_INFO:
                raise TypeError(f"哈希桶数据格式错误，key={key}")
            break
        self.hb2_keys=[]
        min_l_value = min(self.hash_buckets.keys())
        valid_b_values = [b for b, _ in self.factors_of_two(capacity, min_l_value - 1)]

        for b in valid_b_values:
            if b in self.hash_buckets:
                self.hb2_keys.append(b)

        self._logger.info(f"找到 {len(self.hb2_keys)} 个有效的桶键")

    def delete_by_index(self, result: defaultdict[np.ndarray], key: int, index: int) -> None:
        """按索引删除元素"""
        if key in result and 0 <= index < len(result[key]):
            result[key] = np.delete(result[key], index)

    def get_statistics(self) -> Dict[str, Union[int, float]]:
        """获取统计信息"""
        total_items = sum(len(arr) for arr in self.hash_buckets.values())
        memory_gb = self.estimate_memory_usage() / (1024**3)
        
        return {
            "bucket_count": len(self.hash_buckets),
            "total_items": total_items,
            "memory_usage_gb": memory_gb,
            "hb2_keys_count": [len(self.hb2_keys),self.hb2_keys],
            "file_lines": self.total_lines
        }

    def __len__(self) -> int:
        """返回总数据项数"""
        return sum(len(arr) for arr in self.hash_buckets.values())

    def __repr__(self) -> str:
        return f"HashBucketProcessor(buckets={len(self.hash_buckets)}, items={len(self)})"    
        
    def summary(self) -> None:
        """打印摘要信息"""
        stats = self.get_statistics()
        print(f"=== 哈希桶处理摘要 ===")
        print(f"哈希桶数量: {stats['bucket_count']}")
        print(f"总数据项: {stats['total_items']}")
        print(f"内存占用: {stats['memory_usage_gb']:.2f} GB")
        print(f"有效桶键: {stats['hb2_keys_count']}")
        print(f"处理行数: {stats['file_lines']}")    

    def _cleanup_empty_keys(self, verbose: bool = False) -> int:
        """
        清理哈希桶中元素个数为0的key
        
        参数:
            verbose: 是否打印清理详情
        
        返回:
            int: 删除的空key数量
        """
        # 1. 收集需要删除的空key
        empty_keys = []
        for key in list(self.hash_buckets.keys()):
            if len(self.hash_buckets[key]) == 0:
                empty_keys.append(key)
        
        # 2. 删除空key
        for key in empty_keys:
            del self.hash_buckets[key]
        
        # 3. 记录日志
        if verbose or empty_keys:
            self._logger.info(f"清理空key: 删除了 {len(empty_keys)} 个空key")
            if verbose and empty_keys:
                self._logger.debug(f"删除的key: {sorted(empty_keys)}")
        
        return len(empty_keys)    

    def update_hash_buckets(self, remove_empty: bool = True, verbose: bool = False) -> dict:
        """
        更新哈希桶结构，包括清理空key和统计信息
        
        参数:
            remove_empty: 是否删除空key
            verbose: 是否打印详细信息
        
        返回:
            dict: 更新后的统计信息
        """
        # 1. 基础统计
        stats = {
            'before': {
                'total_keys': len(self.hash_buckets),
                'total_items': sum(len(arr) for arr in self.hash_buckets.values()),
                'empty_keys': sum(1 for arr in self.hash_buckets.values() if len(arr) == 0)
            }
        }
        
        # 2. 可选：删除空key
        removed_keys = 0
        if remove_empty:
            removed_keys = self._cleanup_empty_keys(verbose)
        
        # 3. 更新后统计
        stats['after'] = {
            'total_keys': len(self.hash_buckets),
            'total_items': sum(len(arr) for arr in self.hash_buckets.values()),
            'empty_keys': sum(1 for arr in self.hash_buckets.values() if len(arr) == 0)
        }
        
        # 4. 计算变化
        stats['changes'] = {
            'keys_removed': removed_keys,
            'items_removed': stats['before']['total_items'] - stats['after']['total_items']
        }
        
        # 5. 记录日志
        if verbose or stats['changes']['keys_removed'] > 0:
            self._logger.info("哈希桶更新完成:")
            self._logger.info(f"  📊 Key数量: {stats['before']['total_keys']} → {stats['after']['total_keys']}")
            self._logger.info(f"  📦 元素总数: {stats['before']['total_items']} → {stats['after']['total_items']}")
            self._logger.info(f"  🗑️  删除空key: {stats['changes']['keys_removed']}")
        
        return stats

    def get_hash_buckets_summary(self) -> dict:
        """
        获取哈希桶的摘要信息
        
        返回:
            dict: 包含详细统计信息的字典
        """
        # 基础统计
        total_keys = len(self.hash_buckets)
        total_items = sum(len(arr) for arr in self.hash_buckets.values())
        empty_keys = sum(1 for arr in self.hash_buckets.values() if len(arr) == 0)
        
        # 按大小分类统计
        size_distribution = {
            'large': 0,    # >= 8192
            'medium': 0,   # 2048-8192
            'small': 0     # < 2048
        }
        
        items_by_size = {
            'large': 0,
            'medium': 0,
            'small': 0
        }
        
        for key, arr in self.hash_buckets.items():
            count = len(arr)
            if key >= 8192:
                size_distribution['large'] += 1
                items_by_size['large'] += count
            elif key >= 2048:
                size_distribution['medium'] += 1
                items_by_size['medium'] += count
            else:
                size_distribution['small'] += 1
                items_by_size['small'] += count
        
        # 返回完整摘要
        return {
            'basic': {
                'total_keys': total_keys,
                'total_items': total_items,
                'empty_keys': empty_keys,
                'non_empty_keys': total_keys - empty_keys
            },
            'size_distribution': size_distribution,
            'items_by_size': items_by_size,
            'memory_usage': self.estimate_memory_usage()
        }        
    
    def print_example(self, key: int) -> None:
        """打印示例数据"""
        if key in self.hash_buckets:
            arr = self.hash_buckets[key]
            print(f"Key {key} 的数据数量: {len(arr)}")
            print("前3条数据:")
            for item in arr[:3]:
                print(f"  w: {item['w']}, l: {item['l']}, name: {item['name']}")
        else:
            print(f"Key {key} 不存在。")
  
    def pack_with_deletion(self, box_capacity: int = 16384) -> List[np.ndarray]:
        """按容量装箱，优先多样性，装箱后立即从原桶中删除已用元素
        （用于单独处理  (box_capacity/key)==2^n 的 key ）
        中间遇到不满箱时，更换 1次 装箱策略
        """
        from collections import deque
    
        boxes = []
    
        # 为每个 key 维护一个 deque，方便 pop （仅考虑存在元素的桶，后面数据量非常大时可以考虑放入 while 循环）
        key_queues = {k: deque(enumerate(self.hash_buckets[k])) 
                      for k in self.hb2_keys 
                      if k in self.hash_buckets and len(self.hash_buckets[k]) > 0}
    
        while any(key_queues.values()):
            current_box_items = []
            current_sum = 0
            used_indices = defaultdict(list)  # key -> list of indices to delete
    
            keys_to_try = deque(sorted(key_queues.keys()))
    
            while keys_to_try and current_sum < box_capacity:
                key = keys_to_try.popleft()
                queue = key_queues[key]
                if not queue:
                    continue
    
                idx, item = queue[0]
                l_val = key #item['l']
                if current_sum + l_val <= box_capacity:
                    queue.popleft()
                    current_box_items.append(item)
                    current_sum += l_val
                    used_indices[key].append(idx)
    
                    # 如果该 key 还有剩余，放回队列尾部
                    if queue:
                        keys_to_try.append(key)
    
            if current_box_items and current_sum==box_capacity:
                # 满箱：输出并删除
                boxes.append(np.array(current_box_items, dtype=self.DTYPE_SAMPLE_INFO))
    
                # 从 self.hash_buckets 中删除已用元素
                for key, indices in used_indices.items():
                    indices = sorted(indices, reverse=True)
                    for idx in indices:
                        self.hash_buckets[key] = np.delete(self.hash_buckets[key], idx)
    
                    # 更新 key_queues 中的 deque（未删除的又更新到 key_que）
                    # 更新 key_queues 中的 deque（用掉的元素），这样做在很大程度上可以避免完全陷入死循环（除非所有队列剩余元素数目完全相同）
                    key_queues[key] = deque(enumerate(self.hash_buckets[key]))
            else:
                # 加一个判断，如果各个队列元素数完全相同，则改变一次 packing 策略(这种情况跳不出循环♻️) ，再回到原始方法
                self._logger.info(f"当前箱子没有满: {current_sum}")
                self._logger.info(f"当前箱子元素: {current_box_items}")
                
                left_elems = [len(self.hash_buckets[k]) for k in self.hb2_keys if k in self.hash_buckets and len(self.hash_buckets[k])>0]
                # 拼包剩余的 key
                left_keys = [k for k in self.hb2_keys if k in self.hash_buckets and len(self.hash_buckets[k])>0]
                print(f"剩余的key及其元素数量：(keys, nums):({left_keys},{left_elems})")
                if len(set(left_elems)) == 1:
                    self._logger.info(f"改变拼包策略，尝试跳出 循环♻️")
                    b_succeed=False
                    # todo ...... 不考虑多样性的拼包
                    current_box2 = []
                    current_sum2 = 0
                    used_keys_num = defaultdict(int)   # 记录这个桶用了几个元素
                    for key2 in left_keys:   # 取出 1个桶
                        if b_succeed:   # 只拼一个
                            print(f"改变策略拼包成功:✅✅✅✅✅✅✅✅✅✅")
                            break
                        arr2 = self.hash_buckets[key2]
                        l_val2 = key2
                        for item2 in arr2:
                            if current_sum2 + l_val2 <= box_capacity:
                                current_box2.append(item2)
                                current_sum2 += l_val2
                                used_keys_num[key2] += 1
                                
                                if current_sum2==box_capacity:
                                    boxes.append(np.array(current_box2, dtype=self.DTYPE_SAMPLE_INFO))
                                    current_box2 = []
                                    current_sum2 = 0
                                    # 删除元素
                                    for kkey, knum in used_keys_num.items():
                                        for _ in range(knum):
                                            # self.delete_by_index(self.hash_buckets, kkey,0)
                                            self.hash_buckets[kkey] = np.delete(self.hash_buckets[kkey], 0)
                                        key_queues[kkey] = deque(enumerate(self.hash_buckets[key]))
                                    # 重新同步 key_queues 和 self.hash_buckets
                                    # key_queues = {k: deque(enumerate(self.hash_buckets[k])) 
                                    #               for k in self.hb2_keys if k in self.hash_buckets}
                                    print(f"改变策略拼包成功:✅✅✅✅✅{boxes[-1]}")
                                    used_keys_num = defaultdict(int)
                                    b_succeed = True
                                    break
                            else:
                                current_box2 = []
                                current_sum2 = 0
                                used_keys_num = defaultdict(int)
                                b_succeed = False
                                print(f"改变策略拼包失败:❌❌❌❌❌")
                                break
                    pass
                else:
                    print(f"num of left_elems:{left_elems}")
    
        return boxes

    def pack_with_deletion_recursion(self, box_capacity: int = 16384) -> List[np.ndarray]:
        """递归多样性优先装箱：只输出/删除满箱，所有不满箱混合重装，直到只剩一个不满箱。
        （用于单独处理  (box_capacity/key)==2^n 的 key ）
        递归实现
        """
        from collections import deque, defaultdict
        def recursive_diversity_pack(key_queues):
            boxes = []
            not_full_items = []
            print("----------- pack_with_deletion_recursion -----------")
            while any(key_queues.values()):
                current_box = []
                current_sum = 0
                used_indices = defaultdict(list)
                keys_to_try = deque(sorted(key_queues.keys()))
    
                # 多样性优先：每轮从不同桶取
                while keys_to_try and current_sum < box_capacity:
                    key = keys_to_try.popleft()
                    queue = key_queues[key]
                    if not queue:
                        continue
                    idx, item = queue[0]
                    l_val = item['l']
                    if current_sum + l_val <= box_capacity:
                        queue.popleft()
                        current_box.append((key, idx, item))
                        current_sum += l_val
                        used_indices[key].append(idx)
                        if queue:
                            keys_to_try.append(key)
    
                if current_sum == box_capacity:
                    # 满箱，输出并记录要删除的索引
                    boxes.append(np.array([item for _, _, item in current_box], dtype=self.DTYPE_SAMPLE_INFO))
                    for key, indices in used_indices.items():
                        # 删除已用元素
                        indices = sorted(indices, reverse=True)
                        for idx in indices:
                            self.hash_buckets[key] = np.delete(self.hash_buckets[key], idx)
                        # 更新 key_queues
                        key_queues[key] = deque(enumerate(self.hash_buckets[key]))
                elif current_box:
                    # 不满箱，暂存
                    not_full_items.extend(current_box)
    
            return boxes, not_full_items
    
        # 初始化 key_queues
        key_queues = {k: deque(enumerate(self.hash_buckets[k])) for k in self.hb2_keys if k in self.hash_buckets}
        boxes, not_full_items = recursive_diversity_pack(key_queues)
    
        # 混合所有不满箱元素递归装箱
        while not_full_items:
            # 混合所有剩余元素，重新分桶
            mixed = defaultdict(list)
            for _, _, item in not_full_items:
                mixed[item['l']].append(item)
            key_queues = {k: deque(enumerate(np.array(v, dtype=self.DTYPE_SAMPLE_INFO))) for k, v in mixed.items()}
            new_boxes, new_not_full_items = recursive_diversity_pack(key_queues)
            boxes.extend(new_boxes)
            if not new_boxes or not new_not_full_items:
                break
            not_full_items = new_not_full_items
        return boxes, not_full_items

    def pack_large_seed_parallel_multithread(self, box_capacity: int = 16384, min_ratio: float = 0.95, 
                                           max_workers: int = None) -> List[np.ndarray]:
        """
        多线程版本（处理 pack_with_deletion 之后的元素）：大种子并行装箱，小元素作为共享资源，实时删除元素
         （对于一个箱子中的物品数量没有任何限制，速度会比较快一点）
        参数:
            box_capacity: 箱子容量
            min_ratio: 最小装载率阈值
            max_workers: 最大线程数，None时自动设置为CPU核心数
        
        返回:
            List[np.ndarray]: 成功装箱的箱子列表
        """
        if max_workers is None:
            max_workers = min(os.cpu_count(), 8)  # 限制最大线程数
        
        half = box_capacity // 2
        # half = 4096
        large_keys = [k for k in self.hash_buckets.keys() if k >= half]
        small_keys = [k for k in self.hash_buckets.keys() if k < half]
        
        if not large_keys:
            self._logger.warning("没有找到大种子元素")
            return []
    
        # 1. 线程安全的共享资源管理器
        class SharedResourceManager:
            def __init__(self, hash_buckets, small_keys, large_keys):
                self.lock = threading.RLock()  # 可重入锁
                self.hash_buckets = hash_buckets  # 直接引用原始哈希桶
                self.small_keys = small_keys
                self.large_keys = large_keys
                
                # 初始化可用的small keys
                self.available_small_keys = sorted([
                    k for k in small_keys 
                    if k in hash_buckets and len(hash_buckets[k]) > 0
                ])
                
                # 统计信息
                self.total_processed = 0
                self.successful_boxes = 0
                
            def get_seed_item(self, seed_key: int) -> tuple:
                """线程安全地获取种子元素"""
                with self.lock:
                    if (seed_key in self.hash_buckets and 
                        len(self.hash_buckets[seed_key]) > 0):
                        
                        item = self.hash_buckets[seed_key][0]
                        self.hash_buckets[seed_key] = self.hash_buckets[seed_key][1:]
                        return True, item
                    return False, None
            
            def get_item_by_key(self, target_key: int) -> tuple:
                """线程安全地从指定key获取一个元素并删除"""
                with self.lock:
                    if (target_key in self.hash_buckets and 
                        len(self.hash_buckets[target_key]) > 0):
                        
                        item = self.hash_buckets[target_key][0]
                        self.hash_buckets[target_key] = self.hash_buckets[target_key][1:]
                        
                        # 如果这个key的桶空了，从available_small_keys中移除
                        if (len(self.hash_buckets[target_key]) == 0 and 
                            target_key in self.available_small_keys):
                            self.available_small_keys.remove(target_key)
                        
                        return True, item
                    return False, None
            
            def get_available_small_keys(self) -> List[int]:
                """获取当前可用的小key列表"""
                with self.lock:
                    return self.available_small_keys.copy()
            
            def update_stats(self, success: bool):
                """更新统计信息"""
                with self.lock:
                    self.total_processed += 1
                    if success:
                        self.successful_boxes += 1
                    
            def get_stats(self) -> dict:
                """获取统计信息"""
                with self.lock:
                    small_items_count = sum(
                        len(self.hash_buckets[k]) for k in self.small_keys 
                        if k in self.hash_buckets
                    )
                    large_items_count = sum(
                        len(self.hash_buckets[k]) for k in self.large_keys 
                        if k in self.hash_buckets
                    )
                    
                    return {
                        'small_items_remaining': small_items_count,
                        'large_items_remaining': large_items_count,
                        'available_small_keys': len(self.available_small_keys),
                        'total_processed': self.total_processed,
                        'successful_boxes': self.successful_boxes,
                        'success_rate': (self.successful_boxes / max(1, self.total_processed))
                    }
    
        # 2. 二分查找函数
        def search_for_fit_key(available_keys: List[int], remaining_capacity: int) -> int:
            """在可用的key中二分查找最大能装入的key"""
            if not available_keys:
                return -1
            index = bisect.bisect(available_keys, remaining_capacity)
            return -1 if index == 0 else (index - 1)
    
        # 3. 单个种子的装箱函数
        def pack_single_seed(seed_key: int, shared_manager: SharedResourceManager, 
                            thread_id: int) -> tuple:
            """为单个种子进行装箱"""
            try:
                # 获取种子元素
                success, seed_item = shared_manager.get_seed_item(seed_key)
                if not success:
                    return False, None, thread_id, 0, "无可用种子"
                
                current_box = [seed_item]
                remaining_capacity = box_capacity - seed_key
                items_added = 1
            
                # 贪心装箱：优先装入大的元素
                max_iterations = 1000  # 防止死循环
                iteration = 0
                
                while remaining_capacity > 0 and iteration < max_iterations:
                    iteration += 1
                    
                    # 获取当前可用的小key
                    available_keys = shared_manager.get_available_small_keys()
                    if not available_keys:
                        break
                    
                    # 二分查找最大可装入的key
                    best_key_index = search_for_fit_key(available_keys, remaining_capacity)
                    if best_key_index == -1:
                        break
                    
                    best_key = available_keys[best_key_index]
                    
                    # 尝试从该key获取元素
                    success, item = shared_manager.get_item_by_key(best_key)
                    if not success:
                        continue  # 这个key已经被其他线程用完，重试
                    
                    current_box.append(item)
                    remaining_capacity -= best_key
                    items_added += 1
                    
                    # 如果装满就停止
                    if remaining_capacity == 0:
                        break
                
                # 检查装载率
                current_capacity = box_capacity - remaining_capacity
                is_successful = current_capacity >= min_ratio * box_capacity
                
                result_box = current_box if is_successful else None
                load_ratio = current_capacity / box_capacity
                
                return (is_successful, result_box, thread_id, current_capacity, 
                       f"装载率:{load_ratio:.1%}, 物品数:{items_added}")
                
            except Exception as e:
                return False, None, thread_id, 0, f"装箱异常: {str(e)}"
    
        # 4. 准备所有大种子任务
        seed_tasks = []
        total_large_items = 0
        
        for key in large_keys:
            if key in self.hash_buckets:
                count = len(self.hash_buckets[key])
                total_large_items += count
                # 为每个大元素创建一个装箱任务
                for _ in range(count):
                    seed_tasks.append(key)
    
        if not seed_tasks:
            self._logger.warning("没有可用的大种子元素")
            return []
    
        # 5. 初始化共享资源管理器
        shared_manager = SharedResourceManager(self.hash_buckets, small_keys, large_keys)
        initial_stats = shared_manager.get_stats()
        
        self._logger.info(f"开始多线程装箱:")
        self._logger.info(f"  🌱 大种子任务: {len(seed_tasks)} 个")
        self._logger.info(f"  🔧 线程数: {max_workers}")
        self._logger.info(f"  📦 目标容量: {box_capacity}")
        self._logger.info(f"  📊 最小装载率: {min_ratio:.1%}")
        self._logger.info(f"  🗂️ 小元素: {initial_stats['small_items_remaining']} 个")
    
        # 6. 多线程执行
        output_boxes = []
        failed_reasons = defaultdict(int)
        start_time = time.time()
    
        with ThreadPoolExecutor(max_workers=max_workers, 
                               thread_name_prefix="PackWorker") as executor:
            
            # 提交所有任务
            future_to_task = {}
            for i, seed_key in enumerate(seed_tasks):
                future = executor.submit(pack_single_seed, seed_key, shared_manager, i)
                future_to_task[future] = (seed_key, i)
            
            # 处理结果
            with tqdm(total=len(seed_tasks), unit='seed', 
                     desc=f'多线程装箱', dynamic_ncols=True) as pbar:
                
                for future in as_completed(future_to_task):
                    seed_key, task_id = future_to_task[future]
                    
                    try:
                        success, box, thread_id, capacity, info = future.result(timeout=30)
                        
                        shared_manager.update_stats(success)
                        
                        if success and box is not None:
                            output_boxes.append(np.array(box, dtype=self.DTYPE_SAMPLE_INFO))
                        else:
                            failed_reasons[info] += 1
                        
                        pbar.update(1)
                        
                        # 每50个任务更新一次描述
                        if pbar.n % 50 == 0:
                            current_stats = shared_manager.get_stats()
                            pbar.set_description(
                                f'装箱进度(成功:{current_stats["successful_boxes"]}, '
                                f'成功率:{current_stats["success_rate"]:.1%}, '
                                f'剩余小元素:{current_stats["small_items_remaining"]})'
                            )
                            
                    except Exception as e:
                        self._logger.error(f"任务 {task_id} (种子key={seed_key}) 执行失败: {e}")
                        failed_reasons[f"执行异常: {str(e)}"] += 1
                        pbar.update(1)
    
        end_time = time.time()
        
        # 7. 输出详细统计信息
        final_stats = shared_manager.get_stats()
        
        if output_boxes:
            total_items = sum(len(box) for box in output_boxes)
            avg_items_per_box = total_items / len(output_boxes)
            total_capacity_used = len(output_boxes) * box_capacity
            
            self._logger.info(f"多线程装箱完成:")
            self._logger.info(f"  ⏱️  总耗时: {end_time - start_time:.2f}秒")
            self._logger.info(f"  📦 成功箱子: {len(output_boxes)}")
            self._logger.info(f"  📊 总成功率: {final_stats['success_rate']:.2%}")
            self._logger.info(f"  📈 平均每箱物品数: {avg_items_per_box:.1f}")
            self._logger.info(f"  💾 总计使用物品: {total_items}")
            self._logger.info(f"  🔗 剩余小元素: {final_stats['small_items_remaining']}")
            self._logger.info(f"  🔑 剩余小key数: {final_stats['available_small_keys']}")
            
            if failed_reasons:
                self._logger.info(f"  ❌ 失败原因统计:")
                for reason, count in failed_reasons.items():
                    self._logger.info(f"     {reason}: {count}次")
        else:
            self._logger.warning("没有成功装箱任何物品")
            self._logger.info(f"失败原因: {dict(failed_reasons)}")
    
        return output_boxes

    def pack_with_min_items_constraint_multithread(self, box_capacity: int = 16384, 
                                                 min_items: int = 10, min_ratio: float = 0.95,
                                                 max_workers: int = None) -> List[np.ndarray]:
        """
        多线程多约束装箱：容量约束 + 最小物品数量约束
        （对于每个箱子内的最少物品数增加限制,保证后续的 attn 时间尽量接近）
        参数:
            box_capacity: 箱子容量
            min_items: 每箱最少物品数量
            min_ratio: 最小装载率阈值
            max_workers: 最大线程数
        
        返回:
            List[np.ndarray]: 满足所有约束的箱子列表
        """
        if max_workers is None:
            max_workers = min(os.cpu_count(), 6)  # 约束问题计算量大，减少线程数
        
        half = box_capacity // 2
        # half = 4096
        print(f"种子筛选条件参数 half:{half}")
        large_keys = [k for k in self.hash_buckets.keys() if k >= half]
        small_keys = [k for k in self.hash_buckets.keys() if k < half]
        
        if not large_keys:
            self._logger.warning("没有找到大种子元素")
            return []
    
        # 1. 种子潜力评估器
        class SeedPotentialAnalyzer:
            def __init__(self, hash_buckets, small_keys, box_capacity, min_items):
                self.hash_buckets = hash_buckets
                self.small_keys = sorted(small_keys)
                self.box_capacity = box_capacity
                self.min_items = min_items
                
            def calculate_potential(self, seed_key: int) -> float:
                """计算种子的装箱成功潜力"""
                remaining_capacity = self.box_capacity - seed_key
                
                if remaining_capacity <= 0:
                    return 0.0
                
                # 统计小元素的分布
                available_small_items = sum(
                    len(self.hash_buckets[k]) for k in self.small_keys 
                    if k in self.hash_buckets
                )
                
                if available_small_items == 0:
                    return 0.0
                
                # 估算能装入的物品数量（保守估计）
                min_small_key = min(self.small_keys) if self.small_keys else remaining_capacity
                max_possible_items = remaining_capacity // min_small_key
                
                # 考虑实际可用性（不是所有小key都有元素）
                practical_items = min(max_possible_items, available_small_items // 2)
                total_items = practical_items + 1  # +1 for seed
                
                # 潜力评分
                count_score = min(total_items / self.min_items, 1.0) if self.min_items > 0 else 1.0
                capacity_score = seed_key / self.box_capacity
                diversity_score = len([k for k in self.small_keys if k <= remaining_capacity]) / len(self.small_keys) if self.small_keys else 0
                
                return count_score * 0.5 + capacity_score * 0.3 + diversity_score * 0.2

        # 2. 增强的共享资源管理器
        class EnhancedSharedManager:
            def __init__(self, hash_buckets, small_keys, large_keys):
                self.lock = threading.RLock()
                self.hash_buckets = hash_buckets
                self.small_keys = sorted(small_keys)
                self.large_keys = large_keys
                
                # 维护可用key的统计信息
                self.key_stats = {}
                self._update_key_stats()
                
                # 性能统计
                self.stats = {
                    'total_attempts': 0,
                    'successful_boxes': 0,
                    'failed_by_count': 0,
                    'failed_by_ratio': 0,
                    'failed_by_capacity': 0
                }
            
            def _update_key_stats(self):
                """更新key的统计信息"""
                self.key_stats = {}
                for k in self.small_keys:
                    if k in self.hash_buckets and len(self.hash_buckets[k]) > 0:
                        self.key_stats[k] = len(self.hash_buckets[k])
                        
            def get_seed_item(self, seed_key: int) -> Tuple[bool, Optional[np.record]]:
                """获取种子元素"""
                with self.lock:
                    if (seed_key in self.hash_buckets and 
                        len(self.hash_buckets[seed_key]) > 0):
                        item = self.hash_buckets[seed_key][0]
                        self.hash_buckets[seed_key] = self.hash_buckets[seed_key][1:]
                        return True, item
                    return False, None
            
            def get_item_by_key(self, target_key: int) -> Tuple[bool, Optional[np.record]]:
                """获取指定key的元素"""
                with self.lock:
                    if (target_key in self.hash_buckets and 
                        len(self.hash_buckets[target_key]) > 0):
                        item = self.hash_buckets[target_key][0]
                        self.hash_buckets[target_key] = self.hash_buckets[target_key][1:]
                        
                        # 更新统计
                        if target_key in self.key_stats:
                            self.key_stats[target_key] -= 1
                            if self.key_stats[target_key] <= 0:
                                del self.key_stats[target_key]
                        
                        return True, item
                    return False, None
            
            def get_available_keys_with_counts(self) -> Dict[int, int]:
                """获取可用key及其元素数量"""
                with self.lock:
                    return self.key_stats.copy()
            
            def rollback_items(self, items_to_rollback: List[Tuple[int, np.record]]):
                """回滚失败装箱的元素"""
                with self.lock:
                    for key, item in reversed(items_to_rollback):  # 逆序回滚
                        self.hash_buckets[key] = np.insert(self.hash_buckets[key], 0, item)
                        # 更新统计
                        if key in self.small_keys:
                            self.key_stats[key] = self.key_stats.get(key, 0) + 1
            
            def update_stats(self, result_type: str):
                """更新统计信息"""
                with self.lock:
                    self.stats['total_attempts'] += 1
                    if result_type in self.stats:
                        self.stats[result_type] += 1
            
            def get_current_stats(self) -> Dict:
                """获取当前统计"""
                with self.lock:
                    total_small_items = sum(
                        len(self.hash_buckets[k]) for k in self.small_keys 
                        if k in self.hash_buckets
                    )
                    return {
                        **self.stats,
                        'remaining_small_items': total_small_items,
                        'available_key_types': len(self.key_stats)
                    }

        # 3. 智能装箱策略
        def is_feasible_quick_check(remaining_capacity: int, current_items: int, 
                                   available_keys: Dict[int, int], min_items: int) -> bool:
            """快速可行性检查"""
            if current_items >= min_items:
                return True
                
            needed_items = min_items - current_items
            if not available_keys:
                return False
            
            # 贪心估算：优先使用小key
            sorted_keys = sorted(available_keys.keys())
            possible_items = 0
            remaining_cap = remaining_capacity
            
            for key in sorted_keys:
                if remaining_cap <= 0:
                    break
                max_from_this_key = min(remaining_cap // key, available_keys[key])
                possible_items += max_from_this_key
                remaining_cap -= max_from_this_key * key
                
                if possible_items >= needed_items:
                    return True
                    
            return False
    
        def select_optimal_key(strategy: str, available_keys: Dict[int, int], 
                              remaining_capacity: int, current_items: int, min_items: int) -> Optional[int]:
            """根据策略选择最优key"""
            suitable_keys = [k for k in available_keys.keys() 
                            if k <= remaining_capacity and available_keys[k] > 0]
            if not suitable_keys:
                return None
            
            if strategy == "prioritize_count":
                # 优先数量：选择最小的key
                return min(suitable_keys)
            elif strategy == "prioritize_capacity":
                # 优先容量：选择最大的key
                return max(suitable_keys)
            elif strategy == "balanced":
                # 平衡策略：选择中等大小，但考虑可用数量
                suitable_keys.sort()
                # 倾向于选择有较多可用元素的key
                key_scores = [(k, available_keys[k] * (remaining_capacity / k)) for k in suitable_keys]
                key_scores.sort(key=lambda x: x[1], reverse=True)
                return key_scores[0][0]
            else:
                return suitable_keys[0]

        # 4. 核心装箱函数
        def pack_single_seed_with_constraints(seed_key: int, shared_manager: EnhancedSharedManager, 
                                            thread_id: int) -> Tuple:
            """带约束的单种子装箱"""
            try:
                # 获取种子
                success, seed_item = shared_manager.get_seed_item(seed_key)
                if not success:
                    shared_manager.update_stats('failed_by_capacity')
                    return False, None, thread_id, 0, 0, "无可用种子"
                
                current_box = [seed_item]
                used_items = [(seed_key, seed_item)]  # 用于回滚
                remaining_capacity = box_capacity - seed_key
                items_count = 1
                
                max_iterations = min_items * 16  # 防止无限循环
                iteration = 0
                
                # 装箱主循环
                while (remaining_capacity > 0 and 
                       items_count < min_items * 8 and  # 允许超过最小数量
                       iteration < max_iterations):
                    
                    iteration += 1
                    available_keys = shared_manager.get_available_keys_with_counts()
                    
                    # 快速可行性检查
                    if (items_count < min_items and 
                        not is_feasible_quick_check(remaining_capacity, items_count, 
                                                  available_keys, min_items)):
                        # 无法达到最小物品数，提前退出
                        shared_manager.rollback_items(used_items)
                        shared_manager.update_stats('failed_by_count')
                        return False, None, thread_id, 0, items_count, f"无法达到{min_items}个物品"
                    
                    # 动态策略选择
                    if items_count < min_items * 0.8:
                        strategy = "prioritize_count"
                    elif items_count < min_items:
                        strategy = "balanced"
                    else:
                        strategy = "prioritize_capacity"
                    
                    # 选择下一个key
                    target_key = select_optimal_key(strategy, available_keys, 
                                                  remaining_capacity, items_count, min_items)
                    if target_key is None:
                        break
                    
                    # 获取元素
                    success, item = shared_manager.get_item_by_key(target_key)
                    if not success:
                        continue  # 该key已被其他线程用完
                    
                    current_box.append(item)
                    used_items.append((target_key, item))
                    remaining_capacity -= target_key
                    items_count += 1
                    
                    # 如果达到完美装载，可以提前结束
                    if remaining_capacity == 0 and items_count >= min_items:
                        break
                
                # 检查所有约束
                current_capacity = box_capacity - remaining_capacity
                load_ratio = current_capacity / box_capacity
                
                meets_count = items_count >= min_items
                meets_ratio = load_ratio >= min_ratio
                meets_capacity = remaining_capacity >= 0
                
                success = meets_count and meets_ratio and meets_capacity
                
                if success:
                    shared_manager.update_stats('successful_boxes')
                    return True, current_box, thread_id, current_capacity, items_count, f"成功：{items_count}个物品，装载率{load_ratio:.1%}"
                else:
                    # 装箱失败，回滚
                    shared_manager.rollback_items(used_items)
                    if not meets_count:
                        shared_manager.update_stats('failed_by_count')
                        reason = f"物品数不足：{items_count}<{min_items}"
                    elif not meets_ratio:
                        shared_manager.update_stats('failed_by_ratio')
                        reason = f"装载率不足：{load_ratio:.1%}<{min_ratio:.1%}"
                    else:
                        shared_manager.update_stats('failed_by_capacity')
                        reason = "容量约束失败"
                    
                    return False, None, thread_id, current_capacity, items_count, reason
                    
            except Exception as e:
                shared_manager.update_stats('failed_by_capacity')
                return False, None, thread_id, 0, 0, f"装箱异常: {str(e)}"
    
        # 5. 种子预处理和筛选
        analyzer = SeedPotentialAnalyzer(self.hash_buckets, small_keys, box_capacity, min_items)
        
        # 收集并评估所有种子
        seed_candidates = []
        for key in large_keys:
            if key in self.hash_buckets:
                potential = analyzer.calculate_potential(key)
                count = len(self.hash_buckets[key])
                for _ in range(count):
                    seed_candidates.append((key, potential))  # 确保是元组
        if not seed_candidates:
            self._logger.warning("没有可用的种子候选")
            return []
        
        # 按潜力排序，只处理高潜力种子
        seed_candidates.sort(key=lambda x: x[1], reverse=True)
        potential_threshold = 0.2  # 只处理潜力>0.2的种子
        # high_potential_seeds = [seed for seed, potential in seed_candidates if potential > potential_threshold]
        # 修复：正确处理筛选逻辑
        high_potential_candidates = [(seed, potential) for seed, potential in seed_candidates 
                                if potential > potential_threshold]

        # 最后保底：至少保留前50%的种子
        if len(high_potential_candidates) < len(seed_candidates) * 0.5:
            mid_point = len(seed_candidates) // 2
            high_potential_candidates = seed_candidates[:mid_point]
        
        # 修复：正确提取种子列表
        selected_seeds = [seed for seed, potential in high_potential_candidates]
        
        self._logger.info(f"种子筛选完成:")
        self._logger.info(f"  📊 总种子数: {len(seed_candidates)}")
        self._logger.info(f"  🎯 筛选后: {len(selected_seeds)}")
        self._logger.info(f"  🚀 筛选率: {len(selected_seeds)/len(seed_candidates):.1%}")
        self._logger.info(f"  🔧 线程数: {max_workers}")
        self._logger.info(f"  📦 约束: 容量≥{min_ratio:.0%}, 物品≥{min_items}个")
    
        # 6. 初始化共享管理器
        shared_manager = EnhancedSharedManager(self.hash_buckets, small_keys, large_keys)
        initial_stats = shared_manager.get_current_stats()
        
        self._logger.info(f"初始资源状态:")
        self._logger.info(f"  🗂️ 小元素总数: {initial_stats['remaining_small_items']}")
        self._logger.info(f"  🔑 可用小key种类: {initial_stats['available_key_types']}")

        # 7. 多线程执行装箱
        output_boxes = []
        detailed_results = []
        start_time = time.time()
        
        with ThreadPoolExecutor(max_workers=max_workers, 
                               thread_name_prefix="ConstraintPack") as executor:
            
            # 提交所有任务
            future_to_seed = {}
            for i, seed_key in enumerate(selected_seeds):
                future = executor.submit(pack_single_seed_with_constraints, seed_key, shared_manager, i)
                future_to_seed[future] = (seed_key, i)
            
            # 处理结果
            with tqdm(total=len(selected_seeds), unit='seed', 
                     desc='多约束装箱', dynamic_ncols=True) as pbar:
                
                completed_tasks = 0
                for future in as_completed(future_to_seed):
                    seed_key, task_id = future_to_seed[future]
                    
                    try:
                        success, box, thread_id, capacity, item_count, info = future.result(timeout=60)
                        
                        if success and box is not None:
                            output_boxes.append(np.array(box, dtype=self.DTYPE_SAMPLE_INFO))
                        
                        detailed_results.append({
                            'seed_key': seed_key,
                            'success': success,
                            'capacity': capacity,
                            'item_count': item_count,
                            'info': info,
                            'thread_id': thread_id
                        })
    
                        completed_tasks += 1
                        pbar.update(1)
                        
                        # 每100个任务更新一次进度描述
                        if completed_tasks % 100 == 0:
                            current_stats = shared_manager.get_current_stats()
                            success_rate = current_stats['successful_boxes'] / max(1, current_stats['total_attempts'])
                            pbar.set_description(
                                f'多约束装箱(成功:{current_stats["successful_boxes"]}, '
                                f'成功率:{success_rate:.1%}, '
                                f'剩余:{current_stats["remaining_small_items"]})'
                            )
                            
                    except Exception as e:
                        self._logger.error(f"任务 {task_id} (种子={seed_key}) 失败: {e}")
                        detailed_results.append({
                            'seed_key': seed_key,
                            'success': False,
                            'capacity': 0,
                            'item_count': 0,
                            'info': f"任务异常: {str(e)}",
                            'thread_id': -1
                        })
                        pbar.update(1)
    
        end_time = time.time()

        # 8. 详细统计分析
        final_stats = shared_manager.get_current_stats()
        
        # 按失败原因分类
        failure_analysis = defaultdict(int)
        success_details = []
        
        for result in detailed_results:
            if result['success']:
                success_details.append(result)
            else:
                # 简化失败原因
                info = result['info']
                if '物品数不足' in info:
                    failure_analysis['物品数量不足'] += 1
                elif '装载率不足' in info:
                    failure_analysis['装载率不足'] += 1
                elif '无可用种子' in info:
                    failure_analysis['种子耗尽'] += 1
                elif '无法达到' in info:
                    failure_analysis['可行性检查失败'] += 1
                else:
                    failure_analysis['其他原因'] += 1

        # 9. 输出详细报告
        if output_boxes:
            # 成功装箱的统计
            total_items_packed = sum(len(box) for box in output_boxes)
            avg_items_per_box = total_items_packed / len(output_boxes)
            capacities = [result['capacity'] for result in success_details]
            avg_capacity = sum(capacities) / len(capacities) if capacities else 0
            avg_load_ratio = avg_capacity / box_capacity
            
            item_counts = [result['item_count'] for result in success_details]
            min_items_in_box = min(item_counts) if item_counts else 0
            max_items_in_box = max(item_counts) if item_counts else 0
            
            self._logger.info(f"🎉 多约束装箱完成!")
            self._logger.info(f"📊 执行统计:")
            self._logger.info(f"  ⏱️  总耗时: {end_time - start_time:.2f}秒")
            self._logger.info(f"  🎯 处理种子: {len(selected_seeds)}")
            self._logger.info(f"  📦 成功箱子: {len(output_boxes)}")
            self._logger.info(f"  📈 总成功率: {len(output_boxes)/len(selected_seeds):.2%}")
            
            self._logger.info(f"📦 装箱质量:")
            self._logger.info(f"  📊 平均装载率: {avg_load_ratio:.1%}")
            self._logger.info(f"  🔢 平均物品数: {avg_items_per_box:.1f}")
            self._logger.info(f"  📉 物品数范围: {min_items_in_box}-{max_items_in_box}")
            self._logger.info(f"  💾 总打包物品: {total_items_packed}")
            
            self._logger.info(f"🔗 剩余资源:")
            self._logger.info(f"  🗂️ 小元素: {final_stats['remaining_small_items']}")
            self._logger.info(f"  🔑 可用key类型: {final_stats['available_key_types']}")
            
            if failure_analysis:
                self._logger.info(f"❌ 失败分析:")
                for reason, count in failure_analysis.items():
                    percentage = count / len(selected_seeds) * 100
                    self._logger.info(f"     {reason}: {count}次 ({percentage:.1f}%)")
        else:
            self._logger.warning("⚠️  没有成功装箱任何物品!")
            self._logger.info(f"失败原因分布: {dict(failure_analysis)}")
            self._logger.info(f"建议:")
            self._logger.info(f"  1. 降低 min_items (当前: {min_items})")
            self._logger.info(f"  2. 降低 min_ratio (当前: {min_ratio})")
            self._logger.info(f"  3. 检查数据分布是否合理")
    
        return output_boxes

    # '''
    #     # 10. 将函数绑定到类
    #     # HashBucketProcessor.pack_with_min_items_constraint_multithread = pack_with_min_items_constraint_multithread
        
    #     # 11. 使用示例
    #     def demo_constrained_packing():
    #         """演示多约束装箱的使用"""
            
    #         # 创建处理器
    #         processor = HashBucketProcessor("data.txt")
    #         processor.build_buckets()
    #         processor.find_items(16384)
            
    #         print("=== 原始装箱（无物品数量约束） ===")
    #         boxes_original = processor.pack_large_seed_parallel_multithread(
    #             box_capacity=16384,
    #             min_ratio=0.95,
    #             max_workers=4
    #         )
    #         print(f"原始方法成功箱子: {len(boxes_original)}")
            
    #         print("\n=== 多约束装箱（至少10个物品） ===")
    #         boxes_constrained = processor.pack_with_min_items_constraint_multithread(
    #             box_capacity=16384,
    #             min_items=10,
    #             min_ratio=0.95,
    #             max_workers=4
    #         )
    #         print(f"约束方法成功箱子: {len(boxes_constrained)}")
            
    #         # 对比分析
    #         if boxes_constrained:
    #             avg_items_constrained = sum(len(box) for box in boxes_constrained) / len(boxes_constrained)
    #             print(f"约束方法平均每箱物品数: {avg_items_constrained:.1f}")
            
    #         return boxes_original, boxes_constrained
        
    #     # 如果想直接运行演示
    #     if __name__ == "__main__":
    #         # demo_constrained_packing()
    #         pass
    # '''
    def pack_with_flexible_seeds(self, box_capacity: int = 16384,
                               seed_strategy: str = "auto",
                               seed_params: dict = None,
                               min_items: int = 10, min_ratio: float = 0.95,
                               max_workers: int = None) -> List[np.ndarray]:
        """
        自定义种子选择策略 + 背包元素数限制 + 输出背包最小容量
        
        参数:
            seed_strategy: 种子策略
                - "auto": 自动使用 box_capacity // 2
                - "custom_half": 使用自定义的half值
                - "specified_keys": 使用指定的key列表
                - "size_range": 使用大小范围筛选
                - "top_n": 使用最大的N个keys作为种子
                - "capacity_ratio": 指定占用 capacity 的百分比的
            seed_params: 策略参数字典
                - 
        """
        if max_workers is None:
            max_workers = min(os.cpu_count(), 6)
        
        if seed_params is None:
            seed_params = {}
        
        # 🎯 根据策略生成种子
        if seed_strategy == "auto":
            half = box_capacity // 2
            large_keys = [k for k in self.hash_buckets.keys() if k >= half]
            
        elif seed_strategy == "custom_half":
            custom_half = seed_params.get("half", box_capacity // 3)
            # max_elems = seed_params.get("n_max", None)         # 每个 key 中最多取出的种子数÷
            large_keys = [k for k in self.hash_buckets.keys() if k >= custom_half]
            
        elif seed_strategy == "specified_keys":
            specified_keys = seed_params.get("keys", [])
            large_keys = [k for k in specified_keys if k in self.hash_buckets]
            
        elif seed_strategy == "size_range":
            min_size = seed_params.get("min_size", box_capacity // 3)
            max_size = seed_params.get("max_size", box_capacity)
            large_keys = [k for k in self.hash_buckets.keys() 
                         if min_size <= k <= max_size]
            
        elif seed_strategy == "top_n":
            n = seed_params.get("n", 5)
            available_keys = sorted(self.hash_buckets.keys(), reverse=True)
            large_keys = available_keys[:n]
            
        elif seed_strategy == "capacity_ratio":
            min_ratio = seed_params.get("min_ratio", 0.3)  # 至少30%容量
            max_ratio = seed_params.get("max_ratio", 1.0)  # 最多100%容量
            min_size = int(box_capacity * min_ratio)
            max_size = int(box_capacity * max_ratio)
            large_keys = [k for k in self.hash_buckets.keys() 
                         if min_size <= k <= max_size]

        # # 利用分位数 Quartiles 进行种子的选择(Q1,Q2,Q3)
        # elif seed_strategy == "quartiles":
        #     q_n = seed_params.get("q_n", 3)  # 至少30%容量
        #     elems_max_num = seed_params.get("max_num", 20)  # 每一个 key 里面取出作为种子的的最大元素个数
        #     pass
        
        else:
            raise ValueError(f"不支持的种子策略: {seed_strategy}")
        
        # 生成小元素列表
        small_keys = [k for k in self.hash_buckets.keys() if k not in large_keys]
        
        # 策略信息
        self._logger.info(f"种子策略: {seed_strategy}")
        self._logger.info(f"策略参数: {seed_params}")
        self._logger.info(f"  🌱 种子keys(max): {large_keys[-1]}")
        self._logger.info(f"  🔧 填充keys数量: {len(small_keys)}")
        
        if not large_keys:
            self._logger.warning(f"策略 {seed_strategy} 没有生成任何种子")
            return []


        # 1. 种子潜力评估器
        class SeedPotentialAnalyzer:
            def __init__(self, hash_buckets, small_keys, box_capacity, min_items):
                self.hash_buckets = hash_buckets
                self.small_keys = sorted(small_keys)
                self.box_capacity = box_capacity
                self.min_items = min_items
                
            def calculate_potential(self, seed_key: int) -> float:
                """计算种子的装箱成功潜力"""
                remaining_capacity = self.box_capacity - seed_key
                
                if remaining_capacity <= 0:
                    return 0.0
                
                # 统计小元素的分布
                available_small_items = sum(
                    len(self.hash_buckets[k]) for k in self.small_keys 
                    if k in self.hash_buckets
                )
                
                if available_small_items == 0:
                    return 0.0
                
                # 估算能装入的物品数量（保守估计）
                min_small_key = min(self.small_keys) if self.small_keys else remaining_capacity
                max_possible_items = remaining_capacity // min_small_key
                
                # 考虑实际可用性（不是所有小key都有元素）
                practical_items = min(max_possible_items, available_small_items // 2)
                total_items = practical_items + 1  # +1 for seed
                
                # 潜力评分
                count_score = min(total_items / self.min_items, 1.0) if self.min_items > 0 else 1.0
                capacity_score = seed_key / self.box_capacity
                diversity_score = len([k for k in self.small_keys if k <= remaining_capacity]) / len(self.small_keys) if self.small_keys else 0
                
                return count_score * 0.5 + capacity_score * 0.3 + diversity_score * 0.2

        # 2. 增强的共享资源管理器
        class EnhancedSharedManager:
            def __init__(self, hash_buckets, small_keys, large_keys):
                self.lock = threading.RLock()
                self.hash_buckets = hash_buckets
                self.small_keys = sorted(small_keys)
                self.large_keys = large_keys
                
                # 维护可用key的统计信息
                self.key_stats = {}
                self._update_key_stats()
                
                # 性能统计
                self.stats = {
                    'total_attempts': 0,
                    'successful_boxes': 0,
                    'failed_by_count': 0,
                    'failed_by_ratio': 0,
                    'failed_by_capacity': 0
                }
            
            def _update_key_stats(self):
                """更新key的统计信息"""
                self.key_stats = {}
                for k in self.small_keys:
                    if k in self.hash_buckets and len(self.hash_buckets[k]) > 0:
                        self.key_stats[k] = len(self.hash_buckets[k])
                        
            def get_seed_item(self, seed_key: int) -> Tuple[bool, Optional[np.record]]:
                """获取种子元素"""
                with self.lock:
                    if (seed_key in self.hash_buckets and 
                        len(self.hash_buckets[seed_key]) > 0):
                        item = self.hash_buckets[seed_key][0]
                        self.hash_buckets[seed_key] = self.hash_buckets[seed_key][1:]
                        return True, item
                    return False, None
            
            def get_item_by_key(self, target_key: int) -> Tuple[bool, Optional[np.record]]:
                """获取指定key的元素"""
                with self.lock:
                    if (target_key in self.hash_buckets and 
                        len(self.hash_buckets[target_key]) > 0):
                        item = self.hash_buckets[target_key][0]
                        self.hash_buckets[target_key] = self.hash_buckets[target_key][1:]
                        
                        # 更新统计
                        if target_key in self.key_stats:
                            self.key_stats[target_key] -= 1
                            if self.key_stats[target_key] <= 0:
                                del self.key_stats[target_key]
                        
                        return True, item
                    return False, None

            def get_available_keys_with_counts(self) -> Dict[int, int]:
                """获取可用key及其元素数量"""
                with self.lock:
                    return self.key_stats.copy()
            
            def rollback_items(self, items_to_rollback: List[Tuple[int, np.record]]):
                """回滚失败装箱的元素"""
                with self.lock:
                    for key, item in reversed(items_to_rollback):  # 逆序回滚
                        self.hash_buckets[key] = np.insert(self.hash_buckets[key], 0, item)
                        # 更新统计
                        if key in self.small_keys:
                            self.key_stats[key] = self.key_stats.get(key, 0) + 1
            
            def update_stats(self, result_type: str):
                """更新统计信息"""
                with self.lock:
                    self.stats['total_attempts'] += 1
                    if result_type in self.stats:
                        self.stats[result_type] += 1
            
            def get_current_stats(self) -> Dict:
                """获取当前统计"""
                with self.lock:
                    total_small_items = sum(
                        len(self.hash_buckets[k]) for k in self.small_keys 
                        if k in self.hash_buckets
                    )
                    return {
                        **self.stats,
                        'remaining_small_items': total_small_items,
                        'available_key_types': len(self.key_stats)
                    }

        # 3. 智能装箱策略
        def is_feasible_quick_check(remaining_capacity: int, current_items: int, 
                                   available_keys: Dict[int, int], min_items: int) -> bool:
            """快速可行性检查"""
            if current_items >= min_items:
                return True
                
            needed_items = min_items - current_items
            if not available_keys:
                return False
            
            # 贪心估算：优先使用小key
            sorted_keys = sorted(available_keys.keys())
            possible_items = 0
            remaining_cap = remaining_capacity
            
            for key in sorted_keys:
                if remaining_cap <= 0:
                    break
                max_from_this_key = min(remaining_cap // key, available_keys[key])
                possible_items += max_from_this_key
                remaining_cap -= max_from_this_key * key
                
                if possible_items >= needed_items:
                    return True
                    
            return False
    
        def select_optimal_key(strategy: str, available_keys: Dict[int, int], 
                              remaining_capacity: int, current_items: int, min_items: int) -> Optional[int]:
            """根据策略选择最优key"""
            suitable_keys = [k for k in available_keys.keys() 
                            if k <= remaining_capacity and available_keys[k] > 0]
            if not suitable_keys:
                return None
            
            if strategy == "prioritize_count":
                # 优先数量：选择最小的key
                return min(suitable_keys)
            elif strategy == "prioritize_capacity":
                # 优先容量：选择最大的key
                return max(suitable_keys)
            elif strategy == "balanced":
                # 平衡策略：选择中等大小，但考虑可用数量
                suitable_keys.sort()
                # 倾向于选择有较多可用元素的key
                key_scores = [(k, available_keys[k] * (remaining_capacity / k)) for k in suitable_keys]
                key_scores.sort(key=lambda x: x[1], reverse=True)
                return key_scores[0][0]
            else:
                return suitable_keys[0]

        # 4. 核心装箱函数
        def pack_single_seed_with_constraints(seed_key: int, shared_manager: EnhancedSharedManager, 
                                            thread_id: int) -> Tuple:
            """带约束的单种子装箱"""
            try:
                # 获取种子
                success, seed_item = shared_manager.get_seed_item(seed_key)
                if not success:
                    shared_manager.update_stats('failed_by_capacity')
                    return False, None, thread_id, 0, 0, "无可用种子"
                
                current_box = [seed_item]
                used_items = [(seed_key, seed_item)]  # 用于回滚
                remaining_capacity = box_capacity - seed_key
                items_count = 1
                
                max_iterations = min_items * 16  # 防止无限循环  由 5--->15(12 for 16384)
                iteration = 0
                
                # 装箱主循环
                while (remaining_capacity > 0 and 
                       items_count < min_items * 8 and  # 允许超过最小数量（可能有非常小的值）(5 for 16384)
                       iteration < max_iterations):
                    
                    iteration += 1
                    available_keys = shared_manager.get_available_keys_with_counts()
                    
                    # 快速可行性检查
                    if (items_count < min_items and 
                        not is_feasible_quick_check(remaining_capacity, items_count, 
                                                  available_keys, min_items)):
                        # 无法达到最小物品数，提前退出
                        shared_manager.rollback_items(used_items)
                        shared_manager.update_stats('failed_by_count')
                        return False, None, thread_id, 0, items_count, f"无法达到{min_items}个物品"
                    
                    # 动态策略选择
                    if items_count < min_items * 0.8:
                        strategy = "prioritize_count"
                    elif items_count < min_items:
                        strategy = "balanced"
                    else:
                        strategy = "prioritize_capacity"
                    
                    # 选择下一个key
                    target_key = select_optimal_key(strategy, available_keys, 
                                                  remaining_capacity, items_count, min_items)
                    if target_key is None:
                        break
                    
                    # 获取元素
                    success, item = shared_manager.get_item_by_key(target_key)
                    if not success:
                        continue  # 该key已被其他线程用完
                    
                    current_box.append(item)
                    used_items.append((target_key, item))
                    remaining_capacity -= target_key
                    items_count += 1
                    
                    # 如果达到完美装载，可以提前结束
                    if remaining_capacity == 0 and items_count >= min_items:
                        break
                
                # 检查所有约束
                current_capacity = box_capacity - remaining_capacity
                load_ratio = current_capacity / box_capacity
                
                meets_count = items_count >= min_items
                meets_ratio = load_ratio >= min_ratio
                meets_capacity = remaining_capacity >= 0
                
                success = meets_count and meets_ratio and meets_capacity
                
                if success:
                    shared_manager.update_stats('successful_boxes')
                    return True, current_box, thread_id, current_capacity, items_count, f"成功：{items_count}个物品，装载率{load_ratio:.1%}"
                else:
                    # 装箱失败，回滚
                    shared_manager.rollback_items(used_items)
                    if not meets_count:
                        shared_manager.update_stats('failed_by_count')
                        reason = f"物品数不足：{items_count}<{min_items}"
                    elif not meets_ratio:
                        shared_manager.update_stats('failed_by_ratio')
                        reason = f"装载率不足：{load_ratio:.1%}<{min_ratio:.1%}"
                    else:
                        shared_manager.update_stats('failed_by_capacity')
                        reason = "容量约束失败"
                    
                    return False, None, thread_id, current_capacity, items_count, reason
                    
            except Exception as e:
                shared_manager.update_stats('failed_by_capacity')
                return False, None, thread_id, 0, 0, f"装箱异常: {str(e)}"

        # 5. 种子预处理和筛选
        analyzer = SeedPotentialAnalyzer(self.hash_buckets, small_keys, box_capacity, min_items)
        
        # 收集并评估所有种子
        seed_candidates = []
        for key in large_keys:
            if key in self.hash_buckets:
                potential = analyzer.calculate_potential(key)
                count = len(self.hash_buckets[key])
                for _ in range(count):
                    seed_candidates.append((key, potential))  # 确保是元组
        if not seed_candidates:
            self._logger.warning("没有可用的种子候选")
            return []
        
        # 按潜力排序，只处理高潜力种子
        seed_candidates.sort(key=lambda x: x[1], reverse=True)
        potential_threshold = 0.2  # 只处理潜力>0.2的种子
        # high_potential_seeds = [seed for seed, potential in seed_candidates if potential > potential_threshold]
        # 修复：正确处理筛选逻辑
        high_potential_candidates = [(seed, potential) for seed, potential in seed_candidates 
                                if potential > potential_threshold]

        # 最后保底：至少保留前50%的种子
        if len(high_potential_candidates) < len(seed_candidates) * 0.5:
            mid_point = len(seed_candidates) // 2
            high_potential_candidates = seed_candidates[:mid_point]
        
        # 修复：正确提取种子列表
        selected_seeds = [seed for seed, potential in high_potential_candidates]
        
        self._logger.info(f"种子筛选完成:")
        self._logger.info(f"  📊 总种子数: {len(seed_candidates)}")
        self._logger.info(f"  🎯 筛选后: {len(selected_seeds)}")
        self._logger.info(f"  🚀 筛选率: {len(selected_seeds)/len(seed_candidates):.1%}")
        self._logger.info(f"  🔧 线程数: {max_workers}")
        self._logger.info(f"  📦 约束: 容量≥{min_ratio:.0%}, 物品≥{min_items}个")
    
        # 6. 初始化共享管理器
        shared_manager = EnhancedSharedManager(self.hash_buckets, small_keys, large_keys)
        initial_stats = shared_manager.get_current_stats()
        
        self._logger.info(f"初始资源状态:")
        self._logger.info(f"  🗂️ 小元素总数: {initial_stats['remaining_small_items']}")
        self._logger.info(f"  🔑 可用小key种类: {initial_stats['available_key_types']}")

        # 7. 多线程执行装箱
        output_boxes = []
        detailed_results = []
        start_time = time.time()
        
        with ThreadPoolExecutor(max_workers=max_workers, 
                               thread_name_prefix="ConstraintPack") as executor:
            
            # 提交所有任务
            future_to_seed = {}
            for i, seed_key in enumerate(selected_seeds):
                future = executor.submit(pack_single_seed_with_constraints, seed_key, shared_manager, i)
                future_to_seed[future] = (seed_key, i)
            
            # 处理结果
            with tqdm(total=len(selected_seeds), unit='seed', 
                     desc='多约束装箱', dynamic_ncols=True) as pbar:
                
                completed_tasks = 0
                for future in as_completed(future_to_seed):
                    seed_key, task_id = future_to_seed[future]
                    
                    try:
                        success, box, thread_id, capacity, item_count, info = future.result(timeout=60)
                        
                        if success and box is not None:
                            output_boxes.append(np.array(box, dtype=self.DTYPE_SAMPLE_INFO))
                        
                        detailed_results.append({
                            'seed_key': seed_key,
                            'success': success,
                            'capacity': capacity,
                            'item_count': item_count,
                            'info': info,
                            'thread_id': thread_id
                        })
    
                        completed_tasks += 1
                        pbar.update(1)
                        
                        # 每100个任务更新一次进度描述
                        if completed_tasks % 100 == 0:
                            current_stats = shared_manager.get_current_stats()
                            success_rate = current_stats['successful_boxes'] / max(1, current_stats['total_attempts'])
                            pbar.set_description(
                                f'多约束装箱(成功:{current_stats["successful_boxes"]}, '
                                f'成功率:{success_rate:.1%}, '
                                f'剩余:{current_stats["remaining_small_items"]})'
                            )
                            
                    except Exception as e:
                        self._logger.error(f"任务 {task_id} (种子={seed_key}) 失败: {e}")
                        detailed_results.append({
                            'seed_key': seed_key,
                            'success': False,
                            'capacity': 0,
                            'item_count': 0,
                            'info': f"任务异常: {str(e)}",
                            'thread_id': -1
                        })
                        pbar.update(1)
    
        end_time = time.time()

        # 8. 详细统计分析
        final_stats = shared_manager.get_current_stats()
        
        # 按失败原因分类
        failure_analysis = defaultdict(int)
        success_details = []
        
        for result in detailed_results:
            if result['success']:
                success_details.append(result)
            else:
                # 简化失败原因
                info = result['info']
                if '物品数不足' in info:
                    failure_analysis['物品数量不足'] += 1
                elif '装载率不足' in info:
                    failure_analysis['装载率不足'] += 1
                elif '无可用种子' in info:
                    failure_analysis['种子耗尽'] += 1
                elif '无法达到' in info:
                    failure_analysis['可行性检查失败'] += 1
                else:
                    failure_analysis['其他原因'] += 1

        # 9. 输出详细报告
        if output_boxes:
            # 成功装箱的统计
            total_items_packed = sum(len(box) for box in output_boxes)
            avg_items_per_box = total_items_packed / len(output_boxes)
            capacities = [result['capacity'] for result in success_details]
            avg_capacity = sum(capacities) / len(capacities) if capacities else 0
            avg_load_ratio = avg_capacity / box_capacity
            
            item_counts = [result['item_count'] for result in success_details]
            min_items_in_box = min(item_counts) if item_counts else 0
            max_items_in_box = max(item_counts) if item_counts else 0
            
            self._logger.info(f"🎉 多约束装箱完成!")
            self._logger.info(f"📊 执行统计:")
            self._logger.info(f"  ⏱️  总耗时: {end_time - start_time:.2f}秒")
            self._logger.info(f"  🎯 处理种子: {len(selected_seeds)}")
            self._logger.info(f"  📦 成功箱子: {len(output_boxes)}")
            self._logger.info(f"  📈 总成功率: {len(output_boxes)/len(selected_seeds):.2%}")
            
            self._logger.info(f"📦 装箱质量:")
            self._logger.info(f"  📊 平均装载率: {avg_load_ratio:.1%}")
            self._logger.info(f"  🔢 平均物品数: {avg_items_per_box:.1f}")
            self._logger.info(f"  📉 物品数范围: {min_items_in_box}-{max_items_in_box}")
            self._logger.info(f"  💾 总打包物品: {total_items_packed}")
            
            self._logger.info(f"🔗 剩余资源:")
            self._logger.info(f"  🗂️ 小元素: {final_stats['remaining_small_items']}")
            self._logger.info(f"  🔑 可用key类型: {final_stats['available_key_types']}")
            
            if failure_analysis:
                self._logger.info(f"❌ 失败分析:")
                for reason, count in failure_analysis.items():
                    percentage = count / len(selected_seeds) * 100
                    self._logger.info(f"     {reason}: {count}次 ({percentage:.1f}%)")
        else:
            self._logger.warning("⚠️  没有成功装箱任何物品!")
            self._logger.info(f"失败原因分布: {dict(failure_analysis)}")
            self._logger.info(f"建议:")
            self._logger.info(f"  1. 降低 min_items (当前: {min_items})")
            self._logger.info(f"  2. 降低 min_ratio (当前: {min_ratio})")
            self._logger.info(f"  3. 检查数据分布是否合理")


        
        # 只输出 1 个用于状态跟踪，输出3个用于实际应用
        return output_boxes#, failure_analysis, final_stats

    def pack_simplest_strategy(
        self,
        keys: List[int],
        m: int,
        box_capacity: int = 16384,
        min_ratio: float = 0.95,
        max_workers: int = None,
    ) -> List[np.ndarray]:
        """
        极简装箱策略：
        1. 从指定 keys 中随机选 m 个种子；
        2. 其余所有剩余元素作为装填池；
        3. 多线程装箱（成功删，失败回滚）；
        4. 剩余元素单线程兜底，最后一批强制输出并清空。
        """
        import random
        import threading
        from concurrent.futures import ThreadPoolExecutor, as_completed

        if max_workers is None:
            max_workers = min(os.cpu_count(), 8)

        # ---------- 1. 构造种子池 & 装填池 ----------
        seed_pool = []           # [(key, item), ...]
        fill_buckets = defaultdict(list)   # key -> [item, ...]

        # 1.1 收集种子池 & 未选中的同 key 元素
        for k in keys:
            if k not in self.hash_buckets or len(self.hash_buckets[k]) == 0:
                continue
            arr = self.hash_buckets[k]
            # 随机选 m 个，若不足则全选
            chosen = random.sample(list(arr), min(m, len(arr)))
            seed_pool.extend([(k, item) for item in chosen])
            # 未选中的进入装填池
            mask = np.ones(len(arr), dtype=bool)
            idxs = [i for i, it in enumerate(arr) if it in chosen]
            mask[idxs] = False
            fill_buckets[k].extend(arr[mask])

        # 1.2 非 keys 的所有元素归入装填池
        for k in self.hash_buckets:
            if k not in keys:
                fill_buckets[k].extend(self.hash_buckets[k])

        if not seed_pool:
            self._logger.warning("种子池为空，直接输出剩余元素为一箱")
            # 强制输出一箱
            leftover = []
            for k, items in fill_buckets.items():
                leftover.extend(items)
            if leftover:
                box = np.array(leftover, dtype=self.DTYPE_SAMPLE_INFO)
                # 清空 hash_buckets
                for k in list(self.hash_buckets.keys()):
                    del self.hash_buckets[k]
                return [box]
            return []

        # ---------- 2. 线程安全的资源管理器 ----------
        class SimpleManager:
            def __init__(self, seed_items, fill_dict, dtype):
                self.lock = threading.RLock()
                # 种子队列
                self.seed_q = seed_items[:]          # 拷贝
                # 装填池
                self.fill = defaultdict(deque)
                for k, lst in fill_dict.items():
                    self.fill[k] = deque(lst)
                # 统计
                self.boxes = []
                self.attempts = 0
                self.success = 0
                self.dtype = dtype

            def pop_seed(self):
                with self.lock:
                    if not self.seed_q:
                        return None
                    return self.seed_q.pop()

            def pop_fill(self, key):
                with self.lock:
                    if not self.fill[key]:
                        return None
                    return self.fill[key].popleft()

            def add_box(self, box):
                with self.lock:
                    self.boxes.append(np.array(box, dtype=self.dtype))
                    self.success += 1

            def rollback(self, rollback_items):
                with self.lock:
                    for key, item in reversed(rollback_items):
                        self.fill[key].appendleft(item)

            def remaining_elements(self):
                with self.lock:
                    return sum(len(q) for q in self.fill.values())

            def all_items(self):
                with self.lock:
                    items = []
                    for k, q in self.fill.items():
                        items.extend(q)
                    return items

        from collections import deque
        # mgr = SimpleManager(seed_pool, fill_buckets)
        mgr = SimpleManager(seed_pool, fill_buckets, self.DTYPE_SAMPLE_INFO)

        # ---------- 3. 多线程装箱 ----------
        def pack_once(args):
            seed_key, seed_item, tid = args
            box = [seed_item]
            used = [(seed_key, seed_item)]
            rem = box_capacity - seed_key

            # 贪心装填
            for k in sorted(mgr.fill.keys(), reverse=True):
                while rem >= k and mgr.fill[k]:
                    it = mgr.pop_fill(k)
                    if it is None:
                        break
                    box.append(it)
                    used.append((k, it))
                    rem -= k
                    if rem == 0:
                        break

            load = box_capacity - rem
            if load >= min_ratio * box_capacity:
                mgr.add_box(box)
                return True, tid, load
            else:
                mgr.rollback(used)
                return False, tid, load

        # 构造任务列表
        tasks = [(k, it, i) for i, (k, it) in enumerate(mgr.seed_q)]
        mgr.seed_q.clear()   # 清空，由任务列表取代

        with ThreadPoolExecutor(max_workers=max_workers) as exe:
            futs = [exe.submit(pack_once, t) for t in tasks]
            for f in as_completed(futs):
                ok, tid, load = f.result()
                mgr.attempts += 1

        # ---------- 4. 单线程兜底 ----------
        leftover_keys = list(mgr.fill.keys())
        random.shuffle(leftover_keys)

        while mgr.remaining_elements() > 0:
            # 随机找一个种子：从剩余 key 中随机挑一个元素
            candidates = [(k, mgr.fill[k][0]) for k in leftover_keys if mgr.fill[k]]
            if not candidates:
                break
            seed_key, seed_item = random.choice(candidates)
            mgr.pop_fill(seed_key)  # 取出作为种子

            box = [seed_item]
            used = [(seed_key, seed_item)]
            rem = box_capacity - seed_key

            # 继续装填
            for k in sorted(mgr.fill.keys(), reverse=True):
                while rem >= k and mgr.fill[k]:
                    it = mgr.pop_fill(k)
                    if it is None:
                        break
                    box.append(it)
                    used.append((k, it))
                    rem -= k
                    if rem == 0:
                        break

            # 强制输出
            mgr.add_box(box)

        # ---------- 5. 同步回 self.hash_buckets ----------
        # 此时 mgr.fill 已全部清空，hash_buckets 直接置空
        for k in list(self.hash_buckets.keys()):
            del self.hash_buckets[k]

        self._logger.info(
            f"pack_simplest_strategy 完成：多线程任务 {mgr.attempts}，"
            f"成功 {mgr.success}，兜底输出 {len(mgr.boxes) - mgr.success} 箱"
        )
        return mgr.boxes



    
    def check_hash_buckets_state(self):
        """检查哈希桶的当前状态"""
        total_items = sum(len(arr) for arr in self.hash_buckets.values())
        # total_keys = len(self.hash_buckets)
        total_keys = len([key for key in self.hash_buckets if len(self.hash_buckets[key])>0])  # 没有删除 元素为0 的 key
        
        # 按key大小分类统计
        key_distribution = defaultdict(int)
        for key in self.hash_buckets.keys():
            if key >= 8192:
                key_distribution['large'] += len(self.hash_buckets[key])
            elif key >= 2048:
                key_distribution['medium'] += len(self.hash_buckets[key])
            else:
                key_distribution['small'] += len(self.hash_buckets[key])
        
        print(f"当前哈希桶状态:")
        print(f"  📦 总元素数: {total_items}")
        print(f"  🔑 总key数: {total_keys}")
        print(f"  📊 分布情况:")
        for size, count in key_distribution.items():
            print(f"    {size}: {count} 个元素")
        
        return {
            'total_items': total_items,
            'total_keys': total_keys,
            'key_distribution': dict(key_distribution)
        }

# ###----------------------------------------------------------------------------------------------
# ###----------------------------------------------------------------------------------------------
# ###----------------------------------------------------------------------------------------------
# class PackingTracker:
#     """装箱操作追踪器"""
#     def __init__(self, processor):
#         self.processor = processor
#         self.history = []

#     def track_packing(self, strategy_name: str, **kwargs):
#         """记录一次装箱操作"""

#         # 记录操作前状态
#         before_state = self.processor.check_hash_buckets_state()

#         # 执行装箱
#         boxes = getattr(self.processor, strategy_name)(**kwargs)

#         # 记录操作后状态
#         after_state = self.processor.check_hash_buckets_state()

#         # 计算变化
#         change = {
#             'strategy': strategy_name,
#             'kwargs': kwargs,
#             'before': before_state,
#             'after': after_state,
#             'boxes_count': len(boxes),
#             'items_used': before_state['total_items'] - after_state['total_items']
#         }

#         self.history.append(change)
#         return boxes

#     def print_summary(self):
#         """打印装箱历史摘要"""
#         print("\n=== 装箱操作历史 ===")
#         for i, op in enumerate(self.history, 1):
#             print(f"\n操作 {i}: {op['strategy']}")
#             print(f"参数: {op['kwargs']}")
#             print(f"装箱数: {op['boxes_count']}")
#             print(f"使用元素: {op['items_used']}")
#             print(f"成功率: {op['boxes_count'] / op['kwargs'].get('max_workers', 1):.1%}")

class PackingTracker:
    """装箱操作追踪器"""
    def __init__(self, processor):
        self.processor = processor
        self.history = []
        self.snapshots = []  # 添加状态快照(2025.09.12)
        
    def track_packing(self, strategy_name: str, **kwargs):
        """记录一次装箱操作"""
        
        # 保存状态快照 (2025.09.12)
        self.save_current_state()

        before_state = self.processor.check_hash_buckets_state()
        # 支持返回详细统计（如 total_attempts），否则只返回箱子列表
        result = getattr(self.processor, strategy_name)(**kwargs)
        if isinstance(result, tuple) and len(result) >= 2 and isinstance(result[1], dict):
            boxes = result[0]
            stats = result[1]
            total_attempts = stats.get('total_attempts', len(boxes))
        else:
            boxes = result
            total_attempts = len(boxes)
        after_state = self.processor.check_hash_buckets_state()
        change = {
            'strategy': strategy_name,
            'kwargs': kwargs,
            'before': before_state,
            'after': after_state,
            'boxes_count': len(boxes),
            'items_used': before_state['total_items'] - after_state['total_items'],
            'total_attempts': total_attempts
        }
        self.history.append(change)
        return boxes
    
    def print_summary(self):
        """打印装箱历史摘要"""
        print("\n=== 装箱操作历史 ===")
        for i, op in enumerate(self.history, 1):
            print(f"\n操作 {i}: {op['strategy']}")
            print(f"参数: {op['kwargs']}")
            print(f"装箱数: {op['boxes_count']}")
            print(f"使用元素: {op['items_used']}")
            if op.get('total_attempts', 0):
                rate = op['boxes_count'] / op['total_attempts']
                print(f"成功率: {rate:.1%} ({op['boxes_count']}/{op['total_attempts']})")
            else:
                print(f"成功率: N/A")
                
    # (2025.09.12)
    def save_current_state(self):
        """保存当前状态快照"""
        snapshot = {
            'hash_buckets': {k: arr.copy() for k, arr in self.processor.hash_buckets.items()},
            'timestamp': time.time()
        }
        self.snapshots.append(snapshot)  # 这里已经完成了追加操作
        print("save checkpoint........")
        # return snapshot
    
    # (2025.09.12)
    # 
    def restore_state(self, index: int):
        """恢复到指定操作前的状态"""
        if 0 <= index < len(self.snapshots):
            snapshot = self.snapshots[index]
            self.processor.hash_buckets = {
                k: arr.copy() for k, arr in snapshot['hash_buckets'].items()
            }
            # 清理该索引之后的所有状态和历史记录
            self.snapshots = self.snapshots[:index+1]
            self.history = self.history[:index]
            print(f"已恢复到操作 {index} 之前的状态")
            return True
        return False


# # 使用示例
# tracker = PackingTracker(processor)
# tracker.track_packing('pack_large_seed_parallel_multithread', 
#                      box_capacity=16384, min_ratio=0.95)
# tracker.track_packing('pack_with_min_items_constraint_multithread',
#                      box_capacity=16384, min_items=10, min_ratio=0.90)
# tracker.print_summary()


def analyze_packing_history(tracker):
    """分析装箱历史"""
    print("\n=== 详细分析 ===")
    
    total_boxes = sum(op['boxes_count'] for op in tracker.history)
    total_items = sum(op['items_used'] for op in tracker.history)
    
    print(f"总装箱数: {total_boxes}")
    print(f"总使用元素: {total_items}")
    
    # 分析每个策略的效果
    strategy_stats = defaultdict(lambda: {'count': 0, 'items': 0, 'boxes': 0})
    for op in tracker.history:
        strategy = op['strategy']
        strategy_stats[strategy]['count'] += 1
        strategy_stats[strategy]['items'] += op['items_used']
        strategy_stats[strategy]['boxes'] += op['boxes_count']
    
    print("\n策略效果对比:")
    for strategy, stats in strategy_stats.items():
        avg_boxes = stats['boxes'] / stats['count']
        avg_items = stats['items'] / stats['count']
        print(f"{strategy}:")
        print(f"  平均装箱数: {avg_boxes:.1f}")
        print(f"  平均使用元素: {avg_items:.0f}")
        print(f"  平均成功率: {avg_boxes / 4:.1%}")  # 假设使用4个线程
#### --------------------------------------- ####
## CKPT 相关的 tools
import pickle

def save_ckpt(tracker, file_path: str):
    """
    保存 tracker（包含 processor）到文件
    """
    with open(file_path, 'wb') as f:
        pickle.dump(tracker, f)
    print(f"已保存ckpt到 {file_path}")

def load_ckpt(file_path: str):
    """
    加载 tracker（包含 processor）状态
    """
    with open(file_path, 'rb') as f:
        tracker = pickle.load(f)
    print(f"已加载ckpt: {file_path}")
    return tracker

def save_bin_boxes(bin_boxes, file_path: str):
    """
    保存单步装箱结果
    """
    with open(file_path, 'wb') as f:
        pickle.dump(bin_boxes, f)
    print(f"已保存装箱结果到 {file_path}")

def load_bin_boxes(file_path: str):
    """
    加载单步装箱结果
    """
    with open(file_path, 'rb') as f:
        bin_boxes = pickle.load(f)
    print(f"已加载装箱结果: {file_path}")
    return bin_boxes
