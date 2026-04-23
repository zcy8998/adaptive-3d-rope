import bisect
import pdb
import os
import math
import traceback

import h5py
import torch
import torch.utils.data as data
from torch.utils.data.sampler import Sampler
import torch.distributed as dist
import numpy as np
import hdf5storage
from einops import rearrange
from multiprocessing.shared_memory import SharedMemory
from multiprocessing import cpu_count
from concurrent.futures import ThreadPoolExecutor, as_completed
from numpy import random

import util.misc as misc

SPEED_OF_LIGHT = 299792458.0


def _to_float_scalar(value, default=0.0):
    if value is None:
        return float(default)
    arr = np.asarray(value)
    if arr.size == 0:
        return float(default)
    return float(arr.reshape(-1)[0])


def load_dataset_phys_meta(config_path):
    loaded = hdf5storage.loadmat(config_path)
    fc = _to_float_scalar(loaded.get('fc'), default=1.0)
    delta_f = _to_float_scalar(loaded.get('delta_f'), default=1.0)
    delta_t = _to_float_scalar(loaded.get('delta_t'), default=1.0)
    ant_spacing = 0.5 * SPEED_OF_LIGHT / max(fc, 1.0)
    return np.asarray([fc, delta_f, delta_t, ant_spacing], dtype=np.float32)


def noise(H, SNR):
    sigma = 10 ** (- SNR / 10)
    add_noise = np.sqrt(sigma / 2) * (np.random.randn(*H.shape) + 1j * np.random.randn(*H.shape))
    add_noise = add_noise * np.sqrt(np.mean(np.abs(H) ** 2))
    return H + add_noise


class CSIDataset(data.Dataset):
    def __init__(self,
                 dataset,
                 world_size=1,
                 rank=0,
                 dataset_type='train',
                 SNR=20,
                 patch_size=4,
                 data_num=None,
                 max_workers=4,
                 data_dir=None,
                 return_phys_meta=False):
        super(CSIDataset, self).__init__()

        # 分布式信息（目前并不做跨-rank 内存共享；每个 rank 默认各自加载全部数据）
        self.world_size = world_size
        self.rank = rank

        # 基本参数
        self.patch_size = patch_size
        self.max_workers = max_workers
        self.dataset_type = dataset_type
        self.data_dir = data_dir
        self.SNR = SNR
        self.return_phys_meta = return_phys_meta

        # 处理数据集列表
        self.datasets_list = dataset.split(",") if isinstance(dataset, str) else list(dataset)

        # 存储每个数据集的元数据/数组
        self.dataset_bounds = []
        self.dataset_arrays = {}  
        self.dataset_shapes = {}  
        self.dataset_phys_meta = {}

        # 1. 计算元数据（不计算全局最大长度）
        self._calculate_dataset_metadata(data_num)

        # 2. 并行加载数据到内存
        self._load_data_parallel()

        # 阻塞等待（如果使用 torch.distributed）
        if dist is not None and dist.is_available() and dist.is_initialized():
            misc.synchronize()
        print(f"Rank {self.rank} finished in-memory loading")

    @staticmethod
    def _resolve_num_samples(total_samples, data_num):
        if data_num is None:
            return int(total_samples)

        if isinstance(data_num, float) and 0 < data_num <= 1.0:
            return max(1, int(round(total_samples * data_num)))

        requested = int(data_num)
        return max(1, min(int(total_samples), requested))

    def _calculate_dataset_metadata(self, data_num):
        """计算每个数据集的元数据（samples, seq_length, feature_dim 等）"""
        global_start = 0
        self.total_samples = 0

        for name in self.datasets_list:
            path = os.path.join(self.data_dir, name, f"{self.dataset_type}_data.mat")
            if not os.path.exists(path):
                raise FileNotFoundError(f"Data file not found: {path}")

            with h5py.File(path, 'r') as f:
                dset = f[f'H_{self.dataset_type}']
                # 你的原始文件 shape 记为 (U, K, T, B)
                U, K, T, B = dset.shape
                B = self._resolve_num_samples(B, data_num)

            seq_length = T * K * U // (self.patch_size ** 3)
            feature_dim = self.patch_size ** 3
            config_path = os.path.join(self.data_dir, name, 'config.mat')
            phys_meta = load_dataset_phys_meta(config_path) if self.return_phys_meta and os.path.exists(config_path) else None

            self.dataset_bounds.append({
                'name': name,
                'path': path,
                'config_path': config_path,
                'global_start': global_start,
                'global_end': global_start + B,
                'samples': B,
                'dims': (T, K, U),
                'seq_length': seq_length,
                'feature_dim': feature_dim,
                'phys_meta': phys_meta,
            })

            self.total_samples += B
            global_start += B

        print(f"Total samples across all datasets: {self.total_samples}")

    def _load_data_parallel(self):
        """并行加载并处理多个数据集到内存"""
        if not self.dataset_bounds:
            return

        print(f"Rank {self.rank} loading {len(self.dataset_bounds)} datasets into memory")

        with ThreadPoolExecutor(max_workers=min(len(self.dataset_bounds), self.max_workers)) as executor:
            futures = [executor.submit(self._process_dataset, meta) for meta in self.dataset_bounds]
            for fut in as_completed(futures):
                try:
                    fut.result()
                except Exception as e:
                    print(f"[Rank {self.rank}] Error while processing dataset: {e}")
                    traceback.print_exc()

    def _process_dataset(self, meta):
        """加载并处理单个数据集到内存"""
        name = meta['name']
        path = meta['path']
        print(f"[Rank {self.rank}] processing dataset {name} from {path}")

        # 使用 hdf5storage 加载原始数据（与你原实现保持一致）
        loaded = hdf5storage.loadmat(path)
        H_full = loaded[f'H_{self.dataset_type}'] 

        # 如果 data_num < B，裁剪最后一维样本数
        B = meta['samples']
        if H_full.shape[0] > B:
            H_full = H_full[:B, ...]

        power = np.mean(np.abs(H_full) ** 2, axis=(1, 2, 3), keepdims=True)

        power = np.maximum(power, 1e-12)
        H_full = H_full / np.sqrt(power)

        if self.SNR is not None:
            noise = generate_gaussian_noise(H_full, self.SNR)
            H_full = H_full + noise

        patched = patch_maker(H_full, self.patch_size)

        # 校验 patched 的形状是否与 meta 一致；若不一致，尝试适配或抛错
        expected_shape = (meta['samples'], meta['seq_length'], meta['feature_dim'])
        if patched.shape != expected_shape:
            raise ValueError(f"Patched data shape mismatch for {name}: got {patched.shape}, expected {expected_shape}")

        # 处理 dtype：如果 patched 是 complex 就保存 complex64，否则保存 float32
        if np.iscomplexobj(patched):
            print("complex data")
            save_dtype = np.complex64
        else:
            save_dtype = np.float32

        # 强制转换并保存
        patched = np.asarray(patched, dtype=save_dtype)
        self.dataset_arrays[name] = patched
        self.dataset_shapes[name] = meta['dims']
        if meta.get('phys_meta') is not None:
            self.dataset_phys_meta[name] = np.asarray(meta['phys_meta'], dtype=np.float32)

        print(f"[Rank {self.rank}] finished processing dataset {name}; in-memory shape: {self.dataset_arrays[name].shape}")

    def __len__(self):
        return self.total_samples

    def __getitem__(self, idx):
        # 定位 dataset
        for meta in self.dataset_bounds:
            if meta['global_start'] <= idx < meta['global_end']:
                dataset_name = meta['name']
                local_idx = idx - meta['global_start']
                break
        else:
            raise IndexError(f"Index {idx} out of range (total {self.total_samples})")

        # 从内存数组读取（拷贝以避免后续原地修改影响）
        arr = self.dataset_arrays[dataset_name][local_idx]
        # arr shape: (seq_length, feature_dim)
        if self.return_phys_meta:
            phys_meta = self.dataset_phys_meta.get(dataset_name)
            if phys_meta is None:
                phys_meta = np.zeros(4, dtype=np.float32)
            return torch.from_numpy(arr.copy()), meta['seq_length'], meta['dims'], torch.from_numpy(phys_meta.copy())
        return torch.from_numpy(arr.copy()), meta['seq_length'], meta['dims']

    @staticmethod
    def padded_collate_fn(batch):
        """
        填充函数，接收 batch: list of (tensor, seq_len, dims)
        返回: padded_batch (B, max_len, feature_dim), lengths_tensor (B,), dims_tensor (3, B)
        """
        if not batch:
            return None

        # 解包
        data_tensors = [item[0] for item in batch]
        lengths = [int(item[1]) for item in batch]
        feat_dims = [item[2] for item in batch]  # 例如 (T,K,U)

        batch_size = len(batch)
        max_len = max(lengths)
        feature_dim = data_tensors[0].shape[1]

        padded = torch.zeros(batch_size, max_len, feature_dim, dtype=data_tensors[0].dtype)
        for i, t in enumerate(data_tensors):
            L = t.shape[0]
            padded[i, :L, :] = t

        dims_tensor = torch.tensor(feat_dims).T  # shape (3, B) if each dims is length-3
        if len(batch[0]) >= 4:
            phys_meta = torch.stack([item[3] for item in batch], dim=0)
            return padded, torch.tensor(lengths, dtype=torch.long), dims_tensor, phys_meta
        return padded, torch.tensor(lengths, dtype=torch.long), dims_tensor

    def __del__(self):
        try:
            for k in list(self.dataset_arrays.keys()):
                del self.dataset_arrays[k]
            self.dataset_arrays.clear()
            self.dataset_shapes.clear()
            self.dataset_phys_meta.clear()
        except Exception as e:
            print(f"Error during cleanup: {e}")


class CSIDataset_v2(data.Dataset):
    def __init__(self,
                 dataset,
                 world_size=1,
                 rank=0,
                 dataset_type='train',
                 SNR=20,
                 patch_size=4,
                 data_num=None,
                 max_workers=4,
                 data_dir=None):
        super(CSIDataset_v2, self).__init__()

        # 分布式信息
        self.world_size = world_size
        self.rank = rank

        # 基本参数
        self.patch_size = patch_size
        self.max_workers = max_workers
        self.dataset_type = dataset_type
        self.data_dir = data_dir
        self.SNR = SNR

        # 处理数据集列表
        self.datasets_list = dataset.split(",") if isinstance(dataset, str) else list(dataset)

        # 存储每个数据集的元数据/数组
        self.dataset_bounds = []
        self.dataset_arrays = {}   # name -> np.ndarray shaped (B, T, K, U)
        self.dataset_shapes = {}   # name -> 原始 dims (T, K, U)

        # 1. 计算元数据
        self._calculate_dataset_metadata(data_num)

        # 2. 并行加载数据到内存
        self._load_data_parallel()

        # 阻塞等待（如果使用 torch.distributed）
        if dist is not None and dist.is_available() and dist.is_initialized():
            misc.synchronize()
        print(f"Rank {self.rank} finished in-memory loading")

    @staticmethod
    def _resolve_num_samples(total_samples, data_num):
        if data_num is None:
            return int(total_samples)

        if isinstance(data_num, float) and 0 < data_num <= 1.0:
            return max(1, int(round(total_samples * data_num)))

        requested = int(data_num)
        return max(1, min(int(total_samples), requested))

    def _calculate_dataset_metadata(self, data_num):
        """计算每个数据集的元数据"""
        global_start = 0
        self.total_samples = 0

        for name in self.datasets_list:
            path = os.path.join(self.data_dir, name, f"{self.dataset_type}_data.mat")
            if not os.path.exists(path):
                raise FileNotFoundError(f"Data file not found: {path}")

            # 注意：h5py 采用 C 语言底层读取，MATLAB 的 (B, T, K, U) 会被读成逆序的 (U, K, T, B)
            with h5py.File(path, 'r') as f:
                dset = f[f'H_{self.dataset_type}']
                U, K, T, B = dset.shape
                
            B = self._resolve_num_samples(B, data_num)

            # seq_length 仅作为参考信息保留，实际输出是 3D 物理网格
            seq_length = T * K * U // (self.patch_size ** 3)
            feature_dim = self.patch_size ** 3

            self.dataset_bounds.append({
                'name': name,
                'path': path,
                'global_start': global_start,
                'global_end': global_start + B,
                'samples': B,
                'dims': (T, K, U),
                'seq_length': seq_length,
                'feature_dim': feature_dim
            })

            self.total_samples += B
            global_start += B

        print(f"Total samples across all datasets: {self.total_samples}")

    def _load_data_parallel(self):
        """并行加载并处理多个数据集到内存"""
        if not self.dataset_bounds:
            return

        print(f"Rank {self.rank} loading {len(self.dataset_bounds)} datasets into memory")

        with ThreadPoolExecutor(max_workers=min(len(self.dataset_bounds), self.max_workers)) as executor:
            futures = [executor.submit(self._process_dataset, meta) for meta in self.dataset_bounds]
            for fut in as_completed(futures):
                try:
                    fut.result()
                except Exception as e:
                    print(f"[Rank {self.rank}] Error while processing dataset: {e}")
                    traceback.print_exc()

    def _process_dataset(self, meta):
        """加载并处理单个数据集到内存"""
        name = meta['name']
        path = meta['path']
        print(f"[Rank {self.rank}] processing dataset {name} from {path}")

        # 注意：hdf5storage 会自动适配 MATLAB 格式，读取出来的维度是正确的 (B, T, K, U)
        loaded = hdf5storage.loadmat(path)
        H_full = loaded[f'H_{self.dataset_type}']  

        # 如果 data_num < B，裁剪第 0 维 (Batch 维)
        B_meta = meta['samples']
        if H_full.shape[0] > B_meta:
            H_full = H_full[:B_meta, ...]

        # 功率归一化，基于 (T, K, U) 即 axis=(1, 2, 3) 进行求平均
        power = np.mean(np.abs(H_full) ** 2, axis=(1, 2, 3), keepdims=True)
        # 防止除以零
        power = np.maximum(power, 1e-12)
        H_full = H_full / np.sqrt(power)

        # 添加高斯噪声
        if self.SNR is not None:
            noise = generate_gaussian_noise(H_full, self.SNR)
            H_full = H_full + noise

        # 处理 dtype：如果 H_full 是 complex 就保存 complex64，否则 float32
        if np.iscomplexobj(H_full):
            save_dtype = np.complex64
        else:
            save_dtype = np.float32

        # 保存原始 3D 网格数据，保留物理意义
        self.dataset_arrays[name] = np.asarray(H_full, dtype=save_dtype)
        self.dataset_shapes[name] = meta['dims']

        print(f"[Rank {self.rank}] finished processing dataset {name}; in-memory shape: {self.dataset_arrays[name].shape}")

    def __len__(self):
        return self.total_samples

    def __getitem__(self, idx):
        # 定位 dataset
        for meta in self.dataset_bounds:
            if meta['global_start'] <= idx < meta['global_end']:
                dataset_name = meta['name']
                local_idx = idx - meta['global_start']
                break
        else:
            raise IndexError(f"Index {idx} out of range (total {self.total_samples})")

        # 从内存数组读取（拷贝以避免后续原地修改影响）
        # arr 形状是 (B, T, K, U)，取 local_idx 后得到单样本 (T, K, U)
        arr = self.dataset_arrays[dataset_name][local_idx]
        
        return torch.from_numpy(arr.copy()), meta['seq_length'], meta['dims']

    @staticmethod
    def padded_collate_fn(batch):
        """
        三维网格动态填充：将不同维度的 (T, K, U) 填充至当前 Batch 的最大尺寸
        """
        if not batch:
            return None

        data_tensors = [item[0] for item in batch]  # List of (T, K, U)
        lengths = [int(item[1]) for item in batch]
        feat_dims = [item[2] for item in batch]     # List of (T, K, U) tuples

        batch_size = len(batch)
        
        # 寻找当前 Batch 中的最大 T, K, U
        max_T = max(t.shape[0] for t in data_tensors)
        max_K = max(t.shape[1] for t in data_tensors)
        max_U = max(t.shape[2] for t in data_tensors)

        # 创建全零的三维底板 (B, max_T, max_K, max_U)
        padded = torch.zeros(batch_size, max_T, max_K, max_U, dtype=data_tensors[0].dtype)
        
        for i, t in enumerate(data_tensors):
            T, K, U = t.shape
            padded[i, :T, :K, :U] = t

        # feat_dims 是 [B, 3] 的 list，转置后为 [3, B]
        dims_tensor = torch.tensor(feat_dims).T  
        lengths_tensor = torch.tensor(lengths, dtype=torch.long)
        
        # 返回: Padded网格(B, max_T, max_K, max_U), 序列长度(B,), 原始维度(3, B)
        return padded, lengths_tensor, dims_tensor

    def __del__(self):
        try:
            for k in list(self.dataset_arrays.keys()):
                del self.dataset_arrays[k]
            self.dataset_arrays.clear()
            self.dataset_shapes.clear()
        except Exception as e:
            print(f"Error during cleanup: {e}")


# =====================================================================
# 模型内使用的 PyTorch 版 Patch Maker
# =====================================================================
def patch_maker_torch(data: torch.Tensor, patch_size: int = 4) -> torch.Tensor:
    """
    在模型 Forward 阶段将 (B, T, K, U) 的物理网格切分为 Patch 序列。
    
    Args:
        data: 输入的 3D 物理网格，形状为 [B, T, K, U]
        patch_size: 每个 patch 的边长，默认为 4
        
    Returns:
        patched_data: 展平后的 Patch 序列，形状为 [B, num_patches, patch_elements]
    """
    B, T, K, U = data.shape
    
    # 检查维度是否可被 patch_size 整除
    assert T % patch_size == 0 and K % patch_size == 0 and U % patch_size == 0, \
        f"Dimensions ({T}, {K}, {U}) must be divisible by patch_size {patch_size}"
    
    # 计算每个维度的块数
    t_blocks = T // patch_size
    k_blocks = K // patch_size
    u_blocks = U // patch_size
    
    # 将数据重组: [B, t_blocks, patch_size, k_blocks, patch_size, u_blocks, patch_size]
    reshaped = data.view(B, 
                         t_blocks, patch_size, 
                         k_blocks, patch_size, 
                         u_blocks, patch_size)
    
    # 调整维度顺序，将所有外部块维度放在前，内部 patch 维度放在后
    transposed = reshaped.permute(0, 1, 3, 5, 2, 4, 6) 
    
    # 合并块索引维度和内部元素维度
    num_patches = t_blocks * k_blocks * u_blocks
    patch_elements = patch_size ** 3
    
    # 使用 contiguous() 保证内存连续，然后再 view 展平
    patched_data = transposed.contiguous().view(B, num_patches, patch_elements)
    
    return patched_data



class CSIDataset_mmap_nopad(data.Dataset):
    def __init__(self,
                 dataset,
                 world_size=1,
                 rank=0,
                 dataset_type='train',
                 SNR=20,
                 patch_size=4,
                 data_num=None,
                 max_workers=2,
                 mmap_version='adaptive_3d_rope',
                 data_dir='./data/csidata'):
        super(CSIDataset_mmap_nopad, self).__init__()
        
        # 分布式信息
        self.world_size = world_size
        self.rank = rank
        
        # 基本参数
        self.patch_size = patch_size
        self.max_workers = max_workers
        self.dataset_type = dataset_type
        self.mmap_version = mmap_version
        self.data_dir = data_dir
        self.SNR = SNR
        
        # 处理数据集列表
        self.datasets_list = dataset.split(",")
        
        # 存储每个数据集的元数据
        self.dataset_bounds = []
        self.dataset_arrays = {}  # 存储每个数据集的mmap数组
        self.dataset_shapes = {}  # 存储每个数据集的形状
        
        # 1. 取消全局最大长度计算
        self._calculate_dataset_metadata(data_num)
        
        # 2. 创建内存映射文件
        self._create_dataset_mmap_files()
        
        # 3. 并行加载数据
        self._load_data_parallel()

        # 4. 全局数据集同步 (新增加)
        self._sync_datasets_across_ranks()
        
        # 等待所有进程完成加载
        if dist.is_available() and dist.is_initialized():
            misc.synchronize()
        print(f"Rank {self.rank} passed data loading barrier")
            
    @staticmethod
    def _resolve_num_samples(total_samples, data_num):
        if data_num is None:
            return int(total_samples)

        if isinstance(data_num, float) and 0 < data_num <= 1.0:
            return max(1, int(round(total_samples * data_num)))

        requested = int(data_num)
        return max(1, min(int(total_samples), requested))

    def _calculate_dataset_metadata(self, data_num):
        """计算每个数据集的元数据"""
        global_start = 0
        self.total_samples = 0
        
        for name in self.datasets_list:
            path = f"{self.data_dir}/{name}/{self.dataset_type}_data.mat"
            with h5py.File(path, 'r') as f:
                dset = f[f'H_{self.dataset_type}']
                U, K, T, B = dset.shape
                print(name)
                print("U, K, T, B:", U, K, T, B)
                B = self._resolve_num_samples(B, data_num)
            
            # 序列长度计算（基于当前数据集）
            seq_length = T * K * U // (self.patch_size ** 3)
            
            self.dataset_bounds.append({
                'name': name,
                'path': path,
                'global_start': global_start,
                'global_end': global_start + B,
                'samples': B,
                'dims': (T, K, U),
                'seq_length': seq_length,
                'feature_dim': self.patch_size**3
            })
            
            self.total_samples += B
            global_start += B
        
        print(f"Total samples across all datasets: {self.total_samples}")
        
    def _create_dataset_mmap_files(self):
        """为每个数据集创建独立的内存映射文件"""
        if self.rank != 0:
            # 非主进程等待文件创建
            if dist.is_available() and dist.is_initialized():
                misc.synchronize()
            return
        
        # 主进程创建所有内存映射文件
        for meta in self.dataset_bounds:
            # 为每个数据集生成唯一文件路径
            mmap_path = os.path.join(
                self.data_dir, 
                f"csi_{self.dataset_type}_{meta['name']}_{self.mmap_version}.bin"
            )
            
            # 文件大小计算（精确匹配数据集实际形状）
            item_size = meta['seq_length'] * meta['feature_dim']
            file_size = meta['samples'] * item_size * np.dtype(np.float32).itemsize
            
            # 创建文件
            print(f"Creating memmap file for dataset {meta['name']}: "
                  f"{file_size/(1024**2):.2f} MB")
            with open(mmap_path, 'wb') as f:
                f.seek(file_size - 1)
                f.write(b'\0')
        
        # 文件创建完成，通知其他进程
        if dist.is_available() and dist.is_initialized():
            misc.synchronize()
    
    def _load_data_parallel(self):
        # 计算每个rank负责的数据集范围
        datasets_per_rank = math.ceil(len(self.datasets_list) / self.world_size)
        start_idx = self.rank * datasets_per_rank
        end_idx = min((self.rank + 1) * datasets_per_rank, len(self.datasets_list))
        rank_datasets = [meta for meta in self.dataset_bounds 
                         if meta['name'] in self.datasets_list[start_idx:end_idx]]
    
        # print(f"---{self.rank}---")
        # print(rank_datasets)
        
        if not rank_datasets:
            print(f"Rank {self.rank} has no datasets to load")
            return
            
        print(f"Rank {self.rank} loading {len(rank_datasets)} datasets: "
              f"{[d['name'] for d in rank_datasets]}")
        
        # 创建线程池并行处理数据集
        with ThreadPoolExecutor(max_workers=min(len(rank_datasets), self.max_workers)) as executor:
            futures = [executor.submit(self._process_dataset, meta) for meta in rank_datasets]
            for fut in as_completed(futures):
                try:
                    fut.result()  # 如果线程里抛异常，这里会 re-raise
                except Exception as e:
                    print(f"[Rank {self.rank}] Error while processing dataset: {e}")
                    traceback.print_exc()

    def _sync_datasets_across_ranks(self):
        """确保所有rank都加载了全部数据集的内存映射"""
        # 等待所有rank完成数据处理
        if dist.is_available() and dist.is_initialized():
            misc.synchronize()
        
        print(f"Rank {self.rank}: Syncing all datasets")
        
        # 每个rank加载全部数据集
        for meta in self.dataset_bounds:
            name = meta['name']
            mmap_path = os.path.join(
                self.data_dir, 
                f"csi_{self.dataset_type}_{name}_{self.mmap_version}.bin"
            )
            
            # 只读模式访问
            self.dataset_arrays[name] = np.memmap(
                mmap_path,
                dtype=np.float32,
                mode='r',
                shape=(meta['samples'], meta['seq_length'], meta['feature_dim'])
            )
            self.dataset_shapes[name] = meta['dims']
        
        print(f"Rank {self.rank}: Loaded {len(self.dataset_arrays)} datasets")
    
    def _process_dataset(self, meta):
        """加载并处理单个数据集"""
        name = meta['name']
        print(f"Rank {self.rank} processing dataset: {name}")
        
        # 数据集特定内存映射路径
        mmap_path = os.path.join(
            self.data_dir, 
            f"csi_{self.dataset_type}_{name}_{self.mmap_version}.bin"
        )
        
        # 打开内存映射文件 (r+ 模式)
        data_array = np.memmap(
            mmap_path,
            dtype=np.float32,
            mode='r+',
            shape=(meta['samples'], meta['seq_length'], meta['feature_dim'])
        )
        
        # 加载数据
        H_full = hdf5storage.loadmat(meta['path'])[f'H_{self.dataset_type}']
                
        # 功率归一化
        power = np.mean(np.abs(H_full)**2, axis=(1, 2, 3), keepdims=True)
        H_full = H_full / (np.sqrt(power))

        # 添加噪声
        if self.SNR is not None:
            noise = generate_gaussian_noise(H_full, self.SNR)
            H_full = H_full + noise

        # 转换为patch
        patched_data = patch_maker(H_full, self.patch_size)
        
        # 写入内存映射文件
        data_array[:, :, :] = patched_data
        data_array.flush()  # 确保数据写入磁盘
        
        # 关闭并重新以只读模式打开
        del data_array
        self.dataset_arrays[name] = np.memmap(
            mmap_path,
            dtype=np.float32,
            mode='r',
            shape=(meta['samples'], meta['seq_length'], meta['feature_dim'])
        )
        self.dataset_shapes[name] = meta['dims']
        
        print(f"Rank {self.rank} finished processing dataset: {name}")
    
    def __len__(self):
        return self.total_samples
    
    def __getitem__(self, idx):
        # 找到对应数据集
        for meta in self.dataset_bounds:
            if meta['global_start'] <= idx < meta['global_end']:
                dataset_name = meta['name']
                local_idx = idx - meta['global_start']
                break
        else:
            raise IndexError(f"Index {idx} out of range")
        
        # 从内存映射获取数据
        data_arr = self.dataset_arrays[dataset_name][local_idx]
        actual_length = meta['seq_length']
        
        # 返回数据和元信息
        return torch.as_tensor(data_arr.copy()), actual_length, meta['dims']
    
    @staticmethod
    def padded_collate_fn(batch):
        """
        自动填充变长序列的批次处理函数
        参数:
            batch: [(data_tensor1, length1, feat_dim1), ...]
        """
        # 检查批次是否为空
        if not batch:
            return None
            
        # 解压数据、长度和特征维度
        data_list, lengths, feature_dims = zip(*batch)

        length = max(lengths)

        dims = torch.tensor(feature_dims)
        dims = dims.T
        
        # 创建填充后的批次张量
        padded_batch = torch.zeros(
            len(batch), 
            length,
            64,
            dtype=data_list[0].dtype
        )
        
        # 填充每个样本
        for i, (data, length, _) in enumerate(batch):
            padded_batch[i, :length] = data[:length]
            
        return padded_batch, torch.tensor(lengths), dims
    
    def __del__(self):
        """清理资源"""
        try:
            # 只有当所有rank都处理完毕时才删除文件
            if (dist.is_available() and dist.is_initialized() and self.rank == 0):
                misc.synchronize()  # 确保所有rank已完成
                print(f"Rank {self.rank} deleting memmap files")
                for meta in self.dataset_bounds:
                    mmap_path = os.path.join(
                        self.data_dir, 
                        f"csi_{self.dataset_type}_{meta['name']}_{self.mmap_version}.bin"
                    )
                    if os.path.exists(mmap_path):
                        os.unlink(mmap_path)
        except Exception as e:
            print(f"Error during cleanup: {str(e)}")


def data_load_single_nopad(args, dataset_name, SNR=20, dataset_type='test', data_num=1): 

    folder_path = os.path.join(args.data_dir, f'{dataset_name}/{dataset_type}_data.mat')
    config_path = os.path.join(args.data_dir, f'{dataset_name}/config.mat')
    # folder_path = f'/data/zcy_data/zeroshot/{dataset_name}/{dataset_type}_data.mat'

    H_data = hdf5storage.loadmat(folder_path)[f'H_{dataset_type}']
    B, T, K, U = H_data.shape 
    
    if data_num < 1:
        target_len = int(B * data_num)
        H_data = H_data[:target_len]
        print(f"{dataset_name} sampled from {B} to {target_len} (ratio: {data_num})")
        B = target_len # 更新B的大小

    print(dataset_name, 'shape:', H_data.shape)

    power = np.mean(np.abs(H_data)**2, axis=(1, 2, 3), keepdims=True)
    H_data = H_data / (np.sqrt(power)) 
    
    if SNR is not None:
        H_data += generate_gaussian_noise(H_data, SNR)  

    H_data = patch_maker(H_data, 4)
    B, L, C = H_data.shape # 这里会获取切片后最新的B

    token_length = (T / 4) * (K / 4) * (U / 4)
    phys_meta = None
    if getattr(args, 'use_phys_coord', False) and os.path.exists(config_path):
        phys_meta = load_dataset_phys_meta(config_path)
    dataset_test = CSIDataset_Single(H_data, token_length, (T, K, U), dataset_name=dataset_name, phys_meta=phys_meta)

    if args.distributed:
        num_tasks = misc.get_world_size()
        global_rank = misc.get_rank()
        sampler = torch.utils.data.DistributedSampler(
            dataset_test, num_replicas=num_tasks, rank=global_rank, shuffle=True
        )
    else:
        sampler = torch.utils.data.RandomSampler(dataset_test)

    data_loader = torch.utils.data.DataLoader(
        dataset_test,
        shuffle=False, sampler=sampler,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=True,
        # prefetch_factor=8,
    )
    return data_loader, L


class CSIDataset_Single(data.Dataset):
    def __init__(self, X_train, token_length, input_size, dataset_name, phys_meta=None):
        self.X_train = X_train
        self.token_length = token_length
        self.input_size = input_size
        self.dataset_name = dataset_name
        self.phys_meta = None if phys_meta is None else np.asarray(phys_meta, dtype=np.float32)

    def __len__(self):

        return self.X_train.shape[0]

    def __getitem__(self, idx):
        if self.phys_meta is None:
            return self.X_train[idx], self.token_length, self.input_size
        return self.X_train[idx], self.token_length, self.input_size, torch.from_numpy(self.phys_meta.copy())
    
    def get_dataset_name(self):
        return self.dataset_name


class WifoDataset(data.Dataset):
    def __init__(self, X_train, token_length, input_size, dataset_name):
        self.X_train = X_train
        self.token_length = token_length
        self.input_size = input_size
        self.dataset_name = dataset_name

    def __len__(self):
        return self.X_train.shape[0]

    def __getitem__(self, idx):
        return self.X_train[idx], self.token_length, self.input_size

    def get_dataset_name(self):
        return self.dataset_name


def data_load_single_Wifo(args, dataset, SNR=20): # 加载单个数据集

    folder_path_test = f'/data/zcy_data/CSIGPT_Dataset/test_data/{dataset}/test_data.mat'

    X_test = hdf5storage.loadmat(folder_path_test)['X_val']
    # X_test_complex = torch.tensor(np.array(X_test['X_val'], dtype=complex))
    H_data = X_test.transpose(0, 1, 3, 2)  # [B, T, U, K] -> [B, T, K, U]
    if SNR is not None:
        H_data += generate_gaussian_noise(H_data, SNR)   
    B, T, K, U = H_data.shape

    pdb.set_trace()
    H_data = patch_maker(H_data, 4).astype(np.float32)
    # 填充处理
    B, L, C = H_data.shape
    # max_length = args.max_length
    # # 检查长度一致性
    # if L > max_length:
    #     raise ValueError(f"Error in dataset: Sequence length {L} exceeds maximum length {max_length}.")
    
    padded_batch = np.zeros((B, L, C), dtype=H_data.dtype)
    padded_batch[:, :L, :] = H_data
    pdb.set_trace()
    test_data = WifoDataset(padded_batch, L, (T, K, U), dataset_name=dataset)

    batch_size = args.batch_size
    test_data = torch.utils.data.DataLoader(test_data, num_workers=args.num_workers, 
                                            batch_size = batch_size, shuffle=False, pin_memory=True, prefetch_factor=4)

    return test_data, L


def data_load(args, dataset_type, test_type='normal'):

    test_data_all = []
        
    for dataset_name in args.dataset.split(','):
        print(f"Processing {dataset_name} for {dataset_type}")
        if test_type == 'normal':
            # test_data, _ = data_load_single(args, dataset_name, dataset_type=dataset_type)
            test_data, _ = data_load_single_nopad(args, dataset_name, dataset_type=dataset_type, data_num=args.data_num)
        elif test_type == 'wifo':
            test_data, _ = data_load_single_Wifo(args, dataset_name)
        test_data_all.append(test_data)
    
    return test_data_all

def data_load_main(args, dataset_type='val', test_type='normal'):

    test_data = data_load(args, dataset_type, test_type)

    return test_data


def data_load_baseline(args, dataset_type='val', SNR=20, data_num=1.0):
    dataset_name = args.dataset
    folder_path = os.path.join(args.data_dir, f'{dataset_name}/{dataset_type}_data.mat')

    # 加载原始数据
    H_data = hdf5storage.loadmat(folder_path)[f'H_{dataset_type}']
    B, T, K, U = H_data.shape 
    print(f"{dataset_name} original shape: {H_data.shape}")

    # 如果 data_num 在 (0, 1) 之间，则进行采样
    if 0 < data_num < 1.0:
        # 计算需要保留的样本数量
        num_keep = int(B * data_num)
        
        if num_keep > 0:
            # 生成随机索引以打乱数据，避免只取前一部分可能导致的数据偏差
            # 如果不需要随机，可以直接用 H_data = H_data[:num_keep]
            perm_indices = np.random.permutation(B)
            selected_indices = perm_indices[:num_keep]
            H_data = H_data[selected_indices]
            
            # 更新 B 的大小，并打印提示信息
            B = H_data.shape[0]
            print(f"Finetuning sampling: kept {data_num*100:.1f}% data. New shape: {H_data.shape}")
        else:
            print("Warning: data_num implies 0 samples. Keeping original data.")

    power = np.mean(np.abs(H_data)**2, axis=(1, 2, 3), keepdims=True)
    H_data = H_data / (np.sqrt(power)) 
    
    if SNR is not None:
        H_data += generate_gaussian_noise(H_data, SNR)  

    dataset_test = CSIDataset_Single(H_data, 0, (T, K, U), dataset_name=dataset_name)
    
    # 注意：如果数据量减少了，RandomSampler 仍然会从现在的 dataset_test 中采样
    sampler = torch.utils.data.RandomSampler(dataset_test)

    data_loader = torch.utils.data.DataLoader(
        dataset_test,
        shuffle=False, sampler=sampler,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=True, # 如果采样后的数据量小于batch_size，这里可能会丢弃所有数据，需注意
        # prefetch_factor=8,
    )
    return data_loader
    

class DistributedGroupBatchSampler(Sampler):
    def __init__(self, dataset_bounds, batch_size, world_size=None, rank=None, shuffle=True, drop_last=False, seed=42):
        if world_size is None:
            world_size = dist.get_world_size() if dist.is_available() else 1
        if rank is None:
            rank = dist.get_rank() if dist.is_available() else 0

        self.group_bounds = dataset_bounds    
        self.batch_size = batch_size
        self.world_size = world_size
        self.rank = rank
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.seed = seed
        self.epoch = 0  # 添加epoch计数器
        
        # 从dataset获取组信息
        self._create_group_indices()
        self._create_global_index_map()
        
        # 分配样本到各rank（按长度升序排序）
        self._assign_samples_to_ranks()
        
        # 计算batch数量
        self.num_batches = self._calculate_num_batches()
        self.global_num_batches = self._sync_num_batches()
        
    def _create_group_indices(self):
        """创建组索引映射（合并相同长度的组）"""
        self.group_dict = {}
        for bound in self.group_bounds:
            length = bound['seq_length']
            indices = list(range(bound['global_start'], bound['global_end']))
            if length not in self.group_dict:
                self.group_dict[length] = []
            self.group_dict[length].extend(indices)
        
    def _create_global_index_map(self):
        """创建全局索引到序列长度的映射"""
        self.global_index_to_length = {}
        for length, indices in self.group_dict.items():
            for idx in indices:
                self.global_index_to_length[idx] = length
                
    def _assign_samples_to_ranks(self):
        """按长度升序分配样本"""
        # 合并所有样本并按长度升序排序
        all_samples = []
        for length, indices in sorted(self.group_dict.items(), key=lambda x: x[0]):
            if self.shuffle:
                # 使用与epoch无关的随机种子进行初始shuffle
                rng = np.random.RandomState(self.seed)
                rng.shuffle(indices)
            all_samples.extend(indices)
        
        total_samples = len(all_samples)
        
        # 计算每个rank分配的样本数量
        per_rank = total_samples // self.world_size
        remainder = total_samples % self.world_size
        
        # 分配样本
        start = 0
        self.rank_samples = []
        for i in range(self.world_size):
            end = start + per_rank + (1 if i < remainder else 0)
            self.rank_samples.append(all_samples[start:end])
            start = end
        
        # 打印分配信息
        if self.rank == 0:
            print("Sample distribution per rank:")
            for r in range(self.world_size):
                samples = self.rank_samples[r]
                min_len = self.global_index_to_length[samples[0]] if samples else 0
                max_len = self.global_index_to_length[samples[-1]] if samples else 0
                avg_len = (min_len + max_len) / 2 if samples else 0
                print(f"Rank {r}: {len(samples)} samples, "
                      f"min_len={min_len}, max_len={max_len}, avg_len={avg_len:.1f}")
    
    def _calculate_num_batches(self):
        """计算当前rank的batch数"""
        total_samples = len(self.rank_samples[self.rank])
        if self.drop_last:
            return total_samples // self.batch_size
        else:
            return (total_samples + self.batch_size - 1) // self.batch_size
    
    def _sync_num_batches(self):
        """同步所有rank的batch数"""
        if self.world_size == 1:
            return self.num_batches
            
        # 使用分布式操作计算最小batch数
        if dist.is_available() and dist.is_initialized():
            tensor = torch.tensor(self.num_batches, dtype=torch.int).cuda()
            dist.all_reduce(tensor, op=dist.ReduceOp.MIN)
            return tensor.item()
        else:
            return self.num_batches
            
    def set_epoch(self, epoch):
        """设置当前epoch，用于随机种子生成"""
        self.epoch = epoch
    
    def __iter__(self):
        """生成索引列表批次"""
        # 获取当前rank的样本（已按长度升序排序）
        current_samples = self.rank_samples[self.rank]
        
        # 如果需要在批次级别打乱顺序，使用与epoch相关的随机种子
        if self.shuffle:
            # 使用epoch相关的随机种子确保每个epoch的shuffle不同
            rng = np.random.RandomState(self.seed + self.epoch)
            rng.shuffle(current_samples)
        
        # 创建批次
        batches = []
        for i in range(0, len(current_samples), self.batch_size):
            batch_indices = current_samples[i:i+self.batch_size]
            if not self.drop_last or len(batch_indices) == self.batch_size:
                batches.append(batch_indices)
        
        # 对batches进行shuffle（rank内batch级别的shuffle）
        if self.shuffle:
            rng = np.random.RandomState(self.seed + self.epoch + 1)  # 使用不同的种子
            rng.shuffle(batches)
        
        for batch in batches:
            yield batch
    
    def __len__(self):
        return self.global_num_batches


class DistributedBucketBatchSampler(Sampler):
    def __init__(
        self,
        dataset_bounds,
        batch_size,
        accum_steps,
        num_buckets=2,
        world_size=None,
        rank=None,
        shuffle=True,
        drop_last=False,
        seed=0,
    ):
        if num_buckets is None or num_buckets < 1:
            raise ValueError("num_buckets must be a positive integer")

        if world_size is None:
            world_size = dist.get_world_size() if dist.is_initialized() else 1
        if rank is None:
            rank = dist.get_rank() if dist.is_initialized() else 0

        self.dataset_bounds = dataset_bounds
        self.seq_lengths = self._expand_seq_lengths_from_dataset_bounds(dataset_bounds)

        self.num_buckets = num_buckets
        self.batch_size = batch_size
        self.accum_steps = accum_steps
        self.world_size = world_size
        self.rank = rank
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.seed = int(seed)
        self.epoch = 0

        self.bucket_boundaries = self._compute_bucket_boundaries(self.seq_lengths, self.num_buckets)
        self._create_buckets()                   # 保留空桶
        self._adjust_buckets_for_accumulation()  # 截断到 world_size*accum_steps 的倍数
        self._assign_samples_to_ranks()          # 按桶分配到各 rank

        self.num_batches = self._calculate_num_batches()
        self.global_num_batches = self._sync_num_batches()

    @staticmethod
    def _expand_seq_lengths_from_dataset_bounds(dataset_bounds):
        seq_lengths = []
        for entry in dataset_bounds:
            if 'samples' not in entry or 'seq_length' not in entry:
                raise ValueError("Each dataset_bounds entry must contain 'samples' and 'seq_length'.")
            s = int(entry['samples'])
            l = int(entry['seq_length'])
            seq_lengths.extend([l] * s)
        return seq_lengths

    @staticmethod
    def _compute_bucket_boundaries(seq_lengths, num_buckets):
        arr = np.asarray(seq_lengths)
        if num_buckets <= 1:
            return []
        perc = np.linspace(0, 100, num_buckets + 1)[1:-1]
        if len(perc) == 0:
            return []
        bounds = np.percentile(arr, perc)
        print("-------- Bucket Boundaries --------")
        print(bounds)
        return np.unique(bounds).tolist()

    def _create_buckets(self):
        num_bins = len(self.bucket_boundaries) + 1
        buckets = [[] for _ in range(num_bins)]
        for idx, length in enumerate(self.seq_lengths):
            bid = bisect.bisect_left(self.bucket_boundaries, length)
            bid = max(0, min(bid, num_bins - 1))
            buckets[bid].append(idx)
        self.buckets = buckets
        self.global_index_to_length = {i: self.seq_lengths[i] for i in range(len(self.seq_lengths))}

    def _adjust_buckets_for_accumulation(self):
        adjusted = []
        batches_per_cycle = self.world_size * self.accum_steps
        for bucket in self.buckets:
            if not bucket:
                adjusted.append([])
                continue
            num_batches = len(bucket) // self.batch_size
            keep_batches = (num_batches // batches_per_cycle) * batches_per_cycle
            adjusted.append(bucket[: keep_batches * self.batch_size] if keep_batches > 0 else [])
        self.buckets = adjusted
        if all(len(b) == 0 for b in self.buckets):
            raise ValueError("No buckets left after adjustment. Reduce num_buckets or accum_steps.")

    def _assign_samples_to_ranks(self):
        all_samples = []
        self.bucket_boundaries_indices = [0]
        for bid, bucket in enumerate(self.buckets):
            if self.shuffle and bucket:
                rng = np.random.RandomState(self.seed + self.epoch + bid)
                shuffled = bucket.copy()
                rng.shuffle(shuffled)
                all_samples.extend(shuffled)
            else:
                all_samples.extend(bucket.copy())
            self.bucket_boundaries_indices.append(len(all_samples))

        self.rank_samples = [[] for _ in range(self.world_size)]
        for b_idx in range(len(self.buckets)):
            start = self.bucket_boundaries_indices[b_idx]
            end = self.bucket_boundaries_indices[b_idx + 1]
            bucket_samples = all_samples[start:end]
            bucket_batches = len(bucket_samples) // self.batch_size
            if bucket_batches == 0:
                continue
            batches_per_rank = bucket_batches // self.world_size
            if batches_per_rank == 0:
                continue
            for r in range(self.world_size):
                s = start + r * batches_per_rank * self.batch_size
                e = s + batches_per_rank * self.batch_size
                self.rank_samples[r].extend(all_samples[s:e])

        if self.rank == 0:
            num_nonempty = sum(1 for b in self.buckets if len(b) > 0)
            print(f"\nBuckets: total={len(self.buckets)}, non-empty={num_nonempty}")
            print(f"Batch={self.batch_size}, accum_steps={self.accum_steps}, world_size={self.world_size}")
            for i, b in enumerate(self.buckets):
                if b:
                    avg = sum(self.seq_lengths[idx] for idx in b) / len(b)
                    print(f"  Bucket {i}: {len(b)} samples, avg_len={avg:.1f}")
                else:
                    print(f"  Bucket {i}: EMPTY")
            for r in range(self.world_size):
                s = self.rank_samples[r]
                if not s:
                    print(f"  Rank {r}: No samples")
                else:
                    lengths = [self.global_index_to_length[i] for i in s]
                    print(f"  Rank {r}: {len(s)} samples, batches={len(s)//self.batch_size}, min={min(lengths)}, max={max(lengths)}, avg={sum(lengths)/len(lengths):.1f}")

    def _calculate_num_batches(self):
        total = len(self.rank_samples[self.rank])
        return total // self.batch_size if self.drop_last else math.ceil(total / self.batch_size)

    def _sync_num_batches(self):
        if self.world_size <= 1 or not dist.is_initialized():
            return self.num_batches
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        tensor = torch.tensor(self.num_batches, dtype=torch.int32, device=device)
        dist.all_reduce(tensor, op=dist.ReduceOp.MIN)
        return int(tensor.item())

    def set_epoch(self, epoch):
        self.epoch = int(epoch)
        if self.shuffle:
            self._assign_samples_to_ranks()
            self.num_batches = self._calculate_num_batches()
            self.global_num_batches = self._sync_num_batches()

    def __iter__(self):
        current = list(self.rank_samples[self.rank])
        if not current:
            return iter(())
        local_bucket_ids = [bisect.bisect_left(self.bucket_boundaries, self.global_index_to_length[i]) for i in current]

        boundaries = [(0, local_bucket_ids[0])]
        cur = local_bucket_ids[0]
        for i in range(1, len(local_bucket_ids)):
            if local_bucket_ids[i] != cur:
                cur = local_bucket_ids[i]
                boundaries.append((i, cur))
        boundaries.append((len(current), -1))

        bucket_batches = []
        for i in range(len(boundaries) - 1):
            s = boundaries[i][0]
            e = boundaries[i + 1][0]
            bid = boundaries[i][1]
            samples = current[s:e]
            batches_in_bucket = []
            for j in range(0, len(samples), self.batch_size):
                b = samples[j:j + self.batch_size]
                if len(b) == self.batch_size or not self.drop_last:
                    batches_in_bucket.append(b)
            if self.shuffle and batches_in_bucket:
                rng = np.random.RandomState(self.seed + self.epoch + bid)
                rng.shuffle(batches_in_bucket)
            bucket_batches.extend(batches_in_bucket)

        bucket_batches = bucket_batches[: self.global_num_batches]
        for batch in bucket_batches:
            yield batch

    def __len__(self):
        return int(self.global_num_batches)
        

def generate_gaussian_noise(data, snr_db):
    axes = tuple(range(1, data.ndim))
    signal_power = np.mean(np.abs(data) ** 2, axis=axes, keepdims=True)
    
    # Convert SNR to linear scale
    snr_linear = 10 ** (snr_db / 10)
    
    # Ensure SNR has proper shape for broadcasting
    if not isinstance(snr_linear, np.ndarray):
        snr_linear = np.array(snr_linear)
    if snr_linear.ndim == 0 or snr_linear.size == 1:
        snr_linear = snr_linear.reshape((-1,) + (1,)*(data.ndim-1))
    else:
        snr_linear = snr_linear.reshape((-1,) + (1,)*(data.ndim-1))
    
    # Calculate noise power
    noise_power = signal_power / snr_linear
    
    # Generate complex Gaussian noise
    # Real and imaginary parts scaled appropriately
    noise_real = np.random.standard_normal(data.shape) * np.sqrt(noise_power / 2)
    noise_imag = np.random.standard_normal(data.shape) * np.sqrt(noise_power / 2)
    
    # Combine into complex noise
    noise = noise_real + 1j * noise_imag
    
    return noise


def patch_maker(data, patch_size=4):
    B, T, K, U = data.shape
    # 检查维度是否可被patch_size整除
    assert T % patch_size == 0 and K % patch_size == 0 and U % patch_size == 0, \
        "Dimensions must be divisible by patch_size"
    
    # 计算每个维度的块数
    t_blocks = T // patch_size
    k_blocks = K // patch_size
    u_blocks = U // patch_size
    
    # 将数据重组成块结构 [B, t_blocks, k_blocks, u_blocks, patch_size, patch_size, patch_size]
    reshaped = data.reshape(B, 
                            t_blocks, patch_size, 
                            k_blocks, patch_size, 
                            u_blocks, patch_size)
    
    # 调整维度顺序：将块索引维度移到前面，内部patch维度移到后面
    # transposed = reshaped.permute(0, 1, 3, 5, 2, 4, 6)  # [B, t_blocks, k_blocks, u_blocks, patch_size, patch_size, patch_size]
    transposed = reshaped.transpose(0, 1, 3, 5, 2, 4, 6) 
    
    # 合并块索引维度（所有块）和内部维度（每个patch展平）
    num_patches = t_blocks * k_blocks * u_blocks
    patch_elements = patch_size ** 3
    patched_data = transposed.reshape(B, num_patches, patch_elements)

    # real_part = patched_data.real  
    # imag_part = patched_data.imag  

    # combined_patched_data = np.concatenate([real_part, imag_part], axis=-1) 
    
    return patched_data



if __name__ == "__main__":
    # Example usage
    # CSIDataset(file_path='/data/zcy_data/CSIGPT_Dataset/D1/train_data.mat', dataset_type='train')
    pass
