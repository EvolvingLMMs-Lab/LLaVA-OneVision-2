import os
import sys
os.environ["WANDB_MODE"] = "offline"  # 推荐这样设置
cmd = ""
cmd += " CUDA_DEVICE_MAX_CONNECTIONS=1"
cmd += " NCCL_SOCKET_IFNAME=eth0"
cmd += " NCCL_SOCKET_NTHREADS=32"
cmd += " NCCL_NSOCKS_PERTHREAD=4"
cmd += " NCCL_ALGO=Ring"

cmd += " NCCL_IB_GID_INDEX=3"
cmd += " NCCL_IB_DISABLE=0"
cmd += " NCCL_DEBUG=INFO"
# cmd += " NCCL_IB_HCA=mlx5_bond_0,mlx5_bond_1,mlx5_bond_2,mlx5_bond_3,mlx5_bond_4,mlx5_bond_5,mlx5_bond_6,mlx5_bond_7"
cmd += " NCCL_IB_HCA=mlx5_2,mlx5_3,mlx5_4,mlx5_5,mlx5_6,mlx5_7,mlx5_8,mlx5_1"
cmd += " NCCL_NET_GDR_LEVEL=2"
cmd += " NCCL_IB_QPS_PER_CONNECTION=8"
cmd += " NCCL_IB_TC=160"
cmd += " NCCL_IB_TIMEOUT=22"
# cmd += " GLOO_SOCKET_IFNAME=eth0"

cmd += " USE_CHECKPOINT=0"
cmd += " CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 "
cmd += f" bash {sys.argv[1]}"
# cmd += "' &"
print(cmd)
os.system(cmd)
