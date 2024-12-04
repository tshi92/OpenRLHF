#!/bin/bash

#SBATCH --job-name=test
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus-per-task=4
#SBATCH --mem=0
#SBATCH -p dev-g
#SBATCH -t 0-3:00:00

# Getting the node names

module --quiet purge
ml CrayEnv cray-python rocm/6.2.2 gcc

if [ $SLURM_LOCALID -eq 0 ] ; then
    rocm-smi
fi
sleep 2

# MIOPEN needs some initialisation for the cache as the default location
# does not work on LUMI as Lustre does not provide the necessary features.
export MIOPEN_USER_DB_PATH="/tmp/$(whoami)-miopen-cache-$SLURM_NODEID"
export MIOPEN_CUSTOM_CACHE_DIR=$MIOPEN_USER_DB_PATH

if [ $SLURM_LOCALID -eq 0 ] ; then
    rm -rf $MIOPEN_USER_DB_PATH
    mkdir -p $MIOPEN_USER_DB_PATH
fi
sleep 2

export TORCH_NCCL_TRACE_BUFFER_SIZE=1048576 
# export CUDA_VISIBLE_DEVICES=0
export NCCL_DEBUG=INFO
export NCCL_SOCKET_IFNAME=hsn0,hsn1,hsn2,hsn3

export RAY_DEDUP_LOGS=0
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export NUMEXPR_MAX_THREADS=${SLURM_CPUS_PER_TASK}

export RAY_EXPERIMENTAL_NOSET_ROCR_VISIBLE_DEVICES=1
export HIP_FORCE_DEV_KERNARG=1
export TORCHINDUCTOR_MAX_AUTOTUNE=1
export TORCHINDUCTOR_MAX_AUTOTUNE_GEMM_BACKENDS=TRITON
export TORCHINDUCTOR_COORDINATE_DESCENT_TUNING=1
export TORCHINDUCTOR_FREEZING=1
export TORCHINDUCTOR_CPP_WRAPPER=1
export TORCHINDUCTOR_LAYOUT_OPTIMIZATION=1

export VLLM_USE_TRITON_FLASH_ATTN=0
export VLLM_USE_RAY_SPMD_WORKER=1
export VLLM_USE_RAY_COMPILED_DAG=1

export  LIBDRM_RADEON_IDS_PATH=/scratch/project_462000436/tuoshi/python_packages/torch/share/libdrm/

nodes=$(scontrol show hostnames "$SLURM_JOB_NODELIST")
nodes_array=($nodes)

head_node=${nodes_array[0]}
head_node_ip=$(srun --nodes=1 --ntasks=1 -w "$head_node" hostname --ip-address)

# if we detect a space character in the head node IP, we'll
# convert it to an ipv4 address. This step is optional.
if [[ "$head_node_ip" == *" "* ]]; then
IFS=' ' read -ra ADDR <<<"$head_node_ip"
if [[ ${#ADDR[0]} -gt 16 ]]; then
  head_node_ip=${ADDR[1]}
else
  head_node_ip=${ADDR[0]}
fi
echo "IPV6 address detected. We split the IPV4 address as $head_node_ip"
fi

export SBATCH_ACCOUNT=project_462000436
export SALLOC_ACCOUNT=project_462000436

port=6379
ip_head=$head_node_ip:$port
export ip_head
echo "IP Head: $ip_head"

echo "Starting HEAD at $head_node"

srun --nodes=1 --ntasks=1 -w "$head_node" \
    ray start --head --node-ip-address="$head_node_ip" --port=$port \
    --num-cpus "${SLURM_CPUS_PER_TASK}" --num-gpus "${SLURM_GPUS_PER_TASK}" --block &

# optional, though may be useful in certain versions of Ray < 1.0.
sleep 10

echo "I am here"
# number of nodes other than the head node
worker_num=$((SLURM_JOB_NUM_NODES - 1))

for ((i = 1; i <= worker_num; i++)); do
    node_i=${nodes_array[$i]}
    worker_node_ip_i=$(srun --nodes=1 --ntasks=1 -w "$node_i" hostname --ip-address)
    mkdir -p $RAY_LOGGING_DIR/$node_i
    echo "Starting WORKER $i at $node_i of $worker_node_ip_i"
    srun --nodes=1 --ntasks=1 --export=ALL,HOST_IP=${worker_node_ip_i} -w "$node_i" \
        ray start --address "$ip_head" \
        --num-cpus "${SLURM_CPUS_PER_TASK}" --num-gpus "${SLURM_GPUS_PER_TASK}" --block &
    sleep 5
done

ray status
sleep 20


ray job submit --runtime-env-json '{"working_dir": "/scratch/project_462000436/tuoshi/OpenRLHF/openrlhf/test"}' \
    -- python3 -u -m deepspeed_test