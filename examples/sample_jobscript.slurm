#!/bin/bash

#SBATCH -A mp107d_a
#SBATCH --constraint=gpu
#SBATCH --qos=debug
#SBATCH --time=00:00:10
#SBATCH --nodes=1
#SBATCH --job-name=atmo3-test
#SBATCH -o /pscratch/sd/s/shamikg/nyx2cmb/output/slurm_logs/%x_job%j.out

export SLURM_CPU_BIND="cores"

module unload cudatoolkit
module load cudatoolkit/12.2
module load cudnn/8.9.3_cuda12
module load cray-mpich craype-accel-nvidia80

module use /pscratch/sd/s/shamikg/xgsmenv/cuda12-0.0.1/modulefiles
module load xgsmenv
source /pscratch/sd/s/shamikg/xgsmenv/cuda12-0.0.1/conda/bin/activate

let nnode=$SLURM_JOB_NUM_NODES
# 64 cores, 128 hardware threads 
let ngpus_per_node=4
let ncpus_per_task=128/$ngpus_per_node                                                                                                                                                                    
let ntask_node=$ngpus_per_node
let ntask=$nnode*$ntask_node
let tot_gpus=$nnode*$ngpus_per_node

echo "nnode=$nnode"
echo "ngpus_per_node=$ngpus_per_node"
echo "ncpus_per_task=$ncpus_per_task"
echo "ntask_node=$ntask_node"
echo "ntask=$ntask"
echo "tot_gpus=$tot_gpus"

export XGSMENV_NGPUS=$tot_gpus
export XLA_FLAGS=--xla_gpu_force_compilation_parallelism=1

export OMP_NUM_THREADS=$ncpus_per_task
export NUMBA_NUM_THREADS=$OMP_NUM_THREADS

srun -N $nnode -n $ntask --ntasks-per-node=$ntask_node --gres=gpu:$ngpus_per_node --cpus-per-task=$ncpus_per_task python /pscratch/sd/s/shamikg/nyx2cmb/code/ics4nyx.py /pscratch/sd/s/shamikg/nyx2cmb/resources/config/websky_768.yaml
