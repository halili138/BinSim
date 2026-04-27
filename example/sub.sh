#!/bin/bash
#SBATCH -J Nlxp
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -p fat_768
#SBATCH --exclusive
#SBATCH -o error-log/job-%j.log
#SBATCH -e error-log/job-%j.err
# 注意：保留 -c 192 作为默认值，如果命令行传入了新的 -c，命令行会覆盖它
#SBATCH -c 192

# module use /public/software/modulefiles/
# module load common/cpu-affinity-fix

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OMP_PLACES=cores
export OMP_PROC_BIND=close

# 接收从外部传入的第1个和第2个参数
# ${1:-"h2o"} 的意思是：如果 $1 有值则使用 $1，否则默认使用 "h2o"
name=${1:-"h14"}
basis=${2:-"sto-3g"}

echo "Running with name=${name}, basis=${basis}, cpus=${SLURM_CPUS_PER_TASK}"

julia vqe.jl ${name} ${basis} | tee group_${name}_${basis}_${SLURM_CPUS_PER_TASK}.txt