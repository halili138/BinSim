#!/bin/bash
#SBATCH -J Nlxp
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -p normal
#SBATCH --mem=80GB
#SBATCH --gres=gpu:1
#SBATCH -o error-log/job-%j.log
#SBATCH -e error-log/job-%j.err
#SBATCH -c 192

# module use /public/software/modulefiles/
# module load common/cpu-affinity-fix

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OMP_PLACES=cores
export OMP_PROC_BIND=close

name=${1:-"h14"}
ratio=${2:-"1.0"}
basis=${3:-"sto-3g"}
MAX_CORE=$((SLURM_CPUS_PER_TASK - 1))

julia mole_cuda_ite.jl ${name} ${ratio} ${basis} |tee hvec_test_${name}_${ratio}_${basis}_$SLURM_CPUS_PER_TASK.txt




for i in $(seq 0.5 0.1 2.5); do
    sbatch -c 96 sub.sh c2 "$i" cc-pvdz
done


for i in $(seq 0.5 0.1 2.5); do
    sbatch -c 64 4sub.sh h2o "$i" cc-pvdz
done