#!/bin/bash
#SBATCH -J Nlxp
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -p fat_768
#SBATCH --exclusive
#SBATCH -o error-log/job-%j.log
#SBATCH -e error-log/job-%j.err
#SBATCH -c 192

# module use /public/software/modulefiles/
# module load common/cpu-affinity-fix

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OMP_PLACES=cores
export OMP_PROC_BIND=close

name=${1:-"h14"}
basis=${2:-"sto-3g"}

echo "Running with name=${name}, basis=${basis}, cpus=${SLURM_CPUS_PER_TASK}"

MAX_CORE=$((SLURM_CPUS_PER_TASK - 1))

taskset -c 0-${MAX_CORE} \
    likwid-perfctr -c 0-${MAX_CORE} -g MEM -f \
        python3 -u pyscf_test.py ${name} ${basis} > \
        benchmark_pyscf_mem_${name}_${basis}_$SLURM_CPUS_PER_TASK.txt

taskset -c 0-${MAX_CORE} \
    likwid-perfctr -c 0-${MAX_CORE} -g L3 -f \
        python3 -u pyscf_test.py ${name} ${basis} > \
        benchmark_pyscf_l3_${name}_${basis}_$SLURM_CPUS_PER_TASK.txt

