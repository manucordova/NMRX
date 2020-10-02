#!/bin/bash -l


#SBATCH --time=3-00:00:00
#SBATCH --nodes=1
#SBATCH --mem 120G
#SBATCH --ntasks-per-node=28
#SBATCH --cpus-per-task=1
#SBATCH -A lrm

module purge
module load gcc openblas/0.3.6-openmp

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

set +e

conda activate SMLp3

f=$1
f=${f#run_}
f=${f%.py}

d=`grep "out_dir" $1`

dir="$(cut -d'"' -f2 <<< $d )"

echo $dir

if [ ! -d "$dir" ] ;
then
  mkdir $dir
fi

k=$2

for i in `seq 1 28`
do
  srun -n 1 --exclusive --mem=5G python $1 $(( $i + $k )) > output_${f}_$(( $i + $k )).log &
  sleep 5
done
wait
