#!/usr/bin/env bash


devsub () {
  echo $*
  sbatch --job-name=$1 --output=$1-%j.out --error=$1-%j.out --nodes=1 \
         --constraint volta32gb \
         --ntasks-per-node=1 --cpus-per-task=40 --gres=gpu:1 --signal=USR1@600 \
         --open-mode=append --time=2:00:00 --partition=dev --wrap="srun $2"
}


devsub2 () {
  echo $*
  sbatch --job-name=$1 --output=$1-%j.out --error=$1-%j.out --nodes=1 \
         --ntasks-per-node=1 --cpus-per-task=40 --gres=gpu:1 --signal=USR1@600 \
         --open-mode=append --time=24:00:00 --partition=dev --wrap="srun $2"
}


CMD="python ocpmodels/common/efficient_validation/example.py "

#devsub2 "bfgs" "${CMD} --batch-size 2 --relaxopt bfgs"


#for bs in 64 128 192; do
#  for mem in 15 25 50 100; do
#    devsub "lbfgs_${bs}_${mem}" "${CMD} --batch-size ${bs} --lbfgs-mem ${mem} --relaxopt lbfgs"
#  done
#done

for bs in 192; do
  for mem in 25 50; do
    devsub "lbfgs_${bs}_${mem}" "${CMD} --batch-size ${bs} --lbfgs-mem ${mem} --relaxopt lbfgs"
  done
done


#for mem in 15 25 50 100; do
#  for bs in 64 128 192; do
#    mae=$(tail -n 2 lbfgs_${bs}_${mem}-*.out | head -n 1 | grep -oP "[0-9\.]*")
#    tm=$(tail -n 1  lbfgs_${bs}_${mem}-*.out | grep -oP "[0-9\.]*")
#    echo -e $mem $bs $mae $tm
#  done
#done
#
