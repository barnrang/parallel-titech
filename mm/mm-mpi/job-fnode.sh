#!/bin/sh
#$ -cwd
#$ -l f_node=2
#$ -l h_rt=0:10:00

. /etc/profile.d/modules.sh
module load cuda
module load openmpi

mpirun -n 56 -npernode 28 ./mm 2048 2048 2048
