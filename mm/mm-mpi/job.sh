#!/bin/sh
#$ -cwd
#$ -l q_core=2
#$ -l h_rt=0:10:00

. /etc/profile.d/modules.sh
module load cuda
module load openmpi

mpirun -n 8 -npernode 4 ./mm 2048 2048 2048
