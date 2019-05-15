#!/bin/sh
#$ -cwd
#$ -l h_node=1
#$ -l h_rt=00:10:00
echo ",100,500,1000,2000,5000,10000,20000"

for thread in 1 2 4 8
do
    printf $thread
    for thresh in 100 500 1000 2000 5000 10000 20000
    do
        printf ,`./merge_omp 1000000 $thread $thresh | tail -n 1 | awk '{print $NF}'`
    done
    echo
done
