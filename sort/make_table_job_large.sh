#!/bin/sh
#$ -cwd
#$ -l h_node=1
#$ -l h_rt=00:10:00
echo ",50000,100000,200000,500000,1000000,2000000"

for thread in 1 2 4 8
do
    printf $thread
    for thresh in 50000 100000 200000 500000 1000000 2000000
    do
        printf ,`./sort_omp 10000000 $thread $thresh | tail -n 1 | awk '{print $NF}'`
    done
    echo
done
