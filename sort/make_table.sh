rm table.bin || :
touch table.bin

echo ",100,500,1000,2000" >> table.bin

for thread in 1 2 4
do
    printf $thread >> table.bin
    for thresh in 100 500 1000 2000
    do
        printf ,`./sort_omp 1000000 $thread $thresh | tail -n 1 | awk '{print $NF}'` >> table.bin
    done
    echo >> table.bin
done