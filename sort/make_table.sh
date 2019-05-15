outp=table.bin
rm $outp || :
touch $outp

echo ",100,500,1000,2000,5000,10000,20000" >> $outp

for thread in 1 2 4 8
do
    printf $thread >> $outp
    for thresh in 100 500 1000 2000 5000 10000 20000
    do
        printf ,`./sort_omp 1000000 $thread $thresh | tail -n 1 | awk '{print $NF}'` >> $outp
    done
    echo >> $outp
done
