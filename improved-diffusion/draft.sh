for i in {3..9}
do 
	echo $i
	squeue -w gnode00$i
done

for i in {10..99}
do
        echo $i
        squeue -w gnode0$i
done
