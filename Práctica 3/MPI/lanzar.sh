#!/bin/bash

for i in 2 4 8 12 24
do
	for j in {1..10}
	do
		mpirun -np $i ./edge lagarto.jpg
	done
done
