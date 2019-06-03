#!/bin/bash

echo "imagen,tiempo" > resultados_CUDA.csv

for i in {1..3}
do
	bin/edge lena.png >> resultados_CUDA.csv
done

for i in {1..3}
do
	bin/edge lagarto.jpg >> resultados_CUDA.csv
done

