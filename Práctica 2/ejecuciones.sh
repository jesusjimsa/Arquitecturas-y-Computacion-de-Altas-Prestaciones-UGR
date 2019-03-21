#!/bin/sh

echo "Error 0.5" >> 'tiempos.dat'
./cpi-seq 200 0.5
./cpi-seq 2000 0.5
./cpi-seq 20000 0.5
./cpi-seq 200000 0.5
echo "Error 0.2" >> 'tiempos.dat'
./cpi-seq 200 0.2
./cpi-seq 2000 0.2
./cpi-seq 20000 0.2
./cpi-seq 200000 0.2
echo "Error 0.8" >> 'tiempos.dat'
./cpi-seq 200 0.8
./cpi-seq 2000 0.8
./cpi-seq 20000 0.8
./cpi-seq 200000 0.8
