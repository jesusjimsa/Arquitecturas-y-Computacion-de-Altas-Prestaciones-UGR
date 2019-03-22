#!/bin/bash

if test $# -lt 1; then
	echo "Tienes que pasar el número de procesos";
	exit 1;
fi

mpirun -np $1 -hostfile "hosts.txt" "/fenix/alum/d3/jesusjimsa/acap/p2/cpi-mpi" 10000000000
