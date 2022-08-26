#!/bin/bash
echo "Bash version ${BASH_VERSION}..."
inflow=$1
for i in {10..30..10}
do
    for j in {200..600..100}
    do
        python prc/wireles.py anime dyn-yaw-8wt-$i-${j}s
        # cp job/dyn-yaw-8wt-$i-${j}s/output/dyn-yaw-8wt-$i-${j}s_force.h5 .
    done
done
