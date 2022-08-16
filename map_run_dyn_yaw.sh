#!/bin/bash
echo "Bash version ${BASH_VERSION}..."
inflow=$1
for i in {10..30..10}
do
    for j in {100..600..100}
    do
        ./run.sh dyn-yaw-8wt-$i-${j}s
    done
done
