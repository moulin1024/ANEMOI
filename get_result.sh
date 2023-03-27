#!/bin/bash
echo "Bash version ${BASH_VERSION}..."
for wt1 in {10..30..5}
do
    for wt2 in {-30..30..5}
    do  
        for inflow in -5 -3 0 3 5
        do  
            python prc/wireles.py anime rotate-$inflow-$wt1-$wt2
        # python prc/wireles.py anime rotate--2-$wt1-$wt2
        done
    done
done