#!/bin/bash
echo "Bash version ${BASH_VERSION}..."
for wt1 in {-30..30..5}
do
    for wt2 in {-30..30..5}
    do  
        # for inflow in {0}
        # do  
        python prc/wireles.py anime rotate--4-$wt1-$wt2
        # python prc/wireles.py anime rotate--2-$wt1-$wt2
        # done
    done
done