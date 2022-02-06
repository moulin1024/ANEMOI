#!/bin/bash
echo "Bash version ${BASH_VERSION}..."
for wt1 in {0..30..5}
do
    for wt2 in {0..30..5}
    do  
        # for inflow in {0}
        # do  
        python prc/wireles.py anime $wt1-$wt2-rotate-0
        # done
    done
done