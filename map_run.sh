#!/bin/bash
echo "Bash version ${BASH_VERSION}..."
inflow=$1
for wt1 in {0..30..5}
do
    for wt2 in {0..30..5}
    do
        python prc/wireles.py create $wt1-$wt2-rotate-$inflow
        cp -r job/NREL-m/input job/$wt1-$wt2-rotate-$inflow
        python rotate.py $inflow
        python set_yaw.py $wt1-$wt2-rotate-$inflow $wt1 $wt2 
        python prc/wireles.py pre $wt1-$wt2-rotate-$inflow
        python prc/wireles.py solve $wt1-$wt2-rotate-$inflow
    done
done