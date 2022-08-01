#!/bin/bash
echo "Bash version ${BASH_VERSION}..."
inflow=$1
for wt1 in {-30..30..5}
do
    for wt2 in {-30..30..5}
    do
        # python prc/wireles.py create rotate-$inflow-$wt1-$wt2
        cp -r job/NREL-m-test4-n/input job/rotate-$inflow-$wt1-$wt2
        python rotate.py $inflow
        python set_yaw.py rotate-$inflow-$wt1-$wt2 $wt1 $wt2 
        python prc/wireles.py pre rotate-$inflow-$wt1-$wt2
        python prc/wireles.py solve rotate-$inflow-$wt1-$wt2
    done
done
