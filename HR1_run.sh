#!/bin/bash
for inflow in {10..30..5}
do
    ./run.sh HR1-m-$inflow
    # for wt2 in {0..30..5}
    # do
    #     python prc/wireles.py create $wt1-$wt2-rotate-$inflow
    #     # cp -r job/NREL-m/input job/$wt1-$wt2-rotate-$inflow
    #     # python rotate.py $inflow
    #     # python set_yaw.py $wt1-$wt2-rotate-$inflow $wt1 $wt2 
    #     # python prc/wireles.py pre $wt1-$wt2-rotate-$inflow
    #     # python prc/wireles.py solve $wt1-$wt2-rotate-$inflow
    # done
done