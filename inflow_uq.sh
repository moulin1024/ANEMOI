#!/bin/bash
echo "Bash version ${BASH_VERSION}..."
case=$1
for inflow in {-10..10..1}
do
    # python prc/wireles.py create HR1-m-inflow-$inflow
    # cp -r job/HR1-m/input job/HR1-m-inflow-$inflow
    python HornsRev.py $inflow HR1-m-inflow-$inflow
done