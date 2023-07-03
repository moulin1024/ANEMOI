#!/bin/bash
echo "Bash version ${BASH_VERSION}..."
case=$1
basecase=$2
for inflow in {-3..12..1}
do
    # python prc/wireles.py create $case-inflow-$inflow
    cp -r job/$basecase/input job/$case-inflow-$inflow
    python HornsRev.py $inflow $case-inflow-$inflow
    python copy_yaw.py $basecase $case-inflow-$inflow $inflow
    ./debug_run.sh $case-inflow-$inflow
done