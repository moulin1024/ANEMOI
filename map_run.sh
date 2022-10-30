#!/bin/bash
# echo "Bash version ${BASH_VERSION}..."
# inflow=$1
# for wt1 in {-30..30..5}
# do
#     for wt2 in {-30..30..5}
#     do
#         # python prc/wireles.py create rotate-$inflow-$wt1-$wt2
#         cp -r job/NREL-m-test4-n/input job/rotate-$inflow-$wt1-$wt2
#         python rotate.py $inflow
#         python set_yaw.py rotate-$inflow-$wt1-$wt2 $wt1 $wt2 
#         python prc/wireles.py pre rotate-$inflow-$wt1-$wt2
#         python prc/wireles.py solve rotate-$inflow-$wt1-$wt2
#     done
# done

# for idx in {1..1..1}
# do
#     python prc/wireles.py create dyn-yaw-8wt-9-5D-360s-$idx
#     cp -r job/dyn-yaw-8wt-test1/input job/dyn-yaw-8wt-9-5D-360s-$idx/
#     ./run.sh dyn-yaw-8wt-9-5D-360s-$idx
#     # python prc/wireles.py pre dyn-yaw-8wt-9-120s-$idx
# done

# for i in {1..9}
# do
#     ./run.sh ${i}D-baseline
# done

# ./run.sh dyn-yaw-8wt-9-5D-64s-1
# ./run.sh dyn-yaw-8wt-9-5D-72s-1
# ./run.sh dyn-yaw-8wt-9-5D-80s-1
# ./run.sh dyn-yaw-8wt-9-5D-90s-1
# ./run.sh dyn-yaw-8wt-9-5D-128s-1
# ./run.sh dyn-yaw-8wt-9-5D-144s-1
# ./run.sh dyn-yaw-8wt-9-5D-160s-1
