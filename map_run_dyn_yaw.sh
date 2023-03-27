#!/bin/bash
echo "Bash version ${BASH_VERSION}..."
# ./run.sh 7D-baseline
for i in {2..16..2}
do
    # cp -r job/7D-freq2e3-yaw10/input/config job/7D-freq${i}e3-yaw10/input
    ./run.sh 5D-freq${i}e3-yaw10
    # ./run.sh 7D-freq${i}e3-yaw10
    # rm -rf job/7D-freq${i}e3-yaw10-*
    # python prc/wireles.py anime 5D-freq${i}e3-yaw10
    # python prc/wireles.py anime 6D-freq${i}e3-yaw10
    # python prc/wireles.py anime 7D-freq${i}e3-yaw10
    # cp -r job/7D-freq${i}e3-yaw10/input/config job/5D-freq${i}e3-yaw10/input
    # cp -r job/7D-baseline/input/config job/7D-freq${i}e3-yaw10/input
done

# python prc/wireles.py anime 5D-baseline
# python prc/wireles.py anime 6D-baseline
# python prc/wireles.py anime 7D-baseline

# ./run.sh 6D-baseline
# ./run.sh 7D-baseline
# # do
#     for j in {100..600..100}
#     do
#         ./run.sh dyn-yaw-8wt-$i-${j}s
#     done
# done

# for i in {8,10,12,14,16,32}
# do
#     # cp -r job/8D-infinite/input/turb_loc.dat job/${i}D-infinite/input
#     # python prc/wireles.py anime ${i}D-infinite
#     ./run.sh ${i}D-infinite
# done

