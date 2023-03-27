# ./run.sh NREL-m
# ./run.sh NREL-m-10 
# ./run.sh NREL-m-20

# wireles create7D-freq2e3-yaw10
# wireles create7D-freq4e3-yaw10
# wireles create7D-freq6e3-yaw10
# wireles create7D-freq8e3-yaw10
# wireles create7D-freq10e3-yaw10
# wireles create7D-freq12e3-yaw10
# wireles create7D-freq14e3-yaw10
# wireles create7D-freq16e3-yaw10

# for j in {2..16..2}
# do  
#     # python prc/wireles.py anime 7D-freq${j}e3-yaw10
#     # cp -r job/7D-freq${j}e3-yaw10/input job/7D-freq${j}e3-yaw10-${i}
#     # ./run.sh 7D-freq${j}e3-yaw10-${i} 
#     # echo 7D-freq${j}e3-yaw10-${i}
# done


for i in {1..9}
do
    # for j in {2..16..2}
    # do  
    python prc/wireles.py anime 7D-freq10e3-yaw10-${i}
    cp -r job/7D-freq10e3-yaw10-${i}/output/* ./project/paper4/data/
        # python prc/wireles.py create 7D-freq${j}e3-yaw10-${i}
        # cp -r job/7D-freq${j}e3-yaw10/input job/7D-freq${j}e3-yaw10-${i}
        # ./run.sh 7D-freq${j}e3-yaw10-${i} 
        # echo 7D-freq${j}e3-yaw10-${i}
    # done
    # wireles create7D-freq2e3-yaw10
    # ./run.sh ${i}D-baseline
done