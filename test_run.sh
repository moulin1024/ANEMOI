
casename=$(date +%Y-%m-%d_%H-%M-%S)
echo casename
python prc/wireles.py create $casename
python prc/wireles.py pre $casename

cd job/$casename/src
make -j2
make -j2
mpirun -np 1 ./wireles_src
cd ../../..

python prc/wireles.py anime $casename