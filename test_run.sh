
casename=$(date +%Y-%m-%d_%H-%M-%S)
echo casename
python prc/wireles.py create $casename
python prc/wireles.py pre $casename
python prc/wireles.py debug $casename
