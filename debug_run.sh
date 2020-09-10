
casename=$1
python prc/wireles.py clean $casename
python prc/wireles.py pre $casename
python prc/wireles.py debug $casename

