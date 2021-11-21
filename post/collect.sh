casename=$1
cd ..
python prc/wireles.py anime $casename
cp job/$casename/output/$casename.h5 post/data