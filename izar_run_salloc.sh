
casename=$(basename $1)
# echo $casename

# salloc -N 1 --gres gpu:1
# python prc/wireles.py debug $casename
# scancel -u molin

python prc/wireles.py clean $casename
python prc/wireles.py pre $casename
python prc/wireles.py debug $casename
# scancel -u molin
