
cd ~/workspace/WIRE-LES2/
for i in {001..117}
do
   echo "Create case $i"
   python prc/wireles.py create fullwake-$i
   cp prc/*.dat job/fullwake-$i/input
   cp prc/config-haohua job/fullwake-$i/input/config
   cd ~/workspace/WIRE-LES2/app
   python prepare_case.py $i
   cd ..
   ./debug_run.sh fullwake-$i
   python prc/wireles.py anime fullwake-$i
done

cd ~/workspace/WIRE-LES2/app
