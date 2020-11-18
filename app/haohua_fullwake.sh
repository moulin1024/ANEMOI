
cd ~/workspace/WIRE-LES2/
for i in {001..056}
do
   echo "Create case $i"
   python prc/wireles.py create fullwake-lambda-$i
   cp prc/*.dat job/fullwake-lambda-$i/input
   cp prc/config-haohua job/fullwake-lambda-$i/input/config
   cd ~/workspace/WIRE-LES2/app
   python prepare_case.py $i
   cd ..
   ./debug_run.sh fullwake-lambda-$i
   python prc/wireles.py anime fullwake-lambda-$i
done

cd ~/workspace/WIRE-LES2/app
