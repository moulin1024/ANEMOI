cd ~/workspace/WIRE-LES2/
for i in {001..056}
do
    echo "Create case $i"
    python prc/wireles.py anime fullwake-$i
done