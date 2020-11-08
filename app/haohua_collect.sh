cd ~/workspace/WIRE-LES2/
cd app
for i in {01..83}
do
    echo "Create case $i"
    python collect_result_haohua.py $i
done