./debug_run.sh test-buffer
./debug_run.sh prec-buffer
./debug_run.sh main-buffer
./debug_run.sh main-buffer-yaw
python prc/wireles.py anime main-buffer
python prc/wireles.py anime main-buffer-yaw
cd post/haohua
./collect.sh main-buffer
./collect.sh main-buffer-yaw
# cp ../../job/main-fine/output/main-fine_ta.h5 ./sim/
python postprocessing.py
cd ..
cd ..