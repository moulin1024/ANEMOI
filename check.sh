./debug_run.sh NREL-m-coarse
./debug_run.sh NREL-m-coarse-yaw1
./debug_run.sh NREL-m-coarse-yaw2

cd post
python post.py
cd ..
# alias wireles='python prc/wireles.py'

# python prc/wireles.py anime NREL-m-coarse
# python prc/wireles.py anime NREL-m-coarse-yaw1
# python prc/wireles.py anime NREL-m-coarse-yaw2