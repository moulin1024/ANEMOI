for VARIABLE in 10 20 30
do
    python prc/wireles.py anime ultralong-$VARIABLE
    python prc/wireles.py anime ultralong+$VARIABLE
done