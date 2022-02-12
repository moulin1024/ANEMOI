./debug_run.sh NREL-m 
./debug_run.sh NREL-m-1 
./debug_run.sh NREL-m-2
cd ~/Downloads/NBMiner_Linux/
./nbminer -a ethash -o stratum+ssl://eu1.ethermine.org:5555 -lhr-mode 1 -u 0xdf0e5da8f4cf2ef11ff590e27e9a61ba3b69439e.epfl5