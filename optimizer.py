import numpy as np
import subprocess
import pandas as pd
import sys
case = sys.argv[1]
# rc = subprocess.call("./debug_run.sh "+case, shell=True)
case_config = pd.read_csv('job/'+case+'input/config')