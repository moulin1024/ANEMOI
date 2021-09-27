#!/usr/bin/env python
'''
Created on 18.04.2018

@author: trevaz (tristan.revaz@epfl.ch)

---------------------------------------------------------------------------------
app: debug
---------------------------------------------------------------------------------
'''

#################################################################################
# IMPORTS
#################################################################################
import os
from fctlib import get_case_path
from fctlib import get_config
#################################################################################
# CONSTANTS
#################################################################################


#################################################################################
# MAIN FUNCTION
#################################################################################
def debug(PATH, case_name):
    '''
    DEF:    monitor case.
    INPUT:  - case_name: name of the case, type=string
    OUTPUT: - ()
    '''
    case_path = get_case_path(PATH, case_name)
    print('extract config...')
    config = get_config(case_path)

    if config['double_flag'] == 0:
        print('=========================================')
        print('Single precision complie')
        print('=========================================')
        os.environ["PRECISION"] = ""
    else:
        print('=========================================')
        print('Double precision complie')
        print('=========================================')
        os.environ["PRECISION"] = "-DDOUBLE"

    os.chdir(os.path.join(case_path, 'src'))

    os.system('make -j6')
    #os.system('make -j2')
    os.system('mpirun -np ' + str(config['job_np']) +' ./wireles_src')
    # os.system('srun ./wireles_src')
