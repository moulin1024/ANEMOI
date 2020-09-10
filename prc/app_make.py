#!/usr/bin/env python
'''
Created on 18.04.2018

@author: trevaz (tristan.revaz@epfl.ch)

---------------------------------------------------------------------------------
app: make
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
def make(PATH, case_name):
    '''
    DEF:    monitor case.
    INPUT:  - case_name: name of the case, type=string
    OUTPUT: - ()
    '''
    case_path = get_case_path(PATH, case_name)
    ############################################################################
    # EXTRACT CONFIG
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
    # Try to compile twice in case the first compiling falied due to dependency
    os.system('make -j2 -C ' + str(os.path.join(case_path, 'src')))
    os.system('make -j2 -C ' + str(os.path.join(case_path, 'src')))