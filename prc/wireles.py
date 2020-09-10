#!/usr/bin/env python
'''
Created on 18.04.2018

@author: trevaz (tristan.revaz@epfl.ch)

---------------------------------------------------------------------------------
script for wireles
---------------------------------------------------------------------------------
'''

#################################################################################
# IMPORT
#################################################################################
import os
import sys
import argparse

#################################################################################
# CONSTANTS
#################################################################################
PATH = {}
PATH['job'] = os.path.join('job')
PATH['src'] = os.path.join('src')
PATH['prc'] = os.path.join('prc')

#################################################################################
# MAIN FUNCTION
#################################################################################
def wireles_main():

    print_start()
    args = cli()
    print_info(args.app, args.case)

    if args.app == 'create':
        from app_create import create as app

    elif args.app == 'create0':
        from app_create0 import create as app

    elif args.app == 'create1':
        from app_create1 import create as app

    elif args.app == 'create2':
        from app_create2 import create as app

    elif args.app == 'remove':
        from app_remove import remove as app

    elif args.app == 'archive':
        from app_archive import archive as app

    elif args.app == 'unarchive':
        from app_unarchive import unarchive as app

    elif args.app == 'list':
        from app_list import list as app

    elif args.app == 'edit':
        from app_edit import edit as app

    elif args.app == 'clean':
        from app_clean import clean as app

    elif args.app == 'pre':
        from app_pre import pre as app

    elif args.app == 'command':
        from app_command import command as app

    elif args.app == 'make':
        from app_make import make as app

    elif args.app == 'debug':
        from app_debug import debug as app

    elif args.app == 'solve':
        from app_solve import solve as app

    elif args.app == 'monitor1':
        from app_monitor1 import monitor1 as app

    elif args.app == 'monitor2':
        from app_monitor2 import monitor2 as app

    elif args.app == 'post':
        from app_post import post as app

    elif args.app == 'post0':
        from app_post0 import post as app

    elif args.app == 'post1':
        from app_post1 import post as app

    elif args.app == 'post1vali':
        from app_post1vali import post as app

    elif args.app == 'post2':
        from app_post2 import post as app
    
    elif args.app == 'anime':
        from app_anime import anime as app
        
    else:
        print(' ERROR: no app named ' + args.app)
        sys.exit()

    app(PATH, args.case)

    print_end()

#####################################################################
# SECONDARY FUNCTIONS
#####################################################################
def print_start():
    '''
    DEF:    print start
    INPUT:  - ()
    OUTPUT: - ()
    '''
    print('########################################################')
    print('# WiRE-LES')
    print('########################################################')
    print('\n')

def cli():
    '''
    DEF:    command line interface for wireles.
    INPUT:  - ()
    OUTPUT: - ()
    '''
    usage_text = 'wireles.py <app_name> <case_name>' \
               + '\n' \
               + '    <app_name>    : name of application, required (type=string, no default)' + '\n' \
               + '    <case_name>   : name of case, required (type=string, no default)' + '\n' \
               + '\n' \
               + 'for more details, please look at readme_prc, read_me src'

    parser = argparse.ArgumentParser(description='wireles', usage=usage_text, add_help=False)

    parser.add_argument('app', type=str)
    parser.add_argument('case', type=str)
    args = parser.parse_args()

    return args

def print_info(app_name, case_name):
    '''
    DEF:    print info (app_name and case_name)
    INPUT:  - app_name: name of the app, type=string
            - case_name: name of the case, type=string
    OUTPUT: - ()
    '''
    print('# app  = ' + app_name)
    print('# case = ' + case_name)
    print('\n')


def print_end():
    '''
    DEF:    print end
    INPUT:  - ()
    OUTPUT: - ()
    '''
    print('\n')
    print('########################################################')

#####################################################################
# EXECUTE
#####################################################################
if __name__ == '__main__':
    wireles_main()
