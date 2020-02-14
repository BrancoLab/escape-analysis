import numpy
import pandas as pd
import os
from termcolor import colored
import dill as pickle
import time
from helper_code.setup_funcs import create_database



def save_data(save_folder, file_name, object=None, backup = True):
    """ saves an object (the database) to file. If the object is not a dataframe, turns it into one"""
    # get the full file name
    file_name = os.path.join(save_folder, file_name)

    # create backup copy
    if backup: name_mods = ['','_backup']
    else: name_mods = ['']

    # Make sure not to save while the same file is being saved
    for name_mod in name_mods:
        while True:
            try:
                with open(file_name+name_mod, "wb") as dill_file:
                    pickle.dump(object, dill_file)
                break
            except:
                print('file in use'); time.sleep(5)

    print(colored('Database saved', 'yellow'))




def setup_database(analysis):
    """    Creating a new database from scratch using experiments.csv or load a database from a pre-existing file    """
    try:
       # Load existing database, if it exists
        db = pd.read_pickle(os.path.join(analysis.folders['save_folder'], analysis.file_name))
    except:
        # Create new database
        print(colored('Database ' + analysis.file_name + ' not found -- creating it now', 'red'))
        db = create_database(analysis.folders['excel_path'])
        save_data(analysis.folders['save_folder'], analysis.file_name, object=db, backup=True)

    # Add new sessions from experiments.csv
    if analysis.update_database:
        db = create_database(analysis.folders['excel_path'], database=db)
        save_data(analysis.folders['save_folder'], analysis.file_name, object=db, backup=True)

    analysis.db = db




def print_plans(analysis):
    """ When starting a new run, print the options specified in setup.py for the user to check """

    # Display the selected sessions
    if analysis.selector_type == 'all': print(colored('All sessions', 'blue'))
    else: print(colored('Sessions: {}'.format(analysis.selector), 'blue'))

    # Display the plan of action
    if analysis.do_tracking: print(colored('Tracking sessions', 'blue'))
    if analysis.do_registration: print(colored('Registering sessions', 'blue'))
    if analysis.do_processing:
        print(colored('Processing', 'blue'))
        processing_steps = []
        for step in analysis.processing_options.keys():
            if analysis.processing_options[step]: processing_steps.append(step)
        print(colored(processing_steps, 'blue'))
    if analysis.do_analysis:
        print(colored('Analyzing', 'blue'))
        analysis_steps = []
        for step in analysis.analysis_options.keys():
            if analysis.analysis_options[step]: analysis_steps.append(step)
        print(colored(analysis_steps, 'blue'))
