''' '''
import os
from termcolor import colored
import pandas as pd
import numpy as np
from nptdms import TdmsFile
# import matplotlib.pyplot as plt
import dill as pickle

'''
        LOAD FOOD APPROACH TRAINING DATA
'''


def get_tdms_data(foraging_dict, session, base_folder, day, mouse, acq_rate = 30000):
    """
    loops over a list of dictionary with the info for each session and gets all the stimulus onset times
    """
    data_folder = os.path.join(base_folder, day, mouse)
    if os.path.isdir(data_folder):
        # find the tdms
        for file in os.listdir(data_folder):
            if file[-5:] == '.tdms' and 'Food' in file:
                tdmspath = os.path.join(data_folder, file)
                break
        # Loop over each .tdms file and extract stimuli frames
        print(colored('Loading {} -- {}: {}'.format(day, mouse, file), 'yellow'))
        tdms = TdmsFile(tdmspath)

        # extract the raw data
        sound_data = tdms.group_channels('sound_IN_Mantis')[0].data
        lick_data =  tdms.group_channels('licking_TS')[0].data
        pump_data = tdms.group_channels('Pump_pulse_reading')[0].data
        camera_data = tdms.group_channels('Camera_trigger')[0].data

        # get the times when things happen
        session_duration = len(lick_data) / acq_rate / 60
        pump_times = np.where(np.diff(pump_data)<-4)[0] / acq_rate
        lick_times = np.where(np.diff(lick_data)>4)[0] / acq_rate

        # extract sound onset and duration
        sound_on = np.where(abs(sound_data) > 0.1)[0]
        sound_times = np.append(sound_on[0],sound_on[1:][np.diff(sound_on)>acq_rate]) / acq_rate
        sound_end_times = np.append(sound_on[-1],sound_on[:-1][::-1][np.diff(sound_on[::-1])<-acq_rate])[::-1] / acq_rate
        sound_durations = np.round((sound_end_times - sound_times), 1)

        # extract sound onset and duration
        lick_on = np.where(abs(lick_data) > 3)[0]
        lick_times = np.append(lick_on[0],lick_on[1:][np.diff(lick_on)>(acq_rate/7)]) / acq_rate # not > 7 Hz...
        lick_end_times = np.append(lick_on[-1],lick_on[:-1][::-1][np.diff(lick_on[::-1])<-(acq_rate/7)])[::-1] / acq_rate
        lick_durations = lick_end_times - lick_times

        # check how many frames
        camera_on = np.where(abs(camera_data) > 2)[0]
        camera_times = np.append(camera_on[0],camera_on[1:][np.diff(camera_on)>200]) / acq_rate

        # print results
        print(colored('Session duration: {}\nLicks : {}\nPumps: {}\nSounds: {}\nSound duration: {}\nFrames triggered: {}'.\
                      format(session_duration, len(lick_times), len(pump_times), len(sound_times), sound_durations, len(camera_times)), 'green'))

        # put into dictionary
        foraging_dict['session_duration'][session] = session_duration
        foraging_dict['pump_times'][session] = pump_times
        foraging_dict['lick_times'][session] = lick_times
        foraging_dict['lick_duration'][session] = lick_durations
        foraging_dict['sound_times'][session] = sound_times
        foraging_dict['sound_duration'][session] = sound_durations

    return foraging_dict


    '''     MAIN SECTION OF SCRIPT TO GENERATE DATABASE     '''
    # get metadata for *all* sessions
    get_metadata(splitted_all_metadata)

    # get stimulus information for *new* sessions
    splitted_all_metadata = [(all_metadata[i::num_parallel_processes], metadata_dict) for i in range(num_parallel_processes)]
    _ = pool.starmap(get_stim_onset_times, splitted_all_metadata)

    # Create new database, and add to the old one if applicable
    if database is None:
        return generate_database_from_metadatas(metadata_dict, stimulus_dict)
    else:
        new_database = generate_database_from_metadatas(metadata_dict, stimulus_dict)
        for index, row in new_database.iterrows():
            if (index in database.index):
                # loop in case of erroneous duplicate entries (only take the first)
                # for stimuli, registration, tracking in zip(database.loc[index].Stimuli, database.loc[index].Registration, database.loc[index].Tracking):
                stimuli, registration, tracking = database.loc[index].Stimuli, database.loc[index].Registration, database.loc[index].Tracking
                new_database.loc[index].Stimuli = stimuli
                new_database.loc[index].Registration = registration
                new_database.loc[index].Tracking = tracking
                # break
        new_database = new_database.sort_values(by='Number')

        return new_database.sort_values(by='Number')

mice = ['P.1','P.2','P.3','P.4','P.5']
days = ['191127', '191128', '191129', '191130', '191201', '191202'] #'191126',
base_folder = 'D:\\Dropbox (UCL - SWC)\\DAQ\\upstairs_rig\\PS_mousetraining'
foraging_dict = {}
foraging_dict['session_duration'] = {}
foraging_dict['pump_times'] = {}
foraging_dict['lick_times'] = {}
foraging_dict['lick_duration'] = {}
foraging_dict['sound_times'] = {}
foraging_dict['sound_duration'] = {}

for day in days:
    for mouse in mice:
        # get session name
        session = day + '_' + mouse
        # load tdms and fill in foraging dict with data
        foraging_dict = get_tdms_data(foraging_dict, session, base_folder, day, mouse)
        # save foraging dict
        save_file = os.path.join(base_folder, 'foraging_data_IV')
        with open(save_file, "wb") as dill_file: pickle.dump(foraging_dict, dill_file)


# mouse = mice[0]
# day = days[5]
# with open(save_file, 'rb') as dill_file: foraging_dict_load = pickle.load(dill_file)
