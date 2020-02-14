import pyexcel
import os
from multiprocessing.dummy import Pool as ThreadPool
from termcolor import colored
import pandas as pd
import cv2
import numpy as np
from nptdms import TdmsFile

def create_database(excelpath, database=None):
    """
    Creates a new database from scratch loading the info from datalog.csv
    and extracts info from the each session's .tdms file
    If a pre-existing database is passed it will add the sessions not already present
    """

    def generate_database_from_metadatas(metadata_dict, stimulus_dict):
        """
        Given a dictionary of session objects with their metadata creates a new database
        with all the sessions in the database and the associated metadata
        """
        # Create empty database from template class
        indexes = sorted(metadata_dict.keys())
        database = pd.DataFrame(index=indexes, columns=['Number', 'Metadata', 'Tracking', 'Registration', 'Stimuli'])

        # Fill in metadata from the dictionary
        for sessname, metadata in sorted(metadata_dict.items()):
            database['Metadata'][sessname] = metadata
            database['Number'][sessname] = metadata['number']

        for sessname, stimulus in sorted(stimulus_dict.items()):
            database['Stimuli'][sessname] = stimulus

        print(colored('Database initialized.','yellow'))
        return database

    def get_session_videodata(videos):
        """
        Get relevant variables for video files
        """
        # Get first frame of first video for future processing and number of frames in each video
        videos_data = {'Frame rate': [], 'Number frames': []}
        for idx, videofile in enumerate(videos):
            cap = cv2.VideoCapture(videofile)
            videos_data['Frame rate'].append(cap.get(cv2.CAP_PROP_FPS))
            videos_data['Number frames'].append(int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))
        videos_data['Cumu. Num Frames'] = np.cumsum(videos_data['Number frames'])
        return videos_data

    def get_stim_onset_times(sessions, metadata_dict):
        """
        loops over a list of dictionary with the info for each session and gets all the stimulus onset times
        """
        if not isinstance(sessions, list):
            sessions = list(sessions)

        for line in sessions:
            session_id = line['Sess.ID']
            if session_id:  # we loaded a line with session info
                session_name = '{}_{}'.format(line['Experiment'], line['Sess.ID'])

                # Check if session is already in database
                if database is not None and session_name in database.index:
                        continue
                session_stimuli = {}
                session_stimuli['session_id'] = session_id
                session_stimuli['stimuli'] = {}
                session_stimuli['stimuli']['visual'] = []
                session_stimuli['stimuli']['audio'] = []
                session_stimuli['stimuli']['digital'] = []
                videopaths = []
                # load data from .tdms and .avi fils
                for recording in line['Recordings']:
                    path = os.path.join(line['Base fld'], line['Exp fld'], recording)
                    for f in os.listdir(path):
                        if '.avi' in f:
                            videopaths.append(os.path.join(path, f))
                            print(videopaths)
                        elif '.tdms' == f[-5:]:
                            tdmspath = os.path.join(path, f)
                    # Loop over each .tdms file and extract stimuli frames
                    print(colored('Loading {}: {}'.format(session_name,os.path.basename(tdmspath)),'yellow'))
                    tdms = TdmsFile(tdmspath)
                    if metadata_dict[session_name]['software'] == 'behaviour':
                        visual_rec_stims, audio_rec_stims, digital_rec_stims = [], [], []
                        for group in tdms.groups():
                            for obj in tdms.group_channels(group):
                                if 'stimulis' in str(obj).lower():
                                    for idx in obj.as_dataframe().loc[0].index:
                                        if "/'  " in idx:
                                            framen = int(idx.split("/'  ")[1].split('-')[0])
                                        elif "/' " in idx:
                                            framen = int(idx.split("/' ")[1].split('-')[0])
                                        else:
                                            framen = int(idx.split("/'")[2].split('-')[0])
                                        if 'visual' in str(obj).lower():
                                            visual_rec_stims.append(framen)
                                        elif 'audio' in str(obj).lower():
                                            audio_rec_stims.append(framen)
                                        elif 'digital' in str(obj).lower():
                                            digital_rec_stims.append(framen)
                                        else:
                                            print(colored('Couldnt load stim correctly','yellow'))
                        # Now use the AI channels to find the *real* stimulus onset times and replace them
                        if audio_rec_stims:
                            stimulus_on_idx = np.where(tdms.group_channels('AI')[3].data > .55)[0] #in first data sets this is AI 1, later AI 2
                            idx_since_last_stimulus_on = np.diff(stimulus_on_idx)
                            if stimulus_on_idx.size:
                                stimulus_start_idx = stimulus_on_idx[np.append(np.ones(1).astype(bool),idx_since_last_stimulus_on>2*10000)] #usually 10 or 30
                                stimulus_start_frame = np.ceil(stimulus_start_idx / 10000 / (33 + 1 / 3) * 1000).astype(int)
                                stimulus_start_frame = stimulus_start_frame[stimulus_start_frame > 300]
                            else:
                                stimulus_start_frame = np.array(audio_rec_stims)
                                print('NO STIMULI FOUND!!')

                            if len(stimulus_start_frame) != len(audio_rec_stims):
                                print('audio AI channel does not match number of timestamps by ' + str(len(audio_rec_stims)-len(stimulus_start_frame)) )
                            else:
                                discrepancy = stimulus_start_frame - audio_rec_stims
                                if sum(discrepancy>8):
                                    print('audio AI channel does not match values of timestamps')
                                else:
                                    print(discrepancy)
                            # for conditioning experiment, just use what the tdms says
                            # if 'food' in line['Experiment']:
                            #     stimulus_start_frame = np.array(audio_rec_stims)
                            audio_rec_stims = list(stimulus_start_frame)

                        session_stimuli['stimuli']['visual'].append(visual_rec_stims)
                        session_stimuli['stimuli']['audio'].append(audio_rec_stims)
                        session_stimuli['stimuli']['digital'].append(digital_rec_stims)

                    else:
                        """     HERE IS WHERE THE CODE TO GET THE STIM TIMES IN MANTIS WILL HAVE TO BE ADDEDD       """
                        pass

                # Add to dictionary (or update entry)
                stimulus_dict[session_name] = session_stimuli
        return stimulus_dict

    def get_metadata(sessions):
        """
        loops over a list of dictionary with the info for each session and gets all the metadata
        """
        if not isinstance(sessions, list):
            sessions = list(sessions)

        for line in sessions:
            session_id = line['Sess.ID']
            if session_id:  # we loaded a line with session info
                session_name = '{}_{}'.format(line['Experiment'], line['Sess.ID'])

                # Check if session is already in database
                # if database is not None and session_name in database.index:
                        # print(colored('Session is already in database','yellow'))
                        # continue

                # Create the metadata
                session_metadata = {}
                session_metadata['session_id'] = session_id
                session_metadata['experiment'] = line['Experiment']
                session_metadata['date'] = line['Date']
                session_metadata['mouse_id'] = line['MouseID']
                session_metadata['software'] = line['Software']
                session_metadata['number'] = line['Number']

                # initialize video data
                session_metadata['video_file_paths'] = []
                session_metadata['tdms_file_paths'] = []
                session_metadata['videodata'] = []

                # load data from .tdms and .avi fils
                for recording in line['Recordings']:
                    path = os.path.join(line['Base fld'], line['Exp fld'], recording)
                    videopaths = []
                    for f in os.listdir(path):
                        if '.avi' in f:
                            videopaths.append(os.path.join(path, f))
                        elif '.tdms' == f[-5:]:
                            tdmspath = os.path.join(path, f)

                    # add file paths to metadata
                    session_metadata['video_file_paths'].append(videopaths)
                    session_metadata['tdms_file_paths'].append(tdmspath)

                    # Loop over each video and get the relevant data [e.g., number of frames, fps...]
                    session_metadata['videodata'].append(get_session_videodata(videopaths))

                  # Add to dictionary (or update entry)
                metadata_dict[session_name] = session_metadata
        return metadata_dict


    '''     MAIN SECTION OF SCRIPT TO GENERATE DATABASE     '''

    loaded_excel = pyexcel.get_records(file_name=excelpath)

    # Create a dictionary with each session's name as key and its metadata as value
    stimulus_dict, metadata_dict, all_metadata = {}, {}, []
    for line in loaded_excel:   # Read each line in the excel spreadsheet and load info
        temp = {
            'Sess.ID':     line['Sess.ID'],
            'Date':        line['Date'],
            'MouseID':     line['MouseID'],
            'Experiment':  line['Experiment'],
            'Software':    line['Software'],
            'Base fld':    line['Base fld'],
            'Exp fld':     line['Exp fld'],
            'Recordings':  line['Sub Folders'].split('; '),
            'Number':      line['Number']
        }
        all_metadata.append(temp)

    # Loop over each recordings subfolder and check that the paths are correct [fast check]
    for line in all_metadata:
        for recording in line['Recordings']:
            path = os.path.join(line['Base fld'], line['Exp fld'], recording)
            if not os.path.exists(path):
                raise ValueError('Folder not found\n{}'.format(path))
    print(colored('Excel spreadsheet loaded correctly. Now loading metadata.','yellow'))

    # Use loaded metadata to create the database. Threadpooled for faster execution
    num_parallel_processes = 6
    splitted_all_metadata = [all_metadata[i::num_parallel_processes] for i in range(num_parallel_processes)]
    pool = ThreadPool(num_parallel_processes)

    # get metadata for *all* sessions
    _ = pool.map(get_metadata, splitted_all_metadata)

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







