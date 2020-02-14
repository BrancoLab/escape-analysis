from setup import setup
import os
import numpy as np
from helper_code.processing_funcs import set_up_processing_variables, process_DLC_coordinates, get_trial_types, \
                                        set_up_arena_visualizations, setup_session_video, get_trial_details, flatten


class processing():
    def __init__(self, session):
        '''     code to perform processing        '''

        # load the processing parameters
        setup(self)
        # set up class variables
        set_up_processing_variables(self, session)
        # process and visualize DLC coordinates
        self.process_session()


    def process_session(self):
        '''     process and visualize session data (homings, exploration, and escapes)     '''

        # import analysis scripts as necessary
        if self.processing_options['decompose homings']: self.processing_options['spontaneous homings'] = True
        if self.processing_options['spontaneous homings']: from important_code.homing_extraction import extract_homings
        if self.processing_options['decompose homings']: from important_code.homing_decomposition import decompose_homings
        if self.processing_options['simulate']: from important_code.strategy_simulation import simulate_strategies
        if self.processing_options['exploration']: from important_code.exploration import explore
        if self.processing_options['visualize escapes']: from important_code.escape_visualization import visualize_escape
        if self.processing_options['raw clips']: from important_code.escape_visualization import raw_escape_video
        if self.processing_options['summary']: from important_code.summary import summarize

        # Loop over each possible stimulus type
        for stim_type, stims_all in self.session['Stimuli']['stimuli'].items():
            # Only use stimulus modalities that were used during the session
            if not sum(stims_all, []): continue
            self.stim_type = stim_type
            # Get the trial types, video durations, and number of trials
            get_trial_types(self, stims_all)
            # initiate the behavior visualizations
            set_up_arena_visualizations(self)
            # initiate the session videos
            if self.processing_options['visualize escapes']: setup_session_video(self)
            # Loop over each video in the session
            for vid_num, stims in enumerate(stims_all):
                # onlt use stimulus types that were used (e.g. audio, loom)
                if not stims: continue
                else: self.stims = stims
                # update the time based on the end of the previous video
                if vid_num:
                    self.previous_vid_duration += self.video_durations[vid_num-1]
                    self.stims = list(flatten(stims_all))
                # get the dlc data
                process_DLC_coordinates(self, vid_num)
                # if control, use fake frames
                # self.get_control_stim_frames(stims_all[vid_num])
                # Loop over each trial in the video
                for trial in range(len(stims)):
                    # set counters to the next trial
                    self.stim_frame = self.stims[self.trial_num]
                    self.trial_type = self.trial_types[self.trial_num]
                    # get the previous stimulus frame number (add previous video duration if multiple videos)
                    if trial: self.previous_stim_frame = self.stims[self.trial_num-1]
                    elif vid_num: self.previous_stim_frame = 0
                    self.trial_num += 1
                    # get the trial details
                    get_trial_details(self)
                    # if really the same trial, skip
                    if self.stim_frame - self.previous_stim_frame < 30*10: continue

                    # if trial != 3 and trial != 5: continue

                    # extract spontaneous homings
                    if self.processing_options['spontaneous homings']: extract_homings(self, make_vid = False)
                    # decompose homings
                    if self.processing_options['decompose homings']: decompose_homings(self)
                    # do simulations
                    if self.processing_options['simulate']: simulate(self.coordinates, strategy = 'all')
                    # analyze exploration
                    if self.processing_options['exploration']: explore(self, heat_map = False, path_from_shelter = True, silhouette_map = True)
                    # do analysis and video saving
                    if self.processing_options['visualize escapes']: visualize_escape(self)
                    # do basic video saving
                    if self.processing_options['raw clips']: raw_escape_video(self)
            # make a summary plot for each trial
            if self.processing_options['summary']: summarize(self)



    def get_control_stim_frames(self, stims_video):
        '''     get frames when a stim didn't happen but could have         '''
        # initialize values
        y_position = self.coordinates['center_location'][1].copy()
        stim_idx = np.array([], 'int')
        self.scaling_factor = 100 / self.height
        # loop over actual stims, making those times ineligible
        for stim in stims_video:
            stim_idx = np.append(stim_idx, np.arange(stim - int(self.fps*18), stim + int(self.fps*18)))
        y_position[stim_idx] = np.nan; y_position[:int(self.fps*18)] = np.nan; y_position[-int(self.fps*18):] = np.nan
        # must be in threat zone
        eligible_frames = np.where((y_position < (20 / self.scaling_factor)))[0]
        # create fake stim times
        stims_video_control = []
        for i, stim in enumerate(stims_video):
            stims_video_control.append(np.random.choice(eligible_frames))
        stims_video_control.sort()
        # replace real values with fake stims
        self.stims = stims_video_control