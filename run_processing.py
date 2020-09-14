from setup import setup
import os
import numpy as np
from helper_code.processing_funcs import set_up_processing_variables, process_DLC_coordinates, get_trial_types, \
                                        set_up_arena_visualizations, setup_session_video, get_trial_details, flatten


class processing():
    def __init__(self, session, i):
        '''     code to perform processing        '''

        # load the processing parameters
        setup(self)
        # set up class variables
        set_up_processing_variables(self, session)
        # process and visualize DLC coordinates
        self.arena_with_prev_trials = session['temp']

        self.process_session()

        session['temp'] = self.arena_with_prev_trials


    def process_session(self):
        '''     process and visualize session data (homings, exploration, and escapes)     '''

        # import analysis scripts as necessary
        if self.processing_options['decompose homings']: self.processing_options['spontaneous homings'] = True
        if self.processing_options['decompose anti-homings']: self.processing_options['spontaneous anti-homings'] = True
        if self.processing_options['spontaneous homings']: from important_code.homing_extraction import extract_homings
        if self.processing_options['decompose homings']: from important_code.homing_decomposition import decompose_homings
        if self.processing_options['spontaneous anti-homings']: from important_code.homing_extraction import extract_homings
        if self.processing_options['decompose anti-homings']: from important_code.homing_decomposition import decompose_homings
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
                # get the dlc data
                process_DLC_coordinates(self, vid_num)
                # onlt use stimulus types that were used (e.g. audio, loom)
                # if not stims: continue
                # else:
                self.stims = stims
                # update the time based on the end of the previous video
                if vid_num:
                    self.previous_vid_duration += self.video_durations[vid_num-1]
                    self.stims = list(flatten(stims_all))
                elif self.session.Metadata['mouse_id']=='CA8360' and self.session.Metadata['experiment']=='Circle food': continue
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
                    # if really the same trial, skip - but this can screw up homing idxs
                    if self.stim_frame - self.previous_stim_frame < 30*10 and self.stim_frame > 30*10: continue

                    # if trial != 2 and trial != 3 and trial!= 5: continue
                    # if self.trial_type!=1: continue
                    # if trial > 1: continue
                    # mid = self.session.Metadata['mouse_id']
                    # if (mid!='CA8792' and mid!='CA8180' and trial !=3) or ( (mid=='CA8792' or mid == 'CA8180') and trial != 1): continue
                    # if (mid=='CA3720' and trial !=2) or (mid=='CA3740' and trial !=3) or (mid=='CA8541' and trial !=4) or (mid=='CA8551' and trial !=1): continue
                    # if (mid=='CA6950' and trial !=1) or (mid=='CA6990' and trial !=2) or (mid=='CA7220' and trial !=2) or (mid=='CA7501' and trial !=3)or (mid=='CA8180' and (trial <2 or trial > 3)): continue
                    # if (mid=='CA8360' and trial !=15) or (mid=='CA8390' and trial !=14) or (mid=='CA8380' and (trial != 14 and trial != 18)): continue
                    # if (mid == 'CA8360' and trial != 26 and trial!=22) or (mid == 'CA8370' and trial != 26) or (mid == 'CA8380' and trial != 25): continue
                    # if (mid == 'CA3390' and trial != 5 and trial != 6) or (mid == 'CA3410' and trial != 5 and trial != 6) or (mid == 'CA3151' and trial != 4) or (mid == 'CA6970' and trial != 1): continue
                    print(self.stim_frame)
                    # if trial > 2: continue

                    # if trial == 1: self.stim_frame += 46
                    # else: continue

                    # if trial != 2 and trial != 4: continue
                    # if trial > 3 and 'CA8070' in self.videoname:
                    #     continue
                    # if (self.trial_num!= 77 and self.trial_num!= 88 and self.trial_num != 11) and 'CA8791' in self.videoname:
                    #     continue

                    # extract spontaneous homings
                    if self.processing_options['spontaneous homings']: extract_homings(self)
                    # decompose homings
                    if self.processing_options['decompose homings']: decompose_homings(self)
                    # extract spontaneous homings
                    if self.processing_options['spontaneous anti-homings']: extract_homings(self, make_vid = False, anti=True)
                    # decompose homings
                    if self.processing_options['decompose anti-homings']: decompose_homings(self, anti = True)
                    # do simulations
                    if self.processing_options['simulate']: simulate(self.coordinates, strategy = 'all')
                    # analyze exploration
                    if self.processing_options['exploration']: explore(self, heat_map = False, path_from_shelter = False, silhouette_map = True)
                    # do analysis and video saving
                    if self.processing_options['visualize escapes']: visualize_escape(self)
                    # do basic video saving
                    if self.processing_options['raw clips']: raw_escape_video(self)

                if not len(stims): # if first video has no stims
                    self.stim_frame = self.video_durations[vid_num] - 301
                    get_trial_details(self)
                    # extract spontaneous homings
                    if self.processing_options['spontaneous homings']: extract_homings(self, make_vid=False)
                    # decompose homings
                    if self.processing_options['decompose homings']: decompose_homings(self)
                    # extract spontaneous homings
                    if self.processing_options['spontaneous anti-homings']: extract_homings(self, make_vid=False, anti=True)
                    # decompose homings
                    if self.processing_options['decompose anti-homings']: decompose_homings(self, anti=True)
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