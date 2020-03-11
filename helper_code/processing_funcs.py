import os
import cv2
from termcolor import colored
import dill as pickle
import numpy as np
from helper_code.registration_funcs import get_arena_details, model_arena


def set_up_processing_variables(self, session):
    '''     set up processing class variables      '''
    # set up various class variables
    self.session = session
    self.fps = self.session['Metadata']['videodata'][0]['Frame rate'][0]
    get_arena_details(self)

    # Create Tracking entry into session if not present
    if not isinstance(self.session['Tracking'], dict):
        self.session['Tracking'] = {}

    # Check for arena registration
    if not self.session['Registration']: self.registration = []

    # get file paths
    for vid_num, video_path in enumerate(self.session['Metadata']['video_file_paths']):
        self.video_path = video_path[0]
        self.processed_coordinates_file = os.path.join(os.path.dirname(self.video_path), 'coordinates')

    # Initialize variables in case of multi-video sessions
    self.session_trials_plot = None
    self.number_of_trials = 0
    self.trial_num = 0
    self.video_durations = []
    self.previous_vid_duration = 0
    self.previous_stim_frame = 0
    # reset trial types
    # self.session['Tracking']['Trial Types'] = [[],[]]

    # Save to a folder named after the experiment and the mouse
    self.save_folder = os.path.join(self.dlc_settings['clips_folder'], self.session['Metadata']['experiment'], str(self.session['Metadata']['mouse_id']))
    if not os.path.isdir(self.save_folder): os.makedirs(self.save_folder)
    # Also make a summary folder
    self.summary_folder = self.save_folder + ' summary'
    if not os.path.isdir(self.summary_folder): os.makedirs(self.summary_folder)



def set_up_arena_visualizations(self):
    '''     set up arena images for visualizations      '''
    # Set up arena for visualizations
    experiment = self.session['Metadata']['experiment']
    arena, _, _ = model_arena((self.height, self.width), self.trial_types[0], registration = False, obstacle_type = self.obstacle_type, \
                              shelter=not 'no shelter' in experiment and not 'food' in experiment, dark = self.dark_theme)
    _, _, shelter_roi = model_arena((self.height, self.width), self.trial_types[0], registration=False, obstacle_type=self.obstacle_type, shelter=not 'no shelter' in experiment, dark=self.dark_theme)
    # set up for homing extraction
    if self.processing_options['spontaneous homings']: self.homing_arena = cv2.cvtColor(arena.copy(), cv2.COLOR_GRAY2RGB)
    # set up for homing decomposition
    if self.processing_options['decompose homings']: self.decompose_arena = cv2.cvtColor(arena.copy(), cv2.COLOR_GRAY2RGB)
    # set up for escapes
    if self.processing_options['visualize escapes']:
        self.arena_with_prev_trials = cv2.cvtColor(arena.copy(), cv2.COLOR_GRAY2RGB)
        self.shelter_roi = shelter_roi
    # set up for exploration
    if self.processing_options['exploration']:
        arena, _, _ = model_arena((self.height, self.width), self.trial_types[0], registration=False, obstacle_type=self.obstacle_type, \
                                  shelter=not 'no shelter' in experiment and not 'food' in experiment, dark=self.dark_theme) #, shift_wall = True) #temporary
        self.exploration_arena = cv2.cvtColor(arena.copy(), cv2.COLOR_GRAY2RGB)


def process_DLC_coordinates(self, vid_num):
    '''     load or process DLC coordinates        '''
    # find file path for video and saved coordinates
    self.video_path = self.session['Metadata']['video_file_paths'][vid_num][0]
    self.processed_coordinates_file = os.path.join(os.path.dirname(self.video_path), 'coordinates')
    # open saved coordinates if they exist
    if os.path.isfile( self.processed_coordinates_file ) and not self.processing_options['process tracking']:
        print(colored(' - Already processed DLC coordinates', 'green'))
        # open saved coordinates
        with open(self.processed_coordinates_file, "rb") as dill_file: self.coordinates = pickle.load(dill_file)
    # otherwise, extract the coordinates
    else:
        # import DLC processing functions
        from DLC_code.dlc_funcs import extract_dlc, filter_and_transform_dlc, compute_pose_from_dlc
        # open raw coordinates from DLC
        self.coordinates = extract_dlc(self.dlc_settings, self.video_path)
        # process these coordinates
        print(colored(' - Processing DLC coordinates', 'green'))
        # filter coordinates and transform them to the common coordinate space
        self.coordinates = filter_and_transform_dlc(self.dlc_settings, self.coordinates, self.x_offset, self.y_offset,
                                                    self.session['Registration'], plot = False, filter_kernel = 7)
        # compute speed, angles, and pose from coordinates
        self.coordinates = compute_pose_from_dlc(self.dlc_settings['body parts'], self.coordinates, self.shelter_location,
                                                 self.session['Registration'][4][0], self.session['Registration'][4][1], self.subgoal_location)
        # save the processed coordinates to the video folder
        with open(self.processed_coordinates_file, "wb") as dill_file: pickle.dump(self.coordinates, dill_file)
        # input the data from the processing into the global database
        self.session['Tracking']['coordinates'] = self.processed_coordinates_file

    # reset homing indices if re analyzing this
    if self.processing_options['decompose homings']:
        self.coordinates['start_index'] = []
        self.coordinates['end_index'] = []


def get_trial_types(self, stims_all):
    '''    Takes in a video and stimulus information, and outputs the type of trial (obstacle or none)    '''
    # Loop over each video in the session
    for vid_num, stims_video in enumerate(stims_all):
        # initialize trial types array
        self.trial_types = []
        trials_in_video = len(stims_video)
        self.number_of_trials += trials_in_video
        number_of_vids = len(self.session['Metadata']['video_file_paths'])
        # set up the image and video that trial type information will modify
        vid = cv2.VideoCapture(self.session['Metadata']['video_file_paths'][vid_num][0])
        self.width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
        video_duration = int(vid.get(cv2.CAP_PROP_FRAME_COUNT))
        self.video_durations.append(video_duration)
        self.wall_change_frame = False
        # initialize trial type array
        if not ('Trial Types' in self.session['Tracking']):
            self.session['Tracking']['Trial Types'] = [[] for x in range(number_of_vids)]
        # If trial types are already saved correctly, just use those
        if len(list(flatten(self.session['Tracking']['Trial Types']))) == len(list(flatten(stims_all))): # and False:
            self.trial_types = self.session['Tracking']['Trial Types'][vid_num]
        else: # Otherwise, compute them de novo
            arena_specific_trial_types(self, stims_video, vid)

        # input the data from the processing into the global database
        self.session['Tracking']['Trial Types'][vid_num] = self.trial_types
        self.trial_types = sum(self.session['Tracking']['Trial Types'], [])
        # finish up
        print(self.trial_types)
        vid.release()

def arena_specific_trial_types(self, stims_video, vid):
    '''     determine trial types based on the obstacle type        '''
    # for the square arena, just do it based on time
    if self.obstacle_type == 'side wall 14' or self.obstacle_type == 'side wall 32':
        delayed_move_mice = ['CA7494']
        if self.session.Metadata['mouse_id'] in delayed_move_mice: # moved later
            self.trial_types = [2 * int(s > 30 * 30 * 60) for s in stims_video]
        elif self.session.Metadata['mouse_id'] == 'CA6960': #moved back
            self.trial_types = [0,2,2,1,1]
        else:
            self.trial_types = [2 * int(s > 19.5 * 30 * 60) for s in stims_video]

    # for the void arena, just do it based on time
    elif self.obstacle_type == 'void':
        void_up_mice = ['CA7505', 'CA7492', 'CA7502']
        if np.any([mouse == self.session.Metadata['mouse_id'] for mouse in void_up_mice]):
            self.trial_types = [2 * int(s < 36 * 30 * 60) for s in stims_video]
        else:
            self.trial_types = [2 for s in stims_video]
    # loop through each trial in the session
    elif 'wall' in self.obstacle_type:
        for trial_num, stim_frame in enumerate(stims_video):
            # If trial types depend on the trial, determine which trial is of which type
            if self.obstacle_changes and len(self.trial_types) < len(stims_video):
                wall_change_frame, trial_type = initialize_wall_analysis(self, stim_frame, vid)
                if wall_change_frame:
                    self.wall_change_frame = wall_change_frame
                # add this trial's trial type to the list
                self.trial_types.append(trial_type)
            # If all trials are the same, just add a 2 (obstacle) or 0 (no obstacle) to trial type list
            elif len(self.trial_types) < len(stims_video):
                if self.obstacle_type == 'none' or not 'wall' in self.session['Metadata']['experiment']:
                    self.trial_types.append(0)
                else:
                    self.trial_types.append(2)


def setup_session_video(self):
    '''    Initiate the video for the whole sessions    '''
    # set up the session video
    self.fourcc = cv2.VideoWriter_fourcc(*"XVID")
    self.session_videoname = '{}_{}'.format(self.session['Metadata']['experiment'], self.session['Metadata']['mouse_id'])
    # including the dlc visualization
    self.session_video_dlc = cv2.VideoWriter(os.path.join(self.summary_folder, self.session_videoname + ' vid (all trials DLC).mp4'), self.fourcc, self.fps, (self.width, self.height), True)
    # and the actual video
    self.session_video = cv2.VideoWriter(os.path.join(self.summary_folder, self.session_videoname + 'vid (all trials).mp4'), self.fourcc, self.fps, (self.width, self.height), False)
    
    

def get_trial_details(self):
    '''    Get start time, end time, and video name for this trial    '''
    self.start_frame = int(self.stim_frame - (self.processing_parameters['seconds pre stimulus'] * self.fps))
    self.end_frame = int(self.stim_frame + (self.processing_parameters['seconds post stimulus'] * self.fps))

    self.videoname = '{}_{} - trial {} ({}\')'.format(self.session['Metadata']['experiment'],self.session['Metadata']['mouse_id'],
                                                 self.trial_num, round((self.stim_frame + self.previous_vid_duration) / self.fps / 60))


def speed_colors(speed, simulation=False, red=False, plotting = False):
    '''    set up colors for speed-dependent DLC analysis    '''
    # colors depending on speed
    if red:
        slow_color = np.array([240, 240, 240])
        medium_color = np.array([240, 10, 10])
        fast_color = np.array([220, 220, 0])
        super_fast_color = np.array([232, 0, 0])
    else:
        slow_color = np.array([240, 240, 240])
        medium_color = np.array([190, 190, 240])
        fast_color = np.array([0, 232, 120])
        super_fast_color = np.array([0, 232, 0])
    # vary color based on speed
    speed_threshold_3 = 20
    speed_threshold_2 = 14
    speed_threshold = 7
    # apply thresholds
    if speed > speed_threshold_3:
        speed_color = super_fast_color
    elif speed > speed_threshold_2:
        speed_color = ((speed_threshold_3 - speed) * fast_color + (speed - speed_threshold_2) * super_fast_color) / (speed_threshold_3 - speed_threshold_2)
    elif speed > speed_threshold:
        speed_color = ((speed_threshold_2 - speed) * medium_color + (speed - speed_threshold) * fast_color) / (speed_threshold_2 - speed_threshold)
    else:
        speed_color = (speed * medium_color + (speed_threshold - speed) * slow_color) / speed_threshold
    # turn this color into a color multiplier
    speed_color_light = 1 - (1 - speed_color / [255, 255, 255]) / (np.mean(1 - speed_color / [255, 255, 255]) / .08)
    speed_color_dark = 1 - (1 - speed_color / [255, 255, 255]) / (np.mean(1 - speed_color / [255, 255, 255]) / .38)
    if plotting: speed_color_dark = speed_color_dark ** np.max((2, (16 / np.max((2, speed)))))
    else: speed_color_dark = speed_color_dark ** 2
    # gray for simulation
    if simulation:
        speed_color_light, speed_color_dark = np.ones(3) * np.mean(speed_color_light) / 1.2, np.ones(3) * np.mean(speed_color_dark) / 1.2

    return speed_color_light, speed_color_dark


def extract_variables(self, new_thresholds = True):
    '   initialize variables used for homing extrations     '
    if new_thresholds: self.thresholds_passed = np.zeros((2, self.stim_frame - self.previous_stim_frame))
    self.frame_nums = np.arange(self.previous_stim_frame + self.skip_frames, self.stim_frame + self.skip_frames)
    
    self.goal_speeds = self.coordinates['speed_toward_subgoal'][self.frame_nums]
    self.absolute_speeds = self.coordinates['speed'][self.frame_nums]
    self.subgoal_angles = self.coordinates['subgoal_angle'][self.frame_nums]
    self.body_angles = self.coordinates['body_angle'][self.frame_nums]
    self.angular_speed = self.coordinates['angular_speed_shelter'][self.frame_nums]
    self.distance_from_shelter = self.coordinates['distance_from_shelter'][self.frame_nums]

    self.distance_from_subgoal = self.coordinates['distance_from_shelter'][self.frame_nums]
    self.within_subgoal_bound = np.array(self.coordinates['in_subgoal_bound'])[self.frame_nums]

    self.x_location_butt = self.coordinates['butty_location'][0][self.frame_nums].astype(np.uint16)
    self.y_location_butt = self.coordinates['butty_location'][1][self.frame_nums].astype(np.uint16)

    self.x_location_face = self.coordinates['front_location'][0][self.frame_nums].astype(np.uint16)
    self.y_location_face = self.coordinates['front_location'][1][self.frame_nums].astype(np.uint16)

    frame_shape = [self.height, self.width]
    self.shelter_location = [s * frame_shape[1-i]/1000 for i, s in enumerate(self.shelter_location)]
    self.distance_from_shelter_front = np.sqrt( (self.x_location_face - self.shelter_location[0] )**2 + (self.y_location_face - self.shelter_location[1])**2 )

    distance_arena = np.load('.\\arena_files\\distance_arena_' + self.obstacle_type + '.npy')
    angle_arena = np.load('.\\arena_files\\angle_arena_' + self.obstacle_type + '.npy')

    self.distance_from_obstacle = distance_arena[self.y_location_butt, self.x_location_butt]
    self.angles_from_obstacle = angle_arena[self.y_location_butt, self.x_location_butt]
    


def convolve(data, n, sign, time = 'current', time_chase = 20):
    '       convolve data with a uniform filter array       '
    if time == 'past':
        convolved_data = np.concatenate((np.zeros(n - 1), np.convolve(data, np.ones(n) * sign, mode='valid'))) / n
    elif time == 'future':
        convolved_data = np.concatenate((np.convolve(data, np.ones(n) * sign, mode='valid'), np.zeros(n - 1))) / n
    elif time == 'far future':
        convolved_data = np.concatenate((np.convolve(data, np.ones(n) * sign, mode='valid'), np.zeros(n - 1 + time_chase))) / n
        convolved_data = convolved_data[time_chase:]
    else: # current
        convolved_data = np.concatenate((np.zeros(int(n/2 - 1)), np.convolve(data, np.ones(n) * sign, mode='valid'), np.zeros(int(n/2)))) / n

    return convolved_data




def threshold(data, limit, type = '>'):
    '''     threshold an array      '''
    if type == '>':
        passed_threshold = (data > limit).astype(np.uint16)
    elif type == '<':
        passed_threshold = (data < limit).astype(np.uint16)

    return passed_threshold


def register_frame(frame, x_offset, y_offset, registration, map1, map2):
    '''     go from a raw to a registered frame        '''
    # make into 2D
    frame_register = frame[:, :, 0]
    # fisheye correction
    if registration[3]:
        # pad the frame
        frame_register = cv2.copyMakeBorder(frame_register, y_offset, int((map1.shape[0] - frame.shape[0]) - y_offset),
                                            x_offset, int((map1.shape[1] - frame.shape[1]) - x_offset), cv2.BORDER_CONSTANT, value=0)
        # fisheye correct the frame
        frame_register = cv2.remap(frame_register, map1, map2, interpolation=cv2.INTER_LINEAR,
                                   borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        # un-pad the frame
        frame_register = frame_register[y_offset:-int((map1.shape[0] - frame.shape[0]) - y_offset),
                         x_offset:-int((map1.shape[1] - frame.shape[1]) - x_offset)]
    # register the frame
    frame = cv2.cvtColor(cv2.warpAffine(frame_register, registration[0], frame.shape[0:2]),cv2.COLOR_GRAY2RGB)
    # make into 2D again
    frame = frame[:, :, 0]

    return frame


def flatten(iterable):
    '''       flatten a nested list       '''
    it = iter(iterable)
    for e in it:
        if isinstance(e, (list, tuple)):
            for f in flatten(e):
                yield f
        else:
            yield e


def initialize_wall_analysis(self, stim_frame, vid):
    '''     determine whether this is a wall up, wall down, wall, or no wall trial      '''
    # initialize cariables
    start_frame = int(stim_frame - (3 * self.fps))
    end_frame = int(stim_frame + (10 * self.fps))
    experiment = self.session['Metadata']['experiment']
    # load fisheye mapping
    maps = np.load(self.folders['fisheye_map_location']); map1 = maps[:, :, 0:2]; map2 = maps[:, :, 2] * 0
    # set up ROIs for detecting whether the wall is up or down
    x_wall_up_ROI_left = [int(x * self.width / 1000) for x in [223 - 10, 249 + 10]]
    x_wall_up_ROI_right = [int(x * self.width / 1000) for x in [752 - 10, 777 + 10]]
    y_wall_up_ROI = [int(x * self.height / 1000) for x in [494 - 10, 504 + 10]]
    y_wall_up_ROI = [int(x * self.height / 1000) for x in [494 - 30, 504 + 30]]
    # check state of wall on various frames
    frames_to_check = [start_frame, stim_frame - 1, stim_frame + 13, end_frame] # + 13
    wall_darkness = np.zeros((len(frames_to_check), 2))
    # loop over frames to check
    for i, frame_to_check in enumerate(frames_to_check):
        # extract frame
        vid.set(cv2.CAP_PROP_POS_FRAMES, frame_to_check)
        ret, frame = vid.read()
        frame = register_frame(frame, self.x_offset, self.y_offset, self.session['Registration'], map1, map2)
        # compute darkness around wall edge
        wall_darkness[i, 0] = sum(sum(frame[y_wall_up_ROI[0]:y_wall_up_ROI[1], x_wall_up_ROI_left[0]:x_wall_up_ROI_left[1]] < 150))
        wall_darkness[i, 1] = sum(sum(frame[y_wall_up_ROI[0]:y_wall_up_ROI[1], x_wall_up_ROI_right[0]:x_wall_up_ROI_right[1]] < 150))
    # compile darkness before and after trial
    wall_darkness_pre = np.min(wall_darkness[0:int(len(frames_to_check) / 2), 0:2])
    wall_darkness_post = np.min(wall_darkness[int(len(frames_to_check) / 2):len(frames_to_check), 0:2])
    # use these darkness levels to detect whether wall is up, down, rising, or falling:
    # print(wall_darkness_pre)
    # print(wall_darkness_post)
    if 'void' in experiment and 'up' in experiment: experiment = 'void down'
    if (wall_darkness_pre - wall_darkness_post) < -30 and wall_darkness_pre < 200:
        # print(colored('Wall rising trial!', 'green'))
        wall_height_timecourse = [0]
        trial_type = 1
    elif (wall_darkness_pre - wall_darkness_post) > 30 and wall_darkness_post < 200:
        # print(colored('Wall falling trial', 'green'))
        wall_height_timecourse = [1]
        trial_type = -1
    elif ('down' in experiment and (-1 in self.trial_types or 0 in self.trial_types)) or wall_darkness_post < 160: # TEMP ?
        trial_type = 0
        wall_height_timecourse = 0
    elif 'down' in experiment and not (-1 in self.trial_types):
        trial_type = 2
        wall_height_timecourse = 1
    elif 'up' in experiment and 1 in self.trial_types:
        trial_type = 2
        wall_height_timecourse = 1
    elif 'up' in experiment and not (1 in self.trial_types):
        trial_type = 0
        wall_height_timecourse = 0
    else:
        print('Uh-oh -- not sure what kind of trial!')
    # if its a wall rise or wall fall trial, get the timecourse and the index at which the wall rises or walls
    if trial_type == 1 or trial_type == -1:
        wall_change_frame = stim_frame + 14
        # loop across possible frames
        for frame_num in range(stim_frame, stim_frame + 13):
            # read the frame
            vid.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
            ret, frame = vid.read()
            frame = register_frame(frame, self.x_offset, self.y_offset, self.session['Registration'], map1, map2)
            # measure the wall edges
            left_wall_darkness = sum(sum(frame[y_wall_up_ROI[0]:y_wall_up_ROI[1], x_wall_up_ROI_left[0]:x_wall_up_ROI_left[1]] < 200))
            right_wall_darkness = sum(sum(frame[y_wall_up_ROI[0]:y_wall_up_ROI[1], x_wall_up_ROI_right[0]:x_wall_up_ROI_right[1]] < 200))
            # calculate overall change
            left_wall_darkness_change_overall = left_wall_darkness - wall_darkness[1, 0]
            right_wall_darkness_change_overall = right_wall_darkness - wall_darkness[1, 1]
            # show wall down timecourse
            if trial_type == -1:
                wall_height = round(1 - np.mean(
                    [left_wall_darkness_change_overall / (wall_darkness[2, 0] - wall_darkness[1, 0]),
                     right_wall_darkness_change_overall / (wall_darkness[2, 1] - wall_darkness[1, 1])]), 1)
                # get frame when wall is down
                if abs(wall_height-1) >= .9: #wall_height <= .1
                    wall_change_frame = frame_num
                    break
            # show wall up timecourse
            if trial_type == 1:
                wall_height = round(np.mean(
                    [left_wall_darkness_change_overall / (wall_darkness[2, 0] - wall_darkness[1, 0]),
                     right_wall_darkness_change_overall / (wall_darkness[2, 1] - wall_darkness[1, 1])]), 1)
                # get frame when wall is up
                if wall_height >= .6:
                    wall_change_frame = frame_num
                    break
    else:
        wall_change_frame = False
    return wall_change_frame, trial_type
