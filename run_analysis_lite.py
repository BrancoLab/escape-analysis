from setup import setup
import pandas as pd
import os
import cv2
import pickle
from helper_code.registration_funcs import registration, get_arena_details
from DLC_code.dlc_funcs import extract_dlc, filter_and_transform_dlc, compute_pose_from_dlc
from important_code.escape_visualization_lite import visualize_escape


'''
--------------------make dlc object---------------------         
'''
class dlc_video():
    # set up dlc object
    def __init__(self, videos, timing):
        # import settings
        setup(self)

        # run dlc tracking
        self.run_tracking(videos)

        # register arena
        self.run_registration(videos)

        # extract coordinates
        self.get_coordinates(videos)

        # visualize escape
        self.run_rendering(videos, timing)

    # run tracking function
    def run_tracking(self, videos):
        '''     Run DLC to track videos        '''
        # print plans
        print('Tracking: '); print(videos)
        # import DLC tracking
        from deeplabcut.pose_estimation_tensorflow import analyze_videos
        # track each video
        for video_path in videos:
            analyze_videos(self.dlc_settings['config_file'], video_path)

    def run_registration(self, videos):
        '''     Register video to get mouse position in Common Coordinate Behavior space      '''
        # do the registration
        for video_path in videos:
            # files
            registration_file_name = os.path.join(os.path.dirname(video_path), 'registration')
            # to open saved registration
            try:
                with open(registration_file_name, "rb") as dill_file: session = pickle.load(dill_file)
                print('loaded registration for video ' + video_path)
            # make a new registration
            except:
                print('computing registration for video ' + video_path)
                # initialize data structure
                session = pd.Series();
                session['Registration'] = None; session['Metadata'] = {}; session['Metadata']['experiment'] = 'Circle other'
                session['Metadata']['video_file_paths'] = [[video_path]]
                # do the registration
                session, new_registration = registration(session, self.folders['fisheye_map_location'])
                # save the registration
                with open(registration_file_name, "wb") as dill_file: pickle.dump(session, dill_file)


    def get_coordinates(self, videos):
        '''     Extract coordinates using DLC output and the registration       '''
        # get raw coordinates from DLC
        for video_path in videos:
            # files
            coordinates_file_name = os.path.join(os.path.dirname(video_path), 'coordinates')
            registration_file_name = os.path.join(os.path.dirname(video_path), 'registration')
            # to open saved coordinates
            try:
                fail
                with open(coordinates_file_name, "rb") as dill_file: self.coordinates = pickle.load(dill_file)
                print('loaded coordinates for video ' + video_path)
            # make a new coordinates
            except:
                print('computing coordinates for video ' + video_path)
                # open the registration
                with open(registration_file_name, "rb") as dill_file: session = pickle.load(dill_file)
                get_arena_details(session)
                # get the coordinates from dlc
                self.coordinates = extract_dlc(self.dlc_settings, video_path)
                # filter coordinates and transform them to the common coordinate space
                self.coordinates = filter_and_transform_dlc(self.dlc_settings, self.coordinates, 0, 0, session['Registration'], plot=False, filter_kernel=7, confidence = 0, max_error = 9999)
                # compute speed, angles, and pose from coordinates
                self.coordinates = compute_pose_from_dlc(self.dlc_settings['body parts'], self.coordinates, session.shelter_location,
                                                         session['Registration'][4][0], session['Registration'][4][1], session.subgoal_location)
            # save the processed coordinates to the video folder
            with open(coordinates_file_name, "wb") as dill_file: pickle.dump(self.coordinates, dill_file)


    def run_rendering(self, videos, timing):
        '''     Display rendering of the escape         '''
        # visualize each video
        for video_path, times in zip(videos, timing):
            # files
            coordinates_file_name = os.path.join(os.path.dirname(video_path), 'coordinates')
            registration_file_name = os.path.join(os.path.dirname(video_path), 'registration')
            # load coordinates and registration
            with open(coordinates_file_name, "rb") as dill_file: self.coordinates = pickle.load(dill_file)
            with open(registration_file_name, "rb") as dill_file: session = pickle.load(dill_file)
            get_arena_details(session)
            # setup dlc object for visualization
            self.setup_object(times)
            # visualize escape
            visualize_escape(self, video_path, session)

    def setup_object(self, times):
        '''     Get  relevant variables into the dlc object     '''
        self.fps = 30
        self.shelter = True
        self.dark_theme = False
        self.stim_type = 'visual'
        self.end_frame = times[2]
        self.stim_frame = times[1]
        self.start_frame = times[0]
        self.fourcc = cv2.VideoWriter_fourcc(*"XVID")






'''
--------------------run dlc object on data---------------------  
'''
# select videos
videos = ["C:\\Users\\SWC\\Downloads\\8405_pos1_opt1\\8405_pos1_opt1.mp4",
          "C:\\Users\\SWC\\Downloads\\8405_pos1_opt2\\8405_pos1_opt2.mp4",
            "C:\\Users\\SWC\\Downloads\\8405_pos1_opt3\\8405_pos1_opt3.mp4",
            "C:\\Users\\SWC\\Downloads\\8405_pos1_opt4\\8405_pos1_opt4.mp4",
            "C:\\Users\\SWC\\Downloads\\8405_pos2_opt1\\8405_pos2_opt1.mp4",
            "C:\\Users\\SWC\\Downloads\\8405_pos2_opt2\\8405_pos2_opt2.mp4",
            "C:\\Users\\SWC\\Downloads\\8405_pos2_opt3\\8405_pos2_opt3.mp4",
            "C:\\Users\\SWC\\Downloads\\8405_pos2_opt4\\8405_pos2_opt4.mp4"]

# videos = [[] for x in range(3)]
# videos[0] = "D:\\data\\videos_NYU_20202\\RSC_control\\Media3.mp4"
# videos[1] = "D:\\data\\videos_NYU_20202\\RSC_lesion\\Media2.mp4"
# videos[2] = "D:\\data\\videos_NYU_20202\\Ruben\\Vale_movieS2.mp4"

# provide timing parameters for each video
# [start frame, stim frame, end frame]
timing = [[0,3*30, 8*30] for x in range(len(videos))]

# timing = [[] for x in range(3)]
# timing[0] = [0,3*30,8*30]
# timing[1] = [0,3*30,8*30]
# timing[2] = [0,3*30,8*30]

# run dlcing
d = dlc_video(videos, timing)
