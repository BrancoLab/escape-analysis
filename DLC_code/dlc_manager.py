
import os
import platform
import random
import sys
import platform
print('importing DLC...')
import deeplabcut
import yaml



class DLC_manager:
    """
    Collection of useful functions for deeplabcut:

    ADD NEW VIDEOS TO EXISTING PROJECT
    deeplabcut.add_new_videos(`Full path of the project configuration file*',
    [`full path of video 4', `full path of video 5'],copy_videos=True/False)

    MANUALLY EXTRACT MORE FRAMES
    deeplabcut.extract_frames(‘config_path’,‘manual’)
    """

    """
        Typical pipeline for training a DLC network:

            - create project with videos
            - extract frames
            - label frames + check labels
            - creating training sets
            - train
            - evaluate -> make labelled videos and inspect by eye
            - enjoy

    """

    def __init__(self):
        with open('dlcproject_config.yml', 'r') as f:
            self.settings = yaml.load(f)

        if 'windows' in platform.system().lower():
            self.dlc_paths = self.settings['paths-windows']
        else:
            self.dlc_paths = self.settings['paths-mac']

    ### MACROS

    def initialise_project(self):
        """  Create a projec with the training videos, extract the frames and start labeling gui """
        print('Creating project')
        self.create_project()
        print('Extracting frames')
        self.extract_frames()
        print('Labeling frames')
        self.label_frames()
    
    ### UTILS

    def sel_videos_in_folder(self, all=False):
        dr = self.dlc_paths['dr']

        all_videos = [os.path.join(dr, f) for f in os.listdir(
            dr) if self.settings['video_format'] in f]

        if all or self.settings['number_of_training_videos'] >= len(all_videos):
            return all_videos
        else:
            selected_videos = random.sample(
                all_videos, self.settings['number_of_training_videos'])
            return selected_videos

    ### DLC functions

    def create_project(self, date):
        training_videos = self.sel_videos_in_folder()

        self.dlc_paths['cfg_path'] = deeplabcut.create_new_project(self.settings['experiment'], self.settings['experimenter'],
                                      training_videos, date, working_directory=self.dlc_paths['project_path'], copy_videos=True)

    def add_videos_to_project(self):
        vids_to_add = self.sel_videos_in_folder()
        deeplabcut.add_new_videos(self.dlc_paths['cfg_path'], vids_to_add, copy_videos=True)

    def extract_frames(self):
        deeplabcut.extract_frames(self.dlc_paths['cfg_path'], 'automatic', self.settings['extract_frames_mode'], crop=False, checkcropping=False)

    def label_frames(self):
        print('Getting ready to label frames')
        deeplabcut.label_frames(self.dlc_paths['cfg_path']) #, Screens=2)

    def check_labels(self):
        deeplabcut.check_labels(self.dlc_paths['cfg_path'])

    def create_training_dataset(self):
        deeplabcut.create_training_dataset(self.dlc_paths['cfg_path'])

    def train_network(self):
        deeplabcut.train_network(self.dlc_paths['cfg_path'], shuffle=1, gputouse = 0)

    def evaluate_network(self):
        deeplabcut.evaluate_network(self.dlc_paths['cfg_path'], plotting=True)

    def analyze_videos(self, videos=None):     
        if videos is None:
            videos = self.sel_videos_in_folder()
        else: 
            if not isinstance(videos, list):
                videos = [videos]
        deeplabcut.analyze_videos(self.dlc_paths['cfg_path'], videos, shuffle=1, save_as_csv=False)

    def create_labeled_videos(self, videos=None):
        if videos is None:
            videos = self.sel_videos_in_folder()
        else: 
            if not isinstance(videos, list):
                videos = [videos]
        deeplabcut.create_labeled_video(self.dlc_paths['cfg_path'],  videos)

    def extract_outliers(self, videos=None, outlier_algorithm = 'uncertain'):
        if videos is None: videos = self.sel_videos_in_folder()
        deeplabcut.extract_outlier_frames(self.dlc_paths['cfg_path'], videos, outlieralgorithm = outlier_algorithm, epsilon=10, comparisonbodyparts=['nose'], p_bound=.000000001, automatic = True)

    def refine_labels(self):
        deeplabcut.refine_labels(self.dlc_paths['cfg_path'])

    def merge_labels(self):
        deeplabcut.merge_datasets(self.dlc_paths['cfg_path'])

    def add_new_videos(self):
        deeplabcut.add_new_videos(self.dlc_paths['cfg_path'],vid, copy_videos=True)

    def update_training_video_list(self):
        '''
        Updates the config.yaml file to include all videos in your labeled-data folder
        '''
        # load config file
        with open(self.dlc_paths['cfg_path']) as f:
            config_file = yaml.load(f)

        # create dict of labelled data folders
        updated_video_list = {}
        crop_dict_to_use = config_file['video_sets'][list(config_file['video_sets'].keys())[0]]
        training_images_folder = os.path.join(os.path.dirname(self.dlc_paths['cfg_path']), 'labeled-data')
        for i, folder in enumerate(os.listdir(training_images_folder)):
            if folder.find('labeled') < 0:
                updated_video_list[os.path.join(self.dlc_paths['dr'], folder+'.'+self.settings['video_format'])] = crop_dict_to_use

        # replace video list in config file with new list
        config_file['video_sets'] = updated_video_list
        with open(self.dlc_paths['cfg_path'], "w") as f:
            yaml.dump(config_file, f)

    def delete_labeled_outlier_frames(self):
        '''
        Deletes the img.png files that are called 'labeled'
        '''
        # go through folders containing training images
        training_images_folder = os.path.join(os.path.dirname(self.dlc_paths['cfg_path']),'labeled-data')
        for i, folder in enumerate(os.listdir(training_images_folder)):
            if folder.find('labeled') < 0:
                # for the unlabeled folders, delete the png images that are labeled
                trial_images_folder = os.path.join(training_images_folder, folder)
                for image in os.listdir(trial_images_folder):
                    if image.find('labeled.png')>=0:
                        os.remove(os.path.join(trial_images_folder, image))



if __name__ == "__main__":
    '''
    Step 0: create project -> configure the config.yaml file in the project folder 
    '''
    dlc_master = DLC_manager()
    date = '2018-11-22'
    dlc_master.create_project(date)

    '''
    Step 1: extract and label frames from example videos
    '''
    # dlc_master.extract_frames()
    # dlc_master.label_frames()

    '''
    Step 2: check whether the frames were correctly labelled and create training set
    '''
    # dlc_master.check_labels()
    # dlc_master.refine_labels()
    # dlc_master.merge_labels()
    # dlc_master.create_training_dataset()

    '''
    Step 3: update \\dlc-models\\iteration-x\\network_name\\train\\pose_cfg.yaml -> train the network
    '''
    dlc_master.train_network()

    '''
    Step 4: evaluate the network
    '''
    # dlc_master.evaluate_network()
    vids = dlc_master.sel_videos_in_folder(all=True)
    vids = ['D:\\Dropbox (UCL - SWC)\\DAQ\\upstairs_rig\\19JAN27_peacepalace\\CA4060\\cam1.avi']
    #
    # vids = [vids[x] for x in [7]]
    # vids = vids[4:]
    # print(vids)
    # dlc_master.analyze_videos(videos=vids)
    # dlc_master.create_labeled_videos(videos=vids)

    # vids = ['D:\\Dropbox (UCL - SWC)\\DAQ\\upstairs_rig\\19JAN27_peacepalace\\CA4060\\cam1.avi']
    'TO DO: save analysis results in separate folder'
    'TO DO: concatenate machine labels and collected data so can refine the already labeled video'


    '''
    Step 5: refine the network
    '''
    # outlier_algorithm can be 'fitting', 'jump', or 'uncertain' (or all 3 in series)
    # dlc_master.extract_outliers(videos=vids, outlier_algorithm = 'jump')

    # dlc_master.delete_labeled_outlier_frames()
    # dlc_master.label_frames()
    dlc_master.update_training_video_list()

    # dlc_master.refine_labels() # adjust original labels

    # dlc_master.check_labels()


    '''
    Step 6: start next iteration
    '''
    # dlc_master.merge_labels() # increase iteration by 1
    # dlc_master.create_training_dataset()
    # now, repeat steps 3-5 until satisfied




    '''
    If needed: delete labeled frames from databse
    '''
    # import pandas as pd
    # videofolder = 'D:\data\DLC_nets\Barnes-Philip-2018-11-22\labeled-data\\3740_room'
    # dataname = 'CollectedData_Philip'
    # Dataframe = pd.read_hdf(os.path.join(videofolder, dataname + '.h5'))
    # Dataframe = Dataframe.drop('D:\data\DLC_nets\Barnes-Philip-2018-11-22\labeled-data\\3740_room\img001924.png')
    # Dataframe = Dataframe.drop('D:\data\DLC_nets\Barnes-Philip-2018-11-22\labeled-data\\3740_room\img101735.png')
    # Dataframe.to_csv(os.path.join(videofolder, dataname + '.csv'))
    # Dataframe.to_hdf(os.path.join(videofolder, dataname + '.h5'),'df_with_missing',format='table', mode='w')

    #
    #


