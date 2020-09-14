from termcolor import colored

from helper_code.loadsave_funcs import print_plans, setup_database, save_data
from setup import setup


class master():
    def __init__(self):
        '''     master code to perform tracking, registration, processing, and analysis       '''
        # load the parameters
        setup(self)
        # print what we are planning to do
        print_plans(self)
        # Load the database
        setup_database(self)
        # Select sessions
        self.select_sessions()
        # Do DLC tracking
        if self.do_tracking: self.run_tracking()
        # Do registration
        if self.do_registration: self.run_registration()
        # Do processing
        if self.do_processing: self.run_processing_threadpool()
        # Do analysis
        if self.do_analysis: self.run_analysis()
        # Finished
        print(colored('done.', 'green'))


    def select_sessions(self):
        '''     Get a list of the user-selected sessions for analysis      '''
        # initialize selected sessions list
        self.selected_sessions = []
        # loop over all sessions
        for session_name in self.db.index[::-1]:
            # Get the session
            metadata = self.db.loc[session_name].Metadata
            # Check if this is one of the sessions we should be processing
            if (self.selector_type == 'experiment' and metadata['experiment'] in self.selector) \
               or (self.selector_type == 'number' and metadata['number'] in self.selector) or self.selector_type == 'all':
                self.selected_sessions.append(session_name)


    def run_registration(self):
        '''     Register video to get mouse position in Common Coordinate Behavior space      '''
        # import DLC tracking
        from helper_code.registration_funcs import registration
        # loop over all selected sessions
        for session_name in self.selected_sessions:
            # get and report the session and metadata
            session = self.db.loc[session_name]; metadata = session.Metadata
            print(colored('Registering: {} - {} (#{})'.format(metadata['experiment'],metadata['mouse_id'],metadata['number']), 'green'))
            # do the registration
            session, new_registration = registration(session, self.folders['fisheye_map_location'])
            # input the data from the registration into the global database
            if new_registration:
                self.db.loc[session_name]['Registration'] = session['Registration']
                save_data(self.folders['save_folder'], self.file_name, object=self.db, backup=False)

    def run_tracking(self):
        '''     Track video with DLC network to get mouse position at each frame      '''
        # import DLC tracking
        from deeplabcut.pose_estimation_tensorflow import analyze_videos
        # loop over all selected sessions
        for session_name in self.selected_sessions:
            # get and report the session metadata
            metadata = self.db.loc[session_name].Metadata
            print(colored('Running DLC: {} - {} (#{})'.format(metadata['experiment'],metadata['mouse_id'],metadata['number']), 'green'))
            # run the video through DLC network
            for video_path in metadata['video_file_paths']:
                analyze_videos(self.dlc_settings['config_file'], video_path[0])


    def run_processing_threadpool(self):
        '''     Threadpool processing for parallel performance      '''
        # if more than one processes should go on at once, threadpool them
        threads = self.processing_parameters['parallel processes']
        if len(self.selected_sessions)>1 and threads>1:
            from multiprocessing.dummy import Pool as ThreadPool
            splitted_session_list = [self.selected_sessions[i::threads] for i in range(threads)]
            pool = ThreadPool(threads)
            _ = pool.map(self.run_processing, splitted_session_list)
        else:
            self.run_processing(self.selected_sessions)


    def run_processing(self, sessions_to_run):
        '''     Visualize the session's escapes and exploration      '''
        # Import processing code
        from run_processing import processing
        # loop over all selected sessions
        i = 0; arena_with_history = []
        # sessions_to_run = [sessions_to_run[idx] for idx in [3,0,1,2]] #[3,5,0,1,4,2]
        # sessions_to_run = [sessions_to_run[idx] for idx in [1,2,0,3]] # [2,1,0,4,3,5]
        # sessions_to_run = [sessions_to_run[idx] for idx in [2,3,1,0]]
        # sessions_to_run = [sessions_to_run[idx] for idx in [3,1,2,0]]
        # sessions_to_run = [sessions_to_run[idx] for idx in [2,4,3,0,1,5]]
        # sessions_to_run = [sessions_to_run[idx] for idx in [3,2,0,1]]
        # sessions_to_run = [sessions_to_run[idx] for idx in [4,0,1,3,2]]
        # sessions_to_run = [sessions_to_run[idx] for idx in [1, 0, 2]]
        for session_name in sessions_to_run:
            # get and report the session and metadata
            session = self.db.loc[session_name]; metadata = session.Metadata
            print(colored('Processing: {} - {} (#{})'.format(metadata['experiment'],metadata['mouse_id'], metadata['number']), 'green'))
            # run the processing
            session['temp'] = arena_with_history
            processing(session, i)
            # to put all entire history there
            arena_with_history = session['temp']
            i+=1
            # input the data from the processing into the global database
            self.db.loc[session_name]['Tracking'] = session['Tracking']
            # save the data
            save_data(self.folders['save_folder'], self.file_name, object=self.db, backup=False)


    def run_analysis(self):
        '''     Analyze the data and test hypotheses      '''
        # Import processing code
        from run_analysis import analysis
        # run the processing
        analysis(self)



if __name__ == "__main__":
    m = master()