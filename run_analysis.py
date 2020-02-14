from setup import setup
from important_code.do_analysis import analyze_data
from important_code.plot_analysis import *
import os
import dill as pickle

class analysis():
    def __init__(self, dataframe):
        '''     Analyze the data and make summary plots     '''
        # load the processing parameters
        setup(self)
        # select the relevant sessions
        self.select_sessions(dataframe)
        # make analysis directory
        self.make_analysis_directory()
        # loop over analysis types
        for analysis_type in self.analysis_types.keys():
            # only do selected analysis types
            if not self.analysis_types[analysis_type]: continue
            # load analyzed data
            self.load_analysis(analysis_type)
            # analyze the data
            if self.analysis_options['analyze data']: analyze_data(self, dataframe, analysis_type)
            # Plot analysis
            if self.analysis_types['traversals']: plot_traversals(self)
            if self.analysis_types['exploration']: plot_exploration(self)
            if self.analysis_types['escape paths']: plot_escape_paths(self)
            if self.analysis_types['speed traces']: plot_speed_traces(self)
            if self.analysis_types['prediction']: plot_prediction(self)
            if self.analysis_types['edginess']: plot_edginess(self)
            if self.analysis_types['efficiency']: plot_efficiency(self)
            if self.analysis_types['metrics']: plot_metrics_by_strategy(self)

    def load_analysis(self, analysis_type):
        '''     Load analysis that's been done      '''
        # find file path to analysis dictionary
        self.save_file = os.path.join(self.folders['save_folder'], 'analysis_data_' + analysis_type)
        # get edginess variables for prediction
        if analysis_type == 'prediction': analysis_type = 'edginess'
        # load and initialize dictionary
        self.analysis = {}
        for experiment in flatten(self.experiments):
            save_folder = os.path.join( self.dlc_settings['clips_folder'], experiment, analysis_type)
            try:
                with open(save_folder, 'rb') as dill_file:
                    self.analysis[experiment] = pickle.load(dill_file)
            except: self.analysis[experiment] = {}


    def select_sessions(self, dataframe):
        '''     Get a list of the user-selected sessions for analysis      '''
        # initialize selected sessions list
        self.selected_sessions = []
        self.experiments = []
        # loop over all sessions
        for session_name in dataframe.db.index[::-1]:
            # Get the session
            metadata = dataframe.db.loc[session_name].Metadata
            # Check if this is one of the sessions we should be processing
            if metadata['experiment'] in self.flatten(self.analysis_experiments['experiments']):
                self.selected_sessions.append(session_name)
                # add experiment to a list of experiments
                if metadata['experiment'] not in self.experiments: self.experiments.append(metadata['experiment'])

        # set up experiments and conditions to analyze
        self.experiments = self.analysis_experiments['experiments']
        self.conditions = self.analysis_experiments['conditions']
        self.labels = self.analysis_experiments['labels']

    def make_analysis_directory(self):
        '''     Make a folder to store summary analysis     '''
        self.summary_plots_folder = os.path.join(self.folders['save_folder'], 'Summary Plots')
        if not os.path.isdir(self.summary_plots_folder): os.makedirs(self.summary_plots_folder)

    def flatten(self, iterable):
        '''     flatten a nested array      '''
        it = iter(iterable)
        for e in it:
            if isinstance(e, (list, tuple)):
                for f in self.flatten(e): yield f
            else:
                yield e