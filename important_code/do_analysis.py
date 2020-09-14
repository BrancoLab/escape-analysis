import os
import dill as pickle
import numpy as np
import skfmm
import itertools
from scipy.ndimage import gaussian_filter1d
import scipy.signal
from helper_code.analysis_funcs import flatten
from helper_code.registration_funcs import get_arena_details, model_arena


class analyze_data():
    def __init__(self, analysis, dataframe, analysis_type):
        '''     initialize quantities for analyzing data         '''
        # list of quantities to analyze
        if analysis_type == 'exploration': self.quantities_to_analyze =['exploration', 'start time', 'obstacle exploration']
        if analysis_type == 'traversals': self.quantities_to_analyze =['back traversal', 'front traversal']
        if analysis_type == 'escape paths': self.quantities_to_analyze = ['speed', 'start time', 'end time', 'path', 'edginess', 'RT', 'x edge', 'start angle', 'prev homings', 'prev movements']
        if analysis_type == 'prediction': self.quantities_to_analyze = ['start time', 'end time', 'path', 'edginess','RT', 'prev homings', 'prev anti-homings', 'prev movements',
                                                                       'movement', 'x edge', 'start angle', 'x escape', 'optimal path length', 'path', 'prev homings', 'prev movements',
                                                                        'optimal RT path length','full path length','RT path length', 'speed',
                                                                        'time exploring (pre)', 'time exploring (post)', 'distance exploring (pre)', 'distance exploring (post)', 'time exploring obstacle (pre)',
                                                                         'time exploring obstacle (post)', 'time exploring far (pre)', 'time exploring far (post)', 'time exploring L edge (pre)', 'time exploring R edge (pre)']
        if analysis_type == 'edginess': self.quantities_to_analyze = ['start time', 'end time', 'path', 'edginess','RT', 'prev homings', 'prev movements', 'movement', 'x edge', 'start angle']
        if analysis_type == 'efficiency': self.quantities_to_analyze = \
            ['start time', 'end time','optimal path length', 'path', 'prev homings', 'prev movements','prev anti-homings',
            'edginess','optimal RT path length','full path length','RT path length', 'RT', 'x edge', 'speed', 'back traversal', 'front traversal',
            'time exploring (pre)', 'time exploring (post)', 'distance exploring (pre)', 'distance exploring (post)', 'time exploring obstacle (pre)',
             'time exploring obstacle (post)', 'time exploring far (pre)', 'time exploring far (post)', 'time exploring L edge (pre)', 'time exploring R edge (pre)']
        if analysis_type == 'metrics': self.quantities_to_analyze = ['start time', 'end time', 'speed', 'RT','optimal path length', 'path', 'prev homings', 'prev movements',
                                                                     'edginess', 'optimal RT path length','full path length','RT path length', 'x edge', 'movement', 'time exploring (pre)', 'time exploring (post)', 'distance exploring (pre)', 'distance exploring (post)', 'time exploring obstacle (pre)',
             'time exploring obstacle (post)', 'time exploring far (pre)', 'time exploring far (post)', 'time exploring L edge (pre)', 'time exploring R edge (pre)'] #, 'prev anti-homings']
        if analysis_type == 'speed traces': self.quantities_to_analyze = ['speed', 'geo speed', 'start time', 'end time','RT']
        # conditions to analyze them in
        self.conditions = ['obstacle', 'no obstacle', 'probe']
        # run analysis
        self.main(analysis, dataframe, analysis_type)

    def main(self, analysis, dataframe, analysis_type):
        '''    analyze data from each selected session      '''
        # loop over all selected sessions
        for i, session_name in enumerate(analysis.selected_sessions):
            # get the metadata
            self.get_metadata(session_name, analysis, dataframe, analysis_type)
            # format the analysis dictionary
            self.format_analysis_dict(i, analysis)
            # get indices when the stimulus is or was on
            self.get_stim_idx()
            # loop over all videos
            for vid_num in range(len(self.stims_all)):
                self.vid_num = vid_num
                # load the saved coordinates
                coordinates = self.load_coordinates(dataframe, session_name)
                # get control frames if ananlyzing non-escapes
                if analysis.analysis_options['control']: self.get_control_stim_frames(coordinates)
                # get the time periods when the wall is up or down
                self.vid_duration.append(len(coordinates['speed']))
                self.get_obstacle_epochs(self.vid_duration[-1])
                # loop over these epochs
                for i, epoch in enumerate([self.wall_up_epoch, self.wall_down_epoch, self.probe_epoch]):
                    # if empty epoch, don't analyze
                    if not epoch: continue
                    # otherwise, initialize variables
                    else: self.epoch = epoch; self.condition = self.conditions[i]; self.start_point = self.start_points[i]
                    # Analyze exploration
                    if analysis_type == 'exploration': self.analyze_exploration(coordinates, analysis, self.stims_all)
                    # Analyze quantities on a per-trial basis
                    else: self.analyze_escapes(coordinates, analysis)
                    # Get all traversals across the arena
                    if analysis_type == 'traversals' or 'back traversal' in self.quantities_to_analyze: self.analyze_traversals(coordinates, analysis, self.vid_duration[self.vid_num])
                # get previous vid's homings to next vid
                if 'prev homings' in self.quantities_to_analyze:
                    self.trials_condition_prev_vid = {}
                    for condition in self.conditions:
                        self.trials_condition_prev_vid[condition] = len(analysis.analysis[self.experiment][condition]['prev homings'][self.mouse])
        # save analysis
        self.save_analysis(analysis, analysis_type)


    def analyze_escapes(self, coordinates, analysis):
        '''       analyze speed, geodesic speed, trial time, reaction time        '''
        # initialize data
        distance_from_shelter, position, angular_speed, speed, distance_map, shelter_angle, body_angle = self.initialize_escape_data(coordinates, analysis)
        # loop over each trial
        for stim in self.stims_all[self.vid_num]:
            # make sure it's the right epoch
            if not stim in self.epoch: continue
            # make sure it's not a mistaken trial instead of clicking finish session
            if len(position[0]) - stim < 600: continue
            # get peri-stimulus indices
            threat_idx = np.arange(stim - self.fps * 10, stim + self.fps * 18).astype(int)
            stim_idx = np.arange(stim, stim + self.fps * 18).astype(int)
            # get the idx when at shelter, and trim threat idx if applicable, and label as completed escape or nah
            arrived_at_shelter = np.where(distance_from_shelter[stim_idx] < 60)[0]
            # get the geodesic speed
            geo_speed = self.analyze_geo_speed(threat_idx, position, distance_map, analysis)   
            # get the reaction time
            RT_speed = 15
            RT, subgoal_speed_trace = self.get_reaction_time(geo_speed, RT_speed)
            # change the stim time for multiple videos
            stim = stim + self.vid_duration[self.vid_num]
            # get exploration stats
            if 'time exploring (pre)' in self.quantities_to_analyze: self.analyze_exploration_metrics(stim, position, analysis, distance_from_shelter, speed)
            # get the escape path
            if 'path' in self.quantities_to_analyze: analysis.analysis[self.experiment][self.condition]['path'][self.mouse].append([position[0][stim_idx], position[1][stim_idx]])
            # get the start time
            if 'start time' in self.quantities_to_analyze: analysis.analysis[self.experiment][self.condition]['start time'][self.mouse].append(stim / self.fps / 60)
            # get the speed
            if 'speed' in self.quantities_to_analyze: analysis.analysis[self.experiment][self.condition]['speed'][self.mouse].append(list(speed[threat_idx]))
            # get the start angle
            if 'start angle' in self.quantities_to_analyze: analysis.analysis[self.experiment][self.condition]['start angle'][self.mouse].append(shelter_angle[stim_idx[0]])
            # get the prev homings
            if 'prev homings' in self.quantities_to_analyze: self.analyze_prev_homings(coordinates, stim - self.vid_duration[self.vid_num], analysis) #and not 'food' in self.experiment
            # get the prev homings
            if 'prev anti-homings' in self.quantities_to_analyze: self.analyze_prev_anti_homings(coordinates, stim - self.vid_duration[self.vid_num], analysis) #and not 'food' in self.experiment

            # results for successful escapes: to shelter, fast enough, no no-shelter xp, no starting in shelter xp
            if arrived_at_shelter.size and RT.size and not ('no shelter' in self.experiment and not 'down' in self.experiment) and arrived_at_shelter[0] > 10:
                if 'end time' in self.quantities_to_analyze: analysis.analysis[self.experiment][self.condition]['end time'][self.mouse].append(arrived_at_shelter[0])
                if 'RT' in self.quantities_to_analyze: analysis.analysis[self.experiment][self.condition]['RT'][self.mouse].append(RT[0] / 30)
                if 'optimal path length' in self.quantities_to_analyze: self.analyze_optimal_path_lengths(distance_map, position, stim - self.vid_duration[self.vid_num], arrived_at_shelter, speed, RT, analysis)
                if 'edginess' in self.quantities_to_analyze: self.analyze_edginess(RT_speed, angular_speed, arrived_at_shelter, position, subgoal_speed_trace, stim_idx, threat_idx, analysis, body_angle)
                if 'movement' in self.quantities_to_analyze: self.analyze_movements(coordinates, stim - self.vid_duration[self.vid_num], position, distance_map, distance_from_shelter, analysis)
            elif 'x edge' in self.quantities_to_analyze:
                self.analyze_x_edge(position, stim_idx, analysis)
                self.fill_in_nans(analysis)
            else: self.fill_in_nans(analysis)

        if not self.stims_all[self.vid_num] and 'prev homings' in self.quantities_to_analyze and not analysis.analysis[self.experiment]['no obstacle']['prev homings'][self.mouse]:
            self.condition = 'no obstacle' # fix double vid session...
            stim = self.vid_duration[self.vid_num+1] - 301
            # get the prev homings
            if 'prev homings' in self.quantities_to_analyze: self.analyze_prev_homings(coordinates, stim, analysis)  # and not 'food' in self.experiment
            # get the prev homings
            if 'prev anti-homings' in self.quantities_to_analyze: self.analyze_prev_anti_homings(coordinates, stim, analysis)  # and not 'food' in self.experiment



    def analyze_traversals(self, coordinates, analysis, vid_duration):
        '''     analyze traversals across the arena         '''
        # take the position data
        x_location = coordinates['center_location'][0][self.epoch] * self.scaling_factor
        y_location = coordinates['center_location'][1][self.epoch] * self.scaling_factor

        # x_location = coordinates['center_location'][0] * self.scaling_factor
        # y_location = coordinates['center_location'][1] * self.scaling_factor


        void_compare = False

        back = 25
        front = 75
        middle = 50
        back_idx = y_location < back;
        front_idx = y_location > front
        back_half_idx = y_location < (middle - 5 - 5*void_compare); # 5 exploration // 10 if void comparison
        front_half_idx = y_location > (middle + 5 + 5*void_compare)

        #  initialize lists
        center_back_idx = back_half_idx * ~back_idx; center_front_idx = front_half_idx * ~front_idx
        traversals_from_back = []; traversals_from_front = []
        pre_traversals_from_back = []; pre_traversals_from_front = []
        traversals_time_back = []; traversals_time_front = []
        escapes_from_back = []; escape_times = []
        # loop over front and back
        for j, location_idx in enumerate([center_back_idx, center_front_idx]):
            idx = 0;
            group_length = 0
            for k, g in itertools.groupby(location_idx):

                idx += group_length
                frames_grouped = list(g)
                group_length = len(frames_grouped)

                # must not start the video
                if not idx: continue
                # must be in the right section
                if not k: continue
                # must not be too short
                if group_length < 5: continue
                # must start at front or back
                if abs(y_location[idx] - 50) < 15: continue
                # must end in middle
                if abs(y_location[idx + group_length - 1] - 50) > 20: continue # was 10  then 15
                # must not be along the side of arena?
                if np.max(abs(x_location[idx:idx + group_length - 1] - 50)) > (35 + 5*void_compare): continue # 35 exploration // 40 if void comparison
                # a stimulus-evoked escape
                if (idx + self.start_point in self.stim_idx[self.vid_num] or idx + self.start_point + group_length - 1 in self.stim_idx[self.vid_num]):
                    if j == 0:
                        escapes_from_back.append((x_location[idx:idx + group_length - 1] / self.scaling_factor,
                                                  y_location[idx:idx + group_length - 1] / self.scaling_factor))
                        escape_times.append(idx+vid_duration)
                        continue
                # for back traversals
                if j == 0:
                    traversals_from_back.append((x_location[idx:idx + group_length - 1] / self.scaling_factor,
                                                 y_location[idx:idx + group_length - 1] / self.scaling_factor))
                    pre_traversals_from_back.append((x_location[idx - 30:idx] / self.scaling_factor,
                                                      y_location[idx - 30:idx] / self.scaling_factor))
                    traversals_time_back.append(idx+vid_duration)
                # for front traversals
                elif j == 1:
                    traversals_from_front.append((x_location[idx:idx + group_length - 1] / self.scaling_factor,
                                                  y_location[idx:idx + group_length - 1] / self.scaling_factor))
                    pre_traversals_from_front.append((x_location[idx - 30:idx] / self.scaling_factor,
                                                      y_location[idx - 30:idx] / self.scaling_factor))
                    traversals_time_front.append(idx+vid_duration)

        # compute the edginess of each traversal
        edginess = {}; duration = {}
        for traversal_type, traversals in zip(['back traversal', 'front traversal', 'escape'],[traversals_from_back, traversals_from_front,escapes_from_back]):
            edginess[traversal_type] = []; duration[traversal_type] = []
            for traversal in traversals:
                trav_edginess = abs(self.compute_edginess(traversal[0]*self.scaling_factor, traversal[1]*self.scaling_factor, traversal_type))
                edginess[traversal_type].append(trav_edginess)
                duration[traversal_type].append(len(traversal[0])/30)

        if self.vid_num and analysis.analysis[self.experiment][self.condition]['back traversal'][self.mouse]:
            # if previous video, add that videos homings
            for idx, trav_var in enumerate([traversals_from_back, traversals_time_back, edginess['back traversal'], duration['back traversal'],
                                            len(self.epoch) / 30 / 60, escapes_from_back, escape_times, edginess['escape'], duration['escape'], len(self.stim_idx[self.vid_num]), pre_traversals_from_back]):
                trav_var_prev = analysis.analysis[self.experiment][self.condition]['back traversal'][self.mouse][idx]
                trav_var = trav_var_prev + trav_var
                analysis.analysis[self.experiment][self.condition]['back traversal'][self.mouse][idx] = trav_var

            for idx, trav_var in enumerate([traversals_from_front, traversals_time_front, edginess['front traversal'], duration['front traversal'], len(self.epoch) / 30 / 60, [],[],[],[],[], pre_traversals_from_front]):
                trav_var_prev = analysis.analysis[self.experiment][self.condition]['front traversal'][self.mouse][idx]
                trav_var = trav_var_prev + trav_var
                analysis.analysis[self.experiment][self.condition]['front traversal'][self.mouse][idx] = trav_var


        else:
            # add to the analysis dictionary
            analysis.analysis[self.experiment][self.condition]['back traversal'][self.mouse] = [traversals_from_back, traversals_time_back,
                                                                                                edginess['back traversal'], duration['back traversal'],
                                                                                                len(self.epoch) / 30 / 60, escapes_from_back, escape_times, # np.sum(y_location < 25)
                                                                                                edginess['escape'], duration['escape'], len(self.stim_idx[self.vid_num]), pre_traversals_from_back]

            analysis.analysis[self.experiment][self.condition]['front traversal'][self.mouse] = [traversals_from_front, traversals_time_front,
                                                                                                 edginess['front traversal'], duration['front traversal'],
                                                                                                 len(self.epoch) / 30 / 60, #np.sum(y_location > 75),
                                                                                                 [],[],[],[],[], pre_traversals_from_front]

    def compute_edginess(self, x_pos, y_pos, traversal_type):
        '''     compute edginess        '''
        if 'front' in traversal_type:
            y_pos = 100 - y_pos

        y_eval_point = 40 - 5* ('void' in self.experiment)
        y_wall_point = 45 - 5* ('void' in self.experiment)

        x_pos_start = x_pos[0]
        y_pos_start = y_pos[0]
        # do line from starting position to shelter
        y_pos_shelter = self.shelter_location[1] / 10
        x_pos_shelter = self.shelter_location[0] / 10
        slope = (y_pos_shelter - y_pos_start) / (x_pos_shelter - x_pos_start)
        intercept = y_pos_start - x_pos_start * slope
        distance_to_line = abs(y_pos - slope * x_pos - intercept) / np.sqrt((-slope) ** 2 + (1) ** 2)
        # # get index at center point (wall location)
        mouse_at_eval_point = np.argmin(abs(y_pos - y_eval_point))
        homing_vector_at_center = (y_eval_point - intercept) / slope
        # get offset from homing vector
        linear_offset = distance_to_line[mouse_at_eval_point]
        # get line to the closest edge
        mouse_at_wall = np.argmin(abs(y_pos - y_wall_point))
        y_edge = 50
        if x_pos[mouse_at_wall] > 50:
            x_edge = 75
        else:
            x_edge = 25
        # do line from starting position to edge position
        slope = (y_edge - y_pos_start) / (x_edge - x_pos_start)
        intercept = y_pos_start - x_pos_start * slope
        distance_to_line = abs(y_pos - slope * x_pos - intercept) / np.sqrt((-slope) ** 2 + (1) ** 2)
        edge_offset = np.mean(distance_to_line[mouse_at_eval_point])
        # compute the max possible deviation
        edge_vector_at_center = (y_eval_point - intercept) / slope
        line_to_edge_offset = abs(homing_vector_at_center - edge_vector_at_center)  # + 5
        # get edginess
        # edginess = np.min((1, (linear_offset - edge_offset + line_to_edge_offset) / (2 * line_to_edge_offset)))
        edginess = (linear_offset - edge_offset + line_to_edge_offset) / (2 * line_to_edge_offset)  #* np.sign(x_edge-50)
        return edginess

    def analyze_exploration(self, coordinates, analysis, stims):
        '''       analyze explorations         '''
        # Histogram of positions
        position = coordinates['center_location']
        height, width = analysis.analysis[self.experiment][self.condition]['shape'][0], analysis.analysis[self.experiment][self.condition]['shape'][1]
        H, x_bins, y_bins = \
            np.histogram2d(position[0, self.epoch], position[1, self.epoch], [np.arange(0, width + 1), np.arange(0, height + 1)], normed=True)
        H = H.T
        # put into dictionary
        analysis.analysis[self.experiment][self.condition]['exploration'][self.mouse] = H

        '''
        get exploration traces near the obstacle
        '''
        # obstacle
        # end_time = stims[0][0]
        # position = position[0][:end_time], position[1][:end_time]
        #
        # # get idx when mouse is near obstacle
        # near_obstacle_idx = np.where( (position[0]*self.scaling_factor < (50 + x_width)) * (position[0]*self.scaling_factor > (50 - x_width)) * \
        #                     (position[1]*self.scaling_factor < (50 + y_width - 1.5)) * (position[1]*self.scaling_factor > (50 - y_width - 1.5)) )[0]
        #
        # # get coresponding coods
        # near_obstacle_coords = position[0][near_obstacle_idx], position[1][near_obstacle_idx]
        #
        # # put in dict
        # analysis.analysis[self.experiment][self.condition]['obstacle exploration'][self.mouse] = near_obstacle_idx, near_obstacle_coords

        '''
        get exploration traces post removal
        '''
        # obstacle removed
        post_OR_idx = [s in self.epoch for s in stims[0]]
        post_OR_stims = np.array(stims[0])[post_OR_idx]

        if len(post_OR_stims) > 2: end_time = post_OR_stims[2]
        else: end_time = stims[0][-1]

        # self.condition
        post_OR_idx = np.arange(self.epoch[0],end_time)

        # get coresponding coods
        post_OR_coords = position[0][post_OR_idx], position[1][post_OR_idx]


        analysis.analysis[self.experiment][self.condition]['obstacle exploration'][self.mouse] = post_OR_idx, post_OR_coords




    def analyze_exploration_metrics(self, stim, position, analysis, distance_from_shelter, speed):
        '''     get the amount of time spent in the central region      '''

        # if stim in self.probe_epoch:
        #     start_frame = 0
        # else:
        #     start_frame = self.epoch[0]
        # Get epochs for exploration analysis
        exploring_epoch_pre = self.wall_up_epoch
        start_frame = self.epoch[0]
        print(start_frame)

        time_in_center_post = np.sum((position[0][start_frame:stim] * self.scaling_factor > 20) * (position[0][start_frame:stim] * self.scaling_factor < 80) * \
                                  (position[1][start_frame:stim] * self.scaling_factor > (45- 5 * ('void' in self.experiment))) * (position[1][start_frame:stim] * self.scaling_factor < (55 + 5* ('void' in self.experiment)) )) / 30

        time_in_center_pre = np.sum((position[0][exploring_epoch_pre] * self.scaling_factor > 20) * (position[0][exploring_epoch_pre] * self.scaling_factor < 80) * \
                                  (position[1][exploring_epoch_pre] * self.scaling_factor > (45- 5* ('void' in self.experiment))) * (position[1][exploring_epoch_pre] * self.scaling_factor < (55 + 5* ('void' in self.experiment)) )) / 30


        time_in_left_edge = np.sum((position[0][exploring_epoch_pre] * self.scaling_factor > 20) * (position[0][exploring_epoch_pre] * self.scaling_factor < 30) * \
                                  (position[1][exploring_epoch_pre] * self.scaling_factor > (45- 5* ('void' in self.experiment))) * (position[1][exploring_epoch_pre] * self.scaling_factor < (55 + 5* ('void' in self.experiment)) )) / 30


        time_in_right_edge = np.sum((position[0][exploring_epoch_pre] * self.scaling_factor > 70) * (position[0][exploring_epoch_pre] * self.scaling_factor < 80) * \
                                  (position[1][exploring_epoch_pre] * self.scaling_factor > (45- 5* ('void' in self.experiment))) * (position[1][exploring_epoch_pre] * self.scaling_factor < (55 + 5* ('void' in self.experiment)) )) / 30


        time_in_far_side_pre = np.sum( position[1][exploring_epoch_pre] * self.scaling_factor < 50 ) / 30
        time_in_far_side_post = np.sum(position[1][start_frame:stim] * self.scaling_factor < 50) / 30

        exploring_idx_pre = distance_from_shelter[exploring_epoch_pre] * self.scaling_factor > 10
        time_exploring_pre = np.sum(exploring_idx_pre) / 30

        exploring_idx_post = distance_from_shelter[start_frame:stim] * self.scaling_factor > 10
        time_exploring_post = np.sum(exploring_idx_post) / 30

        distance_explored_pre = np.sum(speed[exploring_epoch_pre][exploring_idx_pre]*self.scaling_factor) / 1000
        distance_explored_post = np.sum(speed[start_frame:stim][exploring_idx_post] * self.scaling_factor) / 1000

        analysis.analysis[self.experiment][self.condition]['time exploring (pre)'][self.mouse].append(time_exploring_pre)
        analysis.analysis[self.experiment][self.condition]['time exploring (post)'][self.mouse].append(time_exploring_post)
        analysis.analysis[self.experiment][self.condition]['distance exploring (pre)'][self.mouse].append(distance_explored_pre)
        analysis.analysis[self.experiment][self.condition]['distance exploring (post)'][self.mouse].append(distance_explored_post)
        analysis.analysis[self.experiment][self.condition]['time exploring obstacle (pre)'][self.mouse].append(time_in_center_pre)
        analysis.analysis[self.experiment][self.condition]['time exploring obstacle (post)'][self.mouse].append(time_in_center_post)
        analysis.analysis[self.experiment][self.condition]['time exploring far (pre)'][self.mouse].append(time_in_far_side_pre)
        analysis.analysis[self.experiment][self.condition]['time exploring far (post)'][self.mouse].append(time_in_far_side_post)
        analysis.analysis[self.experiment][self.condition]['time exploring L edge (pre)'][self.mouse].append(time_in_left_edge)
        analysis.analysis[self.experiment][self.condition]['time exploring R edge (pre)'][self.mouse].append(time_in_right_edge)

        if self.vid_num:
            # if previous video, add that videos homings
            for label, explore_var in zip(['time exploring (pre)', 'time exploring (post)', 'distance exploring (pre)', 'distance exploring (post)', 'time exploring obstacle (pre)',
                                           'time exploring obstacle (post)', 'time exploring far (pre)', 'time exploring far (post)', 'time exploring L edge (pre)', 'time exploring R edge (pre)'], \
                                          [time_exploring_pre, time_exploring_post, distance_explored_pre, distance_explored_post, \
                                           time_in_center_pre, time_in_center_post, time_in_far_side_pre, time_in_far_side_post, time_in_left_edge, time_in_right_edge]):

                explore_prev = analysis.analysis[self.experiment][self.condition][label][self.mouse][self.trials_condition_prev_vid[self.condition] - 1]
                explore_var = explore_prev + explore_var
                analysis.analysis[self.experiment][self.condition][label][self.mouse][-1] = explore_var

    def analyze_geo_speed(self, threat_idx, position, distance_map, analysis):
        '''     analyze the geodesic speed      '''
        threat_idx_mod = np.concatenate((np.ones(1, int) * threat_idx[0] - 1, threat_idx))
        threat_position = position[0][threat_idx_mod].astype(int), position[1][threat_idx_mod].astype(int)
        geo_location = distance_map[self.condition][threat_position[1], threat_position[0]]
        geo_speed = np.diff(geo_location)
        if 'geo speed' in self.quantities_to_analyze: analysis.analysis[self.experiment][self.condition]['geo speed'][self.mouse].append(list(geo_speed))
        return geo_speed

    def get_reaction_time(self, geo_speed, RT_speed):
        '''     get the reaction time       '''
        # get the reaction time
        trial_subgoal_speed = [s * self.scaling_factor * 30 for s in geo_speed]
        subgoal_speed_trace = gaussian_filter1d(trial_subgoal_speed, 4)  # 2
        RT = np.where((-subgoal_speed_trace[10 * 30:] > RT_speed))[0]
        return RT, subgoal_speed_trace

    def analyze_optimal_path_lengths(self,distance_map, position, stim, arrived_at_shelter, speed, RT, analysis):
        '''     get the optimal and actual path lengths     '''
        # get the optimal path length
        optimal_path_length = distance_map[self.condition][int(position[1][stim]), int(position[0][stim])]
        optimal_path_length_RT = distance_map[self.condition][int(position[1][stim + RT[0]]), int(position[0][stim + RT[0]])]
        analysis.analysis[self.experiment][self.condition]['optimal path length'][self.mouse].append(optimal_path_length)
        analysis.analysis[self.experiment][self.condition]['optimal RT path length'][self.mouse].append(optimal_path_length_RT)
        # get the actual path length
        full_path_length = np.sum(speed[stim:stim + arrived_at_shelter[0]])
        RT_path_length = np.sum(speed[stim + RT[0]:stim + arrived_at_shelter[0]])
        analysis.analysis[self.experiment][self.condition]['full path length'][self.mouse].append(full_path_length + 60)
        analysis.analysis[self.experiment][self.condition]['RT path length'][self.mouse].append(RT_path_length + 60)


    def fill_in_nans(self, analysis):
        '''     fill in non-escape data with nans       '''
        escape_dependent_fields = ['end time', 'RT', 'edginess', 'optimal path length', 'optimal RT path length', 'full path length', 'RT path length', 'x escape', 'movement']
        for field in escape_dependent_fields:
            if field in self.quantities_to_analyze:
                if field == 'movement':
                    analysis.analysis[self.experiment][self.condition][field][self.mouse].append([(np.nan, np.nan), np.nan, np.nan])
                else:
                    analysis.analysis[self.experiment][self.condition][field][self.mouse].append(np.nan)

    def analyze_prev_homings(self, coordinates, stim, analysis):
        '''     analyze previous homings!       '''
        # get the x values at center
        x_SH = [];
        y_SH = [];
        x_SH_movements = [];
        thru_center = [];
        SH_time = []
        how_long_to_shelter = []
        stim_evoked = []
        x_start = []
        y_start = []
        angle_start = []
        turn_angle = []
        center_y = 45 - 2.5* ('void' in self.experiment)

        combo_paths = 0

        # if not 'U shaped' in self.experiment:
        in_shelter = np.where(coordinates['distance_from_shelter'][:stim] < 60)[0]
        # start_idx = np.where(coordinates['start_index'][:stim])[0]
        # end_idx = coordinates['start_index'][start_idx]
        start_idx = coordinates['end_index'][np.where(coordinates['end_index'][:stim])[0]].astype(int)
        end_idx = coordinates['start_index'][np.where(coordinates['start_index'][:stim])[0]].astype(int)


        for j, (s, e) in enumerate(zip(start_idx, end_idx)):
            # get current path's data
            homing_idx = np.arange(s, e).astype(int)
            path = coordinates['center_location'][0][homing_idx] * self.scaling_factor, \
                   coordinates['center_location'][1][homing_idx] * self.scaling_factor

            # print(path[0][0], path[1][0])

            # if self.mouse == 'CA7010' and path[1][0] < 40:
            #     print(path[1][0])
            #     print(path[0][0])
            #     print(path[1][-1])
            #     print(path[0][-1])
            #     print('hi')
            if self.mouse == 'CA8360' and self.experiment=='Circle food' and not path[1].size: continue

            # mean_speed = np.mean(coordinates['speed'][homing_idx]) * self.scaling_factor * 30 # temp just to check!!
            # exclude if starts too far down or is already used

            if path[1][0] > (35 - 2.5* ('void' in self.experiment)) or combo_paths or s < 150: # or mean_speed < 15: #was 40 TEMP -- then was 35!!!
                if combo_paths>0: combo_paths -= 1
            else:
               # if next path is continuation of this one, see if it qualifies
                continuations_to_test = 3
                for c in range(1, continuations_to_test+1):
                    # if next path is a continuation of this one
                    if (j+c) < len(start_idx) and start_idx[j + c] == end_idx[j+c-1] + 1:
                        combo_paths += 1
                    else:
                        break

                if combo_paths:
                    # combine with next path's data
                    homing_idx = np.arange(s, end_idx[j+combo_paths]).astype(int)
                    path = coordinates['center_location'][0][homing_idx] * self.scaling_factor, \
                           coordinates['center_location'][1][homing_idx] * self.scaling_factor

                #    test eligibility
                #    or never approaches y center      or never is above wall on top part of arena #27.5? TEMP -- WAS 24.5
                if np.sum(abs(path[1] - (50- 10* ('void' in self.experiment)) ) < 5) and np.sum((abs(path[0] - 50) < (24.5+5* ('void' in self.experiment))) * (path[1] < 50)):
                        #                45 TEMP CHANGE BACK TO 45
                    # get x-position along central wall
                    center_y_idx = np.argmin(abs(path[1] - center_y)  )
                    x_SH.append(path[0][center_y_idx])

                    # get y-position closest to center
                    edge_idx = np.argmin(abs(path[1] - 50))
                    y_SH.append(path[1][edge_idx])

                    # get time of homing onset
                    SH_time.append(s / 30 / 60 + self.vid_duration[self.vid_num]/30/60)

                    # get time until next in shelter
                    in_shelter_next = in_shelter[in_shelter > e]
                    if in_shelter_next.size: how_long_to_shelter.append( (in_shelter_next[0] - e) / self.fps)
                    else: how_long_to_shelter.append(60)

                    # get stimulus evoked?
                    stim_evoked.append(s in self.stim_idx[0])

                    # get the initial conditions
                    x_start.append(path[0][0])
                    y_start.append(path[1][0])

                    # get the initial angle
                    start_angle = coordinates['body_angle'][s]
                    end_angle = coordinates['body_angle'][int(e)]
                    angle_start.append(start_angle)

                    # do the turn angle
                    turn = abs(end_angle - start_angle)
                    turn_direction = np.median(np.diff(coordinates['body_angle'])[s:int(e)])  # positive is left, negative is right
                    if abs(turn_direction) > 180: turn_direction *= -1  # if pass the 360, shift sign
                    # correct for crossing zero
                    if turn > 180: turn = 360 - turn
                    # neagtive is left, positive is right
                    turn_angle.append(turn * np.sign(-turn_direction))
        # print(x_SH)

        #
        # else: # for U-shaped experiments define homings differently
        #     position = coordinates['center_location'][0][:stim] * self.scaling_factor, coordinates['center_location'][1][:stim] * self.scaling_factor
        #     duration_of_homing = 30 # in frames
        #     # find instance where the mouse goes from top center past the U within 1 second
        #     in_top_left_center = (position[1] < 30) * (position[0] < 60)
        #     in_top_right_center = (position[1] < 30) * (position[0] > 40)
        #     in_cup = (position[1] > 30) * (abs(position[0] - 50) < 25)
        #     past_cup_left = (position[1] > 20) * (position[0] < 25) * (position[0] > 20)
        #     past_cup_right = (position[1] > 20) * (position[0] > 75) * (position[0] < 80)
        #
        #     # filter in_cup so it includes all one second in future
        #     was_in_cup = scipy.signal.convolve(in_cup, np.ones(duration_of_homing),mode='full', method = 'direct')[:-(duration_of_homing-1)].astype(bool)
        #
        #     # get homings
        #     homing_left = past_cup_left[duration_of_homing:] * ~was_in_cup[duration_of_homing:] * in_top_right_center[:-duration_of_homing]
        #     if np.sum(homing_left): x_SH.append(25)
        #     else: x_SH.append(50)
        #     homing_right = past_cup_right[duration_of_homing:] * ~was_in_cup[duration_of_homing:] * in_top_left_center[:-duration_of_homing]
        #     if np.sum(homing_right): x_SH.append(75)
        #     else: x_SH.append(50)
        #
        #     print(x_SH)


        analysis.analysis[self.experiment][self.condition]['prev homings'][self.mouse].append([x_SH, y_SH, how_long_to_shelter, SH_time, stim_evoked])
        analysis.analysis[self.experiment][self.condition]['prev movements'][self.mouse].append([x_start, y_start, angle_start, turn_angle, x_SH])

        if self.vid_num:
            # if previous video, add that videos homings
            for v, (homing_var, movement_var) in enumerate(zip([x_SH, y_SH, how_long_to_shelter, SH_time, stim_evoked], [x_start, y_start, angle_start, turn_angle, x_SH])):

                homing_var_prev = analysis.analysis[self.experiment][self.condition]['prev homings'][self.mouse][self.trials_condition_prev_vid[self.condition] - 1][v]
                homing_var = homing_var_prev + homing_var
                analysis.analysis[self.experiment][self.condition]['prev homings'][self.mouse][-1][v] = homing_var

                movement_var_prev = analysis.analysis[self.experiment][self.condition]['prev movements'][self.mouse][self.trials_condition_prev_vid[self.condition] - 1][v]
                movement_var = movement_var_prev + movement_var
                analysis.analysis[self.experiment][self.condition]['prev movements'][self.mouse][-1][v] = movement_var

    def analyze_prev_anti_homings(self, coordinates, stim, analysis):
        '''     analyze previous homings!       '''
        # get the x values at center
        x_SH = [];
        y_SH = [];
        x_SH_movements = [];
        thru_center = [];
        SH_time = []
        how_long_to_shelter = []
        stim_evoked = []
        x_start = []
        y_start = []
        angle_start = []
        turn_angle = []
        center_y = 55 + 2.5 * ('void' in self.experiment)

        combo_paths = 0

        in_shelter = np.where(coordinates['anti_distance_from_shelter'][:stim] < 60)[0]
        start_idx = coordinates['anti_end_index'][np.where(coordinates['anti_end_index'][:stim])[0]].astype(int)
        end_idx = coordinates['anti_start_index'][np.where(coordinates['anti_start_index'][:stim])[0]].astype(int)

        for j, (s, e) in enumerate(zip(start_idx, end_idx)):
            # get current path's data
            homing_idx = np.arange(s, e).astype(int)
            path = coordinates['center_location'][0][homing_idx] * self.scaling_factor, \
                   coordinates['center_location'][1][homing_idx] * self.scaling_factor

            # mean_speed = np.mean(coordinates['speed'][homing_idx]) * self.scaling_factor * 30
            # exclude if starts too far down or is already used
            # if path[1][0] > 35 or combo_paths or s < 150:  # was 40 TEMP -- then was 35!!!
            #     if combo_paths > 0: combo_paths -= 1
            if path[1][0] < (100 - 35 + 2.5* ('void' in self.experiment)) or combo_paths or s < 150: # or mean_speed < 15:  # was 40 TEMP -- then was 35!!!
                if combo_paths > 0: combo_paths -= 1
            else:
                # if next path is continuation of this one, see if it qualifies
                continuations_to_test = 3
                for c in range(1, continuations_to_test + 1):
                    # if next path is a continuation of this one
                    if (j + c) < len(start_idx) and start_idx[j + c] == end_idx[j + c - 1] + 1:
                        combo_paths += 1
                    else:
                        break

                if combo_paths:
                    # combine with next path's data
                    homing_idx = np.arange(s, end_idx[j + combo_paths]).astype(int)
                    path = coordinates['center_location'][0][homing_idx] * self.scaling_factor, \
                           coordinates['center_location'][1][homing_idx] * self.scaling_factor

                #    test eligibility
                #    or never approaches y center      or never is above wall on top part of arena
                # if np.sum(abs(path[1] - (50 - 10 * ('void' in self.experiment))) < 5) and np.sum(
                #         (abs(path[0] - 50) < (24.5 + 5 * ('void' in self.experiment))) * (path[1] < 50)):
                if np.sum(abs(path[1] - (50 + 10 * ('void' in self.experiment))) < 5) and np.sum(
                        (abs(path[0] - 50) < (24.5 + 5 * ('void' in self.experiment))) * (path[1] > 50)):
                    #                45 TEMP CHANGE BACK TO 45
                    # get x-position along central wall
                    center_y_idx = np.argmin(abs(path[1] - center_y))
                    x_SH.append(path[0][center_y_idx])

                    # get y-position closest to center
                    edge_idx = np.argmin(abs(path[1] - 50))
                    y_SH.append(path[1][edge_idx])

                    # get time of homing onset
                    SH_time.append(s / 30 / 60 + self.vid_duration[self.vid_num] / 30 / 60)

                    # get time until next in shelter
                    in_shelter_next = in_shelter[in_shelter > e]
                    if in_shelter_next.size:
                        how_long_to_shelter.append((in_shelter_next[0] - e) / self.fps)
                    else:
                        how_long_to_shelter.append(60)

                    # get stimulus evoked?
                    stim_evoked.append(s in self.stim_idx[0])

                    # get the initial conditions
                    x_start.append(path[0][0])
                    y_start.append(path[1][0])

                    # get the initial angle
                    start_angle = coordinates['body_angle'][s]
                    end_angle = coordinates['body_angle'][int(e)]
                    angle_start.append(start_angle)

                    # do the turn angle
                    turn = abs(end_angle - start_angle)
                    turn_direction = np.median(np.diff(coordinates['body_angle'])[s:int(e)])  # positive is left, negative is right
                    if abs(turn_direction) > 180: turn_direction *= -1  # if pass the 360, shift sign
                    # correct for crossing zero
                    if turn > 180: turn = 360 - turn
                    # neagtive is left, positive is right
                    turn_angle.append(turn * np.sign(-turn_direction))


        analysis.analysis[self.experiment][self.condition]['prev anti-homings'][self.mouse].append([x_SH, y_SH, how_long_to_shelter, SH_time, stim_evoked])

        if self.vid_num:
            # if previous video, add that videos homings
            for v, homing_var in enumerate([x_SH, y_SH, how_long_to_shelter, SH_time, stim_evoked]):
                homing_var_prev = analysis.analysis[self.experiment][self.condition]['prev anti-homings'][self.mouse][self.trials_condition_prev_vid[self.condition] - 1][v]
                homing_var = homing_var_prev + homing_var
                analysis.analysis[self.experiment][self.condition]['prev anti-homings'][self.mouse][-1][v] = homing_var




    def analyze_movements(self, coordinates, stim, position, distance_map, distance_from_shelter, analysis):
        '''     get start pos, start orientation, turn angle    '''
        # just the stimulus period
        start_idx = np.where(coordinates['start_index'][stim:int(stim + self.fps * 15)])[0] + stim
        bout_used = False
        bout_start_position, bout_start_angle, turn_angle = np.nan, np.nan, np.nan
        for bout_idx in start_idx:
            end_idx = int(coordinates['start_index'][bout_idx])
            bout_start_position = int(position[0][bout_idx]), int(position[1][bout_idx])
            bout_end_position = int(position[0][end_idx]), int(position[1][end_idx])
            # get euclidean distance change
            euclid_dist_change = distance_from_shelter[bout_idx] - distance_from_shelter[end_idx]

            # get geodesic distance change
            geodesic_dist_change = distance_map[self.condition][bout_start_position[1], bout_start_position[0]] - \
                                   distance_map[self.condition][bout_end_position[1], bout_end_position[0]]

            # use if move toward shelter
            if euclid_dist_change * self.scaling_factor > 7.5 or geodesic_dist_change * self.scaling_factor > 7.5: #TEMP, WAS 10
                # change end idx to where traveled 10 cm
                distance_moved = np.sqrt( (position[0][bout_idx:end_idx] - bout_start_position[0])**2 + \
                                          (position[1][bout_idx:end_idx] - bout_start_position[1])**2 )
                if np.max(distance_moved * self.scaling_factor) > 10:
                    end_idx = bout_idx + np.where(distance_moved * self.scaling_factor > 10)[0][0]
                else:
                    end_idx = bout_idx + len(distance_moved)
                # cache variables
                bout_start_position = tuple(bsp * self.scaling_factor for bsp in bout_start_position)
                bout_start_angle = coordinates['body_angle'][bout_idx]
                bout_end_angle = coordinates['body_angle'][end_idx]
                # get turn angle and direction
                turn_angle = abs(bout_end_angle - bout_start_angle)
                turn_direction = np.median(np.diff(coordinates['body_angle'])[bout_idx:end_idx]) # positive is left, negative is right
                if abs(turn_direction) > 180: turn_direction *= -1 # if pass the 360, shift sign
                # correct for crossing zero
                if turn_angle > 180: turn_angle = 360 - turn_angle
                # neagtive is left, positive is right
                turn_angle = turn_angle * np.sign(-turn_direction)
                break

        analysis.analysis[self.experiment][self.condition]['movement'][self.mouse].append([bout_start_position, bout_start_angle, turn_angle])


    def analyze_edginess(self, RT_speed, angular_speed, arrived_at_shelter, position, subgoal_speed_trace, stim_idx, threat_idx, analysis, body_angle):
        '''     get edginess        '''
        # check for pauses in escape
        RT = np.where((-subgoal_speed_trace[10 * 30:] > RT_speed) * (angular_speed[threat_idx[10 * 30:]] < 5))[0]
        wrong_way = subgoal_speed_trace[10 * 30:10 * 30 + arrived_at_shelter[0]] > 5
        up_top = (position[1][stim_idx[:arrived_at_shelter[0]]] * self.scaling_factor) < 25
        # if goes the wrong way while in the back, doesn't count as RT
        speed_wrong_way = np.where(wrong_way * up_top)[0]  # 15
        if speed_wrong_way.size:
            RT = RT[RT > speed_wrong_way[0]]
        # if no RT but reaches shelter, count RT as beginning (just to avoid errors, don't use this data)
        if arrived_at_shelter[0] < RT[0]: RT[0] = 1
        # if passes the back, counts as the start of trajectory
        if up_top.size:
                if not up_top[RT[0]]:
                    up_top_at_start = np.where(up_top[:RT[0]])[0]
                    if up_top_at_start.size:
                        RT[0] = up_top_at_start[-1]
        x_pos = position[0][stim_idx[RT[0]:arrived_at_shelter[0]]] * self.scaling_factor
        y_pos = position[1][stim_idx[RT[0]:arrived_at_shelter[0]]] * self.scaling_factor
        # switch around if shelter-on-side
        if 'side' in self.experiment:
            if x_pos[0] < 50: x_pos, y_pos = y_pos, x_pos
            else: x_pos, y_pos = y_pos, 100-x_pos

        # get where to evaluate trajectory
        if 'void' in self.experiment: y_eval_point = 35  #compensate for greater girthiness of the void
        else: y_eval_point = 40

        # '''     TEMP - y-eval is 10cm from start point      '''
        # dist_from_start = np.sqrt( (y_pos - y_pos[0])**2 + (x_pos - x_pos[0])**2)
        # y_eval_idx = np.argmin(abs(dist_from_start - 5))
        # y_eval_point = y_pos[y_eval_idx]
        # '''       TEMP use body angle as proxy for path  '''
        # initial_body_angle = body_angle[stim_idx[RT[0]]]
        # y_pos = y_pos[0] + (x_pos - x_pos[0]) / np.cos(np.deg2rad(initial_body_angle))


        if 'U shaped' in self.experiment: y_wall_point = 55 # when it's past the cup
        else: y_wall_point = 47.5 - 5* ('void' in self.experiment) #TEMP WAS 45

        x_pos_start = x_pos[0]
        y_pos_start = y_pos[0]
        # do line from starting position to shelter
        if not 'Square' in self.experiment:
            y_pos_shelter = self.shelter_location[1] / 10
            x_pos_shelter = self.shelter_location[0] / 10
        else: # use the short wall as reference (0)
            y_pos_shelter = 50
            x_pos_shelter = 58


        slope = (y_pos_shelter - y_pos_start) / (x_pos_shelter - x_pos_start)
        intercept = y_pos_start - x_pos_start * slope
        distance_to_line = abs(y_pos - slope * x_pos - intercept) / np.sqrt((-slope) ** 2 + (1) ** 2)
        # # get index at center point (wall location)
        mouse_at_eval_point = np.where( abs(y_pos - y_eval_point) < 1)[0]
        if mouse_at_eval_point.size: mouse_at_eval_point = mouse_at_eval_point[0]
        else: mouse_at_eval_point = np.argmin(abs(y_pos - y_eval_point))

        if 'Square' in self.experiment: # adjust formula for square cuz it's two wall edges
            x_at_eval_point = x_pos[mouse_at_eval_point]
            x_line_at_center = (y_eval_point - intercept) / slope
            line_sign = np.sign(x_at_eval_point - x_line_at_center)
        else: line_sign = np.sign(1)

        homing_vector_at_center = (y_eval_point - intercept) / slope
        # get offset from homing vector
        linear_offset = distance_to_line[mouse_at_eval_point] * line_sign
        # get line to the closest edge
        mouse_at_wall = np.argmin(abs(y_pos - y_wall_point))
        y_edge = 50
        if x_pos[mouse_at_wall] > 50: x_edge = 75  # + 5
        else: x_edge = 25  # - 5
        if 'Square' in self.experiment: x_edge = 76

        # do line from starting position to edge position
        slope = (y_edge - y_pos_start) / (x_edge - x_pos_start)
        intercept = y_pos_start - x_pos_start * slope
        distance_to_line = abs(y_pos - slope * x_pos - intercept) / np.sqrt((-slope) ** 2 + (1) ** 2)

        if 'Square' in self.experiment: # adjust formula for square cuz it's two wall edges
            x_at_eval_point = x_pos[mouse_at_eval_point]
            x_edge_at_center = (y_eval_point - intercept) / slope
            edge_sign = np.sign(x_edge_at_center - x_at_eval_point)
        else:
            edge_sign = np.sign(1)

        edge_offset = np.mean(distance_to_line[mouse_at_eval_point]) * edge_sign
        # compute the max possible deviation
        edge_vector_at_center = (y_eval_point - intercept) / slope
        line_to_edge_offset = abs(homing_vector_at_center - edge_vector_at_center)

        # get edginess
        # edginess = np.min((1, (linear_offset - edge_offset + line_to_edge_offset) / (2 * line_to_edge_offset)))
        edginess = abs( (linear_offset - edge_offset + line_to_edge_offset) / (2 * line_to_edge_offset) )
        if 'Square' in self.experiment:edginess = (linear_offset - edge_offset + line_to_edge_offset) / (2 * line_to_edge_offset)

        # print(linear_offset)
        # print(edge_offset)
        # print(line_to_edge_offset)
        # print(edginess)
        if self.mouse == 'CA7160':
            print('heyyy')

        analysis.analysis[self.experiment][self.condition]['edginess'][self.mouse].append(edginess)
        analysis.analysis[self.experiment][self.condition]['x edge'][self.mouse].append(x_edge)
        if 'x escape' in self.quantities_to_analyze: analysis.analysis[self.experiment][self.condition]['x escape'][self.mouse].append(x_pos[mouse_at_eval_point])

    def analyze_x_edge(self, position, stim_idx, analysis):

        if 'U shaped' in self.experiment: y_wall_point = 55
        else:y_wall_point = 47.5 - 5* ('void' in self.experiment)

        x_pos = position[0][stim_idx]
        x_wall_point = x_pos[np.argmin(abs(position[1][stim_idx] * self.scaling_factor - y_wall_point ))] * self.scaling_factor

        if x_wall_point < 50: x_edge = 25
        else: x_edge = 75

        analysis.analysis[self.experiment][self.condition]['x edge'][self.mouse].append(x_edge)


    def initialize_escape_data(self, coordinates, analysis):
        '''     prep data for escape analysis       '''
        distance_from_shelter = coordinates['distance_from_shelter']
        position = coordinates['center_location']
        # position = coordinates['front_location'] #TEMP CHANGE BACK
        delta_position = np.concatenate((np.zeros((2, 1)), np.diff(position)), axis=1)
        speed = np.sqrt(delta_position[0, :] ** 2 + delta_position[1, :] ** 2)
        angular_speed = coordinates['angular_speed']
        distance_map = self.get_distance_map(analysis)
        shelter_angle = coordinates['shelter_angle']
        body_angle = coordinates['body_angle']

        return distance_from_shelter, position, angular_speed, speed, distance_map, shelter_angle, body_angle


    def load_coordinates(self, dataframe, session_name):
        '''     load the coordinates file       '''
        # get and report the session and metadata
        session = dataframe.db.loc[session_name];
        metadata = session.Metadata
        print('Analyzing: {} - {} (#{})'.format(metadata['experiment'], metadata['mouse_id'], metadata['number']))
        # find file path for video and saved coordinates
        video_path = self.session['Metadata']['video_file_paths'][self.vid_num][0]
        processed_coordinates_file = os.path.join(os.path.dirname(video_path), 'coordinates')
        # open saved coordinates
        with open(processed_coordinates_file, "rb") as dill_file: coordinates = pickle.load(dill_file)
        return coordinates

    def format_analysis_dict(self, i, analysis):
        '''     make sure all the quantities and conditions are listed in the dictionary        '''
        # initialize arena data
        get_arena_details(self)
        # experiment is there
        if not self.experiment in analysis.analysis:
            analysis.analysis[self.experiment] = {}
        # conditions are there
        for condition in self.conditions:
            if not condition in analysis.analysis[self.experiment]:
                analysis.analysis[self.experiment][condition] = {}
                # add the shape and obstacle type of the arena
                shape = tuple(self.session['Registration'][4])
                analysis.analysis[self.experiment][condition]['shape'] = shape
                analysis.analysis[self.experiment][condition]['type'] = self.obstacle_type
            # quantities are there
            for q in self.quantities_to_analyze:
                if not q in analysis.analysis[self.experiment][condition]:
                    analysis.analysis[self.experiment][condition][q] = {}
                if not i or not self.mouse == 'control' or not 'control' in analysis.analysis[self.experiment][condition][q]:
                    analysis.analysis[self.experiment][condition][q][self.mouse] = []

    def get_stim_idx(self):
        '''     get indices when the stimulus is or was just on     '''
        stims_video = []
        # loop over each possible stimulus type
        for stim_type, stims_all in self.session['Stimuli']['stimuli'].items():
            # Only use stimulus modalities that were used during the session
            if not np.any([s for s in stims_all]): continue
            # Add to object
            self.stims_all = stims_all
        # initialize list of stim indices
        self.stim_idx = [[] for x in range(len(self.stims_all))]
        # add each stim time and the following 10 seconds to the list
        for vid_num in range(len(self.stims_all)):
            for stim_frame in self.stims_all[vid_num]:
                self.stim_idx[vid_num] = np.append(self.stim_idx[vid_num], np.arange(stim_frame, stim_frame + 240))
        # set vid length list for multiple vid sessions
        self.vid_duration = [0]

    def get_control_stim_frames(self, coordinates):
        '''     get frames when a stim didn't happen but could have         '''
        # initialize values
        stims_video = self.stims_all[self.vid_num]
        x_position = coordinates['center_location'][0].copy()
        y_position = coordinates['center_location'][1].copy()
        stim_idx = np.array([], 'int')
        # loop over actual stims, making those times ineligible
        for stim in stims_video:
            stim_idx = np.append(stim_idx, np.arange(stim - int(self.fps*18), stim + int(self.fps*18)))
        y_position[stim_idx] = np.nan; y_position[:int(self.fps*18)] = np.nan; y_position[-int(self.fps*18):] = np.nan
        # must be in threat zone
        eligible_frames = y_position < (25 / self.scaling_factor)
        # eligible_frames = (y_position < (55 / self.scaling_factor)) * (y_position > (45 / self.scaling_factor)) * \
        #                   (abs(x_position-50/ self.scaling_factor) > (20 / self.scaling_factor))

        # create fake stim times
        stims_video_control = []
        for i, stim in enumerate(stims_video):
            control_stim = np.random.choice(np.where(eligible_frames)[0])
            stims_video_control.append(control_stim)
            eligible_frames[control_stim - int(self.fps*6): control_stim + int(self.fps*6)] = False
            if not np.sum(eligible_frames): break

        stims_video_control.sort()
        # replace real values with fake stims
        self.stims_all[self.vid_num] = stims_video_control

    def get_obstacle_epochs(self, total_frames):
        '''     get time periods when the obstacle is up or down respectively       '''
        # loop across videos
        trial_types = self.session['Tracking']['Trial Types'][self.vid_num]
        if 'Square' in self.session.Metadata['experiment'] and self.session.Metadata['mouse_id']=='CA6960':
            trial_types = trial_types[:-2]
            total_frames = self.stims_all[self.vid_num][2] + 300
            print(trial_types)
        if not trial_types: trial_types = [np.nan]

        # wall down experiments
        if -1 in trial_types:
            # when did the wall fall
            wall_down_idx = np.where(np.array(trial_types)==-1)[0][0]
            # when the wall was up
            self.wall_up_epoch = list(range(0, self.stims_all[self.vid_num][wall_down_idx]))
            # when the wall was down
            self.wall_down_epoch = list(range(self.stims_all[self.vid_num][wall_down_idx] + 300, total_frames))
            # and no probe trials
            self.probe_epoch = list(range(self.stims_all[self.vid_num][wall_down_idx], self.stims_all[self.vid_num][wall_down_idx] + 300))
            # when the epoch starts
            self.start_points = [0, self.stims_all[self.vid_num][wall_down_idx] + 300, self.stims_all[self.vid_num][wall_down_idx]]
        # wall up experiments
        elif 1 in trial_types:
            # when did the wall rise
            wall_up_idx = np.where(np.array(trial_types)==1)[0][0]
            # when the wall was down
            self.wall_down_epoch = list(range(0,self.stims_all[self.vid_num][wall_up_idx]))
            # when the wall was up
            self.wall_up_epoch = list(range(self.stims_all[self.vid_num][wall_up_idx] + 300, total_frames))
            # and no probe trials
            self.probe_epoch = list(range(self.stims_all[self.vid_num][wall_up_idx], self.stims_all[self.vid_num][wall_up_idx] + 300))
            # when the epoch starts
            self.start_points = [self.stims_all[self.vid_num][wall_up_idx] + 300, 0, self.stims_all[self.vid_num][wall_up_idx]]
        # void up experiments
        elif trial_types[0]==2 and trial_types[-1] == 0:
            # when did the void rise
            last_wall_idx = np.where([t==2 for t in trial_types])[0][-1]
            first_no_wall_idx = np.where([t==0 for t in trial_types])[0][0]
            # when the void was there
            self.wall_up_epoch = list(range(0,self.stims_all[self.vid_num][last_wall_idx]+300))
            # when the void was filled
            self.wall_down_epoch = list(range(self.stims_all[self.vid_num][first_no_wall_idx] - 300, total_frames))
            # and no probe trials
            self.probe_epoch = []
            # when the epoch starts
            self.start_points = [0, self.stims_all[self.vid_num][first_no_wall_idx] - 300, np.nan]
        # square experiments
        elif trial_types[0]==0 and trial_types[-1] == 2:
            # when did the void rise
            last_wall_idx = np.where([t==0 for t in trial_types])[0][-1]
            first_no_wall_idx = np.where([t==2 for t in trial_types])[0][0]
            # when the wall was in starting position
            self.wall_down_epoch = list(range(0,self.stims_all[self.vid_num][last_wall_idx]+300))
            # when the wall moved sideways
            self.wall_up_epoch = list(range(self.stims_all[self.vid_num][first_no_wall_idx] - 300, total_frames))
            # and no probe trials
            self.probe_epoch = []
            # when the epoch starts
            self.start_points = [self.stims_all[self.vid_num][first_no_wall_idx] - 300, 0, np.nan]
        # lights on off baseline experiments
        elif 'lights on off (baseline)' in self.session['Metadata']['experiment']:
            if len(self.stims_all[self.vid_num]) > 4:
                # when the light was there
                self.wall_up_epoch = list(range(0,self.stims_all[self.vid_num][4]-300))
                # when the dark was there
                self.wall_down_epoch = list(range(self.stims_all[self.vid_num][4] - 10, total_frames))
                # and no probe trials
                self.probe_epoch = []
                # when the epoch starts
                self.start_points = [0, self.stims_all[self.vid_num][4] - 300, np.nan]
            else:
                # when the light was there
                self.wall_up_epoch = list(range(0,total_frames))
                # when the dark was there
                self.wall_down_epoch = []
                # and no probe trials
                self.probe_epoch = []
                # when the epoch starts
                self.start_points = [0, np.nan, np.nan]
        # obstacle static experiments
        elif trial_types[0]==2:
            # the wall was always up
            self.wall_up_epoch = list(range(0, total_frames))
            # and never down
            self.wall_down_epoch = []
            # and no probe trials
            self.probe_epoch = []
            # when the epoch starts
            self.start_points = [0, np.nan, np.nan]
        elif 'down (no shelter)' in self.session.Metadata['experiment']:
            # when did the wall fall
            if self.session.Metadata['mouse_id'] == 'CA8501': wall_down_frame = int(23.5 * 30 * 60)
            elif self.session.Metadata['mouse_id']=='CA8511': wall_down_frame = int(18.8 * 30 * 60)
            elif self.session.Metadata['mouse_id']=='CA8521': wall_down_frame = int(21.3 * 30 * 60)
            elif self.session.Metadata['mouse_id']=='CA8541': wall_down_frame = int(19.5 * 30 * 60)
            elif self.session.Metadata['mouse_id']=='CA7220': wall_down_frame = int(25 * 30 * 60)
            # when the wall was up
            self.wall_up_epoch = list(range(0, wall_down_frame))
            # when the wall was down
            self.wall_down_epoch = list(range(wall_down_frame + 300, total_frames))
            # and no probe trials
            self.probe_epoch = list(range(wall_down_frame, wall_down_frame + 300))
            # when the epoch starts
            self.start_points = [0, wall_down_frame + 300, wall_down_frame]

            if self.session.Metadata['mouse_id'] == 'CA7220' and not self.vid_num:
                self.wall_up_epoch = list(range(0, total_frames))
                self.wall_down_epoch = []
                self.probe_epoch = []
                self.start_points = [0, np.nan, np.nan]

        # obstacle static experiments
        elif trial_types[0]==0:
            # the wall was always up
            self.wall_up_epoch = []
            # and never down
            self.wall_down_epoch = list(range(0, total_frames))
            # and no probe trials
            self.probe_epoch = []
            # when the epoch starts
            self.start_points = [np.nan, 0, np.nan]
        else:
            # assume the wall was always up
            self.wall_up_epoch = list(range(0, total_frames))
            print('assuming wall up during this segment!')
            # and never down
            self.wall_down_epoch = []
            # and no probe trials
            self.probe_epoch = []
            # when the epoch starts
            self.start_points = [0, np.nan, np.nan]

    def get_distance_map(self, analysis):
        '''     get the map of geodesic distance to shelter, to compute geodesic speed      '''
        # initialize the arena data
        shape = analysis.analysis[self.experiment][self.condition]['shape']
        arena, _, _ = model_arena(shape, 2, False, self.obstacle_type, simulate=True)
        shelter_location = [int(a * b / 1000) for a, b in zip(self.shelter_location, arena.shape)]
        # initialize the geodesic function
        phi = np.ones_like(arena)
        mask = (arena == 90)
        phi_masked = np.ma.MaskedArray(phi, mask)
        distance_map = {}
        # get the geodesic map of distance from the shelter
        phi_from_shelter = phi_masked.copy()
        phi_from_shelter[shelter_location[1], shelter_location[0]] = 0
        distance_map['obstacle'] = np.array(skfmm.distance(phi_from_shelter))
        # get the euclidean map of distance from the shelter
        phi[shelter_location[1], shelter_location[0]] = 0
        distance_map['no obstacle'] = np.array(skfmm.distance(phi))
        # get the distance map for the probe trial
        trial_types = self.session['Tracking']['Trial Types'][self.vid_num]
        if -1 in trial_types: distance_map['probe'] = distance_map['no obstacle']
        elif 1 in trial_types: distance_map['probe'] = distance_map['obstacle']
        else: distance_map['probe'] = np.nan

        return distance_map

    def get_metadata(self, session_name, analysis, dataframe, analysis_type):
        '''     get metadata for analysis       '''
        self.session = dataframe.db.loc[session_name]
        self.experiment = self.session['Metadata']['experiment']
        self.mouse = self.session['Metadata']['mouse_id']
        if analysis.analysis_options['control']: self.mouse = 'control'
        self.scaling_factor = 100 / self.session['Registration'][4][0]
        self.fps = self.session['Metadata']['videodata'][0]['Frame rate'][0]

        if analysis_type == 'escape paths' and 'dark' in self.experiment: self.quantities_to_analyze.append('prev homings')
        if analysis_type == 'escape paths' and 'dark' in self.experiment: self.quantities_to_analyze.append('prev movements')

    def save_analysis(self, analysis, analysis_type):
        '''     save analysis       '''
        # loop across experiments
        for experiment in flatten(analysis.experiments):
            # save analysis in the folder where clips are saved
            save_folder = os.path.join(analysis.dlc_settings['clips_folder'], experiment, analysis_type)
            with open(save_folder, "wb") as dill_file: pickle.dump(analysis.analysis[experiment], dill_file)

