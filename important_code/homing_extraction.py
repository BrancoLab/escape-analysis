import os
import cv2
import numpy as np
import itertools
import scipy.misc
from helper_code.processing_funcs import extract_variables, convolve, threshold, speed_colors
from important_code.escape_visualization import make_model_mouse_mask

def extract_homings(self, make_vid = False, homing_vectors = False, from_the_back = True):
    '''    compute and display SPONTANEOUS HOMINGS    '''

    # make video
    if make_vid: video_clip = cv2.VideoWriter(os.path.join(self.save_folder, self.videoname + ' homings.mp4'), self.fourcc, 160, (self.width, self.height), True)

    # initialize arrays
    self.skip_frames = 300
    model_mouse_mask_initial = (self.homing_arena[:, :, 0] * 0)#.astype(bool)

    # set the speed parameters
    self.scaling_factor = self.height / 100
    speeds = [[3.6, 5.4, 7.2], [3.6, 7.2, 28.8]]
    smooth_duration = 30

    # set distance parameters
    close_to_shelter_distance = 40
    minimum_shelter_distance = 150
    minimum_distance = 400
    close_to_shelter_angle = 30
    strict_shelter_angle = 85
    back_butt_dist = 13

    # create local variables for the current epoch
    extract_variables(self)

    # do convolutions to get current, future, far future, and past speeds w.r.t the shelter
    current_speed = convolve(self.goal_speeds, 24, -1, time='current')
    past_speed = convolve(self.goal_speeds, 45, -1, time='past')
    future_speed = convolve(self.goal_speeds, 60, -1, time='future')
    far_future_speed = convolve(self.goal_speeds, 60, -1, time='far future', time_chase=20)

    # is the mouse close enough to the shelter
    close_enough_to_shelter = convolve(self.distance_from_shelter < minimum_distance, 60, +1, time='future')
    # is the mouse facing the shelter
    angle_thresholded = threshold(abs(self.subgoal_angles), close_to_shelter_angle, '<')
    # is the mouse closely facing the shelter
    strict_angle_thresholded = threshold(abs(self.subgoal_angles), strict_shelter_angle, '<') + threshold(self.frame_nums, self.stim_frame)
    # is the mouse not already right by the shelter
    distance_thresholded = threshold(self.distance_from_shelter, close_to_shelter_distance)

    # loop across fast and slow homings
    for speed_setting in [0, 1]:

        # use higher speed setting to make faster homings darker
        low_speed, medium_speed, high_speed = \
            speeds[speed_setting][0]/self.scaling_factor, speeds[speed_setting][1]/self.scaling_factor, speeds[speed_setting][2]/self.scaling_factor
        # threshold the convolved speeds to determine which frames to draw
        current_speed_thrsh = threshold(current_speed, high_speed)
        future_speed_thrsh = threshold(future_speed, high_speed) * threshold(current_speed, medium_speed)
        far_future_speed_thrsh = threshold(far_future_speed, high_speed) * threshold(current_speed, medium_speed)
        past_speed_thrsh = threshold(past_speed, high_speed) * threshold(current_speed, medium_speed)
        stimulus_thresholded = threshold(self.frame_nums, self.stim_frame - 1 + smooth_duration) * threshold(current_speed, low_speed)
        # combine speed thresholds into one
        combined_speed_thresholds = (current_speed_thrsh + future_speed_thrsh + far_future_speed_thrsh + past_speed_thrsh)
        # combine all thresholds into one
        self.thresholds_passed[0,:] = distance_thresholded * (angle_thresholded * close_enough_to_shelter * combined_speed_thresholds + stimulus_thresholded)
        # and smooth it *out*
        self.thresholds_passed[0,:] = (strict_angle_thresholded * convolve(self.thresholds_passed[0,:], smooth_duration, +1, time='current') ).astype(bool)
        # finally, add a minimum duration threshold
        minimum_distance = (-.4 + .2 * speed_setting) 
        minimum_duration = smooth_duration + 5
        get_homing_groups(self, minimum_distance, minimum_duration, minimum_shelter_distance)
        # set function output
        self.trial_groups = self.thresholds_passed[0,:] + self.thresholds_passed[1, :]
        # get the index when adequately long locomotor bouts pass the threshold
        self.thresholds_passed[1, :] = self.thresholds_passed[0,:]
        thresholds_passed_idx = np.where(self.thresholds_passed[1, :])[0]
        # more initializing
        group_counter = 0
        bout_starting = True
        # smooth speed trace for coloration
        smoothed_speed = np.concatenate((np.zeros(6 - 1), np.convolve(self.coordinates['speed'], np.ones(12), mode='valid'), np.zeros(6))) / 12
        counter = 0
        # loop over each frame that passed the threshold
        for idx in thresholds_passed_idx:
            # get frame number
            frame_num = self.frame_nums[idx]
            # temporary: don't include stimulus-evoked events in the image
            # if frame_num >= self.stim_frame-10:
            #     break
            # set up new bout
            if self.group_idx[idx]:
                group_length = self.group_idx[idx]
                bout_starting = False
                start_position = self.coordinates['center_location'][0][frame_num-1], self.coordinates['center_location'][1][frame_num-1]
                end_position = self.coordinates['center_location'][0][frame_num-1 + int(group_length)], \
                               self.coordinates['center_location'][1][frame_num - 1 + int(group_length)]
            # don't include bouts that stay in the back of the arena
            if end_position[1] < 320: continue
            if homing_vectors:
                # don't include bouts that don't end up in the shelter
                if end_position[1] < 550 or abs(end_position[0] - 360) > 72: continue
                # don't include bits that start in the back center
                position = self.coordinates['center_location'][0][frame_num - 1], self.coordinates['center_location'][1][frame_num - 1]
                if position[1] < 350 and abs(position[0] - 360) < 220: continue
                if position[1] < 300: continue
            # draw ellipses representing model mouse
            model_mouse_mask, _, _ = make_model_mouse_mask(self.coordinates, frame_num, model_mouse_mask_initial, scale = back_butt_dist)
            # set the color of the silhouette
            speed = smoothed_speed[frame_num - 1] #* 1.7
            speed_color_light, speed_color_dark = speed_colors(speed, red = False) #True
            # emphasize the back
            if from_the_back:
                if start_position[1] > 280: speed_color_light, speed_color_dark = speed_color_light**.3, speed_color_dark**.3
            speed_color_light, speed_color_dark = speed_color_light ** .4, speed_color_dark ** .4
            # apply color to arena image
            if counter*speed > 80: # start it light
                self.homing_arena[model_mouse_mask.astype(bool)] = self.homing_arena[model_mouse_mask.astype(bool)] * speed_color_dark
                counter = 0
            else: # end it dark
                self.homing_arena[model_mouse_mask.astype(bool)] = self.homing_arena[model_mouse_mask.astype(bool)] * speed_color_light
                counter += 1
            # display image
            cv2.imshow(self.save_folder + 'homings', self.homing_arena[:,:,::-1])
            # write frame
            if make_vid: video_clip.write(self.homing_arena)
            # press q to quit
            if cv2.waitKey(1) & 0xFF == ord('q'): break
    # apply the contours and border to the image and save the image
    scipy.misc.imsave(os.path.join(self.save_folder, self.videoname + ' homings image.tif'), self.homing_arena)
    # end video
    if make_vid: video_clip.release()



def get_homing_groups(self, minimum_distance, minimum_duration, minimum_shelter_distance):
    '''     group up locomotor bouts into continuous homing movements       '''
    # initialize variables
    thresholds_passed = self.thresholds_passed[0, :] * (1 - self.thresholds_passed[1, :])
    groups = []
    idx = 0
    group_idx = np.zeros(len(thresholds_passed))

    for k, g in itertools.groupby(thresholds_passed):
        groups.append(list(g))
        group_length = len(groups[len(groups) - 1]);
        idx += group_length
        distance_traveled = (self.distance_from_subgoal[idx - 1] - self.distance_from_subgoal[idx - group_length]) / self.distance_from_subgoal[idx - group_length]
        if k and ((group_length < minimum_duration) or (distance_traveled > minimum_distance) or \
                  (self.distance_from_shelter[idx - group_length] < minimum_shelter_distance)):
            thresholds_passed[idx - group_length: idx] = False
        elif k:
            group_idx[idx - group_length] = group_length
            group_idx[idx - 1] = 2000*(np.mean(self.absolute_speeds[idx - group_length:idx]) / group_length**2)

    # pass back to the object
    self.thresholds_passed[0, :] = thresholds_passed
    self.group_idx = group_idx

