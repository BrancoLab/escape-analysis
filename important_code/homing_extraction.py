import os
import cv2
import numpy as np
import itertools
import scipy.misc
from helper_code.processing_funcs import extract_variables, convolve, threshold, speed_colors
from important_code.escape_visualization import make_model_mouse_mask
from scipy.ndimage import gaussian_filter1d

# Analyze Data
def extract_homings(self, make_vid = False, homing_vectors = False, from_the_back = True, anti = False, red = True):
    '''    compute and display SPONTANEOUS HOMINGS    '''

    # make video
    if make_vid: video_clip = cv2.VideoWriter(os.path.join(self.save_folder, self.videoname + ' homings.mp4'), self.fourcc, 160, (self.width, self.height), True)

    # initialize arrays
    self.skip_frames = 300
    model_mouse_mask_initial = (self.homing_arena[:, :, 0] * 0)#.astype(bool)

    # set the speed parameters
    self.scaling_factor = self.height / 100
    speed = 5.4
    smooth_duration = 30

    # set distance parameters
    close_to_shelter_distance = 40
    minimum_shelter_distance = 150
    minimum_distance = 400
    close_to_shelter_angle = 30
    back_butt_dist = 14
    angle_speed_treshold = -3

    # create local variables for the current epoch
    extract_variables(self)


    # do convolutions to get current, future, far future, and past speeds w.r.t the shelter
    smoothed_speed = gaussian_filter1d(self.goal_speeds, 16.5) * -1
    smoothed_speed_shelter = gaussian_filter1d(self.shelter_speeds, 3) * -1

    # is the mouse close enough to the shelter over next two seconds?
    close_enough_to_shelter_future = convolve(self.distance_from_shelter < minimum_distance, 60, +1, time='future')
    # is the mouse facing the shelter
    facing_shelter = threshold(abs(self.subgoal_angles), close_to_shelter_angle, '<')
    # is the mouse not already right by the shelter
    far_enough_from_shelter = threshold(self.distance_from_shelter, close_to_shelter_distance)
    # use higher speed setting to make faster homings darker
    speed_for_threshold = speed/self.scaling_factor
    # threshold the convolved speeds to determine which frames to draw
    speed_threshold = threshold(smoothed_speed, speed_for_threshold)
    # angular speed threshold
    angular_speed_threshold = threshold(self.angular_speed_subgoal, angle_speed_treshold, '<')

    # is the mouse closely facing the shelter
    # post_filter_angle_thresholded = threshold(abs(self.subgoal_angles), 90, '<')
    # post_filter_speed_thresholded = threshold(smoothed_speed, 0)
    post_filter_speed_thresholded = threshold(smoothed_speed_shelter, 0)
    post_filter_angular_speed_thresholded = threshold(self.angular_speed_subgoal, 0, '<')
    post_filter_thresholds = (post_filter_speed_thresholded + post_filter_angular_speed_thresholded) # post_filter_angle_thresholded +

    # self.thresholds_passed = self.thresholds_passed[0, :]
    self.thresholds_passed = far_enough_from_shelter * close_enough_to_shelter_future * (facing_shelter * speed_threshold + angular_speed_threshold)
    # and smooth it *out*
    self.thresholds_passed = (post_filter_thresholds * convolve(self.thresholds_passed, smooth_duration, +1, time='current')).astype(bool)
    # finally, add a minimum distance and duration threshold
    minimum_distance = -.3 #(-.4 + .2 * speed_setting)
    minimum_duration = smooth_duration + 5

    get_homing_groups(self, minimum_distance, minimum_duration, minimum_shelter_distance)
    # set function output
    self.trial_groups = self.thresholds_passed#[0,:] + self.thresholds_passed[1, :]
    # get the index when adequately long locomotor bouts pass the threshold
    thresholds_passed_idx = np.where(self.thresholds_passed)[0]
    # smooth speed trace for coloration
    smoothed_speed = np.concatenate((np.zeros(6 - 1), np.convolve(self.coordinates['speed'], np.ones(12), mode='valid'), np.zeros(6))) / 12
    counter = 0
    color_trail = np.array([.02, -.02, -.02])
    silhouette_color = (220, 0, 0)
    skip = False
    # loop over each frame that passed the threshold
    for idx in thresholds_passed_idx:
        # get frame number
        frame_num = self.frame_nums[idx]
        # temporary: don't include stimulus-evoked events in the image
        if frame_num > self.stim_frame-10: shift = 10
        else: shift = 0
        #     continue
        # set up new bout
        if self.group_idx[idx]:
            group_length = self.group_idx[idx]
            bout_starting = True
            start_position = self.coordinates['center_location'][0][frame_num-1], self.coordinates['center_location'][1][frame_num-1]
            end_position = self.coordinates['center_location'][0][frame_num-1 + int(group_length)], \
                           self.coordinates['center_location'][1][frame_num - 1 + int(group_length)]
            counter = 0

            # if not counted as 'homing from threat area', then don't show silhouette
            path = self.coordinates['center_location'][0][frame_num - 1:frame_num - 1 + int(group_length)] / self.scaling_factor, \
                   self.coordinates['center_location'][1][frame_num - 1:frame_num - 1 + int(group_length)] / self.scaling_factor
            if (path[1][0] > (30 - 2.5* ('void' in self.videoname)) \
                or not np.sum(abs(path[1] - (45 - 10 * ('void' in self.videoname))) < 5) \
                or not np.sum((abs(path[0] - 50) < (24.5 + 5 * ('void' in self.videoname))) * (path[1] < 50))) and not anti:
                skip = True
            else:
                skip = False
            # print(start_position)
            # print(end_position)
            # print(group_length)
            # print('')
            # for 3390 tr 2
            # if start_position[1] > 240 and not abs(start_position[1] - 508) < 2: skip = True
            # else: skip = False
            # if start_position[1] > 360: bout_starting = False

            # for 6970
            # if start_position[1] > 200: skip = True # 360
            # elif end_position[1] < 300: skip = True
            # else: skip = False

        else: bout_starting = False

        if skip: continue
        # for 3390 tr 2
        # if group_length == 237 and counter < 35: counter+=1; bout_starting = True; continue

        # don't include bouts that stay in the back of the arena
        # if end_position[1] < 320: continue # -temp commented out
        # if start_position[1] > 360 and end_position[1] < 615: continue
        # if abs(start_position[0]- 360) > 220: continue

        if homing_vectors:
            # don't include bouts that don't end up in the shelter
            if end_position[1] < 550 or abs(end_position[0] - 360) > 72: continue
            # don't include bits that start in the back center
            position = self.coordinates['center_location'][0][frame_num - 1], self.coordinates['center_location'][1][frame_num - 1]
            if position[1] < 350 and abs(position[0] - 360) < 220: continue
            if position[1] < 300: continue
        # draw ellipses representing model mouse
        model_mouse_mask, _, body_location = make_model_mouse_mask(self.coordinates, frame_num, model_mouse_mask_initial, scale = back_butt_dist, shift = shift)
        # set the color of the silhouette
        if red:
            if from_the_back and start_position[1] > 280: continue
            # draw red contours on start location
            if bout_starting: #or counter == 35:
                _, contours, _ = cv2.findContours(model_mouse_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                counter += 1
            # if start_position[1] < 300:
            cv2.drawContours(self.homing_arena, contours, 0, (255, 0, 0), thickness=-1)
            # cv2.drawContours(self.homing_arena, contours, 0, (0, 0, 0), thickness=1)
            # get distance traveled
            dist_from_start = np.sqrt((start_position[0] - float(body_location[0])) ** 2 + (start_position[1] - float(body_location[1])) ** 2)
            if body_location[1] > 360: dist_from_start = 150
            dist_to_make_red = 150
            prev_homing_color = np.array([.98, .98, .98]) + np.max((0, dist_to_make_red - dist_from_start)) / dist_to_make_red * color_trail
            self.homing_arena[model_mouse_mask.astype(bool)] = self.homing_arena[model_mouse_mask.astype(bool)] * prev_homing_color

            if not idx+1 in thresholds_passed_idx:
                self.homing_arena[model_mouse_mask.astype(bool)] = self.homing_arena[model_mouse_mask.astype(bool)] * .6


        else:
            speed = smoothed_speed[frame_num - 1] #* 1.7
            speed_color_light, speed_color_dark = speed_colors(speed, red = False) #True
            if bout_starting: speed_color_light, speed_color_dark = speed_color_light*0, speed_color_dark*0
            # emphasize the back
            if from_the_back:
                if start_position[1] > 280:
                    speed_color_light, speed_color_dark = speed_color_light**.3, speed_color_dark**.3
            speed_color_light, speed_color_dark = speed_color_light ** .4, speed_color_dark ** .4
            # apply color to arena image
            # print(counter*speed)
            # if counter*speed > 80: # start it light
            #     self.homing_arena[model_mouse_mask.astype(bool)] = self.homing_arena[model_mouse_mask.astype(bool)] * speed_color_dark
            #     counter = 0
            # else: # end it dark
            #     self.homing_arena[model_mouse_mask.astype(bool)] = self.homing_arena[model_mouse_mask.astype(bool)] * speed_color_light
            #     counter += 1
            self.homing_arena[model_mouse_mask.astype(bool)] = self.homing_arena[model_mouse_mask.astype(bool)] * speed_color_light

        # display image
        cv2.imshow(self.save_folder + 'homings', self.homing_arena[:,:,::-1])
        # write frame
        if make_vid: video_clip.write(self.homing_arena)
        # press q to quit
        if cv2.waitKey(1) & 0xFF == ord('q'): break
    # apply the contours and border to the image and save the image
    if anti: image_name = self.videoname + ' anti-homings image.tif'
    else: image_name = self.videoname + ' homings image.tif'
    scipy.misc.imsave(os.path.join(self.save_folder, image_name), self.homing_arena)
    # end video
    if make_vid: video_clip.release()

#
#
# # GET VIDEO
# def extract_homings(self, make_vid = True, homing_vectors = False, from_the_back = True, anti = False, red = True):
#     '''    compute and display SPONTANEOUS HOMINGS    '''
#
#     # make video
#     self.fourcc = cv2.VideoWriter_fourcc(*"XVID")
#     if make_vid: video_clip = cv2.VideoWriter(os.path.join(self.save_folder, self.videoname + ' homings.mp4'), self.fourcc, 300, (self.width, self.height), True)
#
#     # initialize arrays
#     self.skip_frames = 300
#     model_mouse_mask_initial = (self.homing_arena[:, :, 0] * 0)
#     red_mask = (self.homing_arena[:, :, 0] * 0) + 1
#
#
#     # set the speed parameters
#     self.scaling_factor = self.height / 100
#     speeds = [[3.6, 5.4, 7.2], [3.6, 7.2, 28.8]]
#     smooth_duration = 30
#
#     # set distance parameters
#     close_to_shelter_distance = 40
#     minimum_shelter_distance = 150
#     minimum_distance = 400
#     close_to_shelter_angle = 30
#     strict_shelter_angle = 85
#     back_butt_dist = 14
#
#     # create local variables for the current epoch
#     extract_variables(self)
#
#     # do convolutions to get current, future, far future, and past speeds w.r.t the shelter
#     current_speed = convolve(self.goal_speeds, 24, -1, time='current')
#     past_speed = convolve(self.goal_speeds, 45, -1, time='past')
#     future_speed = convolve(self.goal_speeds, 60, -1, time='future')
#     far_future_speed = convolve(self.goal_speeds, 60, -1, time='far future', time_chase=20)
#
#     # is the mouse close enough to the shelter
#     close_enough_to_shelter = convolve(self.distance_from_shelter < minimum_distance, 60, +1, time='future')
#     # is the mouse facing the shelter
#     angle_thresholded = threshold(abs(self.subgoal_angles), close_to_shelter_angle, '<')
#     # is the mouse closely facing the shelter
#     strict_angle_thresholded = threshold(abs(self.subgoal_angles), strict_shelter_angle, '<') + threshold(self.frame_nums, self.stim_frame)
#     # is the mouse not already right by the shelter
#     distance_thresholded = threshold(self.distance_from_shelter, close_to_shelter_distance)
#
#     self.group_idx = np.zeros_like(current_speed)
#     # loop across fast and slow homings
#     for speed_setting in [0, 1]:
#
#         # use higher speed setting to make faster homings darker
#         low_speed, medium_speed, high_speed = \
#             speeds[speed_setting][0]/self.scaling_factor, speeds[speed_setting][1]/self.scaling_factor, speeds[speed_setting][2]/self.scaling_factor
#         # threshold the convolved speeds to determine which frames to draw
#         current_speed_thrsh = threshold(current_speed, high_speed)
#         future_speed_thrsh = threshold(future_speed, high_speed) * threshold(current_speed, medium_speed)
#         far_future_speed_thrsh = threshold(far_future_speed, high_speed) * threshold(current_speed, medium_speed)
#         past_speed_thrsh = threshold(past_speed, high_speed) * threshold(current_speed, medium_speed)
#         stimulus_thresholded = threshold(self.frame_nums, self.stim_frame - 1 + smooth_duration) * threshold(current_speed, low_speed)
#         # combine speed thresholds into one
#         combined_speed_thresholds = (current_speed_thrsh + future_speed_thrsh + far_future_speed_thrsh + past_speed_thrsh)
#         # combine all thresholds into one
#         self.thresholds_passed[0,:] = distance_thresholded * (angle_thresholded * close_enough_to_shelter * combined_speed_thresholds + stimulus_thresholded)
#         # and smooth it *out*
#         self.thresholds_passed[0,:] = (strict_angle_thresholded * convolve(self.thresholds_passed[0,:], smooth_duration, +1, time='current') ).astype(bool)
#         # finally, add a minimum duration threshold
#         minimum_distance = (-.4 + .2 * speed_setting)
#         minimum_duration = smooth_duration + 5
#
#         # for aesthetic purposes for 3390
#         # self.thresholds_passed[:, 17320:17359] = 0
#         get_homing_groups(self, minimum_distance, minimum_duration, minimum_shelter_distance)
#         # set function output
#         self.trial_groups = self.thresholds_passed[0,:] + self.thresholds_passed[1, :]
#         either_threshold_passed = np.where(self.trial_groups)[0]
#         # get the index when adequately long locomotor bouts pass the threshold
#         self.thresholds_passed[1, :] = self.thresholds_passed[0,:]
#         thresholds_passed_idx = np.where(self.thresholds_passed[1, :])[0]
#         # smooth speed trace for coloration
#         smoothed_speed = np.concatenate((np.zeros(6 - 1), np.convolve(self.coordinates['speed'], np.ones(12), mode='valid'), np.zeros(6))) / 12
#         counter = 0
#         color_trail = np.array([.025, -.025, -.025])
#         silhouette_color = (220, 0, 0)
#         # loop over each frame that passed the threshold
#         start_frames = [4448, 7788, 11857, 15031, 20134, 33736, 34296]
#         skip = False
#         # for idx in thresholds_passed_idx:
#         if speed_setting:
#             for idx in range(0,either_threshold_passed[-1]):
#                 # get frame number
#                 frame_num = self.frame_nums[idx]
#                 if self.coordinates['distance_from_shelter'][frame_num] < 80: continue
#
#                 # draw ellipses representing model mouse
#                 model_mouse_mask, _, body_location = make_model_mouse_mask(self.coordinates, frame_num, model_mouse_mask_initial, scale=back_butt_dist)
#                 model_mouse_mask_gray = model_mouse_mask * red_mask
#                 if self.coordinates['center_location'][1][frame_num-1] > 620: continue
#                 if np.mean(self.homing_arena[model_mouse_mask.astype(bool)]) < 100: continue
#                 red_frame = False
#                 pre_red = start_frames - frame_num
#
#                 if pre_red[np.argmin(abs(pre_red))] > 0 and np.min(abs(pre_red)) < 60: red_frame = True
#
#                 if idx in either_threshold_passed:
#
#                     # temporary: don't include stimulus-evoked events in the image
#                     # if frame_num >= self.stim_frame-10:
#                     #     break
#                     # set up new bout
#                     if self.group_idx[idx]:
#                         group_length = self.group_idx[idx]
#                         bout_starting = True
#                         start_position = self.coordinates['center_location'][0][frame_num-1], self.coordinates['center_location'][1][frame_num-1]
#                         end_position = self.coordinates['center_location'][0][frame_num-1 + int(group_length)], \
#                                        self.coordinates['center_location'][1][frame_num - 1 + int(group_length)]
#                         counter = 0
#
#                         # for 3390 tr 2
#                         # if start_position[1] > 240 and not abs(start_position[1] - 508) < 2: skip = True
#                         # else: skip = False
#                         # if start_position[1] > 360: bout_starting = False
#
#                         # for 6970
#                         if start_position[1] > 200: skip = True # 360
#                         elif end_position[1] < 300: skip = True
#                         else: skip = False
#
#                         if not skip: print(frame_num)
#
#                     else: bout_starting = False
#
#                     if skip:
#                         # draw in light gray
#                         prev_homing_color = np.array([.99, .99, .99])
#                         self.homing_arena[model_mouse_mask_gray.astype(bool)] = self.homing_arena[model_mouse_mask_gray.astype(bool)] * prev_homing_color
#                     else:
#                         # for 3390 tr 2
#                         # if group_length == 237 and counter < 35: counter+=1; bout_starting = True; continue
#
#                         # don't include bouts that stay in the back of the arena
#                         # if end_position[1] < 320: continue # -temp commented out
#                         # if start_position[1] > 360 and end_position[1] < 615: continue
#                         # if abs(start_position[0]- 360) > 220: continue
#
#                         if homing_vectors:
#                             # don't include bouts that don't end up in the shelter
#                             if end_position[1] < 550 or abs(end_position[0] - 360) > 72: continue
#                             # don't include bits that start in the back center
#                             position = self.coordinates['center_location'][0][frame_num - 1], self.coordinates['center_location'][1][frame_num - 1]
#                             if position[1] < 350 and abs(position[0] - 360) < 220: continue
#                             if position[1] < 300: continue
#                         # set the color of the silhouette
#                         if red:
#                             # if from_the_back and start_position[1] > 280: continue
#                             # draw red contours on start location
#                             if bout_starting or counter == 35:
#                                 _, contours, _ = cv2.findContours(model_mouse_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#                                 counter += 1
#                             # if start_position[1] < 300:
#                             cv2.drawContours(self.homing_arena, contours, 0, (255, 100, 100), thickness=-1)
#                             # cv2.drawContours(self.homing_arena, contours, 0, (0, 0, 0), thickness=1)
#                             # get distance traveled
#                             dist_from_start = np.sqrt((start_position[0] - float(body_location[0])) ** 2 + (start_position[1] - float(body_location[1])) ** 2)
#                             if body_location[1] > 360: dist_from_start = 150
#                             dist_to_make_red = 150
#                             prev_homing_color = np.array([.98, .98, .98]) + np.max((50, dist_to_make_red - dist_from_start)) / dist_to_make_red * color_trail
#                             self.homing_arena[model_mouse_mask.astype(bool)] = self.homing_arena[model_mouse_mask.astype(bool)] * prev_homing_color
#                             red_mask[model_mouse_mask.astype(bool)] = 0
#                             red_frame = True
#
#                         else:
#                             speed = smoothed_speed[frame_num - 1] #* 1.7
#                             speed_color_light, speed_color_dark = speed_colors(speed, red = False) #True
#                             if bout_starting: speed_color_light, speed_color_dark = speed_color_light*0, speed_color_dark*0
#                             # emphasize the back
#                             if from_the_back:
#                                 if start_position[1] > 280:
#                                     speed_color_light, speed_color_dark = speed_color_light**.3, speed_color_dark**.3
#                             speed_color_light, speed_color_dark = speed_color_light ** .4, speed_color_dark ** .4
#                             # apply color to arena image
#                             print(counter*speed)
#                             if counter*speed > 80: # start it light
#                                 self.homing_arena[model_mouse_mask.astype(bool)] = self.homing_arena[model_mouse_mask.astype(bool)] * speed_color_dark
#                                 counter = 0
#                             else: # end it dark
#                                 self.homing_arena[model_mouse_mask.astype(bool)] = self.homing_arena[model_mouse_mask.astype(bool)] * speed_color_light
#                                 counter += 1
#                 else: # non homing frames
#                     # draw in light gray
#                     prev_homing_color = np.array([.99, .99, .99])
#                     # if red_frame: prev_homing_color = np.array([.98, .98, .98])
#                     self.homing_arena[model_mouse_mask_gray.astype(bool)] = self.homing_arena[model_mouse_mask_gray.astype(bool)] * prev_homing_color
#                 for repeat in range(red_frame*9+1):
#                     # display image
#                     cv2.imshow(self.save_folder + 'homings', self.homing_arena[:,:,::-1])
#                     # write frame
#                     if make_vid: video_clip.write(self.homing_arena[:,:,::-1])
#                 # press q to quit
#                 if cv2.waitKey(1) & 0xFF == ord('q'): break
#     # apply the contours and border to the image and save the image
#     if anti: image_name = self.videoname + ' anti-homings image.tif'
#     else: image_name = self.videoname + ' homings image.tif'
#     scipy.misc.imsave(os.path.join(self.save_folder, image_name), self.homing_arena)
#     # end video
#     if make_vid: video_clip.release()


def get_homing_groups(self, minimum_distance, minimum_duration, minimum_shelter_distance):
    '''     group up locomotor bouts into continuous homing movements       '''
    # initialize variables
    thresholds_passed = self.thresholds_passed#[0, :] * (1 - self.thresholds_passed[1, :])
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
        elif k and self.trial_num==1 and (idx - group_length) < 300: # can't be at beginning
            thresholds_passed[idx - group_length: idx] = False
        elif k:
            group_idx[idx - group_length] = group_length
            # group_idx[idx - 1] = 2000*(np.mean(self.absolute_speeds[idx - group_length:idx]) / group_length**2)

    # pass back to the object
    self.thresholds_passed = thresholds_passed
    self.group_idx = group_idx #+ self.group_idx

