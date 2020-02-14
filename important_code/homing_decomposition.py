import cv2
import numpy as np
import itertools
import scipy.signal
from scipy.ndimage import gaussian_filter1d
import os
import pickle
from helper_code.processing_funcs import extract_variables, convolve
from important_code.escape_visualization import make_model_mouse_mask

def decompose_homings(self):
    '''     decompose homings into piecewise linear bouts       '''
    '''    note: minimum distance traditionally 50, 200 to separate bouts    '''

    # initialize arrays
    skip_frames = 300
    model_mouse_mask_initial = self.decompose_arena[:, :, 0] * 0

    # create local variables for the current epoch
    # extract_variables(self, new_thresholds = False)

    # set what this mouse considers a high speed
    speeds = [3.6, 7.2, 28.8] #[.5, 1, 4]
    high_speed, medium_speed, low_speed = speeds[2], speeds[1], speeds[0]

    # set distance parameters
    minimum_distance = 50
    max_shelter_proximity = 50
    critical_turn = 25
    back_butt_dist = 18

    # do convolutions to get current, future, far future, and past speeds w.r.t the shelter
    now_speed = convolve(self.goal_speeds, 12, -1, time='current')
    past_speed = convolve(self.goal_speeds, 30, -1, time='past')
    present_angular_speed = convolve(abs(self.angular_speed), 8, +1, time='current')

    # get the homing epochs
    thresholds_passed = self.trial_groups
    # get rid of segments just prior to stimulus
    thresholds_passed[-(skip_frames + 10):-skip_frames] = False

    # get vectors from the first phase of all homings
    # print(self.stim_frame)
    group_idx, distance_from_start, end_idx = multi_phase_phinder(self, thresholds_passed, minimum_distance, max_shelter_proximity, critical_turn)
    thresholds_passed_idx = np.where(group_idx)[0]

    #initialize counter
    group_counter = 0

    # loop over each frame that passed the threshold
    for idx in thresholds_passed_idx:
        # get frame number
        frame_num = self.frame_nums[idx]

        # if frame_num >= stim_frame:
        #     continue

        # draw ellipses representing model mouse
        model_mouse_mask, _, _ = make_model_mouse_mask(self.coordinates, frame_num, model_mouse_mask_initial, scale=back_butt_dist)

        # determine color and darkness
        speed_color = np.array([210, 230, 200])  # blue
        multiplier = 10
        save_multiplier = 10

        # modify color and darkness for stimulus driven escapes
        if frame_num >= self.stim_frame - 30:
            speed_color = np.array([100, 100, 100])
            save_multiplier = 2

        # determine arrow color
        if group_idx[idx]:
            if frame_num < self.stim_frame - 60:
                line_color = [.6 * x for x in [210, 230, 200]]
                save_line_color = line_color
            else:
                line_color = [100, 100, 100]
                save_line_color = [10, 10, 10]

            # compute procedural vector
            origin = np.array([int(self.coordinates['center_body_location'][0][frame_num - 1]), int(self.coordinates['center_body_location'][1][frame_num - 1])])
            endpoint_index = int(group_idx[idx]) #int(frame_num - 1 + group_idx[idx] - end_idx[int(group_idx[idx])])
            endpoint = np.array([int(self.coordinates['center_body_location'][0][endpoint_index]), int(self.coordinates['center_body_location'][1][endpoint_index]) ])

            vector_tip = (origin + 20 * (endpoint - origin) / np.sqrt(np.sum( (endpoint - origin)**2)) ).astype(int)

            cv2.arrowedLine(self.decompose_arena, tuple(origin), tuple(vector_tip), line_color, thickness=1, tipLength=.2)
            # cv2.arrowedLine(save_exploration_arena, tuple(origin), tuple(vector_tip), save_line_color, thickness=2, tipLength=.2)

            group_counter += 1

        # create color multiplier to modify image
        color_multiplier = 1 - (1 - speed_color / [255, 255, 255]) / (np.mean(1 - speed_color / [255, 255, 255]) * multiplier)
        save_color_multiplier = 1 - (1 - speed_color / [255, 255, 255]) / (np.mean(1 - speed_color / [255, 255, 255]) * save_multiplier)

        # apply color to arena image
        self.decompose_arena[model_mouse_mask.astype(bool)] = self.decompose_arena[model_mouse_mask.astype(bool)] * color_multiplier
        # save_exploration_arena[model_mouse_mask.astype(bool)] = save_exploration_arena[model_mouse_mask.astype(bool)] * save_color_multiplier

        # display image
        cv2.imshow(self.save_folder + 'movements', self.decompose_arena)

        # press q to quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # apply the contours and border to the image and save the image
    scipy.misc.imsave(os.path.join(self.save_folder, self.videoname + ' lunging image.tif'), self.decompose_arena)

    if not 'start_index' in self.coordinates:
        self.coordinates['start_index'] = []
        self.coordinates['end_index'] = []
    if len(self.coordinates['start_index']) < self.stim_frame:
        self.coordinates['start_index'] = np.append(self.coordinates['start_index'], group_idx)
        self.coordinates['end_index'] = np.append(self.coordinates['end_index'], end_idx)

    with open(self.processed_coordinates_file, "wb") as dill_file:
        pickle.dump(self.coordinates, dill_file)




def multi_phase_phinder(self, thresholds_passed, minimum_distance_spont, max_shelter_proximity, critical_turn):

    # thresholds_passed, minimum_distance_spont, max_shelter_proximity, body_angles, distance_from_shelter,
    #                         x_location_butt, y_location_butt, x_location_face, y_location_face, critical_turn, first_frame,
    #                             distance_from_obstacle, angles_from_obstacle, stim_on, absolute_speeds, shelter_location_front):

    groups = []; idx = 0
    vectors  = [];
    group_idx = np.zeros(len(thresholds_passed))
    end_idx = np.zeros(len(thresholds_passed))
    distance_from_start = np.zeros(len(thresholds_passed))
    stim_on = self.frame_nums > self.stim_frame
    first_frame = self.previous_stim_frame + self.skip_frames
    # print(self.previous_stim_frame)
    # print(self.skip_frames)
    # print(first_frame)
    # print('')

    for k, g in itertools.groupby(thresholds_passed):
        groups.append(list(g))
        group_length = len(groups[len(groups)-1]);
        idx += group_length
        # for each bout, get the relevant vectors
        if k:
            # get later phases of the escape
            start_index = idx - group_length

            # trim the end off it moves away from shelter
            if stim_on[start_index]:
                closest_to_shelter = np.argmin(self.distance_from_shelter_front[idx - group_length: idx])
                if self.distance_from_shelter_front[idx - group_length + closest_to_shelter] < 50:
                    idx -= (group_length - (closest_to_shelter + 1) )
                    group_length -= (group_length - (closest_to_shelter + 1) )

                #TEMPORARY: IF START TOO CLOSE, THEN SKIP
                if self.y_location_face[np.max(np.where(~stim_on[:start_index]))+1] < 23:
                    continue

            # get the distance travelled so far during the bout
            distance_from_start[idx - group_length: idx] = np.sqrt((self.x_location_butt[idx - group_length: idx] - self.x_location_butt[idx - group_length]) ** 2 + \
                                                                   (self.y_location_butt[idx - group_length: idx] - self.y_location_butt[idx - group_length]) ** 2)

            angle_for_comparison = 0; end_of_turn = 0; start_of_turn = 0
            if stim_on[start_index]: minimum_distance = minimum_distance_spont - 10
            else: minimum_distance = minimum_distance_spont# + 10

            while True:
                # get the cumulative distance traveled
                distance_traveled = np.sqrt((self.x_location_butt[start_index:idx] - self.x_location_butt[start_index]) ** 2
                                            + (self.y_location_butt[start_index:idx] - self.y_location_butt[start_index]) ** 2)

                # has the minimum distance been traveled
                traveled_far_enough = np.where(distance_traveled > minimum_distance)[0]
                if not traveled_far_enough.size: break

                # now check for more phases: find the cumulative turn angle since the start of the bout
                angle_for_comparison_index = np.max((start_index + traveled_far_enough[0], start_index + end_of_turn - start_of_turn))
                angle_for_comparison = self.body_angles[angle_for_comparison_index]

                angle_turned = np.zeros((idx-start_index, 2))
                if stim_on[start_index]:
                    angle_turned[:,0] = (self.body_angles[start_index:idx] - angle_for_comparison) #body_angles[start_index + traveled_far_enough[0]])
                    try:
                        angle_turned[:, 1] = (self.body_angles[start_index:idx] - self.body_angles[start_index - 15:idx - 15]) #15
                    except: break
                else:
                    angle_turned[:,0] = (gaussian_filter1d(self.body_angles[start_index:idx],3) - angle_for_comparison)
                    try: angle_turned[:, 1] = (gaussian_filter1d(self.body_angles[start_index:idx],3) - gaussian_filter1d(self.body_angles[start_index - 15:idx - 15],3))
                    except: break

                angle_turned[angle_turned > 180] = 360 - angle_turned[angle_turned > 180]
                angle_turned[angle_turned < -180] = 360 + angle_turned[angle_turned < -180]

                # not including the beginning
                # angle_turned[:traveled_far_enough[0], 0] = 0
                zero_index = np.max((end_of_turn-start_of_turn, traveled_far_enough[0]))
                angle_turned[:zero_index, 0] = 0
                angle_turned[:traveled_far_enough[0]+15, 1] = 0
                max_turn_idx = np.argmax(abs(angle_turned), 1)
                max_angle_turned = angle_turned[np.arange(0, idx - start_index), max_turn_idx]

                # give a bonus for being near obstacle?.. #RECENTLY REMOVED
                # near_obstacle = distance_from_obstacle[start_index:idx] < 25
                # above_obstacle = angles_from_obstacle[start_index:idx] == -90
                # max_angle_turned[(near_obstacle * above_obstacle).astype(bool)] = max_angle_turned[(near_obstacle * above_obstacle).astype(bool)] * 4 #1.75


                # get the indices of critically large turns
                critically_turned = np.where(abs(max_angle_turned) > critical_turn)[0]

                if critically_turned.size:

                    if stim_on[start_index]:
                        # check if there was a pause in the meanwhile!
                        bc = scipy.signal.boxcar(10) / 10
                        pausing = scipy.signal.convolve(self.absolute_speeds[start_index:idx], bc, mode='same') < 2 # was 3!
                        pausing[:15] = False; pausing[critically_turned[0]:] = False
                        # print(pausing)
                        take_a_break = np.where(pausing)[0]
                        # print(take_a_break)
                    else: take_a_break = np.array([])
                    if take_a_break.size:
                        start_of_turn = take_a_break[0]
                        end_of_turn = take_a_break[-1]

                    else:
                        # get the angular speed
                        angular_speed = np.diff(angle_turned[:critically_turned[0],0])
                        angular_speed_after = np.diff(angle_turned[critically_turned[0]:,0])

                        # find the beginning of the turn
                        try: start_of_turn = np.max(np.where( ((angular_speed*np.sign(angle_turned[critically_turned[0],0]))<-.05) )) + 1
                        except: start_of_turn = np.max(np.where( ((angular_speed*np.sign(angle_turned[critically_turned[0],0]))<=0) )) + 1

                        # find the end of the turn
                        stops_turning = np.where(np.sign(angular_speed_after) != np.sign(angle_turned[critically_turned[0],0]))[0]
                        if stops_turning.size: end_of_turn = stops_turning[0] + critically_turned[0]
                        else: end_of_turn = critically_turned[0]

                    # break if getting too close to shelter
                    if (self.distance_from_shelter[start_index + start_of_turn] < max_shelter_proximity):
                        group_idx[start_index] = idx + first_frame
                        end_idx[idx] = start_index + first_frame
                        break
                    else:
                        group_idx[start_index] = start_index + start_of_turn - 1 + first_frame
                        end_idx[start_index + start_of_turn - 1] = start_index + first_frame
                        start_index += start_of_turn
                else:
                    group_idx[start_index] = idx - 1 + first_frame
                    end_idx[idx - 1] = start_index + first_frame
                    break

    return group_idx, distance_from_start, end_idx


