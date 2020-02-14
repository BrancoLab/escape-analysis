import numpy as np
import cv2
import scipy
import copy
import os
import itertools
from tqdm import tqdm
import scipy.signal
import scipy.ndimage
from Utils.registration_funcs import model_arena
from Utils.obstacle_funcs import set_up_speed_colors
import time
from scipy.ndimage import gaussian_filter1d


'''
FIRST, SOME AUXILIARY FUNCTIONS TO HELP WITH STRATEGY PROCESSING
'''


def trial_start(thresholds_passed, minimum_distance, x_location_butt, y_location_butt):

    groups = []; idx = 0
    vectors  = []; group_idx = np.zeros(len(thresholds_passed))
    trial_start_array = np.zeros(len(thresholds_passed))

    for k, g in itertools.groupby(thresholds_passed):
        groups.append(list(g))
        group_length = len(groups[len(groups)-1]);
        idx += group_length
        if k:
            distance_traveled = np.sqrt((x_location_butt[idx - group_length:idx] - x_location_butt[idx - group_length]) ** 2
                                        + (y_location_butt[idx - group_length:idx] - y_location_butt[idx - group_length]) ** 2)
            far_enough_way = np.where( distance_traveled > minimum_distance)[0]
            if far_enough_way.size:
                trial_start_array[idx - group_length: idx - group_length + far_enough_way[0]] = True

    return trial_start_array


def get_color_settings(frame_num, stim_frame, speed_setting, speed_colors, multipliers, exploration_arena, model_mouse_mask, group_counter, blue_duration):

    # determine color
    if frame_num >= stim_frame - 60:
        speed_color = np.array([200, 200, 200])
    else:
        speed_color = speed_colors[speed_setting] + (group_counter < blue_duration )*np.array([230,-4,30]) #[20, 254, 140]
        # speed_color = speed_colors[speed_setting] + (group_counter < 70) * np.array([230, -4, 30])

    # get darkness
    multiplier = multipliers[speed_setting]

    # make this trial's stimulus response more prominent in the saved version
    if frame_num >= stim_frame:
        save_multiplier = multiplier / 5
    else:
        save_multiplier = multiplier

    # create color multiplier to modify image
    color_multiplier = 1 - (1 - speed_color / [255, 255, 255]) / (np.mean(1 - speed_color / [255, 255, 255]) * multiplier)
    save_color_multiplier = 1 - (1 - speed_color / [255, 255, 255]) / (np.mean(1 - speed_color / [255, 255, 255]) * save_multiplier)

    return color_multiplier, save_color_multiplier


def stimulus_onset(stimulus_started, model_mouse_mask, exploration_arena):

    _, contours, _ = cv2.findContours(model_mouse_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    save_exploration_arena = exploration_arena.copy()
    stimulus_started = True
    first_stim_frame = True

    return save_exploration_arena, stimulus_started, contours

def draw_mouse(model_mouse_mask, model_mouse_mask_total, coordinates, frame_num, back_butt_dist, group_length, stim_frame, stimulus_started, group_counter, time_chase):

    # reset mask at start of bout
    model_mouse_mask = model_mouse_mask * (1 - (group_length > 0)) * (1 - (frame_num >= stim_frame and not stimulus_started))

    # extract DLC coordinates from the saved coordinates dictionary
    body_angle = coordinates['body_angle'][frame_num]
    shoulder_angle = coordinates['shoulder_angle'][frame_num]
    shoulder_location = tuple(coordinates['shoulder_location'][:, frame_num].astype(np.uint16))
    body_location = tuple(coordinates['center_body_location'][:, frame_num].astype(np.uint16))

    # draw ellipses representing model mouse
    model_mouse_mask = cv2.ellipse(model_mouse_mask, body_location, (int(back_butt_dist), int(back_butt_dist * .4)), 180 - body_angle, 0, 360, 100, thickness=-1)
    model_mouse_mask = cv2.ellipse(model_mouse_mask, shoulder_location, (int(back_butt_dist * .8), int(back_butt_dist * .26)), 180 - shoulder_angle, 0, 360, 100, thickness=-1)

    # stop shading after n frames
    if group_counter >= time_chase:
        # extract DLC coordinates from the saved coordinates dictionary
        old_body_angle = coordinates['body_angle'][frame_num - time_chase]
        old_shoulder_angle = coordinates['shoulder_angle'][frame_num - time_chase]
        old_shoulder_location = tuple(coordinates['shoulder_location'][:, frame_num - time_chase].astype(np.uint16))
        old_body_location = tuple(coordinates['center_body_location'][:, frame_num - time_chase].astype(np.uint16))

        # erase ellipses representing model mouse
        model_mouse_mask = cv2.ellipse(model_mouse_mask, old_body_location, (int(back_butt_dist), int(back_butt_dist * .4)), 180 - old_body_angle, 0, 360, 0,thickness=-1)
        model_mouse_mask = cv2.ellipse(model_mouse_mask, old_shoulder_location, (int(back_butt_dist * .8), int(back_butt_dist * .26)), 180 - old_shoulder_angle, 0, 360, 0, thickness=-1)

        # draw ellipses representing model mouse
        model_mouse_mask_total = cv2.ellipse(model_mouse_mask_total, body_location, (int(back_butt_dist), int(back_butt_dist * .33)), 180 - body_angle, 0, 360,100, thickness=-1)
        model_mouse_mask_total = cv2.ellipse(model_mouse_mask_total, shoulder_location, (int(back_butt_dist * .8), int(back_butt_dist * .23)),180 - shoulder_angle, 0, 360, 100, thickness=-1)
    else:
        model_mouse_mask_total = model_mouse_mask.copy()

    # return model_mouse_mask.astype(bool), model_mouse_mask_initial, model_mouse_mask_total
    return model_mouse_mask.astype(bool), model_mouse_mask_total




def dilute_shading(exploration_arena, save_exploration_arena, prior_exploration_arena, model_mouse_mask_initial, stimulus_started, end_index, speed_setting):

    speed = end_index #+ .3 * speed_setting
    if speed > .6: speed = .6
    if speed < .3: speed = .3
    # print(speed)

    exploration_arena[model_mouse_mask_initial.astype(bool)] = \
        ((speed * exploration_arena[model_mouse_mask_initial.astype(bool)].astype(np.uint16) +
          (1 - speed) * prior_exploration_arena[model_mouse_mask_initial.astype(bool)].astype(np.uint16))).astype(np.uint8)

    if stimulus_started:
        save_exploration_arena[model_mouse_mask_initial.astype(bool)] = \
            ((speed * save_exploration_arena[model_mouse_mask_initial.astype(bool)].astype(np.uint16) +
              (1 - speed) * prior_exploration_arena[model_mouse_mask_initial.astype(bool)].astype(np.uint16))).astype(np.uint8)
    # else:
    #     save_exploration_arena = None

    return exploration_arena, save_exploration_arena


'''
NOW, THE ACTUAL STRATEGY PROCESSING FUNCTIONS
'''

