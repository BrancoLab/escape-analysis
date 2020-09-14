import glob
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import copy
import scipy.signal
import warnings
import cv2
import math
warnings.simplefilter("ignore")


def extract_dlc(dlc_config_settings, video_path):
    '''
    .....................................EXTRACT COORDINATES WITH DLC.....................................
    '''
    # read the freshly saved coordinates file
    coordinates_file = glob.glob(os.path.dirname(video_path) + '\\*.h5')[0]
    DLC_network = os.path.basename(coordinates_file)
    DLC_network = DLC_network[DLC_network.find('Deep'):-3]
    body_parts = dlc_config_settings['body parts']

    DLC_dataframe = pd.read_hdf(coordinates_file)

    # plot body part positions over time
    coordinates = {}

    # For each body part, get out the coordinates
    for i, body_part in enumerate(body_parts):
        # initialize coordinates
        coordinates[body_part] = np.zeros((3, len(DLC_dataframe[DLC_network][body_part]['x'].values)))

        # extract coordinates
        for j, axis in enumerate(['x', 'y']):
            coordinates[body_part][j] = DLC_dataframe[DLC_network][body_part][axis].values

        coordinates[body_part][2] = DLC_dataframe[DLC_network][body_part]['likelihood'].values

    return coordinates



def filter_and_transform_dlc(dlc_config_settings, coordinates_copy, x_offset, y_offset, registration, plot = True, filter_kernel = 7, confidence = .999, max_error = 60):
    '''
    .....................................FILTER AND TRANSFORM COORDINATES FROM DLC.....................................
    '''
    # change variable name so this can be rurun for testing
    coordinates = copy.deepcopy(coordinates_copy)

    # get the list of body parts
    body_parts = dlc_config_settings['body parts']

    # fisheye correct the coordinates
    inverse_fisheye_maps = np.load(dlc_config_settings['inverse_fisheye_map_location'])

    # array of all body parts, axis x body part x frame
    all_body_parts = np.zeros((2, len(body_parts), coordinates[body_parts[0]].shape[1]))

    # loop across body parts to remove points with low confidence and to median filter
    for bp, body_part in enumerate(body_parts):
        # loop across axes
        for i in range(2):

            # remove coordinates with low confidence
            coordinates[body_part][i][coordinates[body_part][2] < confidence] = np.nan

            # interpolate nan values
            coordinates[body_part][i] = np.array(pd.Series(coordinates[body_part][i]).interpolate())
            coordinates[body_part][i] = np.array(pd.Series(coordinates[body_part][i]).fillna(method='bfill'))
            coordinates[body_part][i] = np.array(pd.Series(coordinates[body_part][i]).fillna(method='ffill'))

            # median filter coordinates (replace nans with infinity first)
            coordinates[body_part][i] = scipy.signal.medfilt(coordinates[body_part][i], filter_kernel)

            # remove coordinates with low confidence
            coordinates[body_part][i][coordinates[body_part][2] < confidence] = np.nan

        # put all together
        all_body_parts[:, bp, :] = coordinates[body_part][0:2]

    # Get the median position of body parts in all frames (unless many points are uncertain)
    median_positions = np.nanmedian(all_body_parts, axis=1)
    num_of_nans = np.sum(np.isnan(all_body_parts[0, :, :]), 0)
    no_median = num_of_nans > 7

    # Set up plot, if applicable
    if plot:
        fig = plt.figure('DLC coordinates', figsize=(20, 7))
        ax = fig.add_subplot(111)

    # loop across body parts to transform points to CCB
    for bp, body_part in enumerate(coordinates):
        # get distance from median position for all frames
        distance_from_median_position = np.sqrt(
            (coordinates[body_part][0] - median_positions[0, :]) ** 2 + (coordinates[body_part][1] - median_positions[1, :]) ** 2)

        # loop across axes
        for i in range(2):
            # remove coordinates far from rest of body parts
            coordinates[body_part][i][distance_from_median_position > max_error] = np.nan

            # remove coordinates where many body parts are uncertain
            coordinates[body_part][i][no_median] = np.nan

            # correct any negative coordinates
            coordinates[body_part][i][(coordinates[body_part][i] < 0)] = 0

        # get index of uncertain points
        nan_index = np.isnan(coordinates[body_part][i])

        # initialize transformed points array
        transformed_points = np.zeros(coordinates[body_part].shape)

        # loop across axes
        for i in range(2):

            # convert original coordinates to registered coordinates
            transformed_points[i] = inverse_fisheye_maps[coordinates[body_part][1].astype(np.uint16) + y_offset,
                                                         coordinates[body_part][0].astype(np.uint16) + x_offset, i] \
                                                           - (x_offset*(1-i) + y_offset*(i))

        # affine transform to match model arena
        transformed_points = np.matmul(np.append(registration[0], np.zeros((1, 3)), 0),
                                       np.concatenate((transformed_points[0:1], transformed_points[1:2],
                                                       np.ones((1, len(transformed_points[0])))), 0))

        # fill in the coordinates array with the transformed points
        coordinates[body_part][0] = transformed_points[0, :]
        coordinates[body_part][1] = transformed_points[1, :]

        # fill in the coordinates array with the uncertain points as nan
        coordinates[body_part][0][nan_index] = np.nan
        coordinates[body_part][1][nan_index] = np.nan

        # plot, if applicable
        if plot:
            ax.plot(coordinates[body_part][0][18710:18720]) #**2 +  coordinates[body_part][1][18700:18800]**2)


    if plot:
        ax.legend(body_parts)
        plt.pause(2)
        # plt.close('all')

    return coordinates


def compute_pose_from_dlc(body_parts, coordinates, shelter_location, width, height, subgoal_locations):

    # array of all body parts, axis x body part x frame
    all_body_parts = np.zeros((2, len(body_parts), coordinates[body_parts[0]].shape[1]))
    for i, body_part in enumerate(body_parts):
        all_body_parts[:, i, :] = coordinates[body_part][0:2]

    # make sure values are within the proper range
    all_body_parts[all_body_parts >= width] = width - 1

    # compute particular body part locations by taking the nan median of several points
    coordinates['snout_location'] = np.nanmean(all_body_parts[:, 0:3, :], axis=1)
    coordinates['head_location'] = np.nanmean(all_body_parts[:, 0:6, :], axis=1)
    coordinates['neck_location'] = np.nanmean(all_body_parts[:, 3:6, :], axis=1)
    coordinates['center_location'] = np.nanmean(all_body_parts[:, :, :], axis=1)
    coordinates['nack_location'] = np.nanmean(all_body_parts[:, 3:9, :], axis=1)
    coordinates['front_location'] = np.nanmean(all_body_parts[:, 0:9, :], axis=1)
    coordinates['shoulder_location'] = np.nanmean(all_body_parts[:, 6:9, :], axis=1)
    coordinates['center_body_location'] = np.nanmean(all_body_parts[:, 6:12, :], axis=1)
    coordinates['butty_location'] = np.nanmean(all_body_parts[:, 6:, :], axis=1)
    coordinates['butt_location'] = np.nanmean(all_body_parts[:, 9:, :], axis=1)

    # coordinates['front_shelter_location'] = (np.nanmean(all_body_parts[:, 0:9, :], axis=1).T - shelter_location).T
    # coordinates['back_shelter_location'] = (np.nanmean(all_body_parts[:, 6:, :], axis=1).T - shelter_location).T


    # compute speed
    delta_position = np.concatenate( ( np.zeros((2,1)), np.diff(coordinates['center_location']) ) , axis = 1)
    coordinates['speed'] = np.sqrt(delta_position[0,:]**2 + delta_position[1,:]**2)

    # compute distance from shelter
    coordinates['distance_from_shelter'] = np.sqrt((coordinates['center_location'][0] - shelter_location[0] * width / 1000) ** 2 +
                                                   (coordinates['center_location'][1] - shelter_location[1] * height / 1000) ** 2)

    # compute speed w.r.t. shelter
    coordinates['speed_toward_shelter'] = np.concatenate( ([0], np.diff(coordinates['distance_from_shelter'])))

    # linearly interpolate any remaining nan values
    locations = ['speed', 'distance_from_shelter', 'speed_toward_shelter', 'head_location', 'butt_location', 'snout_location',
                'neck_location', 'shoulder_location', 'nack_location', 'center_body_location', 'center_location', 'front_location', 'butty_location']
                 # 'front_shelter_location', 'back_shelter_location']

    for loc_num, loc in enumerate(locations):
        if loc_num < 3:
            coordinates[loc] = np.array(pd.Series(coordinates[loc]).interpolate())
            coordinates[loc] = np.array(pd.Series(coordinates[loc]).fillna(method='bfill'))
            coordinates[loc] = np.array(pd.Series(coordinates[loc]).fillna(method='ffill'))
        else:
            for i in [0,1]:
                coordinates[loc][i] = np.array(pd.Series(coordinates[loc][i]).interpolate())
                coordinates[loc][i] = np.array(pd.Series(coordinates[loc][i]).fillna(method='bfill'))
                coordinates[loc][i] = np.array(pd.Series(coordinates[loc][i]).fillna(method='ffill'))

    # compute angles
    coordinates['body_angle'] = np.angle((coordinates['shoulder_location'][0] - coordinates['butty_location'][0]) + (-coordinates['shoulder_location'][1] + coordinates['butty_location'][1]) * 1j, deg=True)
    coordinates['shoulder_angle'] = np.angle((coordinates['head_location'][0] - coordinates['center_body_location'][0]) + (-coordinates['head_location'][1] + coordinates['center_body_location'][1]) * 1j, deg=True)
    coordinates['head_angle'] = np.angle((coordinates['snout_location'][0] - coordinates['neck_location'][0]) + (-coordinates['snout_location'][1] + coordinates['neck_location'][1]) * 1j, deg=True)
    coordinates['neck_angle'] = np.angle((coordinates['head_location'][0] - coordinates['shoulder_location'][0]) + (-coordinates['head_location'][1] + coordinates['shoulder_location'][1]) * 1j, deg=True)
    coordinates['nack_angle'] = np.angle((coordinates['head_location'][0] - coordinates['nack_location'][0]) + (-coordinates['head_location'][1] + coordinates['nack_location'][1]) * 1j, deg=True)

    coordinates['shelter_angle'] = np.degrees( np.arctan2(coordinates['front_location'][1] - coordinates['butty_location'][1], coordinates['front_location'][0] - coordinates['butty_location'][0]) -\
             np.arctan2(shelter_location[1] * height / 1000 - coordinates['butty_location'][1], shelter_location[0] * width / 1000 - coordinates['butty_location'][0]) )
    coordinates['shelter_angle'][coordinates['shelter_angle'] < -180] = coordinates['shelter_angle'][coordinates['shelter_angle'] < -180] + 360
    coordinates['shelter_angle'][coordinates['shelter_angle'] > 180] = coordinates['shelter_angle'][coordinates['shelter_angle'] > 180] - 360

    # get angular speeds
    coordinates['angular_speed_shelter'] = abs(np.concatenate(([0], np.diff(coordinates['shelter_angle']))))
    coordinates['angular_speed_shelter'][coordinates['angular_speed_shelter'] > 180] = \
        360 - coordinates['angular_speed_shelter'][coordinates['angular_speed_shelter']> 180]
    coordinates['angular_speed_shelter'][coordinates['angular_speed_shelter'] > 12] = 12
    coordinates['angular_speed_shelter'][coordinates['angular_speed_shelter'] < -12] = -12

    coordinates['angular_speed'] = abs(np.concatenate(([0], np.diff(coordinates['body_angle']))))
    coordinates['angular_speed'][coordinates['angular_speed'] > 180] = \
        360 - coordinates['angular_speed'][coordinates['angular_speed']> 180]
    coordinates['angular_speed'][coordinates['angular_speed'] > 12] = 12
    coordinates['angular_speed'][coordinates['angular_speed'] < -12] = -12

    # correct locations out of frame
    locations = ['head_location', 'butt_location', 'snout_location', 'neck_location', 'shoulder_location', 'nack_location',
                  'center_body_location', 'center_location', 'front_location', 'butty_location']
    for loc in locations:
        coordinates[loc][0][coordinates[loc][0] >= width ] = width - 1
        coordinates[loc][1][coordinates[loc][1] >= height] = height - 1
        coordinates[loc][0][coordinates[loc][0] < 0] = 0
        coordinates[loc][1][coordinates[loc][1] < 0] = 0

    # instead of angle to shelter, take min of angle to subgoals
    coordinates['subgoal_angle'] = np.zeros((len(subgoal_locations['sub-goals']) + 1, len(coordinates['speed'])))
    coordinates['subgoal_angle'][0,:] = coordinates['shelter_angle']

    # instead of  distance to shelter, take min of distance to shelter and distance to subgoals
    coordinates['distance_from_subgoal'] = np.zeros((len(subgoal_locations['sub-goals']) + 1, len(coordinates['speed'])))

    # instead of speed to shelter, take max of speed to shelter and speed to subgoals
    coordinates['speed_toward_subgoal'] = np.zeros((len(subgoal_locations['sub-goals'])+1, len(coordinates['speed']) ))
    coordinates['speed_toward_subgoal'][0,:] = coordinates['speed_toward_shelter']
    subgoal_bound = [ (int(x * width / 1000), int(y* height / 1000)) for x, y in subgoal_locations['region'] ]
    # infomark_bound = [(int(x * width / 1000), int(y * height / 1000)) for x, y in infomark_location]

    # compute distance from subgoal
    subgoal_mask = np.zeros((height, width))
    cv2.drawContours(subgoal_mask, [np.array(subgoal_bound)], 0, 100, -1)
    subgoal_mask = subgoal_mask.astype(bool)

    # infomark_mask = np.zeros((height, width))
    # cv2.drawContours(infomark_mask, [np.array(infomark_bound)], 0, 100, -1)
    # infomark_mask = infomark_mask.astype(bool)

    # get the x and y locations to loop through
    x_location = coordinates['front_location'][0].astype(np.uint16)
    y_location = coordinates['front_location'][1].astype(np.uint16)
    x_location[x_location >= width] = width - 1
    y_location[y_location >= height] = height - 1

    for i, sg in enumerate(subgoal_locations['sub-goals']):
        # calculate distance to subgoal
        coordinates['distance_from_subgoal'][i+1, :] = np.sqrt((coordinates['center_location'][0] - sg[0] * width / 1000) ** 2 +
                                                   (coordinates['center_location'][1] - sg[1] * height / 1000) ** 2)

        # compute valid locations to go to subgoal
        within_subgoal_bound = []; within_infomark_bound = [];
        for x, y in zip(x_location, y_location):
            within_subgoal_bound.append(subgoal_mask[y, x]) #+ infomark_mask[y, x])
            # within_infomark_bound.append(infomark_mask[y, x])

        # compute speed w.r.t. subgoal
        coordinates['speed_toward_subgoal'][i+1, :] = np.concatenate( ([0], np.diff(coordinates['distance_from_subgoal'][i+1, :]))) * within_subgoal_bound

        # compute subgoal angle
        coordinates['subgoal_angle'][i + 1, :] = np.degrees(np.arctan2(coordinates['front_location'][1] - coordinates['butty_location'][1], coordinates['front_location'][0] - coordinates['butty_location'][0]) - \
                                                  np.arctan2(sg[1] * height / 1000 - coordinates['butty_location'][1], sg[0] * width / 1000 - coordinates['butty_location'][0]))
        coordinates['subgoal_angle'][i + 1, coordinates['subgoal_angle'][i + 1, :] < -180] = coordinates['subgoal_angle'][i + 1, coordinates['subgoal_angle'][i + 1, :] < -180] + 360
        coordinates['subgoal_angle'][i + 1, coordinates['subgoal_angle'][i + 1, :] >  180] = coordinates['subgoal_angle'][i + 1, coordinates['subgoal_angle'][i + 1, :] >  180] - 360
        coordinates['subgoal_angle'][i + 1, :] = coordinates['subgoal_angle'][i + 1, :] + ( [(1 - x)*9999 for x in within_subgoal_bound] )

    coordinates['speed_toward_subgoal'] = np.min(coordinates['speed_toward_subgoal'], 0)
    coordinates['distance_from_subgoal'] = np.min(coordinates['distance_from_subgoal'], 0)
    coordinates['subgoal_angle'] = np.nanmin(abs(coordinates['subgoal_angle']), 0)
    coordinates['in_subgoal_bound'] = within_subgoal_bound
    # coordinates['infomark_bound'] = within_infomark_bound

    return coordinates



def compute_anti_pose_from_dlc(coordinates, shelter_location, width, height, subgoal_locations):

    # get anti-shelter
    anti_shelter_location = [shelter_location[0], 1000-shelter_location[1]]
    subgoal_locations['anti region'] = [(0, 500), (0, 1000), (1000, 1000), (1000, 500)]

    # compute distance from shelter
    coordinates['anti_distance_from_shelter'] = np.sqrt((coordinates['center_location'][0] - anti_shelter_location[0] * width / 1000) ** 2 +
                                                   (coordinates['center_location'][1] - anti_shelter_location[1] * height / 1000) ** 2)

    # compute speed w.r.t. shelter
    coordinates['anti_speed_toward_shelter'] = np.concatenate( ([0], np.diff(coordinates['anti_distance_from_shelter'])))

    # compute angle w.r.t. shelter
    coordinates['anti_shelter_angle'] = np.degrees( np.arctan2(coordinates['front_location'][1] - coordinates['butty_location'][1], coordinates['front_location'][0] - coordinates['butty_location'][0]) -\
             np.arctan2(anti_shelter_location[1] * height / 1000 - coordinates['butty_location'][1], anti_shelter_location[0] * width / 1000 - coordinates['butty_location'][0]) )
    coordinates['anti_shelter_angle'][coordinates['anti_shelter_angle'] < -180] = coordinates['anti_shelter_angle'][coordinates['anti_shelter_angle'] < -180] + 360
    coordinates['anti_shelter_angle'][coordinates['anti_shelter_angle'] > 180] = coordinates['anti_shelter_angle'][coordinates['anti_shelter_angle'] > 180] - 360

    # instead of angle to shelter, take min of angle to subgoals
    coordinates['anti_subgoal_angle'] = np.zeros((len(subgoal_locations['sub-goals']) + 1, len(coordinates['speed'])))
    coordinates['anti_subgoal_angle'][0,:] = coordinates['anti_shelter_angle']

    # instead of  distance to shelter, take min of distance to shelter and distance to subgoals
    coordinates['anti_distance_from_subgoal'] = np.zeros((len(subgoal_locations['sub-goals']) + 1, len(coordinates['speed'])))

    # instead of speed to shelter, take max of speed to shelter and speed to subgoals
    coordinates['anti_speed_toward_subgoal'] = np.zeros((len(subgoal_locations['sub-goals'])+1, len(coordinates['speed']) ))
    coordinates['anti_speed_toward_subgoal'][0,:] = coordinates['anti_speed_toward_shelter']
    subgoal_bound = [ (int(x * width / 1000), int(y* height / 1000)) for x, y in subgoal_locations['anti region'] ]

    # compute distance from subgoal
    subgoal_mask = np.zeros((height, width))
    cv2.drawContours(subgoal_mask, [np.array(subgoal_bound)], 0, 100, -1)
    subgoal_mask = subgoal_mask.astype(bool)

    # get the x and y locations to loop through
    x_location = coordinates['front_location'][0].astype(np.uint16)
    y_location = coordinates['front_location'][1].astype(np.uint16)
    x_location[x_location >= width] = width - 1
    y_location[y_location >= height] = height - 1

    for i, sg in enumerate(subgoal_locations['sub-goals']):
        # calculate distance to subgoal
        coordinates['anti_distance_from_subgoal'][i+1, :] = np.sqrt((coordinates['center_location'][0] - sg[0] * width / 1000) ** 2 +
                                                   (coordinates['center_location'][1] - sg[1] * height / 1000) ** 2)

        # compute valid locations to go to subgoal
        within_subgoal_bound = []
        for x, y in zip(x_location, y_location):
            within_subgoal_bound.append(subgoal_mask[y, x])


        # compute speed w.r.t. subgoal
        coordinates['anti_speed_toward_subgoal'][i+1, :] = np.concatenate( ([0], np.diff(coordinates['anti_distance_from_subgoal'][i+1, :]))) * within_subgoal_bound

        # compute subgoal angle
        coordinates['anti_subgoal_angle'][i + 1, :] = np.degrees(np.arctan2(coordinates['front_location'][1] - coordinates['butty_location'][1], coordinates['front_location'][0] - coordinates['butty_location'][0]) - \
                                                  np.arctan2(sg[1] * height / 1000 - coordinates['butty_location'][1], sg[0] * width / 1000 - coordinates['butty_location'][0]))
        coordinates['anti_subgoal_angle'][i + 1, coordinates['anti_subgoal_angle'][i + 1, :] < -180] = coordinates['anti_subgoal_angle'][i + 1, coordinates['anti_subgoal_angle'][i + 1, :] < -180] + 360
        coordinates['anti_subgoal_angle'][i + 1, coordinates['anti_subgoal_angle'][i + 1, :] >  180] = coordinates['anti_subgoal_angle'][i + 1, coordinates['anti_subgoal_angle'][i + 1, :] >  180] - 360
        coordinates['anti_subgoal_angle'][i + 1, :] = coordinates['anti_subgoal_angle'][i + 1, :] + ( [(1 - x)*9999 for x in within_subgoal_bound] )

    coordinates['anti_speed_toward_subgoal'] = np.min(coordinates['anti_speed_toward_subgoal'], 0)
    coordinates['anti_distance_from_subgoal'] = np.min(coordinates['anti_distance_from_subgoal'], 0)
    coordinates['anti_subgoal_angle'] = np.nanmin(abs(coordinates['anti_subgoal_angle']), 0)
    coordinates['anti_in_subgoal_bound'] = within_subgoal_bound

    return coordinates