import cv2
import numpy as np
import scipy.misc
from termcolor import colored
from tqdm import tqdm
import glob
import os


def registration(session, fisheye_map_location):
    '''
    ..........................CONTROL BACKGROUND ACQUISITION AND ARENA REGISTRATION................................
    '''
    get_arena_details(session)

    if not np.array(session['Registration']).shape: # or True:
        print(colored(' - Registering session', 'green'))

        # Get background
        background, width, height = get_background(session['Metadata']['video_file_paths'][0][0],start_frame=1000, avg_over=20)

        # Register arena
        session['Registration'] = register_arena(background, fisheye_map_location, session.x_offset, session.y_offset, session.obstacle_type)
        session['Registration'].append([width, height])
        new_registration = True

    else:
        print(colored(' - Already registered session', 'green'))
        new_registration = False

    return session, new_registration


def get_arena_details(self, experiment = 'experiment'):
    '''     get details on the arena using the experiment name    '''
    try: experiment = self.session['Metadata']['experiment']
    except:
        try: experiment = self['Metadata']['experiment']
        except: pass

    if 'U shaped' in experiment:
        x_offset = 300
        y_offset = 120
        obstacle_type = 'U wall'
        shelter_location = [500, 865] #885

        # points of optimal subgoals
        subgoal_location = {}
        subgoal_location['region'] = [(0, 0),(0, 500), (1000, 500), (1000, 0)] # contour of where sub-goal is relevant
        subgoal_location['sub-goals'] = [(250, 500),(750, 500),(250, 250),(750, 250)] # (x, y) for each sub-goal

    elif '11 shaped' in experiment:
        x_offset = 300
        y_offset = 120
        obstacle_type = '11 wall'
        shelter_location = [500, 865] #885

        # points of optimal subgoals
        subgoal_location = {}
        subgoal_location['region'] = [(0, 0),(0, 500), (1000, 500), (1000, 0)] # contour of where sub-goal is relevant
        subgoal_location['sub-goals'] = [(250, 500),(750, 500),(250, 250),(750, 250)] # (x, y) for each sub-goal

    elif 'void' in experiment and 'side' in experiment:
        x_offset = 300
        y_offset = 120
        obstacle_type = 'center void'
        shelter_location = [500, 865] #885

        # points of optimal subgoals
        subgoal_location = {}
        subgoal_location['region'] = [(0, 0),(0, 500), (1000, 500), (1000, 0)] # contour of where sub-goal is relevant
        subgoal_location['sub-goals'] = [(250, 500),(750, 500)] # (x, y) for each sub-goal

    elif 'void' in experiment:
        x_offset = 300
        y_offset = 120
        obstacle_type = 'void'
        shelter_location = [500, 865] #885

        # points of optimal subgoals
        subgoal_location = {}
        subgoal_location['region'] = [(0, 0),(0, 500), (1000, 500), (1000, 0)] # contour of where sub-goal is relevant
        subgoal_location['sub-goals'] = [(250, 500),(750, 500)] # (x, y) for each sub-goal

    elif 'on side' in experiment:
        x_offset = 300
        y_offset = 120
        obstacle_type = 'center wall'
        shelter_location = [500, 865] #885

        # points of optimal subgoals
        subgoal_location = {}
        subgoal_location['region'] = [(0, 0),(0, 500), (1000, 500), (1000, 0)] # contour of where sub-goal is relevant
        subgoal_location['sub-goals'] = [(250, 500),(750, 500)] # (x, y) for each sub-goal

    elif 'moves' in experiment or 'Square' in experiment:
        x_offset = 300
        y_offset = 120

        if 'left' in experiment:
            obstacle_type = 'side wall 14'
        elif 'right' in experiment:
            obstacle_type = 'side wall 32'

        shelter_location = [450, 800] #885

        # points of optimal subgoals
        subgoal_location = {}
        subgoal_location['region'] = [(0, 0),(0, 500), (1000, 500), (1000, 0)] # contour of where sub-goal is relevant
        subgoal_location['sub-goals'] = [(580, 500),(760, 500)] # (x, y) for each sub-goal -- for now putting both


    elif 'one-sided' in experiment:
        x_offset = 300
        y_offset = 120

        obstacle_type = 'side wall'

        shelter_location = [300, 700]

        # points of optimal subgoals
        subgoal_location = {}
        subgoal_location['region'] = [(0, 0),(0, 420), (1000, 420), (1000, 0)] # contour of where sub-goal is relevant
        subgoal_location['sub-goals'] = [(620, 420)] # (x, y) for each sub-goal

    elif ('Circle' in experiment or 'Barnes' in experiment):
        x_offset = 300
        y_offset = 120
        obstacle_type = 'wall'
        shelter_location = [500, 865] #885

        # points of optimal subgoals
        subgoal_location = {}
        subgoal_location['region'] = [(0, 0),(0, 500), (1000, 500), (1000, 0)] # contour of where sub-goal is relevant
        subgoal_location['sub-goals'] = [(250, 500),(750, 500)] # (x, y) for each sub-goal

        # if 'food' in experiment:
        #     obstacle_type = 'none'

    elif 'day' in experiment:
        x_offset = 300
        y_offset = 120

        obstacle_type = 'longer wall'

        shelter_location = [300, 700]

        # points of optimal subgoals
        subgoal_location = {}
        subgoal_location['region'] = [(0, 0),(0, 450), (1000, 450), (1000, 0)] # contour of where sub-goal is relevant
        subgoal_location['sub-goals'] = [(680, 420)] # (x, y) for each sub-goal


    elif 'Mirror' in experiment:
        x_offset = 300
        y_offset = 120

        obstacle_type = 'other side wall'

        shelter_location = [650, 650]

        # points of optimal subgoals
        subgoal_location = {}
        subgoal_location['region'] = [(0, 0),(0, 420), (1000, 420), (1000, 0)] # contour of where sub-goal is relevant
        subgoal_location['sub-goals'] = [(320, 420)] # (x, y) for each sub-goal


    elif 'Sub sub' in experiment:
        x_offset = 300
        y_offset = 120
        obstacle_type = 'T wall'
        shelter_location = [300, 700]

        # points of optimal subgoals
        subgoal_location = {}
        subgoal_location['region'] = [(0, 0),(0, 420), (1000, 420), (1000, 0)] # contour of where sub-goal is relevant
        subgoal_location['sub-goals'] = [(620, 420), (520, 240)] # (x, y) for each sub-goal

    elif 'Peace' in experiment:
        x_offset = 300
        y_offset = 120
        obstacle_type = 'triangle'
        shelter_location = [571, 583]

        subgoal_location = {}
        # subgoal_location['region'] = [(0, 0),(0, 500), (1000, 500), (1000, 0)] # contour of where sub-goal is relevant
        # subgoal_location['sub-goals'] = [(250, 500),(750, 500)] # (x, y) for each sub-goal
        obstacle_location = []
    elif 'The Room' in experiment:
        x_offset = 300
        y_offset = 120
        obstacle_type = 'room'
        shelter_location = [455, 667]

        subgoal_location = {}
        # subgoal_location['region'] = [(0, 0),(0, 500), (1000, 500), (1000, 0)] # contour of where sub-goal is relevant
        # subgoal_location['sub-goals'] = [(250, 500),(750, 500)] # (x, y) for each sub-goal
        obstacle_location = []
    elif 'Anti Room' in experiment:
        x_offset = 300
        y_offset = 120
        obstacle_type = 'anti-room'
        shelter_location = [452, 452]

        subgoal_location = {}
        # subgoal_location['region'] = [(0, 0),(0, 500), (1000, 500), (1000, 0)] # contour of where sub-goal is relevant
        # subgoal_location['sub-goals'] = [(250, 500),(750, 500)] # (x, y) for each sub-goal
        obstacle_location = []
    else:
        print('arena type not identified')
        x_offset = None
        y_offset = None
        obstacle_type = None
        shelter_location = None
        obstacle_location = None
        subgoal_location = None
        infomark_location = None

    if ('up' in experiment) or ('down' in experiment) or ('moves' in experiment):
        obstacle_changes = True
    else:
        obstacle_changes = False

    self.x_offset, self.y_offset, self.obstacle_type, self.shelter_location, self.subgoal_location, self.obstacle_changes = \
         x_offset, y_offset, obstacle_type, shelter_location, subgoal_location, obstacle_changes



def model_arena(size, trial_type, registration = False, obstacle_type = 'wall', simulate = False, shelter = True, dark = True, shift_wall = False):
    '''
    ..........................GENERATE A MODEL ARENA IMAGE................................
    '''

    # initialize model arena
    if dark: model_arena = np.zeros((1000,1000)).astype(np.uint8)
    else: model_arena = 255 * np.ones((1000,1000)).astype(np.uint8)

    # generate arena topography, depending on arena
    if obstacle_type == 'wall' or obstacle_type == 'none':
        # arena outline
        cv2.circle(model_arena, (500, 500), 460, 245, -1)
        if not dark: cv2.circle(model_arena, (500, 500), 460, 0, 1, lineType = 16)

        if shift_wall:
            # add wall - down
            cv2.rectangle(model_arena, (int(500 - 500 / 2), int(500 - 6 / 2 - 10)), (int(500 + 500 / 2), int(500 + 6 / 2 - 10)), 90, thickness=-1)

        elif trial_type:
            # add wall - down
            cv2.rectangle(model_arena, (int(500 - 500 / 2), int(500 - 6 / 2)), (int(500 + 500 / 2), int(500 + 6 / 2)), 90, thickness=-1)

        elif registration:
            # add wall - up
            cv2.rectangle(model_arena, (int(500 - 554 / 2), int(500 - 6 / 2)), (int(500 + 554 / 2), int(500 + 6 / 2)), 60, thickness=-1)
            # add wall - down
            cv2.rectangle(model_arena, (int(500 - 504 / 2), int(500 - 8 / 2)), (int(500 + 504 / 2), int(500 + 8 / 2)), 0, thickness=-1)
    elif obstacle_type == 'U wall':
        # arena outline
        cv2.circle(model_arena, (500, 500), 460, 245, -1)
        if not dark: cv2.circle(model_arena, (500, 500), 460, 0, 1, lineType=16)

        cv2.rectangle(model_arena, (int(250 - 6 / 2), int(250)), (int(250 + 6 / 2), int(500)), 90, thickness=-1)
        cv2.rectangle(model_arena, (int(750 - 6 / 2), int(250)), (int(750 + 6 / 2), int(500)), 90, thickness=-1)
        if trial_type:
            # add wall - down
            cv2.rectangle(model_arena, (int(500 - 500 / 2), int(500 - 6 / 2)), (int(500 + 500 / 2), int(500 + 6 / 2)), 90, thickness=-1)

    elif obstacle_type == '11 wall':
        # arena outline
        cv2.circle(model_arena, (500, 500), 460, 245, -1)
        if not dark: cv2.circle(model_arena, (500, 500), 460, 0, 1, lineType=16)
        cv2.rectangle(model_arena, (int(250 - 6 / 2), int(250)), (int(250 + 6 / 2), int(500)), 90, thickness=-1)
        cv2.rectangle(model_arena, (int(750 - 6 / 2), int(250)), (int(750 + 6 / 2), int(500)), 90, thickness=-1)

    elif obstacle_type == 'center wall':
        # arena outline
        cv2.circle(model_arena, (500, 500), 460, 245, -1)
        if not dark: cv2.circle(model_arena, (500, 500), 460, 0, 1, lineType=16)
        if trial_type:
            # add wall - down
            cv2.rectangle(model_arena, (int(500 - 6 / 2), int(500 - 500 / 2)), (int(500 + 6 / 2), int(500 + 500 / 2)), 90, thickness=-1)

    elif 'other' in obstacle_type:
        # arena outline
        cv2.rectangle(model_arena, (200, 200), (800, 800), 245, -1)

        # add wall - down
        cv2.rectangle(model_arena, (int(320), int(420 - 6 / 2)), (int(800), int(420 + 6 / 2)), 90, thickness=-1)

    elif obstacle_type == 'side wall':
        # arena outline
        cv2.rectangle(model_arena, (200, 200), (800, 800), 245, -1)
        if not dark: cv2.rectangle(model_arena, (200, 200), (800, 800), 0, 1, lineType=16)

        # add wall - down
        cv2.rectangle(model_arena, (int(200), int(420 - 6 / 2)), (int(200 + 420), int(420 + 6 / 2)), 90, thickness=-1)

    elif obstacle_type == 'side wall 14':
        # arena outline
        if not dark:  cv2.rectangle(model_arena, (100, 100), (900, 900), 0, 2, lineType=16)
        cv2.rectangle(model_arena, (100, 100), (900, 900), 245, -1)

        # add wall
        if not trial_type:
            cv2.rectangle(model_arena, (int(100), int(500 - 6 / 2)), (int(100 + 480 + 180 - 20), int(500 + 6 / 2)), 90, thickness=-1)
        else:
            cv2.rectangle(model_arena, (int(100), int(500 - 6 / 2)), (int(100 + 480 - 20), int(500 + 6 / 2)), 90, thickness=-1)

    elif obstacle_type == 'side wall 32':
        # arena outline
        if not dark:  cv2.rectangle(model_arena, (100, 100), (900, 900), 0, 2, lineType=16)
        cv2.rectangle(model_arena, (100, 100), (900, 900), 245, -1)

        # add wall
        if not trial_type:
            cv2.rectangle(model_arena, (int(100), int(500 - 6 / 2)), (int(100 + 480 + 180 - 20), int(500 + 6 / 2)), 90, thickness=-1)
        else:
            cv2.rectangle(model_arena, (int(100), int(500 - 6 / 2)), (int(100 + 480 - 20), int(500 + 6 / 2)), 90, thickness=-1)

    elif obstacle_type == 'side wall':
        # arena outline
        cv2.rectangle(model_arena, (200, 200), (800, 800), 245, -1)

        # add wall - down
        cv2.rectangle(model_arena, (int(200), int(420 - 6 / 2)), (int(200 + 420), int(420 + 6 / 2)), 90, thickness=-1)

    elif obstacle_type == 'longer wall':
        # arena outline
        cv2.rectangle(model_arena, (200, 200), (800, 800), 245, -1)

        # add wall - down
        cv2.rectangle(model_arena, (int(200), int(440 - 6 / 2)), (int(200 + 470), int(440 + 6 / 2)), 90, thickness=-1)

    elif obstacle_type == 'T wall':
        # arena outline
        cv2.rectangle(model_arena, (200, 200), (800, 800), 245, -1)

        # add wall - down
        cv2.rectangle(model_arena, (int(200), int(440 - 6 / 2)), (int(200 + 480), int(440 + 6 / 2)), 90, thickness=-1)
        cv2.rectangle(model_arena, (int(200+320 - 6/2), int(260)), (int(200 + 320 + 6/2), int(440)), 90, thickness=-1)

    elif obstacle_type == 'void':
        # arena outline
        cv2.circle(model_arena, (500, 500), 460, 245, -1)
        if not dark: cv2.circle(model_arena, (500, 500), 460, 0, 1, lineType=16)
        # add void
        cv2.rectangle(model_arena, (int(500 - 500 / 2), int(500 - 100 / 2)), (int(500 + 500 / 2), int(500 + 100 / 2)), 90, thickness=-1)

    elif obstacle_type == 'triangle':
        # arena outline
        triangle_contours = [np.array([(500, int((1000-750)/2)), (int((1000-866)/2), int((1000-750)/2 + 750)), (int((1000-866)/2 + 866), int((1000-750)/2 + 750))])]
        cv2.drawContours(model_arena, triangle_contours, 0, 245, -1)

        # add walls
        wall_contours_1 = [np.array([(int(500), int((1000 - 750) / 2 + 160)), (int(500), int((1000 - 750) / 2 + 160 + 340))])]
        cv2.drawContours(model_arena, wall_contours_1, 0, 90, thickness=5)

        wall_contours_2 = [np.array([(int((1000-866)/2 + 138.55), int((1000 - 750) / 2 + 670)), (int(500), int((1000 - 750) / 2 + 160 + 340))])]
        cv2.drawContours(model_arena, wall_contours_2, 0, 90, thickness=5)

        wall_contours_3 = [np.array([(int((1000-866)/2 + 866 - 138.55), int((1000 - 750) / 2 + 670)), (int(500), int((1000 - 750) / 2 + 160 + 340))])]
        cv2.drawContours(model_arena, wall_contours_3, 0, 90, thickness=5)

    elif obstacle_type == 'room':
        # arena outline
        cv2.rectangle(model_arena, (int(250/2), int(250/2)), (int(1000-250/2), int(1000-250/2)), 245, thickness=-1)

        # add walls
        wall_contours_1 = [np.array([(int(250/2 + 152.5), int(250/2 + 155)), (int(250/2 + 152.5 + 250), int(250/2 + 155))])]
        cv2.drawContours(model_arena, wall_contours_1, 0, 90, thickness=5)

        wall_contours_2 = [np.array([(int(250/2 + 152.5), int(250/2 + 155)), (int(250/2 + 152.5), int(250/2 + 155 + 340))])]
        cv2.drawContours(model_arena, wall_contours_2, 0, 90, thickness=5)

        wall_contours_3 = [np.array([(int(250/2 + 152.5), int(250/2 + 155 + 340)), (int(250/2 + 152.5 + 355), int(250/2 + 155 + 340))])]
        cv2.drawContours(model_arena, wall_contours_3, 0, 90, thickness=5)

        wall_contours_4 = [np.array([(int(250/2 + 152.5 + 355), int(250/2 + 155 + 340)), (int(250/2 + 152.5 + 355), int(250/2 + 155))])]
        cv2.drawContours(model_arena, wall_contours_4, 0, 90, thickness=5)

    elif obstacle_type == 'anti-room':
        # arena outline
        cv2.rectangle(model_arena, (int(250/2), int(250/2)), (int(1000-250/2), int(1000-250/2)), 245, thickness=-1)

        # add void
        anti_room_contours = [np.array([(int(250/2 + 152.5 + 250), int(250/2 + 155)), (int(250/2 + 152.5), int(250/2 + 155)),
                                        (int(250 / 2 + 152.5), int(250 / 2 + 155 + 340)), (int(250/2 + 152.5 + 355), int(250/2 + 155 + 340)),
                                        (int(250 / 2 + 152.5 + 355), int(250 / 2 + 155)), (int(250/2 + 152.5 + 355 - 120), int(250/2 + 155 + 120)),
                                        (int(250 / 2 + 152.5 + 355 - 120), int(250 / 2 + 155 + 120 + 110)), (int(250 / 2 + 152.5 + 355 - 240), int(250 / 2 + 155 + 120 + 110)),
                                        (int(250 / 2 + 152.5 + 355 - 240), int(250 / 2 + 155 + 120)), (int(250/2 + 152.5 + 250), int(250/2 + 155))])]
        cv2.drawContours(model_arena, anti_room_contours, 0, 90, thickness=-1)

    # measure distance and angle to wall from each point
    if simulate:
        get_obstacle_metrics(model_arena, obstacle_type, size)

    # add shelter
    if shelter:
        alpha = .5
        model_arena_shelter = model_arena.copy()
        shelter_roi = np.zeros(model_arena.shape).astype(np.uint8)
        if obstacle_type == 'wall' or obstacle_type == 'none' or obstacle_type == 'void' or obstacle_type == 'center void' or obstacle_type == 'U wall' or obstacle_type == '11 wall' or obstacle_type == 'center wall':
            cv2.rectangle(model_arena_shelter, (int(500 - 54), int(500 + 385 + 25 - 54)), (int(500 + 54), int(500 + 385 + 25 + 54)), (0, 0, 255),thickness=-1)
            cv2.addWeighted(model_arena, alpha, model_arena_shelter, 1 - alpha, 0, model_arena)
            if not dark: cv2.circle(model_arena, (500, 500), 465, 255, 5, lineType=16)
            shelter_roi = cv2.rectangle(shelter_roi, (int(500 - 50), int(500 + 385 + 25 - 50)),(int(500 + 50), int(500 + 385 + 25 + 50)), 1, thickness=-1)
        elif obstacle_type == 'other side wall':
            cv2.rectangle(model_arena_shelter, (int(800 - 200), int(800 - 100)), (int(800 - 200 + 108), int(800 - 100 - 108)), (0, 0, 255),thickness=-1)
            cv2.addWeighted(model_arena, alpha, model_arena_shelter, 1 - alpha, 0, model_arena)

            shelter_roi = cv2.rectangle(shelter_roi, (int(800 - 200), int(800 - 100)), (int(800 - 200 + 108), int(800 - 100 - 108)), 1, thickness=-1)

        elif obstacle_type == 'side wall' or obstacle_type == 'T wall':
            # cv2.rectangle(model_arena_shelter, (int(200 + 50), int(800 - 50)), (int(200 + 50 + 108), int(800 - 50 - 108)), (0, 0, 255),thickness=-1)
            # cv2.addWeighted(model_arena, alpha, model_arena_shelter, 1 - alpha, 0, model_arena)
            #
            # shelter_roi = cv2.rectangle(shelter_roi, (int(200 + 50), int(800 - 50)), (int(200 + 50 + 108), int(800 - 50 - 108)), 1, thickness=-1)

            cv2.rectangle(model_arena_shelter, (int(200 + 50 + 270), int(800 - 50)), (int(200 + 50 + 108 + 270), int(800 - 50 - 108)), (0, 0, 255), thickness=-1)
            cv2.addWeighted(model_arena, alpha, model_arena_shelter, 1 - alpha, 0, model_arena)

            shelter_roi = cv2.rectangle(shelter_roi, (int(200 + 50 + 270), int(800 - 50)), (int(200 + 50 + 108+ 270), int(800 - 50 - 108)), 1, thickness=-1)

        elif obstacle_type == 'side wall 14' or obstacle_type == 'side wall 32':
            cv2.rectangle(model_arena_shelter, (int(396), int(792)), (int(396 + 108), int(900)), (0, 0, 255), thickness=-1)
            cv2.addWeighted(model_arena, alpha, model_arena_shelter, 1 - alpha, 0, model_arena)

            shelter_roi = cv2.rectangle(shelter_roi, (int(396), int(792)), (int(396 + 108), int(900)), 1, thickness=-1)

        elif obstacle_type == 'longer wall':
            cv2.rectangle(model_arena_shelter, (int(200 + 100), int(800 - 100)), (int(200 + 100 + 108), int(800 - 100 - 108)), (0, 0, 255),thickness=-1)
            cv2.addWeighted(model_arena, alpha, model_arena_shelter, 1 - alpha, 0, model_arena)

            shelter_roi = cv2.rectangle(shelter_roi, (int(200 + 100), int(800 - 100)), (int(200 + 100 + 108), int(800 - 100 - 108)), 1, thickness=-1)
        elif obstacle_type == 'triangle':
            shelter_contours = [np.array([(500 , int((1000 - 750) / 2 + 160 + 340 - 60.6)), ( int((1000 - 866) / 2+ 485.5 ),int((1000-750)/2 + 750 - 220) ),
                                          (int((1000 - 866) / 2 + 576.4), int((1000 - 750) / 2 + 477.8)), (int((1000 - 866) / 2+ 523.9 ), int((1000-750)/2 + 386.9))])]

            cv2.drawContours(model_arena, shelter_contours, 0, (0, 0, 255), thickness=-1)
            cv2.addWeighted(model_arena, alpha, model_arena_shelter, 1 - alpha, 0, model_arena)
            shelter_roi = cv2.drawContours(shelter_roi, shelter_contours, 0, (0, 0, 255), thickness=-1)
        elif obstacle_type == 'room':
            cv2.rectangle(model_arena_shelter, (int(250/2 + 152.5 + 355/2 - 50), int(250/2 + 155 + 340 - 2.5)), (int(250/2 + 152.5 + 355/2 +50), int(250/2 + 155 + 340 - 2.5 - 100)), (0, 0, 255),thickness=-1)
            cv2.addWeighted(model_arena, alpha, model_arena_shelter, 1 - alpha, 0, model_arena)

            shelter_roi = cv2.rectangle(shelter_roi, (int(250/2 + 152.5 + 355/2 - 50), int(250/2 + 155 + 340 - 2.5)), (int(250/2 + 152.5 + 355/2 + 50), int(250/2 + 155 + 340 - 2.5 - 100)), 1, thickness=-1)
        elif obstacle_type == 'anti-room':
            cv2.rectangle(model_arena_shelter, (int(250 / 2 + 152.5 + 355 / 2 - 50 - 2.5), int(250 / 2 + 155 + 340 - 2.5 - 115)),
                          (int(250 / 2 + 152.5 + 355 / 2 + 50 - 2.5), int(250 / 2 + 155 + 340 - 2.5 - 100 - 115)), (0, 0, 255), thickness=-1)
            cv2.addWeighted(model_arena, alpha, model_arena_shelter, 1 - alpha, 0, model_arena)

            shelter_roi = cv2.rectangle(shelter_roi, (int(250 / 2 + 152.5 + 355 / 2 - 50 - 2.5), int(250 / 2 + 155 + 340 - 2.5 - 115)),
                          (int(250 / 2 + 152.5 + 355 / 2 + 50 - 2.5), int(250 / 2 + 155 + 340 - 2.5 - 100 - 115)), 1, thickness=-1)

    # add circular wells along edge
    if registration and (obstacle_type == 'wall' or obstacle_type == 'U wall' or obstacle_type == '11 wall' or obstacle_type == 'center wall'):
        number_of_circles = 20
        for circle_num in range(number_of_circles):
            x_center = int(500+385*np.sin(2*np.pi/number_of_circles*circle_num))
            y_center = int(500-385*np.cos(2*np.pi/number_of_circles*circle_num))
            cv2.circle(model_arena,(x_center,y_center),25,200,-1)

    # resize to the size of the image
    if simulate: model_arena = cv2.resize(model_arena, size, interpolation = cv2.INTER_NEAREST)
    else: model_arena = cv2.resize(model_arena, size)
    if shelter: shelter_roi = cv2.resize(shelter_roi, size, interpolation = cv2.INTER_NEAREST)
    else: shelter_roi = 0

    # add points for the user to click during registration
    if obstacle_type == 'wall' or obstacle_type == 'U wall' or obstacle_type == '11 wall' or obstacle_type == 'none':
        points = np.array(([500, 500 + 460 - 75], [500 - 460 + 75, 500], [500, 500 - 460 + 75], [500 + 460 - 75, 500])) * [size[0] / 1000, size[1] / 1000]
    elif obstacle_type == 'center wall' or obstacle_type == 'center void' or obstacle_type == 'void':
        points = np.array(([500, 500 + 460], [500 - 460, 500], [500, 500 - 460], [500 + 460 , 500])) * [size[0] / 1000, size[1] / 1000]
    elif '14' in obstacle_type or '32' in obstacle_type:
        points = np.array(([100, 100], [100, 900], [900, 900], [900, 100])) * [size[0] / 1000, size[1] / 1000]
    elif 'wall' in obstacle_type or obstacle_type == 'T wall':
        points = np.array(([200, 200], [200, 800], [800, 800], [800, 200])) * [size[0] / 1000, size[1] / 1000]
    elif obstacle_type == 'void':
        points = np.array(([int(500 - 750 / 2 * 92 / 100), int(500 - 188 / 2 * 92 / 100)], [int(500 - 750 / 2 * 92 / 100), int(500 + 188 / 2 * 92 / 100)],
                           [int(500 + 750 / 2 * 92 / 100), int(500 - 188 / 2 * 92 / 100)], [int(500 + 750 / 2 * 92 / 100), int(500 - 188 / 2 * 92 / 100)])) * [size[0] / 1000, size[1] / 1000]
    elif obstacle_type == 'triangle':
        points = np.array(( [500, int((1000-750)/2)], [int((1000-866)/2), int((1000-750)/2 + 750)], [int((1000-866)/2 + 866), int((1000-750)/2 + 750)] )) * [size[0] / 1000, size[1] / 1000]
    elif obstacle_type == 'room' or obstacle_type == 'anti-room':
        points = np.array(( [int(250 / 2), int(250 / 2)], [int(250 / 2), int(1000 - 250 / 2)],
                            [int(1000 - 250 / 2), int(1000 - 250 / 2)], [int(1000 - 250 / 2), int(250 / 2)] )) * [size[0] / 1000, size[1] / 1000]

    # cv2.imshow('arena',model_arena)

    return model_arena, points, shelter_roi


def get_obstacle_metrics(model_arena, obstacle_type, size):
    ''' GENERATE A MAP OF THE DISTANCE AND ANGLE TO THE OBSTACLE'''

    # check if the map already exists
    if not os.path.isfile('.\\arena_files\\distance_arena_' + obstacle_type + '.npy'):
        print('generating obstacle metrics...')
        # resize model arena
        model_arena = cv2.resize(model_arena, size,  interpolation=cv2.INTER_NEAREST)

        # just take the white pixels
        pixels_to_calculate = (model_arena == 255)
        pixel_coordinates = np.where(pixels_to_calculate)

        # get the coordinates of the wall
        obstacle_pixels = (model_arena < 255) * (model_arena > 0)
        obstacle_coordinates = np.where(obstacle_pixels)

        # initialize the arena
        distance_arena = model_arena.copy() * np.nan
        angle_arena = model_arena.copy() * np.nan

        # do the calculation for each pixel
        for x, y in tqdm(zip(pixel_coordinates[1], pixel_coordinates[0])):
            dist_to_obstacle = np.zeros(len(obstacle_coordinates[0]))
            # calculate the distance to the obstacle
            dist_to_obstacle = np.sqrt((x - obstacle_coordinates[1]) ** 2 + (y - obstacle_coordinates[0]) ** 2)
            # calculate the minimum distance to the obstacle
            min_dist, min_dist_pixel = np.min(dist_to_obstacle), np.argmin(dist_to_obstacle)
            distance_arena[y, x] = min_dist
            # calculate the angle to the obstacle
            angle_arena[y, x] = np.angle((obstacle_coordinates[1][min_dist_pixel] - x) + (-obstacle_coordinates[0][min_dist_pixel] + y) * 1j, deg=True)

        # save these maps
        np.save('.\\arena_files\\distance_arena_' + obstacle_type + '.npy', distance_arena)
        np.save('.\\arena_files\\angle_arena_' + obstacle_type + '.npy', angle_arena)


def get_background(vidpath, start_frame = 1000, avg_over = 100):
    '''
    ..........................EXTRACT BACKGROUND BY AVERAGING FRAMES THROGHOUT THE VIDEO................................
    '''
    # initialize the video
    vid = cv2.VideoCapture(vidpath)
    width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
    background = np.zeros((height, width))
    num_frames = int(vid.get(cv2.CAP_PROP_FRAME_COUNT))
    vid.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    # initialize the counters
    every_other = int(num_frames / avg_over)
    j = 0

    # loop through num_frames frames
    for i in tqdm(range(num_frames)):
        # only use every other x frames
        if i % every_other == 0:
            vid.set(cv2.CAP_PROP_POS_FRAMES, i)
            ret, frame = vid.read()
            # store the current frame in as a numpy array
            if ret:
                background += frame[:, :, 0]
                j+=1

    # normalize the background intensity to the number of frames summed
    background = (background / (j)).astype(np.uint8)

    # show the background
    cv2.imshow('background', background)
    cv2.waitKey(10)

    # release the video
    vid.release()

    return background, width, height




def register_arena(background, fisheye_map_location, x_offset, y_offset, obstacle_type = 'wall'):
    '''
    ..........................GUI TO REGISTER ARENAS TO COMMON FRAMEWORK................................
    '''

    # create model arena and background
    arena, arena_points, _ = model_arena(background.shape[::-1], 1, True, obstacle_type)

    # load the fisheye correction
    try:
        maps = np.load(fisheye_map_location)
        map1 = maps[:, :, 0:2]
        map2 = maps[:, :, 2]*0

        background_copy = cv2.copyMakeBorder(background, y_offset, int((map1.shape[0] - background.shape[0]) - y_offset),
                                             x_offset, int((map1.shape[1] - background.shape[1]) - x_offset), cv2.BORDER_CONSTANT, value=0)

        background_copy = cv2.remap(background_copy, map1, map2, interpolation=cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        background_copy = background_copy[y_offset:-int((map1.shape[0] - background.shape[0]) - y_offset),
                          x_offset:-int((map1.shape[1] - background.shape[1]) - x_offset)]
    except:
        background_copy = background.copy()
        fisheye_map_location = ''
        print('fisheye correction not available')

    # initialize clicked points
    blank_arena = arena.copy()
    background_data = [background_copy, np.array(([], [])).T]
    arena_data = [[], np.array(([], [])).T]

    alpha = .7
    colors = [[150, 0, 150], [0, 255, 0]]
    color_array = make_color_array(colors, background.shape)

    make_new_transform_immediately = True

    print('\nSelect reference points on the experimental background image in the indicated order')

    # initialize clicked point arrays
    background_data = [background_copy, np.array(([], [])).T]
    arena_data = [[], np.array(([], [])).T]

    # add 1-2-3-4 markers to model arena
    for i, point in enumerate(arena_points.astype(np.uint32)):
        arena = cv2.circle(arena, (point[0], point[1]), 3, 255, -1)
        arena = cv2.circle(arena, (point[0], point[1]), 4, 0, 1)
        cv2.putText(arena, str(i+1), tuple(point), 0, .55, 150, thickness=2)

        point = np.reshape(point, (1, 2))
        arena_data[1] = np.concatenate((arena_data[1], point))

    # initialize GUI
    cv2.startWindowThread()
    cv2.namedWindow('background')
    cv2.imshow('background', background_copy)
    cv2.namedWindow('model arena')
    cv2.imshow('model arena', arena)

    # create functions to react to clicked points
    cv2.setMouseCallback('background', select_transform_points, background_data)  # Mouse callback

    while True: # take in clicked points until four points are clicked
        cv2.imshow('background',background_copy)

        number_clicked_points = background_data[1].shape[0]
        if number_clicked_points == len(arena_data[1]):
            break
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

    # perform projective transform
    M = cv2.estimateRigidTransform(background_data[1], arena_data[1], False)


    # REGISTER BACKGROUND, BE IT WITH LOADED OR CREATED TRANSFORM
    registered_background = cv2.warpAffine(background_copy, M, background.shape[::-1])

    # --------------------------------------------------
    # overlay images
    # --------------------------------------------------
    alpha = .7
    colors = [[150, 0, 150], [0, 255, 0]]
    color_array = make_color_array(colors, background.shape)

    registered_background_color = (cv2.cvtColor(registered_background, cv2.COLOR_GRAY2RGB)
                             * np.squeeze(color_array[:, :, :, 0])).astype(np.uint8)
    arena_color = (cv2.cvtColor(blank_arena, cv2.COLOR_GRAY2RGB)
                   * np.squeeze(color_array[:, :, :, 1])).astype(np.uint8)

    overlaid_arenas = cv2.addWeighted(registered_background_color, alpha, arena_color, 1 - alpha, 0)
    cv2.namedWindow('registered background')
    cv2.imshow('registered background', overlaid_arenas)

    # --------------------------------------------------
    # initialize GUI for correcting transform
    # --------------------------------------------------
    print('\nLeft click model arena // Right click model background // Press ''y'' when finished')
    print('Purple within arena and green along the boundary represent the model arena')

    update_transform_data = [overlaid_arenas,background_data[1], arena_data[1], M, background_copy]

    # create functions to react to additional clicked points
    cv2.setMouseCallback('registered background', additional_transform_points, update_transform_data)

    # take in clicked points until 'q' is pressed
    initial_number_clicked_points = [update_transform_data[1].shape[0], update_transform_data[2].shape[0]]
    M_initial = M

    while True:
        cv2.imshow('registered background',overlaid_arenas)
        cv2.imshow('background', registered_background)
        number_clicked_points = [update_transform_data[1].shape[0], update_transform_data[2].shape[0]]
        update_transform = False
        k = cv2.waitKey(10)
        # If a left and right point are clicked:
        if number_clicked_points[0]>initial_number_clicked_points[0] and number_clicked_points[1]>initial_number_clicked_points[1]:
            initial_number_clicked_points = number_clicked_points
            # update transform and overlay images
            try:
                M = cv2.estimateRigidTransform(update_transform_data[1], update_transform_data[2],False) #True ~ full transform
                update_transform = True
            except:
                continue
            update_transform = True
        elif  k == ord('r'):
            print('Transformation erased')
            update_transform_data[1] = np.array(([],[])).T
            update_transform_data[2] = np.array(([],[])).T
            initial_number_clicked_points = [3,3]
        elif k == ord('q') or k == ord('y'):
            print('Registration completed')
            break

        if update_transform:
            update_transform_data[3] = M

            registered_background = cv2.warpAffine(background_copy, M, background.shape[::-1])
            registered_background_color = (cv2.cvtColor(registered_background, cv2.COLOR_GRAY2RGB)
                                           * np.squeeze(color_array[:, :, :, 0])).astype(np.uint8)
            overlaid_arenas = cv2.addWeighted(registered_background_color, alpha, arena_color, 1 - alpha, 0)
            update_transform_data[0] = overlaid_arenas

    cv2.destroyAllWindows()
    return [M, update_transform_data[1], update_transform_data[2], fisheye_map_location]


# mouse callback function I
def select_transform_points(event,x,y, flags, data):
    if event == cv2.EVENT_LBUTTONDOWN:

        data[0] = cv2.circle(data[0], (x, y), 3, 255, -1)
        data[0] = cv2.circle(data[0], (x, y), 4, 0, 1)

        clicks = np.reshape(np.array([x, y]),(1,2))
        data[1] = np.concatenate((data[1], clicks))

# mouse callback function II
def additional_transform_points(event,x,y, flags, data):
    if event == cv2.EVENT_RBUTTONDOWN:

        data[0] = cv2.circle(data[0], (x, y), 2, (200,0,0), -1)
        data[0] = cv2.circle(data[0], (x, y), 3, 0, 1)

        M_inverse = cv2.invertAffineTransform(data[3])
        transformed_clicks = np.matmul(np.append(M_inverse,np.zeros((1,3)),0), [x, y, 1])
        # M_inverse = np.linalg.inv(data[3])
        # M_inverse = cv2.findHomography(data[2][:len(data[1])], data[1])
        # transformed_clicks = np.matmul(M_inverse[0], [x, y, 1])

        data[4] = cv2.circle(data[4], (int(transformed_clicks[0]), int(transformed_clicks[1])), 2, (0, 0, 200), -1)
        data[4] = cv2.circle(data[4], (int(transformed_clicks[0]), int(transformed_clicks[1])), 3, 0, 1)

        clicks = np.reshape(transformed_clicks[0:2],(1,2))
        data[1] = np.concatenate((data[1], clicks))
    elif event == cv2.EVENT_LBUTTONDOWN:

        data[0] = cv2.circle(data[0], (x, y), 2, (0,200,200), -1)
        data[0] = cv2.circle(data[0], (x, y), 3, 0, 1)

        clicks = np.reshape(np.array([x, y]),(1,2))
        data[2] = np.concatenate((data[2], clicks))

def make_color_array(colors, image_size):
    color_array = np.zeros((image_size[0],image_size[1], 3, len(colors)))  # create coloring arrays
    for c in range(len(colors)):
        for i in range(3):  # B, G, R
            color_array[:, :, i, c] = np.ones((image_size[0],image_size[1])) * colors[c][i] / sum(
                colors[c])
    return color_array


def register_frame(frame, x_offset, y_offset, registration, map1, map2):
    '''
    ..........................GO FROM A RAW TO A REGISTERED FRAME................................
    '''
    frame_register = frame[:, :, 0]

    frame_register = cv2.copyMakeBorder(frame_register, y_offset,
                                        int((map1.shape[0] - frame.shape[0]) - y_offset),
                                        x_offset, int((map1.shape[1] - frame.shape[1]) - x_offset),
                                        cv2.BORDER_CONSTANT, value=0)
    frame_register = cv2.remap(frame_register, map1, map2, interpolation=cv2.INTER_LINEAR,
                               borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    frame_register = frame_register[y_offset:-int((map1.shape[0] - frame.shape[0]) - y_offset),
                     x_offset:-int((map1.shape[1] - frame.shape[1]) - x_offset)]



    frame = cv2.cvtColor(cv2.warpAffine(frame_register, registration[0], frame.shape[0:2]),cv2.COLOR_GRAY2RGB)

    return frame



def invert_fisheye_map(registration, inverse_fisheye_map_location):
    '''
    ..........................GO FROM A NORMAL FISHEYE MAP TO AN INVERTED ONE.........................
    '''

    if len(registration) == 5:
        pass
    elif os.path.isfile(inverse_fisheye_map_location):
        registration.append(inverse_fisheye_map_location)
    elif len(registration) == 4:  # setup fisheye correction
        print('creating inverse fisheye map')
        inverse_maps = np.load(registration[3])
        # invert maps
        inverse_maps[inverse_maps < 0] = 0

        maps_x_orig = inverse_maps[:, :, 0]
        maps_x_orig[maps_x_orig > 1279] = 1279
        maps_y_orig = inverse_maps[:, :, 1]
        maps_y_orig[maps_y_orig > 1023] = 1023

        map_x = np.ones(inverse_maps.shape[0:2]) * np.nan
        map_y = np.ones(inverse_maps.shape[0:2]) * np.nan
        for x in range(inverse_maps.shape[1]):
            for y in range(inverse_maps.shape[0]):
                map_x[maps_y_orig[y, x], maps_x_orig[y, x]] = x
                map_y[maps_y_orig[y, x], maps_x_orig[y, x]] = y

        grid_x, grid_y = np.mgrid[0:inverse_maps.shape[0], 0:inverse_maps.shape[1]]
        valid_values_x = np.ma.masked_invalid(map_x)
        valid_values_y = np.ma.masked_invalid(map_y)

        valid_idx_x_map_x = grid_x[~valid_values_x.mask]
        valid_idx_y_map_x = grid_y[~valid_values_x.mask]

        valid_idx_x_map_y = grid_x[~valid_values_y.mask]
        valid_idx_y_map_y = grid_y[~valid_values_y.mask]

        map_x_interp = interpolate.griddata((valid_idx_x_map_x, valid_idx_y_map_x), map_x[~valid_values_x.mask],
                                            (grid_x, grid_y), method='linear').astype(np.uint16)
        map_y_interp = interpolate.griddata((valid_idx_x_map_y, valid_idx_y_map_y), map_y[~valid_values_y.mask],
                                            (grid_x, grid_y), method='linear').astype(np.uint16)

        fisheye_maps_interp = np.zeros((map_x_interp.shape[0], map_x_interp.shape[1], 2)).astype(np.uint16)
        fisheye_maps_interp[:, :, 0] = map_x_interp
        fisheye_maps_interp[:, :, 1] = map_y_interp

        np.save('C:\\Drive\\DLC\\transforms\\inverse_fisheye_maps.npy', fisheye_maps_interp)

    return registration