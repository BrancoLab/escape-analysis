import cv2
import numpy as np
import scipy.misc
import imageio
import os
import warnings
from helper_code.registration_funcs import model_arena
from helper_code.processing_funcs import speed_colors, register_frame

def visualize_escape(self):
    '''    Generate and save escape video clip and dlc tracking clip     '''

    warnings.filterwarnings("ignore")

    '''     Initialize variables and backgrounds      '''
    # open the behaviour video
    vid = cv2.VideoCapture(self.video_path)
    vid.set(cv2.CAP_PROP_POS_FRAMES, self.start_frame)
    # set up the escape clips for saving
    video = cv2.VideoWriter(os.path.join(self.save_folder, self.videoname + ' vid.mp4'), self.fourcc, self.fps, (self.width, self.height), False)
    video_dlc = cv2.VideoWriter(os.path.join(self.save_folder, self.videoname + ' vid (DLC).mp4'), self.fourcc,self.fps, (self.width , self.height), True)
    # load fisheye mapping
    maps = np.load(self.folders['fisheye_map_location']); map1 = maps[:, :, 0:2]; map2 = maps[:, :, 2] * 0
    # set up model arena
    arena, _, _ = model_arena((self.height, self.width), self.trial_type != 0, registration = False, #self.trial_type > 0
                                        obstacle_type = self.obstacle_type, shelter = ('down' in self.videoname or not 'no shelter' in self.videoname) and not 'food' in self.videoname, dark = self.dark_theme)
    arena = cv2.cvtColor(arena, cv2.COLOR_GRAY2RGB)
    # initialize arenas for mouse mask
    model_mouse_mask_previous = np.zeros(arena.shape[0:2]).astype(np.uint8)
    model_mouse_mask_initial = np.zeros(arena.shape[0:2]).astype(np.uint8)
    # initialize more quantities
    trial_plot = arena.copy()
    frames_past_stimulus = 0
    arrived_at_shelter = False
    count_down = np.inf

    color_trail = np.array([.02, -.02, -.02]) # red
    # color_trail = np.array([-.1, -.1, -.1]) # gray
    # color_trail = np.array([-.02, -.01, .02]) # blue

    contour_color = (255, 100, 100) # red
    # contour_color = (150, 150, 150) # gray
    # contour_color = (100, 200, 250)  # blue
    # when is the stimulus on, and how is the arena shaped
    stim_timing_array, shape, size = initialize_stim_visualization(self.obstacle_type)
    # make a smooth speed trace for coloration
    smoothed_speed = np.concatenate((np.zeros(6 - 1), np.convolve(self.coordinates['speed'], np.ones(12), mode='valid'), np.zeros(6))) / 12
    # set up background arena with previous obstacle dulled out
    if self.trial_type==0 and (('down' in self.videoname) or ('up' in self.videoname and 'void' in self.videoname)) and False:
        former_arena, _, _ = model_arena((self.height, self.width), 1, registration = False,
                                         obstacle_type = self.obstacle_type, shelter = not 'no shelter' in self.videoname and not 'food' in self.videoname, dark = self.dark_theme)
        former_arena = cv2.cvtColor(former_arena, cv2.COLOR_GRAY2RGB)
        discrepancy = ~((arena - former_arena)==0)
        self.arena_with_prev_trials[discrepancy] = 255 #((former_arena[discrepancy] * 1 + arena[discrepancy].astype(float) * 9) / 10).astype(np.uint8)
        trial_plot[discrepancy] = 255 #                  ((former_arena[discrepancy] * 1 + arena[discrepancy].astype(float) * 9) / 10).astype(np.uint8)

    '''     Loop over each frame, making the video and images       '''
    # loop over each frame
    while True:
        # get the frame
        ret, frame = vid.read()
        #get the frame number
        frame_num = int(vid.get(cv2.CAP_PROP_POS_FRAMES))
        frames_past_stimulus = frame_num - self.stim_frame
        frames_til_abort = count_down - frame_num
        # stop after 2 secs of shelter
        if not frames_til_abort: break
        # apply the registration and fisheye correction
        if False: frame = register_frame(frame, self.x_offset, self.y_offset, self.session.Registration, map1, map2)
        # prior to stimulus onset, refresh frame to initialized frame
        if frames_past_stimulus < 0:
            video_arena = self.arena_with_prev_trials.copy() #TEMPORARY: arena.copy() #
            model_mouse_mask_previous = 0
        # extract DLC coordinates and make a model mouse mask
        model_mouse_mask, large_mouse_mask, body_location = make_model_mouse_mask(self.coordinates, frame_num, model_mouse_mask_initial)
        # use speed to determine model mouse coloration
        speed_color_light, speed_color_dark = speed_colors(smoothed_speed[frame_num - 1], blue = True)
        # at the stimulus onset
        if (frame_num+1) == self.stim_frame:
            # get a contour of the mouse mask
            _, contours, _ = cv2.findContours(model_mouse_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            # and store the position of the mouse
            x_stim = float(body_location[0]); y_stim = float(body_location[1])
        # add dark mouse silhouette if distant from the previous dark one
        elif np.sum(large_mouse_mask * model_mouse_mask_previous) == 0 and not arrived_at_shelter and frames_past_stimulus:
            # add dark mouse to the video arena
            video_arena[model_mouse_mask.astype(bool)] = video_arena[model_mouse_mask.astype(bool)] * speed_color_dark
            if frames_past_stimulus >= -1: trial_plot[model_mouse_mask.astype(bool)] = trial_plot[model_mouse_mask.astype(bool)] * speed_color_dark
            # set the current model mouse mask as the one to be compared to to see if dark mouse should be added
            model_mouse_mask_previous = model_mouse_mask
        # continuous shading, after stimulus onset
        elif frames_past_stimulus >= 0:
            # once at shelter, end it
            if np.sum(self.shelter_roi * large_mouse_mask) \
                    and not 'no shelter' in self.videoname and not 'Circle shelter moved' in self.videoname: # or 'down' in self.videoname:
                if not arrived_at_shelter:
                    # end video in 2 seconds
                    arrived_at_shelter = True
                    count_down = frame_num + self.fps * 2
            else:
                # add light mouse to video arena
                video_arena[model_mouse_mask.astype(bool)] = video_arena[model_mouse_mask.astype(bool)] * speed_color_light
                # add light mouse to escape image
                trial_plot[model_mouse_mask.astype(bool)] = trial_plot[model_mouse_mask.astype(bool)] * speed_color_light
        # add red trail to the previous trials' arena
        if frames_past_stimulus > 0 and not arrived_at_shelter:
            dist_from_start = np.sqrt((x_stim - float(body_location[0]))**2 + (y_stim - float(body_location[1]))**2)
            dist_to_make_red = 150
            prev_homing_color = np.array([.98, .98, .98]) + np.max((0,dist_to_make_red - dist_from_start))/dist_to_make_red * color_trail
            self.arena_with_prev_trials[model_mouse_mask.astype(bool)] = self.arena_with_prev_trials[model_mouse_mask.astype(bool)] * prev_homing_color
        # redraw this contour on each frame after the stimulus
        if frame_num >= self.stim_frame:
            cv2.drawContours(video_arena, contours, 0, (255,255,255), thickness = 1)
            cv2.drawContours(trial_plot, contours, 0, (255,255,255), thickness = 1)
        # if wall falls or rises, color mouse in RED
        if (self.trial_type==1 or self.trial_type==-1) and (frame_num == self.wall_change_frame):
            _, contours_wall_change, _ = cv2.findContours(model_mouse_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            # cv2.drawContours(video_arena, contours_wall_change, 0, (220,0,220), thickness=-1)
            cv2.drawContours(trial_plot, contours_wall_change, 0, (220,0,220), thickness=-1)
            if self.trial_type==1:
                video_arena[arena==90] = arena[arena==90]
                self.arena_with_prev_trials[arena == 90] = arena[arena == 90]
            else:
                middle_replace_arena = self.arena_with_prev_trials.copy()
                video_arena[arena==90] = self.arena_with_prev_trials[arena==90] * (255 / 90)
                video_arena[video_arena==122] = 255

        # add a looming spot - for actual loom
        self.stim_type = 'visual'
        if self.stim_type == 'visual': stim_radius = 30 * (frame_num - self.stim_frame) * ( (frame_num - self.stim_frame) < 10) * (frame_num > self.stim_frame)
        else: stim_radius = size * stim_timing_array[frame_num - self.stim_frame] #218 for planning #334 for barnes #347 for void
        stim_radius = 0
        session_trials_plot_show = video_arena.copy()
        trial_plot_show = trial_plot.copy()
        if stim_radius:
            frame = frame.copy()
            loom_frame = frame.copy()
            loom_arena = video_arena.copy()
            loom_arena2 = trial_plot.copy()

            if self.stim_type == 'visual':
                # for actual loom
                stimulus_location = tuple(self.coordinates['center_body_location'][:, self.stim_frame - 1].astype(np.uint16))
                cv2.circle(loom_frame, stimulus_location, stim_radius, 100, -1)
                cv2.circle(loom_arena, stimulus_location, stim_radius, (100,100,100), -1)
            else:
                center = (int(frame.shape[1] / 2), int(frame.shape[0] / 2))
                if shape == 'circle':
                    cv2.circle(loom_frame, center, stim_radius, 200, 12)
                    cv2.circle(loom_arena, center, stim_radius, (200, 200, 200), 12)
                    cv2.circle(loom_arena2, center, stim_radius, (200, 200, 200), 12)

                elif shape == 'square':
                    cv2.rectangle(loom_frame, tuple([c+stim_radius for c in center]), tuple([c-stim_radius for c in center]), 200, 12)
                    cv2.rectangle(loom_arena,  tuple([c+stim_radius for c in center]), tuple([c-stim_radius for c in center]), (200, 200, 200), 12)
                    cv2.rectangle(loom_arena2,  tuple([c+stim_radius for c in center]), tuple([c-stim_radius for c in center]), (200, 200, 200), 12)

            alpha = .3
            cv2.addWeighted(frame, alpha, loom_frame, 1 - alpha, 0, frame)
            cv2.addWeighted(video_arena, alpha, loom_arena, 1 - alpha, 0, session_trials_plot_show)
            cv2.addWeighted(trial_plot, alpha, loom_arena2, 1 - alpha, 0, trial_plot_show)

        # invert colors for opencv display
        arena_with_prev_trials_show = cv2.cvtColor(self.arena_with_prev_trials, cv2.COLOR_BGR2RGB)
        video_arena_show = cv2.cvtColor(video_arena, cv2.COLOR_BGR2RGB)
        # put minute of stimulation in clip
        # stim_minute = str(int(np.round(self.stim_frame / 60 / 30))) + "'"
        # frame = frame.copy()
        # cv2.putText(frame, stim_minute, (20, 50), 0, 1, (255, 255, 255), thickness=2)
        # display current frames
        cv2.imshow(self.session_videoname, frame);
        cv2.imshow(self.session_videoname + ' DLC', video_arena_show)
        if cv2.waitKey(1) & 0xFF == ord('q'): break
        # write current frame to video
        video.write(frame); self.session_video.write(frame)
        video_dlc.write(video_arena_show); self.session_video_dlc.write(video_arena_show)
        # end video
        if frame_num >= self.end_frame: break
    # wrap up
    vid.release(); video.release(); video_dlc.release()
    # draw red silhouette for previous trial arena
    cv2.drawContours(self.arena_with_prev_trials, contours, 0, contour_color, thickness=-1)
    # cv2.drawContours(self.arena_with_prev_trials, contours, 0, (0,0,0), thickness=1)
    # draw purple silhouette for wall change
    if (self.trial_type == 1 or self.trial_type == -1):
        try:
            cv2.drawContours(video_arena, contours_wall_change, 0, (220, 0, 220), thickness=-1)
            cv2.drawContours(trial_plot, contours_wall_change, 0, (220, 0, 220), thickness=-1)
        except:
            print('Obstacle change not picked up for' + self.videoname)
    # save trial images
    imageio.imwrite(os.path.join(self.save_folder, self.videoname + ' image.tif'), trial_plot)
    imageio.imwrite(os.path.join(self.save_folder, self.videoname + ' image with history.tif'), video_arena)
    # after the last trial, save the session workspace image
    if self.trial_num == self.number_of_trials:
        # make all the trials the last frame of the DLC video
        self.session_video.write(self.arena_with_prev_trials)
        # save the all trials image
        scipy.misc.imsave(os.path.join(self.summary_folder, self.videoname + ' image (all trials).tif'), self.arena_with_prev_trials)
        # wrap up
        self.session_video.release()
        self.session_video_dlc.release()
        cv2.destroyAllWindows()

    # cv2.drawContours(trial_plot_show, contours, 0, (255, 255, 255), thickness=1)




def raw_escape_video(processing):
    '''    Generate and save peri-stimulus video clip    '''
    # set up border colors
    pre_stim_color = [0]
    post_stim_color = [200]
    border_size = 40

    # open the behaviour video
    vid = cv2.VideoCapture(processing.video_path)
    # set up the trial clip for saving
    video = cv2.VideoWriter(os.path.join(processing.save_folder,videoname+'.mp4'), processing.fourcc, processing.fps,
                                 (processing.width, processing.height), False)
    vid.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    # set up fisheye correction
    if registration:
        maps = np.load(registration[3]); map1 = maps[:, :, 0:2]; map2 = maps[:, :, 2] * 0

    # run peri-stimulus video
    if display_clip:
        cv2.namedWindow(videoname); cv2.moveWindow(videoname,100,100)
    while True:
        # get the frame
        ret, frame = vid.read()
        if ret:
            # get the frame number
            frame_num = vid.get(cv2.CAP_PROP_POS_FRAMES)

            # apply the fisheye correction
            if registration:
                frame = register_frame(frame, x_offset, y_offset, registration, map1, map2)

            # apply the border and count-down
            if counter:
                frame = apply_border_and_countdown(frame, frame_num, stim_frame, start_frame, end_frame, pre_stim_color, post_stim_color, border_size, videoname, fps, width)
            # just use a normal grayscale image instead
            else:
                frame = frame[:,:,0]

            # display the frame
            if display_clip:
                cv2.imshow(videoname, frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            # write the frame to video
            video.write(frame)

            # end the video
            if frame_num >= end_frame:
                break
        else:
            print('Problem with movie playback'); cv2.waitKey(1000); break
    # wrap up
    vid.release()
    video.release()



def make_model_mouse_mask(coordinates, frame_num, model_mouse_mask_initial, scale = 16):
    '''     extract DLC coordinates and make a model mouse mask     '''
    # extract coordinates
    body_angle = coordinates['body_angle'][frame_num - 1]
    shoulder_angle = coordinates['shoulder_angle'][frame_num - 1]
    head_angle = coordinates['head_angle'][frame_num - 1]
    neck_angle = coordinates['neck_angle'][frame_num - 1]
    nack_angle = coordinates['nack_angle'][frame_num - 1]
    head_location = tuple(coordinates['head_location'][:,frame_num-1].astype(np.uint16))
    nack_location = tuple(coordinates['nack_location'][:, frame_num - 1].astype(np.uint16))
    front_location = tuple(coordinates['front_location'][:, frame_num - 1].astype(np.uint16))
    shoulder_location = tuple(coordinates['shoulder_location'][:, frame_num - 1].astype(np.uint16))
    body_location = tuple(coordinates['center_body_location'][:, frame_num - 1].astype(np.uint16))
    # make sure angles aren't too different
    if abs(neck_angle - head_angle) > 45: head_angle = neck_angle
    if abs(nack_angle - neck_angle) > 45: neck_angle = nack_angle
    if abs(body_angle - nack_angle) > 45: nack_angle = body_angle
    # draw ellipses representing model mouse
    model_mouse_mask = cv2.ellipse(model_mouse_mask_initial.copy(), body_location, (int(scale * .9), int(scale * .5)), 180 - body_angle, 0, 360, 100, thickness=-1)
    model_mouse_mask = cv2.ellipse(model_mouse_mask, nack_location, (int(scale * .7), int(scale * .35)), 180 - nack_angle, 0, 360, 100, thickness=-1)
    # if turning, down't show the shoulder bit
    if abs(neck_angle - nack_angle) < 40: model_mouse_mask = cv2.ellipse(model_mouse_mask, front_location, (int(scale * .5), int(scale * .33)), 180 - neck_angle, 0, 360, 100, thickness=-1)
    if abs(head_angle - nack_angle) < 40: model_mouse_mask = cv2.ellipse(model_mouse_mask, head_location, (int(scale * .6), int(scale * .3)), 180 - head_angle, 0, 360, 100, thickness=-1)
    if abs(body_angle - shoulder_angle) < 30: model_mouse_mask = cv2.ellipse(model_mouse_mask, shoulder_location, (int(scale), int(scale * .4)), 180 - shoulder_angle, 0, 360, 100, thickness=-1)
    # make a single large ellipse used to determine when do use the flight_color_dark
    large_mouse_mask = cv2.ellipse(model_mouse_mask_initial.copy(), body_location, (int(scale * 2.5), int(scale * 1.5)), 180 - shoulder_angle, 0, 360, 100,thickness=-1)
    return model_mouse_mask, large_mouse_mask, body_location




def initialize_stim_visualization(obstacle_type):
    '''     get the settings from the stimulus visualization        '''
    # blink for 3 seconds
    stim_timing_array = np.concatenate( (np.tile(np.concatenate((np.ones(5), np.zeros(5))), 9), np.zeros(999)))  .astype(int)
    if obstacle_type == 'wall' or obstacle_type == 'void':
        shape = 'circle'
        size = 340  # 334
    elif obstacle_type == 'side wall':
        shape = 'square'
        size = 224  # 218
    elif obstacle_type == 'side wall 32' or obstacle_type == 'side wall 14':
        shape = 'square'
        size = 295
    else:
        shape = 'circle'
        size = 340  # 334
        # blink for 9 seconds
        stim_timing_array = np.concatenate( (np.tile(np.concatenate((np.ones(5), np.zeros(5))), 27), np.zeros(999)))  .astype(int)

    return stim_timing_array, shape, size