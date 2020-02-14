import cv2
import numpy as np
import scipy.misc
import imageio
import os
import warnings
warnings.filterwarnings("ignore")
from helper_code.registration_funcs import model_arena
from helper_code.processing_funcs import speed_colors, register_frame

def visualize_escape(self, video_path, session):
    '''    Generate and save escape video clip and dlc tracking clip     '''

    '''     Initialize variables and backgrounds      '''
    # open the behaviour video
    vid = cv2.VideoCapture(video_path)
    vid.set(cv2.CAP_PROP_POS_FRAMES, self.start_frame)
    self.height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
    self.width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
    # set up the escape clips for saving
    rendering_file_name = os.path.join(os.path.dirname(video_path), 'rendering.mp4')
    video_file_name = os.path.join(os.path.dirname(video_path), 'registered_video.mp4')
    video = cv2.VideoWriter(video_file_name, self.fourcc, self.fps, (self.width, self.height), False)
    video_dlc = cv2.VideoWriter(rendering_file_name, self.fourcc, self.fps, (self.width , self.height), True)
    # load fisheye mapping
    if session['Registration'][3]:
        maps = np.load(self.folders['fisheye_map_location']); map1 = maps[:, :, 0:2]; map2 = maps[:, :, 2] * 0
    else: map1 = None; map2 = None
    # set up model arena
    arena, _, shelter_roi = model_arena((self.height, self.width), 0, registration = False,
                            obstacle_type = session.obstacle_type, shelter = self.shelter, dark = self.dark_theme)
    arena = cv2.cvtColor(arena, cv2.COLOR_GRAY2RGB)
    # initialize arenas for mouse mask
    model_mouse_mask_previous = np.zeros(arena.shape[0:2]).astype(np.uint8)
    model_mouse_mask_initial = np.zeros(arena.shape[0:2]).astype(np.uint8)
    # initialize more quantities
    trial_plot = arena.copy()
    frames_past_stimulus = 0
    arrived_at_shelter = False
    count_down = np.inf
    # make a smooth speed trace for coloration
    smoothed_speed = np.concatenate((np.zeros(6 - 1), np.convolve(self.coordinates['speed'], np.ones(12), mode='valid'), np.zeros(6))) / 12
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
        frame = register_frame(frame, session.x_offset, session.y_offset, session['Registration'], map1, map2)
        # prior to stimulus onset, refresh frame to initialized frame
        if frames_past_stimulus < 0:
            video_arena = arena.copy() #self.arena_with_prev_trials.copy() #TEMPORARY
            model_mouse_mask_previous = 0
        # extract DLC coordinates and make a model mouse mask
        model_mouse_mask, large_mouse_mask, body_location = make_model_mouse_mask(self.coordinates, frame_num, model_mouse_mask_initial)
        # use speed to determine model mouse coloration
        speed_color_light, speed_color_dark = speed_colors(smoothed_speed[frame_num - 1])
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
            if np.sum(shelter_roi * large_mouse_mask):
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
            prev_homing_color = np.array([.98, .98, .98]) + np.max((0,dist_to_make_red - dist_from_start))/dist_to_make_red * np.array([.02, -.02, -.02])
        # redraw this contour on each frame after the stimulus
        if frame_num >= self.stim_frame:
            cv2.drawContours(video_arena, contours, 0, (255,255,255), thickness = 1)
            cv2.drawContours(trial_plot, contours, 0, (255, 255, 255), thickness=1)
        # draw dots corresponding to each body part
        body_parts = ['L ear', 'R ear']
        for bp in body_parts:
            cv2.circle(video_arena, (int(self.coordinates[bp][0][frame_num]), int(self.coordinates[bp][1][frame_num])), 2, 100, -1 )

        # add a looming spot - for actual loom
        if self.stim_type == 'visual': stim_radius = 30 * (frame_num - self.stim_frame) * ( (frame_num - self.stim_frame) < 10) * (frame_num > self.stim_frame)

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

            alpha = .3
            cv2.addWeighted(frame, alpha, loom_frame, 1 - alpha, 0, frame)
            cv2.addWeighted(video_arena, alpha, loom_arena, 1 - alpha, 0, session_trials_plot_show)
            cv2.addWeighted(trial_plot, alpha, loom_arena2, 1 - alpha, 0, trial_plot_show)

        # invert colors for opencv display
        video_arena_show = cv2.cvtColor(video_arena, cv2.COLOR_BGR2RGB)
        # display current frames
        cv2.imshow(video_path, frame);
        cv2.imshow(video_path + ' DLC', video_arena_show)
        if cv2.waitKey(1) & 0xFF == ord('q'): break
        # write current frame to video
        video.write(frame)
        video_dlc.write(video_arena_show)
        # end video
        if frame_num >= self.end_frame: break
    # wrap up
    vid.release(); video.release(); video_dlc.release()
    # save trial images
    escape_image_file_name = os.path.join(os.path.dirname(video_path), 'escape image.tif')
    imageio.imwrite(escape_image_file_name, trial_plot)



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

