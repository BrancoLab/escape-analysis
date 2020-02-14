import cv2
import numpy as np
import scipy
import os


def explore(self, heat_map = True, path_from_shelter = True, silhouette_map = True):
    '''    compute and display exploration    '''

    # get the frame numbers to analyze
    frame_nums = np.arange(self.previous_stim_frame,self.stim_frame)

    # find when last in shelter
    distance_from_shelter = self.coordinates['distance_from_shelter'][frame_nums]
    last_in_shelter = np.where(distance_from_shelter < 30)[0]

    # set scale for size of model mouse
    back_butt_dist = 20

    # loop over silhouette map and path from shelter
    for o, option in enumerate([silhouette_map, path_from_shelter]):
        # only do selected options
        if not option: continue
        # start with a blank copy
        exploration_arena = self.exploration_arena.copy()
        # initialize the mouse mask
        model_mouse_mask_initial = self.exploration_arena[:, :, 0] * 0
        # for silhouette map
        if o and last_in_shelter.size:
            # set up timing for path from shelter
            total_time = len(frame_nums[start_idx:])
            start_frame = frame_nums[start_idx]
            start_idx = last_in_shelter[-1]
            suffix = '_from_last_trial'
        # for path from shelter
        else:
            start_idx = 0
            suffix = 'from_shelter'
        # loop over each frame that passed the threshold
        for frame_num in frame_nums[start_idx:]:
            # extract DLC coordinates from the saved coordinates dictionary
            body_angle = self.coordinates['body_angle'][frame_num]
            shoulder_angle = self.coordinates['shoulder_angle'][frame_num]
            shoulder_location = tuple(self.coordinates['shoulder_location'][:, frame_num].astype(np.uint16))
            body_location = tuple(self.coordinates['center_body_location'][:, frame_num].astype(np.uint16))
            # draw ellipses representing model mouse
            model_mouse_mask = cv2.ellipse(model_mouse_mask_initial.copy(), body_location, (int(back_butt_dist * .7), int(back_butt_dist * .35)), 180 - body_angle, 0, 360, 100, thickness=-1)
            model_mouse_mask = cv2.ellipse(model_mouse_mask , shoulder_location, (int(back_butt_dist * .8), int(back_butt_dist*.23)), 180 - shoulder_angle ,0, 360, 100, thickness=-1)
            # determine color by time since shelter
            if o:
                f1 = (frame_num - start_frame)/total_time
                f2 = 1 - f1
                time_color = f1 * np.array([190, 240, 190]) + f2 * np.array([250, 220, 223])
                multiplier = f1 * 40 + f2 * 80
            #grayscale for silhouettes
            else:
               time_color = np.array([220,220,220])
               multiplier = 60
            # create color multiplier to modify image
            color_multiplier = 1 - (1 - time_color / [255, 255, 255]) / (np.mean(1 - time_color / [255, 255, 255]) * multiplier)
            # prevent any region from getting too dark
            # if np.mean(exploration_arena[model_mouse_mask.astype(bool)]) < 100:
            #     continue
            # apply color to arena image
            exploration_arena[model_mouse_mask.astype(bool)] = exploration_arena[model_mouse_mask.astype(bool)] * color_multiplier
            # display image
            cv2.imshow(self.session.Metadata['mouse_id'] + 'trial explore', exploration_arena)
            # press q to quit
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        # color in final position
        if o:
            _, contours, _ = cv2.findContours(model_mouse_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(exploration_arena, contours, 0, time_color, thickness=-1)
        # apply the contours and border to the image and save the image
        scipy.misc.imsave(os.path.join(self.save_folder, self.videoname + '_exploration' + suffix + '.tif'), cv2.cvtColor(exploration_arena, cv2.COLOR_BGR2RGB))

    '''   heat map    '''
    if heat_map:
        # Make a position heat map as well
        scale = 1
        # average all mice data
        position = coordinates['butty_location']
        H, x_bins, y_bins = np.histogram2d(position[0, 0:stim_frame], position[1, 0:stim_frame],
                                           [np.arange(0, exploration_arena_trial.shape[1] + 1, scale), np.arange(0, exploration_arena_trial.shape[0] + 1, scale)], normed=True)
        exploration_all = H.T

        # gaussian blur
        exploration_blur = cv2.GaussianBlur(exploration_all, ksize=(201, 201), sigmaX=13, sigmaY=13)

        # normalize
        exploration_blur = (exploration_blur / np.percentile(exploration_blur, 98.7) * 255)
        exploration_blur[exploration_blur > 255] = 255
        exploration_all[exploration_all>0] = 255

        # change color map
        exploration_blur = cv2.applyColorMap(exploration_blur.astype(np.uint8), cv2.COLORMAP_OCEAN)
        exploration_all = cv2.cvtColor(exploration_all.astype(np.uint8), cv2.COLOR_GRAY2BGR)
        # exploration_blur[arena == 0] = 0

        # make composite image
        # exploration_image = ((exploration_all.astype(float) + exploration_blur.astype(float)) / 2)
        exploration_image = exploration_all.astype(float)*.8 + exploration_blur.astype(float) / 2
        exploration_image[exploration_image > 255] = 255
        exploration_image = exploration_image.astype(np.uint8)

        exploration_image[(arena > 0) * (exploration_image[:,:,0] < 10)] = 10
        # exploration_all[(arena > 0) * (exploration_all[:, :, 0] == 0)] = 20
        # exploration_image_save = cv2.copyMakeBorder(exploration_image, border_size, 0, 0, 0, cv2.BORDER_CONSTANT, value=0)
        # textsize = cv2.getTextSize(videoname, 0, .55, 1)[0]
        # textX = int((arena.shape[1] - textsize[0]) / 2)
        # cv2.putText(exploration_image_save, videoname, (textX, int(border_size * 3 / 4)), 0, .55, (255, 255, 255), thickness=1)

        cv2.imshow('heat map', exploration_image)
        # cv2.imshow('traces', exploration_all)
        cv2.waitKey(1)
        exploration_image = cv2.cvtColor(exploration_image, cv2.COLOR_RGB2BGR)
        scipy.misc.imsave(os.path.join(savepath, videoname + '_exploration.tif'), exploration_image)
