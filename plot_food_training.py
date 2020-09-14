import os
from termcolor import colored
import numpy as np
import matplotlib.pyplot as plt
import dill as pickle
import cv2
from helper_code.registration_funcs import get_background

# data labels
mice = ['P.1','P.2','P.3','P.4','P.5']
days = ['191127', '191128', '191129', '191130', '191201', '191202'] #'191126',
base_folder = 'D:\\Dropbox (UCL - SWC)\\DAQ\\upstairs_rig\\PS_mousetraining'
summary_plots_folder = 'D:\\data\\Summary Plots'

# options
plot_sessions = False
plot_data = True
show_video = False
plot_trajectory = True

bins = np.arange(-63, 64, 4.5) # idx 14 and 15 are stimulus
bins = np.arange(-63, 64, 2.25) # 29 and 30 are stimulus
trials_to_examine = 50

# open data dictionary
save_file = os.path.join(base_folder, 'foraging_data_IV')
with open(save_file, 'rb') as dill_file: foraging_dict = pickle.load(dill_file)


def get_biggest_contour(frame):
    _, contours, _ = cv2.findContours(frame, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cont_count = len(contours)

    big_cnt_ind = 0
    if cont_count > 1:
        areas = np.zeros(cont_count)
        for c in range(cont_count):
            areas[c] = cv2.contourArea(contours[c])
        big_cnt_ind = np.argmax(areas)

    cnt = contours[big_cnt_ind]
    M = cv2.moments(cnt)
    cx = int(M['m10'] / M['m00'])
    cy = int(M['m01'] / M['m00'])

    return contours, big_cnt_ind, cx, cy, cnt
'''
examine the behavior video
'''

if show_video:
    # foraging_dict['video trajectories'] = {}
    # foraging_dict['trajectories'] = {}
    # day = '191129'
    # mouse = 'P.2'
    # vid_path = 'D:\\Dropbox (UCL - SWC)\\DAQ\\upstairs_rig\\PS_mousetraining\\191129\\P.2\\Camera_rig5.mp4'

    for mouse in mice:
        for day in days:

            session = day + '_' + mouse
            if session in foraging_dict['video trajectories'] and session in foraging_dict['trajectories']: continue

            vid_path = 'D:\\Dropbox (UCL - SWC)\\DAQ\\upstairs_rig\\PS_mousetraining\\' + day + '\\' + mouse + '\\Camera_rig5.mp4'
            vid = cv2.VideoCapture(vid_path)
            ret, frame = vid.read()
            if not ret: continue
            else: print(session)

            vid_window = [-1, 9]
            fps = 40
            end_vids = False
            save_vid = False
            get_trajectory = True

            # get stim times
            sound_times = foraging_dict['sound_times'][session]
            pump_times = foraging_dict['pump_times'][session]
            lick_times = foraging_dict['lick_times_II'][session]
            # camera_times = foraging_dict['camera_times'][session]
            trajectories = np.zeros((2, 40*9, len(sound_times))) # x/y, 9 seconds, number of trials

            # extract background
            if get_trajectory:
                background, _, _ = get_background(vid_path, start_frame=0, avg_over=100)
                mask_thresh = .8 # .7
                kernel = [4, 3]
                kernel_er = np.ones((kernel[0], kernel[0]), np.uint8)
                kernel_dil = np.ones((kernel[1], kernel[1]), np.uint8)

                vid_frames = int(vid.get(cv2.CAP_PROP_FRAME_COUNT))
                video_trajectory = np.zeros(vid_frames)
                vid.set(cv2.CAP_PROP_POS_FRAMES, 0)

            '''     go thru entire thing '''
            # for frame_num in range(vid_frames):
            #     ret, frame = vid.read()
            #     if ret:
            #         frame_norm = (frame[:, :, 0] / background)
            #         # use the thresholds, erosion, and dilation set above to extract a mask coinciding with the mouse
            #         frame_norm_mask = (frame_norm < mask_thresh).astype(np.uint8)
            #         # frame_norm_mask = cv2.erode(frame_norm_mask, kernel_er, iterations=1)
            #         # frame_norm_mask = cv2.dilate(frame_norm_mask, kernel_dil, iterations=3)
            #         # extract the largest contour in this mask -- this should correspond to the mouse
            #         try: _, _, x_center, y_center, _ = get_biggest_contour(frame_norm_mask)
            #         except: print('Contour failure')
            #         video_trajectory[frame_num] = x_center

            # foraging_dict['video trajectories'][session] = video_trajectory

            # loop across trials
            for i, sound_time in enumerate(sound_times):
                # if end_vids or i > 20: break
                if save_vid:
                    # set up video writer
                    fourcc_data = cv2.VideoWriter_fourcc(*"XVID")  # LJPG for lossless, XVID for compressed
                    save_video = cv2.VideoWriter('D:\\Dropbox (UCL - SWC)\\DAQ\\upstairs_rig\\PS_mousetraining\\191129\\P.2\\tone trial ' + str(i) + '.mp4', fourcc_data, 40, (1280, 1024), False)
                # get relevant lick and pump times
                relevant_pump_idx = ((pump_times - sound_time) > vid_window[0]) * ((pump_times - sound_time) < vid_window[1])
                relevant_lick_idx = ((lick_times - sound_time) > vid_window[0]) * ((lick_times - sound_time) < vid_window[1])
                relevant_pump_times = np.ceil((pump_times[relevant_pump_idx] - sound_time) * fps)
                relevant_lick_times = np.ceil((lick_times[relevant_lick_idx] - sound_time) * fps)
                pump_on = 0
                lick_on = 0

                stim_frame = np.round(sound_time * fps)
                # proper_frame = np.argmin(abs(camera_times - sound_time))
                # print(str(int(stim_frame)) + ' -- ' + str(proper_frame))

                vid.set(cv2.CAP_PROP_POS_FRAMES, stim_frame)
                # loop across frames
                for rel_frame_num in range(vid_window[0]*fps, vid_window[1]*fps):
                    ret, frame = vid.read()
                    if ret:
                        '''     get centroid of mouse       '''
                        if get_trajectory and rel_frame_num >= 0 and rel_frame_num < 40*9:
                            frame_norm = (frame[:,:,0] / background)
                            # use the thresholds, erosion, and dilation set above to extract a mask coinciding with the mouse
                            frame_norm_mask = (frame_norm < mask_thresh).astype(np.uint8)
                            frame_norm_mask = cv2.erode(frame_norm_mask, kernel_er, iterations=1)
                            frame_norm_mask = cv2.dilate(frame_norm_mask, kernel_dil, iterations=3)
                            # extract the largest contour in this mask -- this should correspond to the mouse
                            _, _, x_center, y_center, _ = get_biggest_contour(frame_norm_mask)
                            trajectories[:, rel_frame_num, i] = np.array([x_center, y_center])
                        else: x_center, y_center = 0, 0


                        if save_vid:
                            # add text of seconds rel to stim
                            cv2.putText(frame, str(rel_frame_num / fps), (20, 50), 0, 1, 255, thickness=2)
                            # say when pump goes
                            if rel_frame_num in relevant_pump_times or pump_on:
                                cv2.putText(frame, 'GET PUMPED!!!', (20, 100), 0, 1, 255, thickness=2)
                                if pump_on: pump_on -= 1
                                else: pump_on = 80
                            # say when TONGUE goes
                            if rel_frame_num in relevant_lick_times or lick_on:
                                cv2.putText(frame, '~~LICK DAT~~', (20, 200), 0, 1, 255, thickness=2)
                                if lick_on: lick_on -= 1
                                else: lick_on = 5

                            if rel_frame_num > 0:
                                cv2.putText(frame, '----------SOUND ON-----------------------', (20, 400), 0, 1, 255, thickness=2)

                            if x_center:
                                cv2.circle(frame, (x_center, y_center), 10, 255, -1)

                            # show frame
                            cv2.imshow(session + ' - trial ' + str(i), frame)
                            save_video.write(frame[:,:,0])
                        # time frames / stop
                        if cv2.waitKey(10) & 0xFF == ord('q'):
                            end_vids = True
                            break
                if save_vid: save_video.release()
            foraging_dict['trajectories'][session] = trajectories
            save_file = os.path.join(base_folder, 'foraging_data_IV')
            with open(save_file, "wb") as dill_file: pickle.dump(foraging_dict, dill_file)

'''     PLOT TRAJCTORIES        '''
show_trajectories = False
traj_success_array = np.zeros((len(mice), len(days)-1))
pseudo_traj_success_array = np.zeros((len(mice), len(days)-1))
seconds_to_examine = 6.75
lick_location = 950 #950
start_location = 500 #550
if plot_trajectory:
    for m, mouse in enumerate(mice):
        for d, day in enumerate(days):

            session = day + '_' + mouse
            if session in foraging_dict['trajectories']: print(session)
            else: continue

            # get trajectories
            trajectories = foraging_dict['trajectories'][session][:,:,:50]
            num_trials = trajectories.shape[2]

            if show_trajectories:
                # create figure
                fig, ax = plt.subplots(figsize = (9,6))
                ax.set_title(session + ' trajectories')
                ax.set_ylim([num_trials, -1])
                shift = 0
                # x/y, 9 seconds, number of trials
                for trial in range(num_trials):
                    # get the start x pos
                    x_start = np.min((1000, trajectories[0,0,trial]))
                    if x_start > 900: shift+=1; continue
                    # get the rightmost x pos in 9 sec
                    x_furthest = np.min((1000, np.max(trajectories[0,:int(4.5*40),trial])))
                    # plot a line between the two
                    ax.plot([x_start, x_furthest], [trial - shift, trial - shift], color = [0,0,0])

            # get stats
            # num of trials starting on the left
            eligible_trial_idx = trajectories[0,0,:] < start_location
            num_eligible_trials = np.sum(eligible_trial_idx)
            # get rightmost point
            rightmost_point = np.max(trajectories[0,:int(seconds_to_examine*40),:][:, eligible_trial_idx], axis = 0)
            # num of trials going to right
            num_go_right_trials = np.sum(rightmost_point > lick_location)
            # print/save results
            print(session + ': ' + str(num_go_right_trials) + ' / ' + str(num_eligible_trials))
            if d > 1 and mouse == 'P.2': d-=1
            elif d > 2: d-= 1
            traj_success_array[m, d] = num_go_right_trials / num_eligible_trials

            # get stats
            # points where start < 500
            num_pseudo_trials = 10000
            pseudo_trajectories = np.zeros((9*40, num_pseudo_trials))
            video_trajectory = foraging_dict['video trajectories'][session]
            eligible_frame_idx = np.where(video_trajectory[:-9*40] < start_location)[0]
            # choose start frames for trials
            pseudo_trial_idx = np.random.choice(eligible_frame_idx, num_pseudo_trials)
            for p, pt in enumerate(pseudo_trial_idx):
                pseudo_trajectories[:, p] = video_trajectory[pt:pt+9*40]

            # get rightmost point
            pseudo_rightmost_point = np.max(pseudo_trajectories[:int(seconds_to_examine*40),:], axis = 0)
            # num of trials going to right
            psuedo_num_go_right_trials = np.sum(pseudo_rightmost_point > lick_location)
            # print/save results
            print(session + ': ' + str(psuedo_num_go_right_trials) + ' / ' + str(num_pseudo_trials))
            pseudo_traj_success_array[m, d] = psuedo_num_go_right_trials / num_pseudo_trials

    # plot stats
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_title('% successful trajectories')
    for d in range(traj_success_array.shape[1]):
        # plot mean
        ax.scatter(d + 1, np.mean(traj_success_array[traj_success_array[:, d]>0, d]), s = 30, color = [0,1,1,.5])
        # plot all points
        ax.scatter(np.ones(len(mice))*d+1, traj_success_array[:, d], s = 10, color = [0,0,0])

    # plot stats
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_title('Relative likelihood of running to lick port')
    ax.set_ylim([.3, 2.6])
    ax.set_xlim([.8, 5.2])
    ax.plot([0, 6], [1, 1], color=[0, 0, 0, .3], linestyle='--')
    for m in range(traj_success_array.shape[0]):
        # plot progression
        ax.plot(np.arange(1, 6), traj_success_array[m, :] / pseudo_traj_success_array[m, :], color=[0, .6, .6, .7], linewidth=3)


    # plot mean across all mice
    mean_rel_success = np.nanmean(traj_success_array / pseudo_traj_success_array, axis = 0)
    ax.plot(np.arange(1, 6), mean_rel_success, color=[0, 0, 1], linewidth = 5)
    plt.show()


day1 = [[t] for t in traj_success_array[:, 0] / pseudo_traj_success_array[:, 0]]
day5 = [[t] for t in traj_success_array[:, -1] / pseudo_traj_success_array[:, -1]]
from important_code.shuffle_test import permutation_test_paired, permutation_test
permutation_test(day5, day1, iterations = 10000, two_tailed = False)


'''     MAKE LICK PROB ARRAYS       '''
# initialize data arrays
if plot_data:
    lick_counts_all = {}
    lick_prob_all = {}
    foraging_dict['lick_times_II'] = {}
    for mouse in mice:
        lick_counts_all[mouse] = np.zeros((5, len(bins) - 1))
        lick_prob_all[mouse] = np.zeros((5, len(bins) - 1))

# loop over mice
for mouse in mice:
    day_num = 0
    # loop over days
    for day in days:
        # get session name
        session = day + '_' + mouse
        print(session)

        # skip sessions that didn't happen
        if not session in foraging_dict['session_duration']: continue

        # extract data
        session_duration = foraging_dict['session_duration'][session]
        pump_times = foraging_dict['pump_times'][session]
        lick_times = foraging_dict['lick_times'][session]
        lick_times_all = foraging_dict['lick_times_II'][session]
        lick_durations = foraging_dict['lick_duration'][session]
        sound_times = foraging_dict['sound_times'][session]
        sound_durations = foraging_dict['sound_duration'][session]
        num_trials = len(sound_times)
        # print(np.median(np.diff(sound_times[:trials_to_examine] / 60)))

        if plot_data:
            lick_counts = np.zeros(len(bins)-1)
            lick_prob = np.zeros(len(bins)-1)

            for i, sound_time in enumerate(sound_times[:trials_to_examine]):

                # find relevant LICKS
                relevant_lick_idx = ((lick_times - sound_time) > -65) * ((lick_times - sound_time) < 65)
                relevant_lick_times = lick_times[relevant_lick_idx]
                relevant_lick_durations = lick_durations[relevant_lick_idx]

                added_licks = np.array([])
                for j, lick in enumerate(relevant_lick_times):
                    duration = relevant_lick_durations[j]
                    if duration > .2:
                        lick_times_all = np.concatenate((lick_times_all, lick + np.arange(.2, duration, .2)))

                relevant_lick_idx = ((lick_times_all - sound_time) > -65) * ((lick_times_all - sound_time) < 65)
                relevant_lick_times = lick_times_all[relevant_lick_idx]
                # copy licks corresponding to how long the lick bout is
                # relevant_lick_times_II = np.array([])
                # for j, time in enumerate(relevant_lick_times):
                #     num_licks = int(min(2.25, relevant_lick_durations[j]) * 7)
                #     relevant_licks_copied = np.ones(num_licks) * time
                #     relevant_lick_times_II = np.concatenate((relevant_lick_times_II, relevant_licks_copied))

                # get hist of relevant LICKS
                counts, bins = np.histogram(relevant_lick_times-sound_time, bins = bins)

                # put into data arrays
                lick_counts = lick_counts + counts
                lick_prob = lick_prob + (counts>0)/num_trials

            # add to global data arrays
            lick_counts_all[mouse][day_num, :] = lick_counts
            lick_prob_all[mouse][day_num, :] = lick_prob
            day_num += 1

            foraging_dict['lick_times_II'][session] = lick_times_all



        if plot_sessions:
            # plot raster of sound, pump, lick
            fig1, ax1 = plt.subplots(figsize=(12, 8))
            ax1.set_ylim([len(sound_times)+1, -1])
            ax1.set_xlim([-20, 20])
            ax1.set_xlabel('Time relative to sound onset (sec)')
            ax1.set_ylabel('Trial number')
            ax1.set_title(session + ' training results IV')
            for i, sound_time in enumerate(sound_times):
                # find relevant PUMP TIMES
                relevant_pump_idx = ((pump_times - sound_time) > 0) * ((pump_times - sound_time) < 10)
                relevant_pump_times = pump_times[relevant_pump_idx]
                # plot TONE (different color depending on trial result)
                if not np.sum(relevant_pump_idx): color = [0, 0, 0, .1]
                elif np.sum(relevant_pump_idx): color = [0, 1, 0, .2]

                # find relevant LICK TIMES
                relevant_lick_idx = ((lick_times_all - sound_time) > 0) * ((lick_times_all - sound_time) < 9)
                relevant_lick_times = lick_times_all[relevant_lick_idx]
                # plot TONE (different color depending on trial result)
                if not np.sum(relevant_lick_idx): color = [0, 0, 0, .1]
                elif np.sum(relevant_lick_idx): color = [0, 1, 0, .2]

                tone_on = plt.Rectangle((0, i),9, .8, color=color, linewidth = 0, fill=True) # sound_durations[i] -> 9
                ax1.add_artist(tone_on)

                # find relevant LICKS
                relevant_lick_idx = ((lick_times_all - sound_time) > -60) * ((lick_times_all - sound_time) < 30)
                relevant_lick_times = lick_times_all[relevant_lick_idx]

                # plot the LICKS
                ax1.eventplot(relevant_lick_times - sound_time, color=[0, 0, 0], lineoffsets=i + .4, linelengths=.8)

                # plot the PUMP ACTION
                ax1.eventplot(relevant_pump_times - sound_time, color=[1,0,0], lineoffsets=i + .4, linelengths=.8)

            # save figure
            fig1.savefig(os.path.join(summary_plots_folder, session + ' training IV.png'), format='png')
            fig1.savefig(os.path.join(summary_plots_folder, session + ' training IV.eps'), format='eps')

'''
    PLOT OVERALL TRAINING DATA      
'''

if plot_data:
    # plot relative lick rates
    fig3, ax3 = plt.subplots(figsize=(12, 6))
    ax3.set_ylim([0, 5.1])
    ax3.set_xlim([.8, 5.2])
    ax3.plot([0,6],[1,1],color=[0,0,0,.3], linestyle = '--')
    ax3.set_xlabel('Session no.')
    ax3.set_xticks(np.arange(1, 6))
    ax3.set_ylabel('Lick probability during stimulus relative to baseline period')
    ax3.set_title('Lick prob across days')

    bins_pre = np.arange(0, 7)
    bins_pre = np.arange(11,14)
    bins_tone = 14


    bins_pre = np.arange(20,29)
    bins_tone = [29,30]

    # bins_pre = np.concatenate((np.arange(20,29), np.arange(33, 42)))

    relative_licks_all = np.zeros((len(mice), 5))

    # loop across mice
    for m, mouse in enumerate(mice):
        # get licks data
        if len(bins_tone) == 1: licks_tone = lick_prob_all[mouse][:, bins_tone[0]]
        else: licks_tone = np.mean(lick_prob_all[mouse][:, bins_tone], 1)

        licks_pre = np.mean(lick_prob_all[mouse][:, bins_pre], 1)

        # put licks in global array
        relative_licks_all[m, :] = licks_tone / licks_pre
        print(mouse)
        print(licks_tone / licks_pre)

        # plot progression of licks for each mouse
        ax3.plot(np.arange(1, 6), licks_tone / licks_pre, color=[0, .6, .6, .7], linewidth = 3)

    # get mean lick progression
    mean_relative_licks = np.mean(relative_licks_all, 0)

    # plot mean progression of licks
    ax3.plot(np.arange(1, 6), mean_relative_licks, color=[0, 0, 1], linewidth = 5)

    fig3.savefig(os.path.join(summary_plots_folder,'training summary ' + str(trials_to_examine) + '.png'), format='png')
    fig3.savefig(os.path.join(summary_plots_folder,'training summary ' + str(trials_to_examine) + '.eps'), format='eps')






'''
old stuff
'''
if False:
    # set colors by bin
    colors = np.zeros((28,3))
    colors[:14, 0] = np.linspace(0,.8,14)
    colors[13, :] = [1,1,0]
    colors[14, :] = [0,1,0]
    colors[15, :] = [0,1,1]
    colors[16:, 2] = np.linspace(1,.2,12)
    colors = np.zeros((57,3))
    colors[:28, 0] = np.linspace(0,.8,28)
    colors[28, :] = [0,0,0]
    colors[29, :] = [0,1,0]
    colors[30, :] = [0,1,1]
    colors[31:, 2] = np.linspace(1,.2,26)

    # loop across mice
    for mouse in mice:
        # plot lick rates across days
        fig2, ax2 = plt.subplots(figsize=(12, 6))
        ax2.set_ylim([0, 400])
        # ax2.set_ylim([0, 1])
        ax2.set_xlim([.8, 5.2])
        ax2.set_xlabel('Session no.')
        ax2.set_xticks(np.arange(1,6))
        ax2.set_ylabel('Licks probability')
        ax2.set_title('Lick prob across days - ' + mouse)
        # loop across time bins
        for bin in range(lick_counts_all[mouse].shape[1]):
            ax2.plot(np.arange(1,6), lick_counts_all[mouse][:, bin], color = colors[bin, :])
            # ax2.plot(np.arange(1, 6), lick_prob_all[mouse][:, bin], color=colors[bin, :])

        fig2.savefig(os.path.join(summary_plots_folder, mouse + ' licking II.png'), format='png')
        fig2.savefig(os.path.join(summary_plots_folder, mouse + ' licking II.eps'), format='eps')



plt.close('all')




'''
foraging_dict['session_duration'][session] = session_duration
foraging_dict['pump_times'][session] = pump_times
foraging_dict['lick_times'][session] = lick_times
foraging_dict['sound_times'][session] = sound_times
foraging_dict['sound_duration'][session] = sound_durations
'''
