'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

-----------#                                          Display a saved video                            --------------------------------


'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

import cv2
import numpy as np
import glob

# ------------------------------------------
# Select video file name and folder location
# ------------------------------------------
save_folder = 'D:\\data\\Summary Plots\\videos\\'


'''     SV1 - visualization   '''
# file_loc = 'D:\\data\\Paper\\Circle wall down dark\\'
# mouse_names = ['CA4030']
# save_name = 'SV1 real.mp4'

'''     SV2 - homing vector   '''
# file_loc = 'D:\\data\\Paper\\Circle wall up\\'
# mouse_names = ['CA3482', 'CA7190', 'CA7170'] #, 'CA3210']  # 'CA3471', 'CA3483',
# save_name = 'SV1.mp4'


'''     SV3 - obstacle        '''
# file_loc1 = 'D:\\data\\Paper\\Circle wall down\\'
# file_loc2 = 'D:\\data\\Paper\\Circle lights on off (baseline)\\'
# # file_locs = [file_loc1, file_loc2, file_loc2, file_loc2, file_loc1, file_loc1, file_loc2, file_loc2]
# # mouse_names = ['CA6940', 'CA7503', 'CA8110', 'CA8140', 'CA3400', 'CA3151', 'CA7491', 'CA8190']
# file_locs = [file_loc1, file_loc2, file_loc2, file_loc1, file_loc1, file_loc2]  # file_loc2, file_loc2,
# mouse_names = ['CA6940', 'CA7503','CA8110', 'CA3400', 'CA3380'] # 'CA7491', 'CA8140', 'CA8190',
# save_name = 'SV_trial1.mp4'
#
# file_locs = [file_loc1, file_loc1, file_loc1]  # file_loc2, file_loc2,
# mouse_names = ['CA3400', 'CA3410','CA3380'] # 'CA7491', 'CA8140', 'CA8190',
# save_name = 'SV_trial3.mp4'

'''     SV4 - wall up   '''
# file_loc = 'D:\\data\\Paper\\Circle wall up\\'
# mouse_names = ['CA3210', 'CA3471', 'CA7170','CA7190']
# save_name = 'SV4.mp4'

'''     SV5 - dark   '''
# file_loc = 'D:\\data\\Paper\\Circle wall down dark\\'
# mouse_names = ['CA8180','CA8792','CA8462','CA8794']
# save_name = 'SV5.mp4'
# file_loc = 'D:\\data\\Paper\\Circle wall down (dark non naive)\\'
# mouse_names = ['CA3720','CA8541','CA3740','CA8551']
# save_name = 'SV5II.mp4'

'''     SV6 - wall down   '''
# file_loc = 'D:\\data\\Paper\\Circle wall down\\'
# mouse_names = ['CA3380', 'CA3410', 'CA3400','CA6950', 'CA3390', 'CA3151']
# save_name = 'SV6.mp4'
file_loc1 = 'D:\\data\\Paper\\Circle wall down\\'
file_loc2 = 'D:\\data\\Paper\\Circle wall down (no baseline)\\'
mouse_names = ['CA3151', 'CA3410', 'CA3410', 'CA6970','CA3390','CA3390']
file_locs = [file_loc1, file_loc1, file_loc1, file_loc2, file_loc1, file_loc1]
save_name = 'SV6 II.mp4'

'''     SV7 - wall left   '''
# file_loc = 'D:\\data\\Paper\\Square wall moves left\\'
# mouse_names = ['CA6950', 'CA8180', 'CA8180', 'CA7501', 'CA6990', 'CA7220']
# save_name = 'SV7.mp4'

'''     SV8 - food      '''
# file_loc = 'D:\\data\\Paper\\Circle food wall down\\'
# mouse_names = ['CA8380','CA8380', 'CA8390', 'CA8360']
# save_name = 'SV8.mp4'

# mouse_names = ['CA8380', 'CA8370', 'CA8360', 'CA8360']
# save_name = 'SV8 II.mp4'


save_fps = 30
color = True

# more options
show_video = True
display_frame_rate = 1000

# set up video writer
fourcc_data = cv2.VideoWriter_fourcc(*"XVID")  # LJPG for lossless, XVID for compressed
data_video = cv2.VideoWriter(save_folder + save_name, fourcc_data, save_fps, (720, 720), color)

# loop across all videos
for m, mouse in enumerate(mouse_names):

    # vids to concatenate
    # vid_paths = glob.glob(file_loc + mouse + '\\' + '*vid (DLC)*')[:2] # 2
    # vid_paths = [glob.glob(file_locs[m] + mouse + '\\' + '*vid (DLC)*')[0]] # 3
    # vid_paths = [glob.glob(file_locs[m] + mouse + '\\' + '*vid (DLC)*')[2]]  # 3 II
    # vid_paths = [glob.glob(file_loc + mouse + '\\' + '*vid (DLC)*')[3]] # 4
    # if not mouse == 'CA8792' and not mouse == 'CA8180': vid_paths = [glob.glob(file_loc + mouse + '\\' + '*vid (DLC)*')[3]]  # 5 I
    # else: vid_paths = [glob.glob(file_loc + mouse + '\\' + '*vid (DLC)*')[1]]  # 5 I
    # if m==0: i = 2
    # elif m == 1: i = 4
    # elif m == 2: i = 3
    # elif m == 3: i = 1
    # vid_paths = [glob.glob(file_loc + mouse + '\\' + '*vid (DLC)*')[i]]  # 5 II
    # vid_paths = [glob.glob(file_loc + mouse + '\\' + '*vid (DLC)*')[3]] # 6
    if m==0: i = 4
    elif m == 1: i = 6
    elif m == 2: i = 7
    elif m == 3: i = 1
    elif m == 4: i = 5
    elif m == 5: i = 6
    vid_paths = [glob.glob(file_locs[m] + mouse + '\\' + '*vid (DLC)*')[i]]  # 6 II
    # if m==0: i = 1
    # elif m == 1: i = 2
    # elif m == 2: i = 3
    # elif m == 3: i = 3
    # elif m == 4: i = 2
    # elif m == 5: i = 2
    # vid_paths = [glob.glob(file_loc + mouse + '\\' + '*vid (DLC)*')[i]]  # 7
    # if m==0: i = 6
    # elif m == 1: i = 10
    # elif m == 2: i = 6
    # elif m == 3: i = 7
    # vid_paths = [glob.glob(file_loc + mouse + '\\' + '*vid (DLC)*')[i]]  # 8
    # if m==0: i = 18
    # elif m == 1: i = 17
    # elif m == 2: i = 15
    # elif m == 3: i = 19
    # vid_paths = [glob.glob(file_loc + mouse + '\\' + '*vid (DLC)*')[i]]  # 8 II

    for vid_path in vid_paths:

        # ---------------------------
        # Play video and save video
        # ---------------------------
        vid = cv2.VideoCapture(vid_path)
        while True:
            ret, frame = vid.read()  # read the frame
            frame_num = vid.get(cv2.CAP_PROP_POS_FRAMES)
            if ret:
                # write the new video
                data_video.write(frame)

                # display the video
                if show_video:
                    cv2.imshow('movie', frame)
                    if cv2.waitKey(int(1000 / display_frame_rate)) & 0xFF == ord('q'): break
            # if can't read the video
            else:
                break
        # close the video files
        vid.release()

data_video.release()
print('done')


