'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

-----------#                                          Display a saved video                            --------------------------------


'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

# ------------------------------------------
# Select video file name and folder location
# ------------------------------------------
'''
User options go here
'''
file_loc = 'D:\\Dropbox (UCL - SWC)\\DAQ\\upstairs_rig\\19JAN27_peacepalace\\CA4060\\'
video_file_name = 'peace_palace.avi'
dlc_file_name = 'cam1DeepCut_resnet50_Barnes2018-11-22shuffle1_750000.h5'
save_file_name = 'peace_palace_labeled.avi'

show_video = True
display_frame_rate = 100

start_frame = 0
end_frame = 5 #np.inf #30*60*5 #np.inf means will go to end of video








# set up video writer using opencv
vid = cv2.VideoCapture(file_loc + video_file_name)
ret, frame = vid.read()
fourcc_data = cv2.VideoWriter_fourcc(*"XVID")  # LJPG for lossless, XVID for compressed
data_video = cv2.VideoWriter(file_loc + save_file_name, fourcc_data, 30, (frame.shape[1], frame.shape[0]), True)
end_frame = int(min(end_frame,vid.get(cv2.CAP_PROP_FRAME_COUNT)))

# ----------------------------
# Get the coordinates from DLC
# ----------------------------
'''
read the coordinates for each body part into an array
'''
# list body parts
body_parts = ['nose','L eye','R eye','L ear','neck','R ear','L shoulder','upper back','R shoulder','L hind limb','Lower back','R hind limb','derriere']

# open coordinates from dlc file
DLC_dataframe = pd.read_hdf(file_loc + dlc_file_name)
DLC_network = os.path.basename(file_loc + dlc_file_name)
DLC_network = DLC_network[DLC_network.find('Deep'):-3]

# array of all body parts, axis x body part x frame
all_body_parts = np.zeros((2, len(body_parts), end_frame - start_frame))
for bp, body_part in enumerate(body_parts):
    # extract coordinates
    for j, axis in enumerate(['x', 'y']):
        all_body_parts[j, bp, :] = DLC_dataframe[DLC_network][body_part][axis].values[start_frame:end_frame]



# ---------------------------
# Play video and save video
# ---------------------------
'''
save the video, with colored circles superimposed onto each body part
press 'q' to stop the video
'''
# create colormap for body parts
colors = plt.cm.get_cmap('hsv', len(body_parts))

# open video
vid = cv2.VideoCapture(file_loc + video_file_name)
vid.set(cv2.CAP_PROP_POS_FRAMES,start_frame)

while True:
    ret, frame = vid.read() # get the frame

    if ret:
        frame_num = int(vid.get(cv2.CAP_PROP_POS_FRAMES))

        # draw circles
        for bp, body_part in enumerate(body_parts):
            cv2.circle(frame, tuple(all_body_parts[:,bp,frame_num-1].astype(np.uint16)), 1, [255*x for x in colors(bp)], -1)

        # write video to file
        data_video.write(frame)


        if show_video:
            cv2.imshow('movie',frame)
            if cv2.waitKey(int(1000/display_frame_rate)) & 0xFF == ord('q'):
                break
        
            if (frame_num-start_frame)%1000==0:
                print(str(int(frame_num-start_frame)) + ' out of ' + str(int(end_frame-start_frame)) + ' frames complete')
            
        if frame_num >= end_frame:
            break 
    else:
        print('Problem with movie playback')
        cv2.waitKey(1000)
        break
        
vid.release()
# Display number of last frame
print('Stopped at frame ' + str(frame_num)) #show the frame number stopped at
data_video.release()

 