'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

-----------#                                          Display a saved video                            --------------------------------


'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

import cv2
import numpy as np

# ------------------------------------------
# Select video file name and folder location
# ------------------------------------------
file_locs = ['D:\\Dropbox (UCL - SWC)\DAQ\\upstairs_rig\\19JUL09_wallupdown (food)\\CA6161\\']
video_file_names = ['cam1.avi']
save_fps = 30
color = [False]
save_name = 'whiteout_'

# more options
show_video = False
display_frame_rate = 1000
start_frame = 0
end_frame = np.inf

# loop across all videos
for vid_num in range(len(file_locs)):

    # get the file location
    file_loc = file_locs[vid_num]
    video_file_name = video_file_names[vid_num]
    save_file_location = file_loc + save_name + video_file_name

    # set up video writer
    vid = cv2.VideoCapture(file_loc + video_file_name)
    ret, frame = vid.read()
    fourcc_data = cv2.VideoWriter_fourcc(*"XVID")  # LJPG for lossless, XVID for compressed
    data_video = cv2.VideoWriter(save_file_location, fourcc_data, save_fps, (frame.shape[1], frame.shape[0]), color[vid_num])

    # modify the frame in some way - make mask to cover dark areas
    vid.set(cv2.CAP_PROP_POS_FRAMES, 200)
    ret, frame = vid.read()
    modified_top = frame[:240, :, 0]
    modified_top_mask = modified_top < 40


    # ---------------------------
    # Play video and save video
    # ---------------------------
    vid.set(cv2.CAP_PROP_POS_FRAMES,start_frame)
    frame_num = vid.get(cv2.CAP_PROP_POS_FRAMES)
    end_frame = min(end_frame,vid.get(cv2.CAP_PROP_FRAME_COUNT))
    while True:
        ret, frame = vid.read() # read the frame
        frame_num = vid.get(cv2.CAP_PROP_POS_FRAMES)
        if ret:
            if not color[vid_num]:
                frame = frame[:,:,0]

            # modify the frame in some way
            modified_top = frame[:240, :]
            modified_top[modified_top_mask] = 180
            frame[:240, :] = modified_top

            # write the new video
            data_video.write(frame)

            # display the video
            if show_video:
                cv2.imshow('movie',frame)
                if cv2.waitKey(int(1000/display_frame_rate)) & 0xFF == ord('q'): break

            # end the video
            if frame_num >= end_frame: break
        # if can't read the video
        else: break
    # close the video files
    vid.release()
    data_video.release()
    # Display number of last frame
    print('Stopped at frame ' + str(frame_num))

 