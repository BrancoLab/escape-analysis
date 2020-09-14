'''     setup the parameters for tracking, registration, coordinate extraction, and visualization      '''
def setup(options):

    '''     SELECT DATA (User suggested to programatically streamline these parameters instead of inputting manually)   '''
    # select videos
    options.videos = ["C:\\Users\\SWC\\Downloads\\8405_pos1_opt1\\8405_pos1_opt1.mp4",
                      "C:\\Users\\SWC\\Downloads\\8405_pos1_opt2\\8405_pos1_opt2.mp4"]
    # provide timing parameters for each video -- [start frame, stim frame, end frame]
    options.start_stim_end_frames = [[[0, 3 * 30, 8 * 30], [2 * 30, 4 * 30, 6 * 30]] for x in range(len(options.videos))] # this example: 2 trials for each video
    # if the frames are cropped from the full camera image size (if not cropped: [0,0]; if no fisheye correction: doesn't matter)
    options.offset = [[0, 0] for x in range(len(options.videos))]
    # set the arena of the common coordinate behavior reference frame - you need to input this arena in model_arena() in registration_lite.py
    options.arena_type = ['circle with shelter' for x in range(len(options.videos))]

    '''    SELECT WHAT WE ARE DOING (later steps require that earlier steps are selected or have been done previously)   '''
    options.do_DLC_tracking = False
    options.do_registration = False
    options.do_coordinate_processing = False
    options.do_visualization = True

    options.overwrite_saved_registration = False # if you need to redo the registration step
    options.overwrite_saved_processing = False # if you need to redo the processing step

    '''     FOLDER LOCATIONS    '''
    options.dlc_config_file = 'D:\\data\\DLC_nets\\Barnes-Philip-2018-11-22\\config.yaml' # generated in the DLC pipeline
    # '.\\arena_files\\fisheye_maps.npy' and '.\\arena_files\\inverse_fisheye_maps.npy' are included in repo for the standard TRABLAB camera
    # However, if you do not want to apply a fisheye correction, write None for the file names
    options.fisheye_correction_file = None
    options.inverted_fisheye_correction_file = None

    '''     PROCESSING PARAMETERS       '''
    options.show_extracted_parameters_plot = False
    options.median_filter_duration_in_frames = 7 # (7 in my well labeled setup)
    options.minimum_confidence_from_dlc = 0 # if under this, set tracking coordinate to nan and interpolate (.999 in my well labeled setup)
    options.max_error_from_dlc = 100 # in pixels. if under this, set tracking coordinate to nan and interpolate (60 in my well labeled setup)

    '''     VISUALIZATION PARAMETERS        '''
    options.show_loom_in_video_clip = True
    options.mouse_silhouette_size = 18 # how big is the mouse in the rendering
    options.speed_thresholds = [7.2, 10.8, 14.4] # for coloration by speed, in pixel / frame; corresponds to 30 / 45 / 60 cm/s in my setup
    options.dist_to_make_red = 150 # leave a red trace for previous escapes