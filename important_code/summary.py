import cv2
import imageio
import numpy as np
import os
import glob
# from Config import dlc_options
# dlc_config_settings = dlc_options()


'''
Mine the extant plots to create a new Super Plot
'''

def summarize(self):

    # Find the old folder, named after the experiment and the mouse
    original_save_folder = os.path.join(dlc_config_settings['clips_folder'], session['Metadata'].experiment,
                               str(session['Metadata'].mouse_id))

    # Save to a new folder named after the experiment and the mouse with the word 'summary'
    new_save_folder = os.path.join(dlc_config_settings['clips_folder'], session['Metadata'].experiment,
                               str(session['Metadata'].mouse_id) + '_summary')
    if not os.path.isdir(new_save_folder):
        os.makedirs(new_save_folder)

    # List the dlc_history, spont_homings, exploration, and expl_recent contents of the old folder
    dlc_history_files = glob.glob(original_save_folder + '\\*history.tif')
    spont_homings_files = glob.glob(original_save_folder + '\\*spont_homings.tif')
    exploration_files = glob.glob(original_save_folder + '\\*exploration.tif')
    journey_files = glob.glob(original_save_folder + '\\*exploration_recent.tif')
    SR_files = glob.glob(original_save_folder + '\\*SR.tif')

    # make the summary plot for each trial
    for trial in range(len(dlc_history_files)):

        # parameters
        border_size = 40

        videoname = os.path.basename(dlc_history_files[trial]).split("')_")[0] + "')"

        # open the images
        dlc_history_image = cv2.imread(dlc_history_files[trial])
        spont_homings_image = cv2.imread(spont_homings_files[trial])
        exploration_image = cv2.imread(exploration_files[trial])
        journey_image = cv2.imread(journey_files[trial])
        SR_image = cv2.imread(SR_files[trial])

        # create a new, super image
        images_shape = dlc_history_image.shape
        super_image = np.zeros((int(images_shape[0] * 1.5), int(images_shape[1] * 1.5), 3)).astype(np.uint8)

        # add the dlc image
        super_image[int(1.5 * border_size):int(.5 * border_size) + images_shape[0], 0:images_shape[1] - border_size ] = \
            dlc_history_image[border_size:, :-border_size]

        # add the auxiliary images
        super_image[ int(1.5 * border_size + (images_shape[0]-border_size)/4) : int(.5 * border_size + images_shape[0] - (images_shape[0]-border_size)/4),
         int(images_shape[1] - border_size) : -int(1.5 * border_size) ] = \
            cv2.resize( spont_homings_image[border_size:, :-border_size], (int( (images_shape[0]-border_size)/2), int((images_shape[1]-border_size)/2)), cv2.INTER_CUBIC)

        # super_image[ int(1.5 * border_size + (images_shape[0]-border_size)/4) : int(.5 * border_size + images_shape[0] - (images_shape[0]-border_size)/4),
        #  int(images_shape[1] - border_size) : -int(1.5 * border_size) ] = \
        #     cv2.resize( SR_image, (int( (images_shape[0]-border_size)/2), int((images_shape[1]-border_size)/2)), cv2.INTER_CUBIC)

        super_image[ int(.5 * border_size + images_shape[0]) : int(1.5 * images_shape[0]),
         int((images_shape[1])/4) : int( (images_shape[1]) *3/4) ] = \
            cv2.resize( exploration_image[border_size:,:], (int((images_shape[0])/2), int((images_shape[1]-border_size)/2)), cv2.INTER_CUBIC)

        super_image[ int(.5 * border_size + images_shape[0]-50) : int(1.5 * images_shape[0]-50),
            int(images_shape[1] - border_size)-50: -int(1.5 * border_size)-50 ] = \
            cv2.resize( journey_image[border_size:, :-border_size], (int((images_shape[0]-border_size)/2), int((images_shape[1]-border_size)/2)), cv2.INTER_CUBIC)

        # super_image[ int(.5 * border_size + images_shape[0]-50) : int(1.5 * images_shape[0]-50),
        #     int(images_shape[1] - border_size)-50: -int(1.5 * border_size)-50 ] = \
        #     cv2.resize( SR_image, (int((images_shape[0]-border_size)/2), int((images_shape[1]-border_size)/2)), cv2.INTER_CUBIC)


        # add the title and border
        super_image[:int(1.5 * border_size), :-int(1.5 * border_size)] = \
            cv2.resize(dlc_history_image[:border_size, :-border_size], (super_image.shape[1] - int(1.5 * border_size), int(1.5 * border_size)), cv2.INTER_CUBIC)
        super_image[:, -int(1.5 * border_size):] = \
            cv2.resize(dlc_history_image[:, -border_size:], (int(1.5 * border_size), super_image.shape[0]), cv2.INTER_CUBIC)

        # make it a bit smaller
        cv2.imshow('super image', super_image)
        cv2.waitKey(100)

        # recolor and save image
        super_image = cv2.cvtColor(super_image, cv2.COLOR_BGR2RGB)
        imageio.imwrite(os.path.join(new_save_folder, videoname + '.tif'), super_image)

    print('Experiment summary plots saved.')

