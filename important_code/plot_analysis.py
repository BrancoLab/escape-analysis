import numpy as np
import cv2
import warnings
warnings.filterwarnings('ignore')
import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
import os
import scipy
import imageio
from scipy.ndimage import gaussian_filter1d, gaussian_filter
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from matplotlib.colors import ListedColormap
import statsmodels.api as sm
import pandas as pd
from statsmodels.stats.anova import AnovaRM
from sklearn import linear_model
from helper_code.registration_funcs import model_arena, get_arena_details
from helper_code.processing_funcs import speed_colors
from helper_code.analysis_funcs import *
from important_code.shuffle_test import permutation_test, permutation_correlation
plt.rcParams.update({'font.size': 30})

def plot_traversals(self):
    '''     plot all traversals across the arena        '''
    # initialize parameters
    sides = ['back', 'front']
    # sides = ['back']
    types = ['spontaneous'] #, 'evoked']
    fast_color = np.array([.5, 1, .5])
    slow_color = np.array([1, .9, .9])
    edge_vector_color = np.array([1, .95, .85])
    homing_vector_color = np.array([.725, .725, .725])
    edge_vector_color =  np.array([.98, .9, .6])**4
    homing_vector_color = np.array([0, 0, 0])

    non_escape_color = np.array([0,0,0])
    condition_colors = [[.5,.5,.5], [.3,.5,.8], [0,.7,1]]
    time_thresh = 15 #20 for ev comparison
    speed_thresh = 2
    p = 0
    HV_cutoff = .681 # .5 for exploratory analysis
    # initialize figures
    fig, fig2, fig3, ax, ax2, ax3 = initialize_figures_traversals(self) #, types = len(types)+1)
    # initialize lists for stats
    all_data = []
    all_conditions = []
    edge_vector_time_all = np.array([])
    # loop over spontaneous vs evoked
    for t, type in enumerate(types):
        # loop over experiments and conditions
        for c, (experiment, condition) in enumerate(zip(self.experiments, self.conditions)):
            strategies = [0, 0, 0]
            # extract experiments from nested list
            sub_experiments, sub_conditions = extract_experiments(experiment, condition)
            # initialize the arena
            arena, arena_color, scaling_factor, obstacle = initialize_arena(self, sub_experiments, sub_conditions)
            path_ax, path_fig = get_arena_plot(obstacle, sub_conditions, sub_experiments)
            # initialize edginess
            all_traversals_edgy = {}
            all_traversals_homy = {}
            proportion_edgy = {}
            for s in sides:
                all_traversals_edgy[s] = []
                all_traversals_homy[s] = []
                proportion_edgy[s] = []

            m = 0
            # loop over each experiment and condition
            for e, (experiment, condition) in enumerate(zip(sub_experiments, sub_conditions)):
                # loop over each mouse in the experiment
                for i, mouse in enumerate(self.analysis[experiment][condition]['back traversal']):
                    mouse_data = []
                    print(mouse)

                    # loop over back and front sides
                    for s, start in enumerate(sides):
                        if start == 'front' and type == 'evoked': continue

                        # find all the paths across the arena
                        traversal = self.analysis[experiment][condition][start + ' traversal'][mouse]
                        # get the duration of those paths
                        # duration = traversal[t*5+3]
                        if traversal:

                            if traversal[t*5]:
                                x_end_loc = np.array([x_loc[-1] * scaling_factor for x_loc in np.array(traversal[t * 5 + 0])[:, 0]])
                                if traversal[4] < 10: continue

                                number_of_edge_vectors = np.sum((np.array(traversal[t*5+3]) < speed_thresh) * \
                                                                       (np.array(traversal[t*5+2]) > HV_cutoff) * \
                                                                        # (abs(x_end_loc - 50) < 30) * \
                                                                       (np.array(traversal[t*5+1]) < time_thresh*30*60) ) / min(traversal[4], time_thresh) * time_thresh
                                # print(traversal[4])
                                number_of_homing_vectors = np.sum((np.array(traversal[t*5+3]) < speed_thresh) * \
                                                                       (np.array(traversal[t*5+2]) < HV_cutoff) * \
                                                                        # (abs(x_end_loc - 50) < 30) * \
                                                                       (np.array(traversal[t*5+1]) < time_thresh*30*60) )/ min(traversal[4], time_thresh) * time_thresh
                                all_traversals_edgy[start].append( number_of_edge_vectors )
                                all_traversals_homy[start].append(number_of_homing_vectors)
                                # print(number_of_edge_vectors)
                                mouse_data.append(number_of_edge_vectors)


                                # get the time of edge vectors
                                if condition == 'obstacle' and 'wall' in experiment:
                                    edge_vector_idx = ( (np.array(traversal[t * 5 + 3]) < speed_thresh) * (np.array(traversal[t * 5 + 2]) > HV_cutoff) )
                                    edge_vector_time = np.array(traversal[t*5+1])[edge_vector_idx] / 30 / 60
                                    edge_vector_time_all = np.concatenate((edge_vector_time_all, edge_vector_time))

                                # prop_edgy = np.sum((np.array(traversal[t*5 + 3]) < speed_thresh) * \
                                #                                       (np.array(traversal[t*5 + 2]) > HV_cutoff) * \
                                #                                         (np.array(traversal[t * 5 + 1]) < time_thresh * 30 * 60)) / \
                                #                               np.sum((np.array(traversal[t * 5 + 3]) < speed_thresh) * \
                                #                                      (np.array(traversal[t * 5 + 1]) < time_thresh * 30 * 60))
                            else:
                                all_traversals_edgy[start].append(0)
                                all_traversals_homy[start].append(0)


                            # if np.isnan(prop_edgy): prop_edgy = .5
                            # prop_edgy =  prop_edgy / .35738
                            # proportion_edgy[start].append(prop_edgy)


                            traversal_coords = np.array(traversal[t*5+0])
                            pre_traversal = np.array(traversal[10])
                        else:
                            # all_traversals_edginess[start].append(0)
                            continue
                        m += .5

                        # loop over all paths
                        show = False
                        if show and traversal:
                            for trial in range(traversal_coords.shape[0]):
                                # make sure it qualifies
                                if traversal[t * 5 + 3][trial] > speed_thresh: continue
                                if traversal[t*5+1][trial] > time_thresh*30*60: continue
                                if not len(pre_traversal[0][0]): continue
                                # if abs(traversal_coords[trial][0][-1]*scaling_factor - 50) > 30: continue

                                # downsample to get even coverage
                                # if c == 2 and np.random.random() > (59 / 234): continue
                                # if c == 1 and np.random.random() > (59 / 94): continue

                                if traversal[t*5+2][trial]> HV_cutoff: plot_color = edge_vector_color
                                else: plot_color = homing_vector_color

                                display_traversal(scaling_factor, traversal_coords, pre_traversal, trial, path_ax, plot_color)



                    if mouse_data:
                        # all_data.append(mouse_data)
                        all_conditions.append(c)
            # save image
            path_fig.savefig(os.path.join(self.summary_plots_folder, self.labels[c] + ' traversals.eps'), format='eps', bbox_inches='tight', pad_inches=0)

            # plot the data
            if type == 'spontaneous' and len(sides) > 1:
                plot_number_edgy = np.array(all_traversals_edgy['front']).astype(float) + np.array(all_traversals_edgy['back']).astype(float)
                plot_number_homy = np.array(all_traversals_homy['front']).astype(float) + np.array(all_traversals_homy['back']).astype(float)
                print(np.sum(plot_number_edgy + plot_number_homy))
                # plot_proportion_edgy = (np.array(proportion_edgy['front']).astype(float) + np.array(proportion_edgy['back']).astype(float)) / 2
                plot_proportion_edgy = plot_number_edgy / (plot_number_edgy + plot_number_homy)
                all_data.append(plot_number_edgy)
            else:
                plot_number_edgy = np.array(all_traversals_edgy[sides[0]]).astype(float)
                plot_number_homy = np.array(all_traversals_homy[sides[0]]).astype(float)

                plot_proportion_edgy = plot_number_edgy / (plot_number_edgy + plot_number_homy)
                # plot_proportion_edgy = np.array(proportion_edgy[sides[0]]).astype(float)

            for i, (plot_data, ax0) in enumerate(zip([plot_number_edgy, plot_number_homy], [ax, ax3])): #, plot_proportion_edgy , ax2
                print(plot_data)
                print(np.sum(plot_data))
                # plot each trial
                # scatter_axis = scatter_the_axis( (p*4/3+.5/3), plot_data)
                ax0.scatter(np.ones_like(plot_data)* (p*4/3+.5/3)* 3 - .2, plot_data, color=[0,0,0, .4], edgecolors='none', s=25, zorder=99)
                # do kde
                # if i==0: bw = .5
                # else: bw = .02
                bw = .5

                kde = fit_kde(plot_data, bw=bw)
                plot_kde(ax0, kde, plot_data, z=4 * p + .8, vertical=True, normto=.3, color=[.5, .5, .5], violin=False, clip=True)
                ax0.plot([4 * p + -.2, 4 * p + -.2], [np.percentile(plot_data, 25), np.percentile(plot_data, 75)], color = [0,0,0])
                ax0.plot([4 * p + -.4, 4 * p + -.0], [np.percentile(plot_data, 50), np.percentile(plot_data, 50)], color = [1,1,1], linewidth = 2)
                # else:
                #     # kde = fit_kde(plot_data, bw=.03)
                #     # plot_kde(ax0, kde, plot_data, z=4 * p + .8, vertical=True, normto=1.2, color=[.5, .5, .5], violin=False, clip=True)
                #     bp = ax0.boxplot([plot_data, [0, 0]], positions=[4 * p + -.2, -10], showfliers=False, zorder=99)
                #     ax0.set_xlim([-1, 4 * len(self.experiments) - 1])

            p+=1

            # plot a stacked bar of strategies
            # fig3 = plot_strategies(strategies, homing_vector_color, non_escape_color, edge_vector_color)
            # fig3.savefig(os.path.join(self.summary_plots_folder, 'Traversal categories - ' + self.labels[c] + '.png'), format='png', bbox_inches = 'tight', pad_inches = 0)
            # fig3.savefig(os.path.join(self.summary_plots_folder, 'Traversal categories - ' + self.labels[c] + '.eps'), format='eps', bbox_inches = 'tight', pad_inches = 0)

    # make timing hist
    plt.figure()
    bins = np.arange(0,22.5,2.5)
    plt.hist(edge_vector_time_all, bins = bins, color = [0,0,0], weights = np.ones_like(edge_vector_time_all) / 2.5 / m) #condition_colors[c])
    plt.ylim([0,2.1])
    plt.show()

    # # save the plot
    fig.savefig(os.path.join(self.summary_plots_folder, 'Traversal # EVS comparison.png'), format='png', bbox_inches='tight', pad_inches=0)
    fig.savefig(os.path.join(self.summary_plots_folder, 'Traversal # EVS comparison.eps'), format='eps', bbox_inches='tight', pad_inches=0)
    fig3.savefig(os.path.join(self.summary_plots_folder, 'Traversal # HVS comparison.png'), format='png', bbox_inches='tight', pad_inches=0)
    fig3.savefig(os.path.join(self.summary_plots_folder, 'Traversal # HVS comparison.eps'), format='eps', bbox_inches='tight', pad_inches=0)



    group_A = [[d] for d in all_data[0]]
    group_B = [[d] for d in all_data[2]]
    permutation_test(group_A, group_B, iterations = 100000, two_tailed = False)

    group_A = [[d] for d in all_data[2]]
    group_B = [[d] for d in all_data[1]]
    permutation_test(group_A, group_B, iterations = 10000, two_tailed = True)

    # fig2.savefig(os.path.join(self.summary_plots_folder, 'Traversal proportion edgy.png'), format='png', bbox_inches='tight', pad_inches=0)
    # fig2.savefig(os.path.join(self.summary_plots_folder, 'Traversal proportion edgy.eps'), format='eps', bbox_inches='tight', pad_inches=0)
    plt.show()


def plot_speed_traces(self, speed = 'absolute'):
    '''     plot the speed traces       '''
    max_speed = 60
    # loop over experiments and conditions
    for c, (experiment, condition) in enumerate(zip(self.experiments, self.conditions)):
        # extract experiments from nested list
        sub_experiments, sub_conditions = extract_experiments(experiment, condition)
        # get the number of trials
        number_of_trials = get_number_of_trials(sub_experiments, sub_conditions, self.analysis)
        number_of_mice = get_number_of_mice(sub_experiments, sub_conditions, self.analysis)
        RT, end_idx, scaling_factor, speed_traces, subgoal_speed_traces, time, time_axis, trial_num = \
            initialize_variables(number_of_trials, self,sub_experiments)
        # create custom colormap
        colormap = speed_colormap(scaling_factor, max_speed, n_bins=256, v_min=0, v_max=max_speed)
        # loop over each experiment and condition
        for e, (experiment, condition) in enumerate(zip(sub_experiments, sub_conditions)):
            # loop over each mouse
            for i, mouse in enumerate(self.analysis[experiment][condition]['speed']):
                # control analysis
                if self.analysis_options['control'] and not mouse=='control': continue
                if not self.analysis_options['control'] and mouse=='control': continue
                # loop over each trial
                for trial in range(len(self.analysis[experiment][condition]['speed'][mouse])):
                    if trial > 2: continue
                    trial_num = fill_in_trial_data(RT, condition, end_idx, experiment, mouse, scaling_factor, self,
                                       speed_traces, subgoal_speed_traces, time, trial, trial_num)
        # print some useful metrics
        print_metrics(RT, end_idx, number_of_mice, number_of_trials)
        # put the speed traces on the plot
        fig = show_speed_traces(colormap, condition, end_idx, experiment, number_of_trials, speed, speed_traces, subgoal_speed_traces, time_axis, max_speed)
        # save the plot
        fig.savefig(os.path.join(self.summary_plots_folder,'Speed traces - ' + self.labels[c] + '.png'), format='png', bbox_inches = 'tight', pad_inches = 0)
        fig.savefig(os.path.join(self.summary_plots_folder,'Speed traces - ' + self.labels[c] + '.eps'), format='eps', bbox_inches = 'tight', pad_inches = 0)
    plt.show()
    print('done')




def plot_escape_paths(self):
    '''     plot the escape paths       '''
    # initialize parameters
    edge_vector_color = [np.array([1, .95, .85]), np.array([.98, .9, .6])**4]
    homing_vector_color = [ np.array([.725, .725, .725]), np.array([0, 0, 0])]
    non_escape_color = np.array([0,0,0])

    fps = 30
    escape_duration = 18 #6 #9 for food # 18 for U
    min_distance_to_shelter = 30
    HV_cutoff = 0.681  #.75 #.7

    # initialize all data for stats
    all_data = [[], [], [], []]
    all_conditions = []

    # loop over experiments and conditions
    for c, (experiment, condition) in enumerate(zip(self.experiments, self.conditions)):
        # extract experiments from nested list
        sub_experiments, sub_conditions = extract_experiments(experiment, condition)
        # initialize the arena
        arena, arena_color, scaling_factor, obstacle = initialize_arena(self, sub_experiments, sub_conditions)
        # more arena stuff for this analysis type
        arena_reference = arena_color.copy()
        arena_color[arena_reference == 245] = 255
        get_arena_details(self, experiment=sub_experiments[0])
        shelter_location = [s / scaling_factor / 10 for s in self.shelter_location]
        # initialize strategy array
        strategies = np.array([0,0,0])

        path_ax, path_fig = get_arena_plot(obstacle, sub_conditions, sub_experiments)
        # loop over each experiment and condition
        for e, (experiment, condition) in enumerate(zip(sub_experiments, sub_conditions)):
            if 'void' in experiment or 'dark' in experiment or ('off' in experiment and condition == 'no obstacle') or 'quick' in experiment:
                escape_duration = 18
            elif 'food' in experiment:
                escape_duration = 9
            else:
                escape_duration = 12
            # loop over each mouse
            for i, mouse in enumerate(self.analysis[experiment][condition]['speed']):
                print(mouse)
                # control analysis
                if self.analysis_options['control'] and not mouse=='control': continue
                if not self.analysis_options['control'] and mouse=='control': continue

                # color based on visual vs tactile obst avoidance
                # if mouse == 'CA7190' or mouse == 'CA3210' or mouse == 'CA3155' or mouse == 'CA8100':
                #     edge_vector_color = [np.array([.6, .4, .99]),np.array([.6, .4, .99])]
                #     homing_vector_color = [np.array([.6, .4, .99]),np.array([.6, .4, .99])]
                # else:
                #     edge_vector_color = [np.array([.8, .95, 0]),np.array([.8, .95, 0])]
                #     homing_vector_color = [np.array([.8, .95, 0]),np.array([.8, .95, 0])]

                # show escape paths
                show_escape_paths(HV_cutoff, arena, arena_color, arena_reference, c, condition, edge_vector_color, escape_duration, experiment, fps,
                                  homing_vector_color, min_distance_to_shelter, mouse, non_escape_color, scaling_factor, self, shelter_location, strategies, path_ax,
                                  determine_strategy = False) #('dark' in experiment and condition=='obstacle'))


        # save image
        # scipy.misc.imsave(os.path.join(self.summary_plots_folder, 'Escape paths - ' + self.labels[c] + '.png'), arena_color[:,:,::-1])
        imageio.imwrite(os.path.join(self.summary_plots_folder, 'Escape paths - ' + self.labels[c] + '.png'), arena_color[:,:,::-1])
        path_fig.savefig(os.path.join(self.summary_plots_folder, 'Escape plot - ' + self.labels[c] + '.png'), format='png', bbox_inches='tight', pad_inches=0)
        path_fig.savefig(os.path.join(self.summary_plots_folder, 'Escape plot - ' + self.labels[c] + '.eps'), format='eps', bbox_inches='tight', pad_inches=0)
        # plot a stacked bar of strategies
        fig = plot_strategies(strategies, homing_vector_color, non_escape_color, edge_vector_color)
        fig.savefig(os.path.join(self.summary_plots_folder, 'Escape categories - ' + self.labels[c] + '.png'), format='png', bbox_inches = 'tight', pad_inches = 0)
        fig.savefig(os.path.join(self.summary_plots_folder, 'Escape categories - ' + self.labels[c] + '.eps'), format='eps', bbox_inches = 'tight', pad_inches = 0)
    plt.show()
    print('escape')




# strategies = np.array([4,5,0])
# fig = plot_strategies(strategies, homing_vector_color, non_escape_color, edge_vector_color)
# plt.show()
# fig.savefig(os.path.join(self.summary_plots_folder, 'Trajectory by previous edge-vectors 2.png'), format='png', bbox_inches='tight', pad_inches=0)
# fig.savefig(os.path.join(self.summary_plots_folder, 'Trajectory by previous edge-vectors 2.eps'), format='eps', bbox_inches='tight', pad_inches=0)

# group_A = [[0],[1],[0,0,0],[0,0],[0,1],[1,0],[0,0,0]]
# group_B = [[1,0,0],[0,0,0,0],[0,0,0],[1,0,0],[0,0,0]]
# permutation_test(group_B, group_A, iterations = 10000, two_tailed = False)

obstacle = [[0],[1],[0,0,0],[0,0],[0,1],[1],[0,0,0], [1]]
# obstacle_exp = [[0,1],[0,0,0,0,1],[0,1],[0]]
open_field = [[1,0,0,0,0],[0,0,0,0,0],[0,0,0,0],[1,0,0,0,0,0],[0,0,0,0,0,0],[0,0,0,0,0,0,0,0]]
# U_shaped = [[0,1],[1,1], [1,1], [0,0,1], [0,0,0], [0], [1], [0], [0,1], [0,1,0,0], [0,0,0]]
# permutation_test(open_field, obstacle, iterations = 10000, two_tailed = False)


# do same edgy homing then stop to both
obstacle = [[0],[1],[0,0,0],[0,0],[0,1],[1],[0,0,0], [1], [1], [0,0,0]]
open_field = [[1],[0,0,0],[0,0,0],[1,0,0],[0,0,0],[0,0,1]] #stop at 3 trials

# do same edgy homing then stop to both --> exclude non escapes
obstacle = [[0],[1],[0,0,0],[0],[0,1],[1],[0,0,0], [1], [1], [0,0,0]]
open_field = [[1],[0,0],[0,0,0],[1,0,0],[0,0,0],[0,1]] #stop at 3 trials



def plot_edginess(self):
    # initialize parameters
    fps = 30
    escape_duration = 12 #9 #6
    HV_cutoff = .681 #.681
    ETD = 10 #10
    traj_loc = 40

    edge_vector_color = np.array([.98, .9, .6])**5
    edge_vector_color = np.array([.99, .94, .6]) ** 3
    # edge_vector_color = np.array([.99, .95, .6]) ** 5
    homing_vector_color = np.array([0, 0, 0])

    # homing_vector_color = np.array([.85, .65, .8])
    # edge_vector_color = np.array([.65, .85, .7])


    # colors for diff conditions
    colors = [np.array([.7, 0, .3]), np.array([0, .8, .5])]
    colors = [np.array([.3,.3,.3]), np.array([1, .2, 0]), np.array([0, .8, .4]), np.array([0, .7, .9])]
    colors = [np.array([.3, .3, .3]), np.array([1, .2, 0]), np.array([.7, 0, .7]), np.array([0, .7, .9]), np.array([0,1,0])]
    # colors = [np.array([0, 0, 0]), np.array([0, 0, 0]),np.array([0, 0, 0]), np.array([0, 0, 0])]
    offset = [0,.2, .2, 0]

    # initialize figures
    fig, fig2, fig3, fig4, _, ax, ax2, ax3 = initialize_figures(self)
    # initialize all data for stats
    all_data = [[],[],[],[]]
    all_conditions = []
    mouse_ID = []; m = 1
    dist_data_EV_other_all = []

    delta_ICs, delta_x_end = [], []
    time_to_shelter, was_escape = [], []

    repetitions = 1
    for rand_select in range(repetitions):
        m = -1
        # loop over experiments and conditions
        for c, (experiment, condition) in enumerate(zip(self.experiments, self.conditions)):
            num_trials_total = 0
            num_trials_escape = 0
            # extract experiments from nested list
            sub_experiments, sub_conditions = extract_experiments(experiment, condition)
            # get the number of trials
            number_of_trials = get_number_of_trials(sub_experiments, sub_conditions, self.analysis)
            number_of_mice = get_number_of_mice(sub_experiments, sub_conditions, self.analysis)
            t_total = 0
            # initialize array to fill in with each trial's data
            edginess, end_idx, time_since_down, time_to_shelter, time_to_shelter_all, prev_edginess, scaling_factor, time_in_center, trial_num, _, _, dist_to_SH, dist_to_other_SH = \
                initialize_variable_edginess(number_of_trials, self, sub_experiments)
            mouse_ID_trial = edginess.copy()
            # loop over each experiment and condition
            for e, (experiment, condition) in enumerate(zip(sub_experiments, sub_conditions)):
                if 'void' in experiment or 'dark' in experiment or ('off' in experiment and condition == 'no obstacle') or 'quick' in experiment:
                    escape_duration = 18
                elif 'food' in experiment:
                    escape_duration = 12
                else: escape_duration = 12
                # elif 'up' in experiment and 'probe' in condition:
                #     escape_duration = 12
                # loop over each mouse
                for i, mouse in enumerate(self.analysis[experiment][condition]['start time']):
                    m+=1
                    # initialize mouse data for stats
                    mouse_data = [[],[],[],[]]
                    print(mouse)
                    skip_mouse = False
                    if self.analysis_options['control'] and not mouse=='control': continue
                    if not self.analysis_options['control'] and mouse=='control': continue
                    # loop over each trial
                    prev_homings = []
                    x_edges_used = []
                    t = 0

                    for trial in range(len(self.analysis[experiment][condition]['end time'][mouse])):
                        trial_num += 1
                        # impose conditions
                        if 'food' in experiment:
                            if t > 12: continue
                            if condition == 'no obstacle' and self.analysis[experiment][condition]['start time'][mouse][trial] < 20: continue
                            num_trials_total += 1
                        elif 'void' in experiment:
                            if t > 5: continue
                        else:
                            if t>2: continue
                            # if trial > 2: continue
                            num_trials_total += 1
                            # if trial!=2: continue
                            # if 'off' in experiment and trial: continue

                            # if trial < 3 and 'wall down' in experiment: continue
                            # if condition == 'obstacle' and not 'non' in experiment and \
                            #         self.analysis[experiment][condition]['start time'][mouse][trial] < 20: continue
                            # if c == 0 and not (trial > 0): continue
                            # if c == 1 and not (trial): continue
                            # if c == 2 and not (trial == 0): continue
                        # if trial and ('lights on off' in experiment and not 'baseline' in experiment): continue
                        if 'Square' in experiment:
                            HV_cutoff = .56
                            HV_cutoff = 0

                            y_idx = self.analysis[experiment][condition]['path'][mouse][trial][1]
                            if y_idx[0] * scaling_factor > 50: continue

                        else:
                            # skip certain trials
                            y_start = self.analysis[experiment][condition]['path'][mouse][trial][1][0] * scaling_factor
                            x_start = self.analysis[experiment][condition]['path'][mouse][trial][0][0] * scaling_factor
                            # print(y_start)
                            # print(x_start)
                            if y_start > 25: continue
                            if abs(x_start-50) > 30: continue

                        end_idx[trial_num] = self.analysis[experiment][condition]['end time'][mouse][trial]
                        RT = self.analysis[experiment][condition]['RT'][mouse][trial]
                        if np.isnan(end_idx[trial_num]) or (end_idx[trial_num] > escape_duration * fps):
                            # if not ('up' in experiment and 'probe' in condition and not np.isnan(RT)):
                            # mouse_data[3].append(0)
                            continue


                        '''     check for previous edgy homings         '''

                        # if 'dark' in experiment or True:
                        # num_prev_edge_vectors, x_edge = get_num_edge_vectors(self, experiment, condition, mouse, trial)
                        # #     print(num_prev_edge_vectors)
                        # if num_prev_edge_vectors and c: continue
                        # if not num_prev_edge_vectors and not c: continue


                            # if num_prev_edge_vectors < 3 and (c==0):  continue
                            # if num_prev_edge_vectors > 0 and c < 4:  continue
                            # if t>1 and c == 2: continue
                            # if num_prev_edge_vectors >= 2:  print('prev edgy homing'); continue

                            # if x_edge in x_edges_used: print('prev edgy escape'); continue
                            #
                            # print('-----------' + mouse + '--------------')
                            #
                            # if self.analysis[experiment][condition]['edginess'][mouse][trial] <= HV_cutoff:
                            #     print('       HV        ')
                            # else:
                            #     print('       EDGY        ')
                            #     # edgy trial has occurred
                            #     print('EDGY TRIAL ' + str(trial))
                            #     x_edges_used.append(x_edge)
                            #
                            # # select only *with* prev homings
                            # if not num_prev_edge_vectors:
                            #     if not x_edge in x_edges_used:
                            #         if self.analysis[experiment][condition]['edginess'][mouse][trial] > HV_cutoff:
                            #             x_edges_used.append(x_edge)
                            #         continue
                        # print(t)
                        num_trials_escape += 1
                        # add data
                        edginess[trial_num] = self.analysis[experiment][condition]['edginess'][mouse][trial]
                        time_since_down[trial_num] = np.sqrt((x_start - 50)**2 + (y_start - 50)**2  )# self.analysis[experiment][condition]['start angle'][mouse][trial]

                        print(edginess[trial_num])
                        if 'Square' in experiment:
                            if edginess[trial_num] <=-.3: # and False: #.15
                                edginess[trial_num] = np.nan
                                continue

                            # edginess to current edge as opposed to specific edge
                            if (('moves left' in experiment and condition == 'no obstacle') \
                                or ('moves right' in experiment and condition== 'obstacle')): # and False:
                                if edginess[trial_num] <= -0: # and False:
                                    edginess[trial_num] = np.nan
                                    continue

                                edginess[trial_num] = edginess[trial_num] - 1

                            # shelter edginess
                            if False:
                                y_pos = self.analysis[experiment][condition]['path'][mouse][trial][1][:int(end_idx[trial_num])] * scaling_factor
                                x_pos = self.analysis[experiment][condition]['path'][mouse][trial][0][:int(end_idx[trial_num])] * scaling_factor

                                # get the latter phase traj
                                y_pos_1 = 55
                                y_pos_2 = 65
                                x_pos_1 = x_pos[np.argmin(abs(y_pos - y_pos_1))]
                                x_pos_2 = x_pos[np.argmin(abs(y_pos - y_pos_2))]

                                #where does it end up
                                slope = (y_pos_2 - y_pos_1) / (x_pos_2 - x_pos_1)
                                intercept = y_pos_1 - x_pos_1 * slope
                                x_pos_proj = (80 - intercept) / slope

                                # compared to
                                x_pos_shelter_R = 40    #40.5 # defined as mean of null dist

                                # if 'long' in self.labels[c]:
                                #     x_pos_shelter_R += 18

                                # compute the metric
                                shelter_edginess = (x_pos_proj - x_pos_shelter_R) / 18
                                edginess[trial_num] = -shelter_edginess


                            # if condition == 'obstacle' and 'left' in experiment:edginess[trial_num] = -edginess[trial_num] # for putting conditions together

                            # get previous edginess #TEMPORARY COMMENT
                        # if not t:
                        #     SH_data = self.analysis[experiment][condition]['prev homings'][mouse][-1]
                        #     time_to_shelter.append(np.array(SH_data[2]))
                        #     was_escape.append(np.array(SH_data[4]))

                        if False: # or True:
                            time_to_shelter, SR = get_prev_edginess(ETD, condition, experiment, mouse, prev_edginess, dist_to_SH, dist_to_other_SH,
                                                                    scaling_factor, self, traj_loc, trial, trial_num, edginess, delta_ICs, delta_x_end)
                            print(prev_edginess[trial_num])
                            print(trial + 1)
                            print('')
                            # get time in center
                            # time_in_center[trial_num] = self.analysis[experiment][condition]['time exploring obstacle'][mouse][trial]
                            # time_in_center[trial_num] = num_PORHVs
                            # if num_PORHVs <= 1:
                            #     edginess[trial_num] = np.nan
                            #     continue
                            # if (prev_edginess[trial_num] < HV_cutoff and not t) or skip_mouse:
                            #     edginess[trial_num] = np.nan
                            #     skip_mouse = True
                            #     continue



                            '''     qualify by prev homings         '''
                            # if prev_edginess[trial_num] < .4: # and c:
                            #     edginess[trial_num] = np.nan
                            #     prev_edginess[trial_num] = np.nan
                            #     continue


                            num_prev_edge_vectors, x_edge = get_num_edge_vectors(self, experiment, condition, mouse, trial, ETD = 10)
                            # print(str(num_prev_edge_vectors) + ' EVs')
                            #
                            # if not num_prev_edge_vectors >= 1 and c ==0:
                            #     edginess[trial_num] = np.nan
                            #     t+=1
                            #     continue
                            # if not num_prev_edge_vectors < 1 and c ==1:
                            #     edginess[trial_num] = np.nan
                            #     t+=1
                            #     continue

                            # print(num_prev_edge_vectors)
                            # if num_prev_edge_vectors !=0 and c==3:
                            #     edginess[trial_num] = np.nan
                            #     t+=1
                            #     continue
                            # if num_prev_edge_vectors != 1 and c == 2:
                            #     edginess[trial_num] = np.nan
                            #     t += 1
                            #     continue
                            # if num_prev_edge_vectors != 2 and num_prev_edge_vectors != 3 and c ==1:
                            #     edginess[trial_num] = np.nan
                            #     t += 1
                            #     continue
                            #
                            # if num_prev_edge_vectors < 4 and c ==0:
                            #     edginess[trial_num] = np.nan
                            #     t += 1
                            #     continue
                            #
                            # print(trial + 1)
                            # print(prev_edginess[trial_num])
                            # print(edginess[trial_num])
                            # print('')

                            # print(t)

                        # get time since obstacle removal?
                        # time_since_down[trial_num] = self.analysis[experiment][condition]['start time'][mouse][trial] - self.analysis[experiment]['probe']['start time'][mouse][0]
                        # add data for stats

                        mouse_data[0].append(int(edginess[trial_num] > HV_cutoff))
                        mouse_data[1].append(edginess[trial_num])
                        mouse_data[2].append(prev_edginess[trial_num])
                        mouse_data[3].append(self.analysis[experiment][condition]['start time'][mouse][trial] - self.analysis[experiment][condition]['start time'][mouse][0])
                        mouse_ID_trial[trial_num] = m

                        t += 1
                        t_total += 1

                    #append data for stats
                    if mouse_data[0]:
                        all_data[0].append(mouse_data[0])
                        all_data[1].append(mouse_data[1])
                        all_data[2].append(mouse_data[2])
                        all_data[3].append(mouse_data[3])
                        all_conditions.append(c)
                        mouse_ID.append(m); m+= 1

                    else:
                        print(mouse)
                        print('0 trials')
                    # get prev homings
                    time_to_shelter_all.append(time_to_shelter)

            dist_data_EV_other_all = np.append(dist_data_EV_other_all, dist_to_other_SH[edginess > HV_cutoff])

            # print(t_total)

            '''     plot edginess by condition     '''
            # get the data
            # data = abs(edginess)
            data = edginess

            plot_data = data[~np.isnan(data)]
            # print(np.percentile(plot_data, 25))
            # print(np.percentile(plot_data, 50))
            # print(np.percentile(plot_data, 75))
            # print(np.mean(plot_data > HV_cutoff))

            # plot each trial
            scatter_axis = scatter_the_axis(c, plot_data)
            ax.scatter(scatter_axis[plot_data>HV_cutoff], plot_data[plot_data>HV_cutoff], color=edge_vector_color[::-1], s=15, zorder = 99)
            ax.scatter(scatter_axis[plot_data<=HV_cutoff], plot_data[plot_data<=HV_cutoff], color=homing_vector_color[::-1], s=15, zorder = 99)
            bp = ax.boxplot([plot_data, [0,0]], positions = [3 * c - .2, -10], showfliers=False, zorder=99)
            plt.setp(bp['boxes'], color=[.5,.5,.5], linewidth = 2)
            plt.setp(bp['whiskers'], color=[.5,.5,.5], linewidth = 2)
            plt.setp(bp['medians'], linewidth=2)

            ax.set_xlim([-1, 3 * len(self.experiments) - 1])
            # ax.set_ylim([-.1, 1.15])
            ax.set_ylim([-.1, 1.3])
            #do kde
            try:
                if 'Square' in experiment:
                    kde = fit_kde(plot_data, bw=.06)
                    plot_kde(ax, kde, plot_data, z=3*c + .3, vertical=True, normto=.8, color=[.5,.5,.5], violin=False, clip=False, cutoff = HV_cutoff+0.0000001, cutoff_colors = [homing_vector_color[::-1], edge_vector_color[::-1]])
                    ax.set_ylim([-1.5, 1.5])
                else:
                    kde = fit_kde(plot_data, bw=.04)
                    plot_kde(ax, kde, plot_data, z=3*c + .3, vertical=True, normto=1.3, color=[.5,.5,.5], violin=False, clip=True, cutoff = HV_cutoff, cutoff_colors = [homing_vector_color[::-1], edge_vector_color[::-1]])
            except: pass

            # plot the polar plot or initial trajectories
            # plt.figure(fig4.number)
            fig4 = plt.figure(figsize=( 5, 5))

            # ax4 = plt.subplot(1,len(self.experiments),len(self.experiments) - c, polar=True)
            ax4 = plt.subplot(1, 1, 1, polar=True)
            plt.axis('off')
            ax.margins(0, 0)
            ax.xaxis.set_major_locator(plt.NullLocator())
            ax.yaxis.set_major_locator(plt.NullLocator())
            ax4.set_xlim([-np.pi / 2 - .1, 0])
            # ax4.set_xlim([-np.pi - .1, 0])

            mean_value_color = max(0, min(1, np.mean(plot_data)))
            mean_value_color = np.sum(plot_data > HV_cutoff) / len(plot_data)
            mean_value = np.mean(plot_data)

            value_color = mean_value_color * edge_vector_color[::-1] + (1 - mean_value_color) * homing_vector_color[::-1]


            ax4.arrow(mean_value + 3 * np.pi / 2, 0, 0, 1.9, color=[abs(v)**1 for v in value_color], alpha=1, width = 0.05, linewidth=2)
            ax4.plot([0, 0 + 3 * np.pi / 2], [0, 2.25], color=[.5,.5,.5], alpha=1, linewidth=1, linestyle = '--')
            ax4.plot([0, 1 + 3 * np.pi / 2], [0, 2.25], color=[.5,.5,.5], alpha=1, linewidth=1, linestyle = '--')
            # ax4.plot([0, -1 + 3 * np.pi / 2], [0, 2.25], color=[.5, .5, .5], alpha=1, linewidth=1, linestyle='--')

            scatter_axis_EV = scatter_the_axis_polar(plot_data[plot_data > HV_cutoff], 2.25, 0) #0.05
            scatter_axis_HV = scatter_the_axis_polar(plot_data[plot_data <= HV_cutoff], 2.25, 0)
            ax4.scatter(plot_data[plot_data > HV_cutoff] + 3 * np.pi/2, scatter_axis_EV, s = 30, color=edge_vector_color[::-1], alpha = .8, edgecolors = None)
            ax4.scatter(plot_data[plot_data <= HV_cutoff] + 3 * np.pi/2, scatter_axis_HV, s = 30, color=homing_vector_color[::-1], alpha=.8, edgecolors = None)

            fig4.savefig(os.path.join(self.summary_plots_folder, 'Angle comparison - ' + self.labels[c] + '.png'), format='png', transparent=True, bbox_inches='tight', pad_inches=0)
            fig4.savefig(os.path.join(self.summary_plots_folder, 'Angle comparison - ' + self.labels[c] + '.eps'), format='eps', transparent=True, bbox_inches='tight', pad_inches=0)

            # print(len(plot_data))
            if len(plot_data) > 1 and False: # or True:
                '''     plot the correlation    '''
                # do both prev homings and time in center # np.array(time_since_down) # 'Time since removal'
                for plot_data_corr, fig_corr, ax_corr, data_label in zip([prev_edginess, time_in_center], [fig2, fig3], [ax2, ax3], ['Prior homings','Exploration']): #
                    plot_data_corr = plot_data_corr[~np.isnan(data)]
                    # plot data
                    ax_corr.scatter(plot_data_corr, plot_data, color=colors[c], s=60, alpha=1, edgecolors=colors[c]/2, linewidth=1) #color=[.5, .5, .5] #edgecolors=[.2, .2, .2]
                    # do correlation
                    r, p = scipy.stats.pearsonr(plot_data_corr, plot_data)
                    print(r, p)
                    # do linear regression
                    plot_data_corr, prediction = do_linear_regression(plot_data, plot_data_corr)
                    # plot linear regresssion
                    ax_corr.plot(plot_data_corr, prediction['Pred'].values, color=colors[c], linewidth=1, linestyle='--', alpha=.7) #color=[.0, .0, .0]
                    ax_corr.fill_between(plot_data_corr, prediction['lower'].values, prediction['upper'].values, color=colors[c], alpha=.075) #color=[.2, .2, .2]
                    fig_corr.savefig(os.path.join(self.summary_plots_folder, 'Edginess by ' + data_label + ' - ' + self.labels[c] + '.png'), format='png')
                    fig_corr.savefig(os.path.join(self.summary_plots_folder, 'Edginess by ' + data_label + ' - ' + self.labels[c] + '.eps'), format='eps')

                # test correlation and stats thru permutation test
                # data_x = list(np.array(all_data[2])[np.array(all_conditions) == c])
                # data_y = list(np.array(all_data[1])[np.array(all_conditions) == c])
                # permutation_correlation(data_x, data_y, iterations=10000, two_tailed=False, pool_all = True)

            print(num_trials_escape)
            print(num_trials_total)
            print(num_trials_escape / num_trials_total)
    # save the plot
    fig.savefig(os.path.join(self.summary_plots_folder, 'Edginess comparison.png'), format='png', bbox_inches='tight', pad_inches=0)
    fig.savefig(os.path.join(self.summary_plots_folder, 'Edginess comparison.eps'), format='eps', bbox_inches='tight', pad_inches=0)


    # fig5.savefig(os.path.join(self.summary_plots_folder, 'Angle dist comparison.png'), format='png', bbox_inches='tight', pad_inches=0)
    # fig5.savefig(os.path.join(self.summary_plots_folder, 'Angle dist comparison.eps'), format='eps', bbox_inches='tight', pad_inches=0)

    plt.show()
    time_to_shelter_all = np.concatenate(list(flatten(time_to_shelter_all))).astype(float)
    np.percentile(time_to_shelter_all, 25)
    np.percentile(time_to_shelter_all, 75)

    group_A = list(np.array(all_data[0])[np.array(all_conditions) == 2])
    group_B = list(np.array(all_data[0])[np.array(all_conditions) == 3])
    permutation_test(group_A, group_B, iterations = 10000, two_tailed = False)

    group_A = list(np.array(all_data[1])[(np.array(all_conditions) == 1) + (np.array(all_conditions) == 2)])
    group_B = list(np.array(all_data[1])[np.array(all_conditions) == 3])
    permutation_test(group_A, group_B, iterations = 10000, two_tailed = False)

    import pandas
    df = pandas.DataFrame(data={"mouse_id": mouse_ID, "condition": all_conditions, "x-data": all_data[2], "y-data": all_data[1]})
    df.to_csv("./Foraging Path Types.csv", sep=',', index=False)


    group_B = list(flatten(np.array(all_data[0])[np.array(all_conditions) == 1]))
    np.sum(group_B) / len(group_B)

    np.percentile(abs(time_since_down[edginess < HV_cutoff]), 50)
    np.percentile(abs(time_since_down[edginess < HV_cutoff]), 25)
    np.percentile(abs(time_since_down[edginess < HV_cutoff]), 75)
    np.percentile(abs(time_since_down[edginess > HV_cutoff]), 50)
    np.percentile(abs(time_since_down[edginess > HV_cutoff]), 25)
    np.percentile(abs(time_since_down[edginess > HV_cutoff]), 75)

    group_A = [[d] for d in abs(time_since_down[edginess > HV_cutoff])]
    group_B = [[d] for d in abs(time_since_down[edginess < HV_cutoff])]
    permutation_test(group_A, group_B, iterations=10000, two_tailed=True)

    WE = np.concatenate(was_escape)
    TTS_spont = np.concatenate(time_to_shelter)[~WE]
    TTS_escape = np.concatenate(time_to_shelter)[WE]

    trials = np.array(list(flatten(all_data[3])))
    edgy = np.array(list(flatten(all_data[0])))

    np.mean(edgy[trials == 0])
    np.mean(edgy[trials == 1])
    np.mean(edgy[trials == 2])
    np.mean(edgy[trials == 3])
    np.mean(edgy[trials == 4])
    np.mean(edgy[trials == 5])
    np.mean(edgy[trials == 6])
    np.mean(edgy[trials == 7])
    np.mean(edgy[trials == 8])
    np.mean(edgy[trials == 9])
    np.mean(edgy[trials == 10])
    np.mean(edgy[trials == 11])
    np.mean(edgy[trials == 12])
    np.mean(edgy[trials == 13])

'''
TRADITIONAL METRICS
'''
def plot_metrics_by_strategy(self):
    '''     plot the escape paths       '''
    # initialize parameters
    edge_vector_color = np.array([1, .95, .85])
    homing_vector_color = np.array([.725, .725, .725])
    non_escape_color = np.array([0,0,0])

    ETD = 10#0
    traj_loc = 40

    fps = 30
    # escape_duration = 12 #12 #9 #12 9 for food 12 for dark
    HV_cutoff = .681 #.65
    edgy_cutoff = .681

    # loop over experiments and conditions
    for c, (experiment, condition) in enumerate(zip(self.experiments, self.conditions)):
        # extract experiments from nested list
        sub_experiments, sub_conditions = extract_experiments(experiment, condition)
        # get the number of trials
        number_of_trials = get_number_of_trials(sub_experiments, sub_conditions, self.analysis)
        number_of_mice = get_number_of_mice(sub_experiments, sub_conditions, self.analysis)
        # initialize array to fill in with each trial's data
        efficiency, efficiency_RT, end_idx, num_prev_homings_EV, duration_RT, duration, prev_edginess, edginess, _, _, _, _, \
            _, _, _, _, _, scaling_factor, time, trial_num, trials, edginess, avg_speed, avg_speed_RT, peak_speed, RT, escape_speed, strategy = \
            initialize_variables_efficiency(number_of_trials, self, sub_experiments)

        mouse_id = efficiency.copy()
        m = 0
        # loop over each experiment and condition
        for e, (experiment, condition) in enumerate(zip(sub_experiments, sub_conditions)):
            # loop over each mouse
            for i, mouse in enumerate(self.analysis[experiment][condition]['speed']):
                print(mouse)
                # control analysis
                if self.analysis_options['control'] and not mouse=='control': continue
                if not self.analysis_options['control'] and mouse=='control': continue
                # loop across all trials
                t = 0
                for trial in range(len(self.analysis[experiment][condition]['end time'][mouse])):
                    if 'food' in experiment: escape_duration = 9
                    else: escape_duration = 12
                    trial_num += 1
                    # impose coniditions - escape duration
                    end_time = self.analysis[experiment][condition]['end time'][mouse][trial]
                    if np.isnan(end_time) or (end_time > (escape_duration * fps)): continue
                    # skip certain trials
                    y_start = self.analysis[experiment][condition]['path'][mouse][trial][1][0] * scaling_factor
                    x_start = self.analysis[experiment][condition]['path'][mouse][trial][0][0] * scaling_factor
                    # needs to start at top
                    if y_start > 25: continue
                    if abs(x_start - 50) > 30: continue
                    # get the strategy used
                    # edgy_escape = self.analysis[experiment][condition]['edginess'][mouse][trial] > edgy_cutoff
                    # is it a homing vector
                    # strategy_code = 0
                    # TEMPORARY COMMENTING
                    # if not edgy_escape:
                    #     if self.analysis[experiment][condition]['edginess'][mouse][trial] < HV_cutoff: strategy_code = 0 # homing vector
                    #     else: continue
                    # else:
                    # get the strategy used -- NUMBER OF PREVIOUS EDGE VECTOR HOMINGS
                    time_to_shelter, SR = get_prev_edginess(ETD, condition, experiment, mouse, prev_edginess, [], [],
                                                            scaling_factor, self, traj_loc, trial, trial_num, edginess, [], [])
                    if t > 2: continue

                    # if c == 0 and trial: continue
                    # if c == 1 and trial  != 2: continue

                    t+=1

                    # if prev_edginess[trial_num] >= HV_cutoff: strategy_code = 1 # path learning
                    # elif prev_edginess[trial_num] < HV_cutoff: strategy_code = 2 # map-based
                    # else: continue


                    # how many prev homings to that edge: if 0, then map-based, if >1, then PL
                    if len(self.analysis[experiment]['probe']['start time'][mouse]):
                        edge_time = self.analysis[experiment]['probe']['start time'][mouse][0] - 1
                    else: edge_time = 19
                    edge_time = np.min((edge_time, self.analysis[experiment][condition]['start time'][mouse][trial]))
                    # print(edge_time)

                    num_edge_vectors, _ = get_num_edge_vectors(self, experiment, condition, mouse, trial, ETD=ETD, time_threshold=edge_time, other_side = False)
                    num_edge_vectors = get_num_homing_vectors(self, experiment, condition, mouse, trial, spontaneous = False, time_threshold = edge_time)
                    print(num_edge_vectors)
                    # if 'wall up' in experiment and 'no' in condition: num_edge_vectors = 0
                    # print(num_edge_vectors)

                    if False or True:
                        if num_edge_vectors == 1:
                            strategy_code = 1
                            # print('EV  -- ' + mouse + ' - trial ' + str(trial))
                        elif num_edge_vectors == 0:
                            strategy_code = 0
                            # print('NO EV -- ' + mouse + ' - trial ' + str(trial))
                        else: continue
                    else:
                        strategy_code = 0



                    strategy[trial_num] = strategy_code
                    # add data for each metric
                    RT[trial_num] = self.analysis[experiment][condition]['RT'][mouse][trial]
                    avg_speed[trial_num] = np.mean(self.analysis[experiment][condition]['speed'][mouse][trial][10*fps : 10*fps+int(end_time)]) * scaling_factor * 30
                    avg_speed_RT[trial_num] = np.mean(self.analysis[experiment][condition]['speed'][mouse][trial][10*fps + int(RT[trial_num]*30) : 10*fps+int(end_time)]) * scaling_factor * 30
                    peak_speed[trial_num] = np.max(self.analysis[experiment][condition]['speed'][mouse][trial][10*fps : 10*fps+int(end_time)])*fps*scaling_factor
                    escape_speed[trial_num] = self.analysis[experiment][condition]['optimal path length'][mouse][trial] * scaling_factor / (end_time/30)
                    efficiency[trial_num] = np.min((1, self.analysis[experiment][condition]['optimal path length'][mouse][trial] / \
                                                    self.analysis[experiment][condition]['full path length'][mouse][trial]))
                    efficiency_RT[trial_num] = np.min((1, self.analysis[experiment][condition]['optimal RT path length'][mouse][trial] / \
                                                       self.analysis[experiment][condition]['RT path length'][mouse][trial]))

                    duration_RT[trial_num] = (end_time / fps - RT[trial_num]) / self.analysis[experiment][condition]['optimal RT path length'][mouse][trial] / scaling_factor * 100
                    duration[trial_num] = end_time / fps / self.analysis[experiment][condition]['optimal path length'][mouse][trial] / scaling_factor * 100

                    # duration[trial_num] = trial
                    # duration_RT[trial_num] = self.analysis[experiment][condition]['start time'][mouse][trial]
                    avg_speed[trial_num] = self.analysis[experiment][condition]['time exploring far (pre)'][mouse][trial] / 60
                    # add data for stats
                    mouse_id[trial_num] = m

                m+=1

        # for metric, data in zip(['Reaction time', 'Peak speed', 'Avg speed', 'Path efficiency - RT','Duration - RT', 'Duration'],\
        #                         [RT, peak_speed, avg_speed_RT, efficiency_RT, duration_RT, duration]):
        # for metric, data in zip(['Reaction time', 'Avg speed', 'Path efficiency - RT'], #,'Peak speed',  'Duration - RT', 'Duration'], \
        #                         [RT, avg_speed_RT, efficiency_RT]): #peak_speed, , duration_RT, duration
        for metric, data in zip(['Path efficiency - RT'], [efficiency_RT]):
        # for metric, data in zip([ 'Duration - RT'],
        #                         [ duration_RT]):
        # for metric, data in zip(['trial', 'time', 'time exploring back'],
        #                         [duration, duration_RT, avg_speed]):
            # format data
            x_data = strategy[~np.isnan(data)]
            y_data = data[~np.isnan(data)]

            if not c: OF_data = y_data

            # make figure
            fig, ax = plt.subplots(figsize=(11, 9))
            plt.axis('off')
            # ax.margins(0, 0)
            ax.xaxis.set_major_locator(plt.NullLocator())
            ax.yaxis.set_major_locator(plt.NullLocator())
            # ax.set_title(metric)

            if 'Reaction time' in metric:
                ax.plot([-.75, 3], [0, 0], linestyle='--', color=[.5, .5, .5, .5])
                ax.plot([-.75, 3], [1, 1], linestyle='--', color=[.5, .5, .5, .5])
                ax.plot([-.75, 3], [2, 2], linestyle='--', color=[.5, .5, .5, .5])
                ax.plot([-.75, 3], [3, 3], linestyle='--', color=[.5, .5, .5, .5])
                ax.plot([-.75, 3], [4, 4], linestyle='--', color=[.5, .5, .5, .5])
            elif 'Peak speed' in metric:
                ax.plot([-.75, 3], [40, 40], linestyle='--', color=[.5, .5, .5, .5])
                ax.plot([-.75, 3], [80, 80], linestyle='--', color=[.5, .5, .5, .5])
                ax.plot([-.75, 3], [120, 120], linestyle='--', color=[.5, .5, .5, .5])
            elif 'Avg speed' in metric:
                ax.plot([-.75, 3], [25, 25], linestyle='--', color=[.5, .5, .5, .5])
                ax.plot([-.75, 3], [50, 50], linestyle='--', color=[.5, .5, .5, .5])
                ax.plot([-.75, 3], [75, 75], linestyle='--', color=[.5, .5, .5, .5])
                ax.plot([-.75, 3], [0, 0], linestyle='--', color=[.5, .5, .5, .5])
            elif 'Path efficiency' in metric:
                ax.plot([-.75, 3], [.5,.5], linestyle='--', color=[.5, .5, .5, .5])
                ax.plot([-.75, 3], [.75, .75], linestyle='--', color=[.5, .5, .5, .5])
                ax.plot([-.75, 3], [1, 1], linestyle='--', color=[.5, .5, .5, .5])
            elif 'Duration' in metric:
                ax.plot([-.75, 3], [0, 0], linestyle='--', color=[.5, .5, .5, .5])
                ax.plot([-.75, 3], [10, 10], linestyle='--', color=[.5, .5, .5, .5])
                ax.plot([-.75, 3], [5, 5], linestyle='--', color=[.5, .5, .5, .5])
            elif 'time' == metric:
                ax.plot([-.75, 3], [0, 0], linestyle='--', color=[.5, .5, .5, .5])
                ax.plot([-.75, 3], [10, 10], linestyle='--', color=[.5, .5, .5, .5])
                ax.plot([-.75, 3], [20, 20], linestyle='--', color=[.5, .5, .5, .5])
                ax.plot([-.75, 3], [30, 30], linestyle='--', color=[.5, .5, .5, .5])
                ax.plot([-.75, 3], [40, 40], linestyle='--', color=[.5, .5, .5, .5])
            elif 'exploring' in metric:
                ax.plot([-.75, 3], [2.5, 2.5], linestyle='--', color=[.5, .5, .5, .5])
                ax.plot([-.75, 3], [5.0, 5.0], linestyle='--', color=[.5, .5, .5, .5])
                ax.plot([-.75, 3], [7.5, 7.5], linestyle='--', color=[.5, .5, .5, .5])
                ax.plot([-.75, 3], [0, 0], linestyle='--', color=[.5, .5, .5, .5])

            #initialize stats array
            stats_data = [[], [], []]

            # go thru each strategy
            for s in [0,1,2]:
                # format data
                if not np.sum(x_data==s): continue
                plot_data = y_data[x_data==s]
                median = np.percentile(plot_data, 50);
                third_quartile = np.percentile(plot_data, 75);
                first_quartile = np.percentile(plot_data, 25)
                # print(first_quartile)
                # print(median)
                # print(third_quartile)

                # if 'Reaction' in metric: print(str(first_quartile), str(median), str(third_quartile))
                IQR = third_quartile - first_quartile
                # remove outliers
                if not metric == 'trial':
                    outliers = abs(plot_data - median) > 2*IQR
                    # plot_data = plot_data[~outliers]

                # plot all data
                ax.scatter(np.ones_like(plot_data)*s, plot_data, color=[0,0,0], s=30, zorder = 99)
                # plot kde
                if 'efficiency' in metric: bw_factor = .02
                elif 'speed' in metric or 'efficiency' in metric or metric == 'time': bw_factor = .04
                elif 'exploring' in metric: bw_factor = .06
                elif 'Duration' in metric: bw_factor = .07
                else: bw_factor = .09

                kde = fit_kde(plot_data, bw=np.median(y_data)*bw_factor)
                plot_kde(ax, kde, plot_data, z= s + .1, vertical=True, normto=.4, color=[.75, .75, .75], violin=False, clip=True)
                # plot errorbar
                ax.errorbar(s - .15, median, yerr=np.array([[median - first_quartile], [third_quartile - median]]), color=[0, 0, 0], capsize=10, capthick=3, alpha=1, linewidth=3)
                ax.scatter(s - .15, median, color=[0, 0, 0], s=175, alpha=1)

                # print(len(plot_data))

                # get mouse ids for stats
                mouse_id_stats = mouse_id[~np.isnan(data)]
                mouse_id_stats = mouse_id_stats[x_data==s]
                if not metric == 'trial': mouse_id_stats = mouse_id_stats[~outliers]

                # for m in np.unique(mouse_id_stats):
                #     stats_data[s].append( list(plot_data[mouse_id_stats==m]) )

            print(metric)
            # for ss in [[0,1]]: #, [0,2], [1,2]]:
            #     group_A = stats_data[ss[0]]
            #     group_B = stats_data[ss[1]]
            #     permutation_test(group_A, group_B, iterations=10000, two_tailed=True)

            # save figure
            fig.savefig(os.path.join(self.summary_plots_folder, metric + ' - ' + self.labels[c] + '.png'), format='png', bbox_inches='tight', pad_inches=0)
            fig.savefig(os.path.join(self.summary_plots_folder, metric + ' - ' + self.labels[c] + '.eps'), format='eps', bbox_inches='tight', pad_inches=0)

    plt.show()

    plt.close('all')


    group_A = [[e] for e in tr1_eff]
    group_B = [[e] for e in tr3_eff]
    group_C = [[e] for e in OF_eff]
    permutation_test(group_A, group_B, iterations=10000, two_tailed=True)
    permutation_test(group_A, group_C, iterations=10000, two_tailed=True)
    permutation_test(group_B, group_C, iterations=10000, two_tailed=True)

'''
DIST OF TURN ANGLES
'''
# def plot_metrics_by_strategy(self):
#     '''     plot the escape paths       '''
#
#     ETD = 10
#     traj_loc = 40
#
#     fps = 30
#     escape_duration = 12
#
#     colors = [[.3,.3,.3,.5], [.5,.5,.8, .5]]
#
#     # make figure
#     fig, ax = plt.subplots(figsize=(11, 9))
#     fig2, ax2 = plt.subplots(figsize=(11, 9))
#     # plt.axis('off')
#     # ax.margins(0, 0)
#     # ax.xaxis.set_major_locator(plt.NullLocator())
#     # ax.yaxis.set_major_locator(plt.NullLocator())
#     all_angles_pre = []
#     all_angles_escape = []
#
#
#     # loop over experiments and conditions
#     for c, (experiment, condition) in enumerate(zip(self.experiments, self.conditions)):
#         # extract experiments from nested list
#         sub_experiments, sub_conditions = extract_experiments(experiment, condition)
#         # get the number of trials
#         number_of_trials = get_number_of_trials(sub_experiments, sub_conditions, self.analysis)
#         number_of_mice = get_number_of_mice(sub_experiments, sub_conditions, self.analysis)
#         # initialize array to fill in with each trial's data
#         shape = self.analysis[sub_experiments[0]]['obstacle']['shape']
#         scaling_factor = 100 / shape[0]
#         turn_angles_pre = []
#         turn_angles_escape = []
#
#         # loop over each experiment and condition
#         for e, (experiment, condition) in enumerate(zip(sub_experiments, sub_conditions)):
#             # loop over each mouse
#             for i, mouse in enumerate(self.analysis[experiment][condition]['speed']):
#                 print(mouse)
#                 # control analysis
#                 if self.analysis_options['control'] and not mouse=='control': continue
#                 if not self.analysis_options['control'] and mouse=='control': continue
#                 # loop across all trials
#                 t = 0
#                 for trial in range(len(self.analysis[experiment][condition]['end time'][mouse])):
#                     # impose coniditions - escape duration
#                     end_time = self.analysis[experiment][condition]['end time'][mouse][trial]
#                     if np.isnan(end_time) or (end_time > (escape_duration * fps)): continue
#
#
#                     ## COMMENT ONE OR THE OTHER IF TESTING PRE OR ESCAPE
#                     #pre
#                     # if trial < 2: continue
#                     # if t: continue
#
#                     # escape
#                     if t > 2: continue
#
#                     # skip certain trials
#                     y_start = self.analysis[experiment][condition]['path'][mouse][trial][1][0] * scaling_factor
#                     x_start = self.analysis[experiment][condition]['path'][mouse][trial][0][0] * scaling_factor
#                     # needs to start at top
#                     if y_start > 25: continue
#                     if abs(x_start - 50) > 30: continue
#
#                     turn_angles_pre.append(list(abs(np.array(self.analysis[experiment][condition]['prev movements'][mouse][trial][3])))) # >145
#                     turn_angles_escape.append(abs(self.analysis[experiment][condition]['movement'][mouse][trial][2])) # >145
#                     #
#                     # turn_angles_pre.append(list(np.array(self.analysis[experiment][condition]['prev movements'][mouse][trial][3])))
#                     # turn_angles_escape.append(self.analysis[experiment][condition]['movement'][mouse][trial][2])
#
#                     t += 1
#
#
#
#         # format data
#         hist_data_pre = np.array(list(flatten(turn_angles_pre)))
#         hist_data_escape = np.array(list(flatten(turn_angles_escape)))
#
#         # for permutation test
#         # all_angles_pre.append(turn_angles_pre)
#         # all_angles_escape.append([[tae] for tae in turn_angles_escape])
#
#         ax.set_title('Prior movement angles')
#         ax2.set_title('Escape movement angles')
#         ax.plot([0, 0], [0, .4], linestyle='--', color=[.5, .5, .5, .5])
#         ax.plot([90, 90],[0, .4], linestyle='--', color=[.5, .5, .5, .5])
#         ax.plot([180, 180],[0, .4], linestyle='--', color=[.5, .5, .5, .5])
#         ax2.plot([0, 0], [0, .4], linestyle='--', color=[.5, .5, .5, .5])
#         ax2.plot([90, 90],[0, .4], linestyle='--', color=[.5, .5, .5, .5])
#         ax2.plot([180, 180],[0, .4], linestyle='--', color=[.5, .5, .5, .5])
#
#         # format data
#         bin_width = 30
#         hist_pre, n, _ = ax.hist(hist_data_pre, bins=np.arange(-0, 180+bin_width, bin_width), color=colors[c], weights = np.ones_like(hist_data_pre) * 1/ len(hist_data_pre))
#         hist_escape, n, _ = ax2.hist(hist_data_escape, bins=np.arange(-0, 180+bin_width, bin_width), color=colors[c], weights = np.ones_like(hist_data_escape) * 1/ len(hist_data_escape))
#
#         count_pre, n = np.histogram(hist_data_pre, bins=np.arange(-0, 180+bin_width, bin_width))
#         count_escape, n = np.histogram(hist_data_escape, bins=np.arange(-0, 180+bin_width, bin_width))
#
#         # for chi squared
#         all_angles_pre.append(count_pre)
#         all_angles_escape.append(count_escape)
#
#
#      # save figure
#     fig.savefig(os.path.join(self.summary_plots_folder, 'Prior Angle dist.png'), format='png', bbox_inches='tight', pad_inches=0)
#     fig.savefig(os.path.join(self.summary_plots_folder, 'Prior Angle dist.eps'), format='eps', bbox_inches='tight', pad_inches=0)
#      # save figure
#     fig2.savefig(os.path.join(self.summary_plots_folder, 'Escape Angle dist.png'), format='png', bbox_inches='tight', pad_inches=0)
#     fig2.savefig(os.path.join(self.summary_plots_folder, 'Escape Angle dist.eps'), format='eps', bbox_inches='tight', pad_inches=0)
#
#     plt.show()
#
#
#     scipy.stats.chi2_contingency(all_angles_pre)
#     scipy.stats.chi2_contingency(all_angles_escape)
#
#
#     group_A = all_angles_pre[0]
#     group_B = all_angles_pre[1]
#     permutation_test(group_A, group_B, iterations = 10000, two_tailed = True)
#
#     group_A = all_angles_escape[0]
#     group_B = all_angles_escape[1]
#     permutation_test(group_A, group_B, iterations = 10000, two_tailed = True)
#
#     plt.close('all')




#
# '''
# DIST OF EDGE VECTORS
# '''
# def plot_metrics_by_strategy(self):
#     '''     plot the escape paths       '''
#
#     ETD = 10
#     traj_loc = 40
#
#     fps = 30
#     escape_duration = 12
#
#     dist_thresh = 5
#     time_thresh = 20
#
#     colors = [[.3,.3,.3,.5], [.5,.5,.8, .5]]
#
#     # make figure
#     fig1, ax1 = plt.subplots(figsize=(11, 9))
#     fig2, ax2 = plt.subplots(figsize=(11, 9))
#     # plt.axis('off')
#     # ax.margins(0, 0)
#     # ax.xaxis.set_major_locator(plt.NullLocator())
#     # ax.yaxis.set_major_locator(plt.NullLocator())
#     all_EVs = []
#     all_HVs = []
#
#
#     # loop over experiments and conditions
#     for c, (experiment, condition) in enumerate(zip(self.experiments, self.conditions)):
#         # extract experiments from nested list
#         sub_experiments, sub_conditions = extract_experiments(experiment, condition)
#         # get the number of trials
#         number_of_trials = get_number_of_trials(sub_experiments, sub_conditions, self.analysis)
#         number_of_mice = get_number_of_mice(sub_experiments, sub_conditions, self.analysis)
#         # initialize array to fill in with each trial's data
#         shape = self.analysis[sub_experiments[0]]['obstacle']['shape']
#         scaling_factor = 100 / shape[0]
#         EVs = []
#         HVs = []
#         edge_vector_time_exp = []
#
#         # loop over each experiment and condition
#         for e, (experiment, condition) in enumerate(zip(sub_experiments, sub_conditions)):
#             # loop over each mouse
#             for i, mouse in enumerate(self.analysis[experiment][condition]['speed']):
#                 print(mouse)
#                 # control analysis
#                 if self.analysis_options['control'] and not mouse=='control': continue
#                 if not self.analysis_options['control'] and mouse=='control': continue
#                 # just take the last trial
#                 trial = len(self.analysis[experiment][condition]['start time'][mouse])-1
#                 if trial < 0:
#                     if condition == 'obstacle':
#                         condition_use = 'no obstacle'
#                         trial = 0
#                     elif condition == 'no obstacle':
#                         condition_use = 'obstacle'
#                         trial = len(self.analysis[experiment][condition]['start time'][mouse])-1
#                     if mouse == 'CA7220': trial = 1 #compensate for extra vid
#                 else: condition_use = condition
#
#                 # get the prev homings
#                 SH_data = self.analysis[experiment][condition_use]['prev homings'][mouse][trial]
#
#                 # get their start time
#                 homing_time = np.array(SH_data[3])
#                 edge_vector_time_exp.append(list(homing_time))
#
#                 # get their x value
#                 SH_x = np.array(SH_data[0])
#
#                 # only use spontaneous
#                 stim_evoked = np.array(SH_data[4])
#                 SH_x = SH_x[~stim_evoked]
#                 homing_time = homing_time[~stim_evoked]
#
#                 # normalize to 20 min
#                 SH_x = SH_x[homing_time < time_thresh] / np.min((time_thresh, self.analysis[experiment][condition_use]['start time'][mouse][trial])) * 20
#
#                 # get number of edge vectors
#                 num_edge_vectors = np.sum(abs(SH_x - 25) < dist_thresh) + np.sum(abs(SH_x - 75) < dist_thresh)
#                 num_homing_vectors = np.sum(abs(SH_x - 50) < dist_thresh)
#                 print(num_edge_vectors)
#
#
#                 # get the prev anti homings
#                 anti_SH_data = self.analysis[experiment][condition_use]['prev anti-homings'][mouse][trial]
#
#                 # get their start time
#                 homing_time = np.array(anti_SH_data[3])
#                 edge_vector_time_exp.append(list(homing_time))
#
#                 # get their x value
#                 anti_SH_x = np.array(anti_SH_data[0])
#
#                 # limit to 20 min
#                 anti_SH_x = anti_SH_x[homing_time < time_thresh] / np.min((time_thresh, self.analysis[experiment][condition_use]['start time'][mouse][trial])) * 20
#
#                 # get number of edge vectors
#                 num_anti_edge_vectors = np.sum(abs(anti_SH_x - 25) < dist_thresh) + np.sum(abs(anti_SH_x - 75) < dist_thresh)
#                 num_anti_homing_vectors = np.sum(abs(anti_SH_x - 50) < dist_thresh)
#                 print(num_anti_edge_vectors)
#
#                 # append to list
#                 EVs.append(num_edge_vectors   + num_anti_edge_vectors  )
#                 HVs.append(num_edge_vectors   + num_anti_edge_vectors  - (num_homing_vectors + num_anti_homing_vectors))
#         print(EVs)
#         all_EVs.append(EVs)
#         all_HVs.append(HVs)
#
#         # make timing hist
#         plt.figure()
#         plt.hist(list(flatten(edge_vector_time_exp)), bins=np.arange(0, 22.5, 2.5)) #, color=condition_colors[c])
#
#         # plot EVs and HVs
#         for plot_data, ax, fig in zip([EVs, HVs], [ax1, ax2], [fig1, fig2]):
#
#             scatter_axis = scatter_the_axis(c * 4 / 3 + .5 / 3, plot_data)
#             ax.scatter(scatter_axis, plot_data, color=[0, 0, 0], s=25, zorder=99)
#             # do kde
#             kde = fit_kde(plot_data, bw=.5)
#             plot_kde(ax, kde, plot_data, z=4 * c + .8, vertical=True, normto=1.2, color=[.5, .5, .5], violin=False, clip=False)  # True)
#
#             # save figure
#             fig.savefig(os.path.join(self.summary_plots_folder, 'EV dist - ' + self.labels[c] + '.png'), format='png', bbox_inches='tight', pad_inches=0)
#             fig.savefig(os.path.join(self.summary_plots_folder, 'EV dist - ' + self.labels[c] + '.eps'), format='eps', bbox_inches='tight', pad_inches=0)
#
#
#     plt.show()
#
#
#     group_A = all_EVs[1]
#     group_B = all_EVs[2]
#     permutation_test(group_A, group_B, iterations = 10000, two_tailed = True)
#
#     group_A = all_HVs[0]
#     group_B = all_HVs[1]
#     permutation_test(group_A, group_B, iterations = 10000, two_tailed = True)
#
#     plt.close('all')


'''
PREDICTION PLOTS, BY TURN ANGLE OR EXPLORATION/EDGINESS
  |
  |
  v
'''
def plot_prediction(self):

    by_angle_not_edginess = False

    if by_angle_not_edginess:

        # initialize parameters
        fps = 30
        escape_duration = 12
        ETD = 10 #4
        traj_loc = 40

        # initialize figures
        fig1, ax1, fig2, ax2, fig3, ax3 = initialize_figures_prediction(self)
        plt.close(fig2); plt.close(fig3)
        # loop over experiments and conditions
        for c, (experiment, condition) in enumerate(zip(self.experiments, self.conditions)):
            # extract experiments from nested list
            sub_experiments, sub_conditions = extract_experiments(experiment, condition)
            # get the number of trials
            number_of_trials = get_number_of_trials(sub_experiments, sub_conditions, self.analysis)
            number_of_mice = get_number_of_mice(sub_experiments, sub_conditions, self.analysis)

            mouse_trial_list = []

            IC_x_all, IC_y_all, IC_angle_all, IC_time_all, turn_angles_all = [], [], [], [], []

            # initialize array to fill in with each trial's data
            efficiency, efficiency_RT, end_idx, x_pred, y_pred, angle_pred, time_pred, mean_pred, initial_body_angle, initial_x, initial_y, x_edge, _, \
            _, _, _, _, scaling_factor, time, trial_num, trials, edginess, prev_edginess, dist_to_SH, dist_to_other_SH, RT_all, avg_speed, _ = \
                initialize_variables_efficiency(number_of_trials, self, sub_experiments)

            # initialize array to fill in with each trial's data
            edginess, end_idx, angle_turned, _, _, prev_edginess, scaling_factor, _, trial_num, _, _, dist_to_SH, dist_to_other_SH = \
                initialize_variable_edginess(number_of_trials, self, sub_experiments)

            for shuffle_time in [False, True]:

                angle_turned_all, x_pred_all, y_pred_all, angle_pred_all, time_pred_all, mean_pred_all = [], [], [], [], [], []


                num_repeats = shuffle_time * 499 + 1 #* 19
                num_repeats = shuffle_time * 19 + 1  # * 19

                prediction_scores_all = []

                for r in range(num_repeats):

                    trial_num = -1
                    # loop over each experiment and condition
                    for e, (experiment_real, condition_real) in enumerate(zip(sub_experiments, sub_conditions)):
                        # loop over each mouse
                        for i, mouse_real in enumerate(self.analysis[experiment_real][condition_real]['start time']):
                            if self.analysis_options['control'] and not mouse_real=='control': continue
                            if not self.analysis_options['control'] and mouse_real=='control': continue
                            # loop over each trial
                            prev_homings = []
                            t = 0
                            for trial_real in range(len(self.analysis[experiment_real][condition_real]['end time'][mouse_real])):
                                trial_num += 1
                                # impose conditions
                                if t > 2: continue
                                end_idx[trial_num] = self.analysis[experiment_real][condition_real]['end time'][mouse_real][trial_real]
                                if np.isnan(end_idx[trial_num]): continue
                                if (end_idx[trial_num] > escape_duration * fps): continue
                                # skip certain trials
                                y_start = self.analysis[experiment_real][condition_real]['path'][mouse_real][trial_real][1][0] * scaling_factor
                                x_start = self.analysis[experiment_real][condition_real]['path'][mouse_real][trial_real][0][0] * scaling_factor
                                if y_start > 25: continue
                                if abs(x_start-50) > 30: continue
                                # use different data if shuffle:
                                # if shuffle_time:
                                #     experiment, condition, mouse, trial = mouse_trial_list[np.random.randint(len(mouse_trial_list))]
                                # else:
                                #     experiment, condition, mouse, trial = experiment_real, condition_real, mouse_real, trial_real

                                '''     just use real mouse      '''
                                experiment, condition, mouse, trial = experiment_real, condition_real, mouse_real, trial_real

                                '''     control ICs, real escape        '''
                                # # get the angle turned during the escape
                                angle_turned[trial_num] = self.analysis[experiment_real][condition_real]['movement'][mouse_real][trial_real][2]
                                # angle_turned[trial_num] = abs(self.analysis[experiment_real][condition_real]['edginess'][mouse_real][trial_real])

                                # get the angle turned, delta x, delta y, and delta phi of previous homings
                                bout_start_angle = self.analysis[experiment_real][condition_real]['movement'][mouse_real][trial_real][1]
                                bout_start_position  = self.analysis[experiment_real][condition_real]['movement'][mouse_real][trial_real][0]
                                start_time = self.analysis[experiment_real][condition_real]['start time'][mouse_real][trial_real]

                                # get initial conditions and endpoint quantities
                                IC_x = np.array(self.analysis[experiment][condition]['prev movements'][mouse][trial][0][-ETD:])
                                IC_y = np.array(self.analysis[experiment][condition]['prev movements'][mouse][trial][1][-ETD:])
                                IC_angle = np.array(self.analysis[experiment][condition]['prev movements'][mouse][trial][2][-ETD:])
                                IC_time = np.array(self.analysis[experiment][condition]['prev homings'][mouse][trial][3][-ETD:])
                                turn_angles = np.array(self.analysis[experiment][condition]['prev movements'][mouse][trial][3][-ETD:])

                                # MOE = 10
                                # x_edge_trial = self.analysis[experiment][condition]['x edge'][mouse][trial]
                                # SH_x = np.array(self.analysis[experiment][condition]['prev homings'][mouse][trial][0][-ETD:])
                                # if x_edge_trial > 50 and np.sum(SH_x > 25 + MOE):
                                #     IC_x = IC_x[SH_x > 25 + MOE]
                                #     IC_y = IC_y[SH_x > 25 + MOE]
                                #     IC_angle = IC_angle[SH_x > 25 + MOE]
                                #     IC_time = IC_time[SH_x > 25 + MOE]
                                #     turn_angles = turn_angles[SH_x > 25 + MOE]
                                # elif np.sum(SH_x > 75 - MOE):
                                #     IC_x = IC_x[SH_x > 75 - MOE]
                                #     IC_y = IC_y[SH_x > 75 - MOE]
                                #     IC_angle = IC_angle[SH_x > 75 - MOE]
                                #     IC_time = IC_time[SH_x > 75 - MOE]
                                #     turn_angles = turn_angles[SH_x > 75 - MOE]

                                if not shuffle_time: # gather previous movements
                                    IC_x_all = np.concatenate((IC_x_all, IC_x))
                                    IC_y_all = np.concatenate((IC_y_all, IC_y))
                                    IC_angle_all = np.concatenate((IC_angle_all, IC_angle))
                                    IC_time_all = np.concatenate((IC_time_all, IC_time))
                                    turn_angles_all = np.concatenate((turn_angles_all, turn_angles))
                                else:
                                    # sample randomly from these movements
                                    random_idx = np.random.choice(len(IC_x_all), len(IC_x_all), replace = False)
                                    IC_x = IC_x_all[random_idx]
                                    IC_y = IC_y_all[random_idx]
                                    IC_angle = IC_angle_all[random_idx]
                                    IC_time = IC_time_all[random_idx]
                                    turn_angles = turn_angles_all[random_idx]

                                # calculate difference in ICs
                                delta_x = abs( np.array(IC_x - bout_start_position[0]) )
                                delta_y = abs( np.array(IC_y - bout_start_position[1]) )
                                delta_angle = abs( np.array(IC_angle - bout_start_angle) )
                                delta_angle[delta_angle > 180] = 360 - delta_angle[delta_angle > 180]
                                delta_time = start_time - np.array(IC_time)

                                ''' prediction data -- angle turned is a function of prev movement and ICs '''
                                x_weights = (1 / (delta_x+.0001)) / np.sum(1/(delta_x+.0001))
                                y_weights = (1 / (delta_y+.0001)) / np.sum(1 / (delta_y+.0001))
                                angle_weights = (1 / (delta_angle+.0001)) / np.sum(1 / (delta_angle+.0001))
                                time_weights = (1 / (delta_time+.0001)) / np.sum(1 / (delta_time+.0001))

                                x_pred[trial_num] = np.sum(turn_angles * x_weights)
                                y_pred[trial_num] = np.sum(turn_angles * y_weights)
                                angle_pred[trial_num] = np.sum(turn_angles * angle_weights)
                                time_pred[trial_num] = np.sum(turn_angles * time_weights) * 0
                                mean_pred[trial_num] = np.mean(turn_angles) * 0

                                # try mean pred is the *closest* angle to real
                                # x_pred[trial_num] = 0
                                # y_pred[trial_num] = 0
                                # angle_pred[trial_num] = 0
                                # time_pred[trial_num] = 0
                                # mean_pred[trial_num] = turn_angles[np.argmin( abs(turn_angles - angle_turned[trial_num]) )]


                                # '''     turn angle prediction to edginess prediction    '''
                                if not shuffle_time:
                                    edginess[trial_num] = abs(self.analysis[experiment][condition]['edginess'][mouse][trial])
                                    initial_body_angle[trial_num] = self.analysis[experiment_real][condition_real]['movement'][mouse_real][trial_real][1]
                                    initial_x[trial_num] = self.analysis[experiment_real][condition_real]['movement'][mouse_real][trial_real][0][0]
                                    initial_y[trial_num] = self.analysis[experiment_real][condition_real]['movement'][mouse_real][trial_real][0][1]
                                    x_edge[trial_num] = self.analysis[experiment][condition]['x edge'][mouse][trial_real]

                                # add mouse and trial to list of mice and trials
                                if not shuffle_time:
                                    mouse_trial_list.append([experiment, condition, mouse, trial])

                                t+=1

                    '''     concatenate??...        '''
                    # angle_turned_all = np.concatenate((angle_turned_all, angle_turned))
                    #
                    # x_pred_all = np.concatenate((x_pred_all, x_pred))
                    # y_pred_all = np.concatenate((y_pred_all, y_pred))
                    # angle_pred_all = np.concatenate((angle_pred_all, angle_pred))
                    # time_pred_all = np.concatenate((time_pred_all, time_pred ))
                    # mean_pred_all = np.concatenate((mean_pred_all, mean_pred ))
                    #
                    #
                    # IC_angle_array = np.ones((len(angle_turned_all[~np.isnan(angle_turned_all)]), 5))
                    # angle_metrics = [x_pred_all[~np.isnan(angle_turned_all)], y_pred_all[~np.isnan(angle_turned_all)], angle_pred_all[~np.isnan(angle_turned_all)], \
                    #                 time_pred_all[~np.isnan(angle_turned_all)], mean_pred_all[~np.isnan(angle_turned_all)]]
                    # for i, angle_metric in enumerate(angle_metrics): #
                    #     IC_angle_array[:, i] = angle_metric
                    #
                    # # get the data
                    # predict_data_y_all = [ angle_turned_all[~np.isnan(angle_turned_all)].reshape(-1, 1)] # for the movements input data

                    '''     don't concatenate...    '''

                    IC_angle_array = np.ones((len(angle_turned[~np.isnan(angle_turned)]), 5))
                    angle_metrics = [x_pred[~np.isnan(angle_turned)], y_pred[~np.isnan(angle_turned)],
                                     angle_pred[~np.isnan(angle_turned)], \
                                     time_pred[~np.isnan(angle_turned)], mean_pred[~np.isnan(angle_turned)]]
                    for i, angle_metric in enumerate(angle_metrics):  #
                        IC_angle_array[:, i] = angle_metric

                    # get the data
                    predict_data_y_all_angle = [angle_turned[~np.isnan(angle_turned)].reshape(-1, 1)]  # for the movements input data
                    predict_data_y_all_edgy = [edginess[~np.isnan(edginess)].reshape(-1, 1)]  # for the movements input data

                    data_y_labels = ['angle']

                    predict_data_x_all = [IC_angle_array]  # turn angles
                    predict_data_y_all = predict_data_y_all_angle # angles

                    '''     predict edginess from turn angle       '''
                    predict_edginess = True
                    if predict_edginess:
                        if not shuffle_time:
                            initial_body_angle = initial_body_angle[~np.isnan(initial_body_angle)].reshape(-1, 1)
                            initial_x = initial_x[~np.isnan(initial_x)].reshape(-1, 1)
                            initial_y = initial_y[~np.isnan(initial_y)].reshape(-1, 1)
                            x_edge = x_edge[~np.isnan(x_edge)].reshape(-1, 1)
                        # create the model
                        LR = linear_model.Ridge(alpha=.1)
                        # train the model
                        LR.fit(predict_data_x_all[0], predict_data_y_all_angle[0])
                        print(LR.score(predict_data_x_all[0], predict_data_y_all_angle[0]))
                        # get the model prediction
                        # model_prediction = LR.predict(predict_data_x_all[0])
                        model_prediction = predict_data_y_all_angle[0]
                        # predict body angles after turn
                        predicted_body_angle = initial_body_angle[~np.isnan(initial_body_angle)].reshape(-1, 1) - model_prediction
                        predicted_body_angle[predicted_body_angle >180] = predicted_body_angle[predicted_body_angle >180] - 360
                        predicted_body_angle[(predicted_body_angle > 0) * (predicted_body_angle < 90)] = -1 # super edgy to the right
                        predicted_body_angle[(predicted_body_angle > 0) * (predicted_body_angle > 90)] = 1 # super edgy to the right
                        # predict position at y = 40; set reasonable boundaries
                        x_at_40 = np.maximum(15 * np.ones_like(initial_x), np.minimum(90 * np.ones_like(initial_x),
                            initial_x - (40 - initial_y) / np.tan(np.deg2rad(predicted_body_angle)) ))
                        # get edginess
                        y_pos_end = 86.5; x_pos_end = 50; y_edge = 50
                        slope = (y_pos_end - initial_y) / (x_pos_end - (initial_x+.0001))
                        intercept = initial_y - initial_x * slope
                        distance_to_line = abs(40 - slope * x_at_40 - intercept) / np.sqrt((-slope) ** 2 + (1) ** 2)
                        homing_vector_at_center = (40 - intercept) / slope
                        # do line from starting position to edge position
                        slope = (y_edge - initial_y) / (x_edge - initial_x)
                        intercept = initial_y - initial_x * slope
                        distance_to_edge = abs(40 - slope * x_at_40 - intercept) / np.sqrt((-slope) ** 2 + (1) ** 2)
                        # compute the max possible deviation
                        edge_vector_at_center = (40 - intercept) / slope
                        line_to_edge_offset = abs(homing_vector_at_center - edge_vector_at_center)  # + 5
                        # get index at center point (wall location)
                        # prev_edginess = np.maximum(np.zeros_like(distance_to_line), np.minimum(1.2*np.ones_like(distance_to_line),
                        #                   (distance_to_line - distance_to_edge + line_to_edge_offset) / (2 * line_to_edge_offset) ))

                        prev_edginess = abs((distance_to_line - distance_to_edge + line_to_edge_offset) / (2 * line_to_edge_offset))

                        predict_data_x_all = [prev_edginess] # predicted prev edginess  #scipy.stats.zscore(
                        predict_data_y_all = predict_data_y_all_edgy # edginess




                    # edgy input colors
                    input_colors = [ [[0, .6, .4], [.5,.5,.5]], [[0, .6, .4], [.5,.5,.5]], [[.6, 0, .4], [.5,.5,.5]] ]
                    # split the data for cross val
                    num_trials = 1000 - 985 * shuffle_time #985
                    # loop acros angle prediction and traj prediction
                    for i, (fig, ax, predict_data_x) in enumerate(zip([fig1, fig2, fig3],[ax1, ax2, ax3], predict_data_x_all)):
                        # get prediction data
                        predict_data_y = predict_data_y_all[i]
                        # get color
                        color = input_colors[i][int(shuffle_time)]
                        # initialize prediction arrays
                        prediction_scores = np.zeros(num_trials)
                        for j in range(num_trials):
                            test_size = 0.5

                            # test_size = 0.25

                            # if shuffle_time: test_size = 0.25
                            # get x-val set
                            X_train, X_test, y_train, y_test = train_test_split(predict_data_x, \
                                                        predict_data_y, test_size=test_size, random_state=j)
                            # create the model
                            LR = linear_model.Ridge(alpha = .1) # .15, .5

                            # train the model
                            LR.fit(X_train, y_train)

                            # get the score
                            prediction_scores[j] = LR.score(X_test, y_test)
                        # exclude super negative ones
                        # prediction_scores = prediction_scores[prediction_scores > np.percentile(prediction_scores, 10)]
                        # put into larger array
                        prediction_scores_all = np.concatenate((prediction_scores_all, prediction_scores))
                print(np.median(prediction_scores_all))
                # exclude super negative ones
                # prediction_scores_all = prediction_scores_all[prediction_scores_all > np.percentile(prediction_scores_all, 5)]
                #do kde
                kde = fit_kde(prediction_scores_all, bw=.03)  # .04
                plot_kde(ax, kde, prediction_scores_all, z = 0, vertical=False, color=color, violin=False, clip=False)  # True)

                #plt.show()
                fig.savefig(os.path.join(self.summary_plots_folder,'Predictions of ' + data_y_labels[i] + ' - ' + self.labels[c] + '.png'), format='png')
                fig.savefig(os.path.join(self.summary_plots_folder,'Predictions of ' + data_y_labels[i] + ' - ' + self.labels[c] + '.eps'), format='eps')
        plt.show()
        print('hi')
    else:

        '''
        PREDICTION PLOTS EDGINESS OR BY **EXPLORATION**
        '''

        fps = 30
        escape_duration = 12
        ETD = 10 #4
        traj_loc = 40
        # mean_types = ['even', 'space', 'angle'] #, 'time', 'shelter time']
        mean_types = ['space', 'angle', 'shelter time'] #, 'escape']
        mean_type = 'even'
        mean_colors = [[0, .6, .4], [0, .6, .8], [0, .6, .8], [.4, 0, 1] ]
        mean_colors = [[0, .6, .4], [.4, 0, .8], [0, .6, .8], [.5, .5, .5]]

        # initialize figures
        fig1, ax1, fig2, ax2, fig3, ax3 = initialize_figures_prediction(self)

        for m, mean_type in enumerate(mean_types):
            # loop over experiments and conditions
            for c, (experiment, condition) in enumerate(zip(self.experiments, self.conditions)):
                # extract experiments from nested list
                sub_experiments, sub_conditions = extract_experiments(experiment, condition)
                # get the number of trials
                number_of_trials = get_number_of_trials(sub_experiments, sub_conditions, self.analysis)
                number_of_mice = get_number_of_mice(sub_experiments, sub_conditions, self.analysis)

                mouse_trial_list = []

                # initialize array to fill in with each trial's data
                edginess, end_idx, angle_turned, _, _, prev_edginess, scaling_factor, _, trial_num, prev_movement_and_ICs, data_y_for_prev_movement, dist_to_SH, dist_to_other_SH = \
                    initialize_variable_edginess(number_of_trials, self, sub_experiments)

                # initialize array to fill in with each trial's data
                efficiency, efficiency_RT, end_idx, num_prev_homings_EV, num_prev_homings_front_EV, num_prev_homings_other_EV, num_prev_homings_HV, time_exploring_pre, time_exploring_post, distance_exploring_pre, distance_exploring_post, time_exploring_obstacle_pre, \
                time_exploring_obstacle_post, time_exploring_far_pre, time_exploring_far_post, time_exploring_edge, time_exploring_other_edge, scaling_factor, time, trial_num, trials, edginess, prev_edginess, dist_to_SH, dist_to_other_SH, RT_all, avg_speed, _ = \
                    initialize_variables_efficiency(number_of_trials, self, sub_experiments)

                for shuffle_time in [False]:

                    num_repeats = shuffle_time * 19 + 1

                    for r in range(num_repeats):
                        trial_num = -1
                        # loop over each experiment and condition
                        for e, (experiment_real, condition_real) in enumerate(zip(sub_experiments, sub_conditions)):
                            # loop over each mouse
                            for i, mouse_real in enumerate(self.analysis[experiment_real][condition_real]['start time']):
                                if self.analysis_options['control'] and not mouse_real=='control': continue
                                if not self.analysis_options['control'] and mouse_real=='control': continue
                                # loop over each trial
                                prev_homings = []
                                t = 0
                                for trial_real in range(len(self.analysis[experiment_real][condition_real]['end time'][mouse_real])):
                                    trial_num += 1
                                    # impose conditions
                                    if t > 2: continue
                                    end_idx[trial_num] = self.analysis[experiment_real][condition_real]['end time'][mouse_real][trial_real]
                                    if np.isnan(end_idx[trial_num]): continue
                                    if (end_idx[trial_num] > escape_duration * fps): continue
                                    # skip certain trials
                                    y_start = self.analysis[experiment_real][condition_real]['path'][mouse_real][trial_real][1][0] * scaling_factor
                                    x_start = self.analysis[experiment_real][condition_real]['path'][mouse_real][trial_real][0][0] * scaling_factor
                                    if y_start > 25: continue
                                    if abs(x_start-50) > 30: continue
                                    # use different data if shuffle:
                                    if shuffle_time:
                                        experiment, condition, mouse, trial = mouse_trial_list[np.random.randint(len(mouse_trial_list))]
                                    else:
                                        experiment, condition, mouse, trial = experiment_real, condition_real, mouse_real, trial_real

                                    # just add real data for edginess etc
                                    if not shuffle_time:
                                    # add data
                                        edginess[trial_num] = abs(self.analysis[experiment][condition]['edginess'][mouse][trial])
                                        # get previous edginess
                                        time_to_shelter, SR = get_prev_edginess(ETD, condition_real, experiment_real, mouse_real, prev_edginess, dist_to_SH,
                                                                                dist_to_other_SH, scaling_factor, self, traj_loc, trial_real, trial_num, edginess,
                                                                                [], [], mean = mean_type, get_initial_conditions=True)
                                        # _, _, prev_edginess_all, elig_idx = get_all_prev_edginess(ETD, condition, experiment, mouse, prev_edginess, dist_to_SH, dist_to_other_SH, scaling_factor, self, traj_loc, trial, trial_num, edginess, [], [])
                                        # add data
                                        fill_in_trial_data_efficiency(ETD, condition, efficiency, efficiency_RT, experiment, mouse, num_prev_homings_EV,
                                                                      num_prev_homings_front_EV, num_prev_homings_other_EV,
                                                                      num_prev_homings_HV,
                                                                      time_exploring_pre, time_exploring_post, distance_exploring_pre, distance_exploring_post,
                                                                      time_exploring_obstacle_pre,
                                                                      time_exploring_obstacle_post, time_exploring_far_pre, time_exploring_far_post, time_exploring_edge,
                                                                      time_exploring_other_edge,
                                                                      self, time, trial, trial_num, trials, edginess, t)

                                    # add mouse and trial to list of mice and trials
                                    if not shuffle_time:
                                        mouse_trial_list.append([experiment, condition, mouse, trial])

                                    t+=1

                    # format mean prior trajectory
                    if not shuffle_time:
                        prev_edginess = prev_edginess[~np.isnan(edginess)]

                    exploration_array = np.ones((len(edginess[~np.isnan(edginess)]), 2))
                    exploration_metrics = [time_exploring_far_pre[~np.isnan(edginess)], time_exploring_far_post[~np.isnan(edginess)]]
                    for i, exploration_metric in enumerate(exploration_metrics): #
                        exploration_array[:, i] = exploration_metric

                        if shuffle_time: # regress out other variable
                            m = (((np.mean(prev_edginess) * np.mean(exploration_array[:, i])) - np.mean(prev_edginess * exploration_array[:, i])) /
                                 ((np.mean(prev_edginess) ** 2) - np.mean(prev_edginess ** 2)))

                            regressed_data = exploration_array[:, i] - prev_edginess * m

                            exploration_array[:, i] = regressed_data

                    if shuffle_time:  # regress out exploration from mean prior traj
                        for exploration_metric in exploration_metrics:
                            m = (((np.mean(exploration_metric) * np.mean(prev_edginess)) - np.mean(exploration_metric * prev_edginess)) /
                                 ((np.mean(exploration_metric) ** 2) - np.mean(exploration_metric ** 2)))

                            regressed_data = prev_edginess - exploration_array[:, 0] * m

                            prev_edginess = regressed_data

                    # get the data
                    predict_data_y_all = [ edginess[~np.isnan(edginess)].reshape(-1, 1), # for the EXPLORATION input data
                                           edginess[~np.isnan(edginess)].reshape(-1, 1)]  # for the mean edginess input data
                                           # turn_angle_for_prev_movement ] # for the movements input data
                    data_y_labels = ['exploration','trajectory'] #, 'angle']

                    predict_data_x_all = [exploration_array, # exploration data
                                          prev_edginess.reshape(-1, 1)]#,  # mean prev edginess
                                          # prev_movements_and_ICs_array] # all prev homing movements


                    # edgy input colors
                    input_colors = [ [[0, .6, .4], [.5,.5,.5]], [[0, .6, .4], [.5,.5,.5]], [[.6, 0, .4], [.5,.5,.5]] ]
                    # split the data for cross val
                    num_trials = 1000
                    # loop acros angle prediction and traj prediction
                    for i, (fig, ax, predict_data_x) in enumerate(zip([fig1, fig2, fig3],[ax1, ax2, ax3], predict_data_x_all)):
                        # get prediction data
                        predict_data_y = predict_data_y_all[i]
                        # get color
                        color = input_colors[i][int(shuffle_time)]
                        # color = mean_colors[m]
                        # initialize prediction arrays
                        prediction_scores = np.zeros(num_trials)
                        for j in range(num_trials):
                            test_size = 0.5
                            if shuffle_time and i==2:
                                test_size = .025
                            # get x-val set
                            X_train, X_test, y_train, y_test = train_test_split(predict_data_x, \
                                                        predict_data_y, test_size=test_size, random_state=j)
                            # create the model
                            # LR = linear_model.LinearRegression()
                            # if i:
                            #     LR = linear_model.LogisticRegression()
                            # else:
                            LR = linear_model.Ridge(alpha = .1) # .15, .5

                            # train the model
                            # try:
                            LR.fit(X_train, y_train)
                            # except:
                            #     print('i=h')
                            # print(LR.coef_)
                            # get the score
                            prediction_scores[j] = LR.score(X_test, y_test)
                        print(data_y_labels[i])
                        print(np.median(prediction_scores))
                        # exclude super negative ones
                        prediction_scores = prediction_scores[prediction_scores > np.percentile(prediction_scores, 10)]
                        # plot the scores
                        # ax.scatter(prediction_scores, np.zeros_like(prediction_scores), color=color, s=20, alpha = .1)
                        #do kde
                        kde = fit_kde(prediction_scores, bw=.04)  # .04
                        plot_kde(ax, kde, prediction_scores, z = 0, vertical=False, color=color, violin=False, clip=False)  # True)

                        fig.savefig(os.path.join(self.summary_plots_folder,'Prediction of ' + data_y_labels[i] + ' - ' + self.labels[c] + '.png'), format='png')
                        fig.savefig(os.path.join(self.summary_plots_folder,'Precition of ' + data_y_labels[i] + ' - ' + self.labels[c] + '.eps'), format='eps')
        plt.show()
        print('hi')

#     # get the correlation
#     r, p = scipy.stats.pearsonr(exploration_array[:, 0], edginess)
#     print('r = ' + str(np.round(r, 3)) + '\np = ' + str(np.round(p, 3)))
#
#     m = (((np.mean(prev_edginess) * np.mean(exploration_array[:, 0])) - np.mean(prev_edginess * exploration_array[:, 0])) /
#          ((np.mean(prev_edginess) ** 2) - np.mean(prev_edginess ** 2)))
#
#     regressed_data = exploration_array[:, 0] - prev_edginess * m
#     r, p = scipy.stats.pearsonr(prev_edginess, regressed_data)
#     print('r = ' + str(np.round(r, 3)) + '\np = ' + str(np.round(p, 3)))
#
#     # get the correlation after regressing out prev edginess
#     r, p = scipy.stats.pearsonr(regressed_data, edginess)
#     print('r = ' + str(np.round(r, 3)) + '\n= ' + str(np.round(p, 3)))

# #
# def plot_efficiency(self):
#     # initialize parameters
#     fps = 30
#     traj_loc = 40
#     escape_duration = 12  # 12 #6
#     HV_cutoff = .681
#     ETD = 10
#     # ax2, fig2, ax3, fig3 = initialize_figures_efficiency(self)
#     efficiency_data = [[], [], [], []]
#     duration_data = [[], [], [], []]
#     # initialize arrays for stats
#     efficiency_data_all = []
#     duration_data_all = []
#     prev_homings_data_all = []
#     all_conditions = []
#     mouse_ID = [];
#     m = 1
#     data_condition = ['naive', 'experienced']
#     # data_condition = ['food','escape']
#     # data_condition = ['OR - EV', 'OR - HV', 'OF']
#     fig1, ax1 = plt.subplots(figsize=(13, 5))
#
#     colors = [[1,0,0],[0,0,0]]
#     kde_colors = [ [1, .4, .4], [.75, .75, .75]]
#
#     # loop over experiments and conditions
#     for c, (experiment, condition) in enumerate(zip(self.experiments, self.conditions)):
#         # extract experiments from nested list
#         sub_experiments, sub_conditions = extract_experiments(experiment, condition)
#         # get the number of trials
#         number_of_trials = get_number_of_trials(sub_experiments, sub_conditions, self.analysis)
#         number_of_mice = get_number_of_mice(sub_experiments, sub_conditions, self.analysis)
#         # initialize array to fill in with each trial's data
#         efficiency, efficiency_RT, end_idx, num_prev_homings_EV, num_prev_homings_other_EV, num_prev_homings_HV, time_exploring, distance_exploring, time_exploring_obstacle, time_exploring_far, \
#         scaling_factor, time, trial_num, trials, edginess, prev_edginess, dist_to_SH, dist_to_other_SH, RT_all, avg_speed, _ = \
#             initialize_variables_efficiency(number_of_trials, self, sub_experiments)
#         # loop over each experiment and condition
#         for e, (experiment, condition) in enumerate(zip(sub_experiments, sub_conditions)):
#             if 'void' in experiment or 'dark' in experiment:
#                 escape_duration = 12
#             # loop over each mouse
#             for i, mouse in enumerate(self.analysis[experiment][condition]['full path length']):
#                 # initialize arrays for stats
#                 efficiency_data_mouse = []
#                 duration_data_mouse = []
#                 prev_homings_data_mouse = []
#                 # control analysis
#                 if self.analysis_options['control'] and not mouse == 'control': continue
#                 if not self.analysis_options['control'] and mouse == 'control': continue
#                 # loop over each trial
#                 t = 0
#                 for trial in range(len(self.analysis[experiment][condition]['end time'][mouse])):
#
#                     trial_num += 1
#                     if t > 2 and not 'food' in experiment and not 'void' in experiment: continue
#
#                     if t > 8: continue
#                     # print(t)
#                     # impose coniditions
#                     end_idx[trial_num] = self.analysis[experiment][condition]['end time'][mouse][trial]
#                     if (end_idx[trial_num] > escape_duration * fps) or np.isnan(end_idx[trial_num]): continue
#                     # skip certain trials
#                     y_start = self.analysis[experiment][condition]['path'][mouse][trial][1][0] * scaling_factor
#                     x_start = self.analysis[experiment][condition]['path'][mouse][trial][0][0] * scaling_factor
#                     if y_start > 25: continue
#                     if abs(x_start - 50) > 25: continue  # 25
#
#                     # get prev edginess
#                     _, _ = get_prev_edginess(ETD, condition, experiment, mouse, prev_edginess, dist_to_SH, dist_to_other_SH,
#                                              scaling_factor, self, traj_loc, trial, trial_num, edginess, [], [])
#
#                     # only do predict edgy:
#                     # if c == 0:
#                     #     if prev_edginess[trial_num] <= HV_cutoff and 'down' in experiment: continue
#                     # elif c == 1:
#                     #     if prev_edginess[trial_num] > HV_cutoff and 'down' in experiment: continue
#
#                     # add data
#                     fill_in_trial_data_efficiency(ETD, condition, efficiency, efficiency_RT, experiment, mouse, num_prev_homings_EV, num_prev_homings_other_EV, num_prev_homings_HV,
#                                                   time_exploring, distance_exploring, time_exploring_obstacle, time_exploring_far,
#                                                   self, time, trial, trial_num, trials, edginess, t)
#
#                     # normalize end idx to
#                     RT = self.analysis[experiment][condition]['RT'][mouse][trial]
#                     if not RT:
#                         print(RT)
#                         continue
#                     RT_all[trial_num] = RT
#
#                     avg_speed[trial_num] = self.analysis[experiment][condition]['RT path length'][mouse][trial] * scaling_factor / (
#                                 (end_idx[trial_num] - RT) / fps)
#                     # avg_speed[trial_num] = self.analysis[experiment][condition]['full path length'][mouse][trial] * scaling_factor / (end_idx[trial_num] / fps)
#
#                     end_idx[trial_num] = (end_idx[trial_num] / fps) / self.analysis[experiment][condition]['optimal path length'][mouse][
#                         trial] / scaling_factor * 100
#
#                     # add data for stats
#                     efficiency_data_mouse.append(efficiency[trial_num])
#                     # duration_data_mouse.append(end_idx[trial_num]) #TEMP COMMENTING
#                     duration_data_mouse.append(RT)
#                     prev_homings_data_mouse.append(num_prev_homings_EV[trial_num])
#
#                     t += 1
#
#                 # append data for stats
#                 if efficiency_data_mouse:
#                     efficiency_data_all.append(efficiency_data_mouse)
#                     duration_data_all.append(duration_data_mouse)
#                     prev_homings_data_all.append(prev_homings_data_mouse)
#                     all_conditions.append(data_condition[c])
#                     mouse_ID.append(m);
#                     m += 1
#
#         # format end ind
#         # end_idx = np.array([e/30 for e in end_idx])
#         end_idx[np.isnan(efficiency)] = np.nan
#         # loop over data to plot
#         for i, (data, data_label) in enumerate(zip([efficiency_RT, end_idx, RT_all, avg_speed, edginess],
#                                                    ['Efficiency'])):  # , 'Duration',  'Reaction Time', 'Speed', 'Trajectory'])): #edginess, 'Trajectory',
#             # for i, (data, data_label) in enumerate(zip([edginess], ['Trajectory'])):  # edginess, 'Trajectory',
#
#             # for i, (data, data_label) in enumerate(zip([edginess, efficiency, end_idx], ['Trajectory', 'Efficiency', 'Duration'])):
#             #     for x_data, x_data_label in zip([num_prev_homings], ['Prior homings']):
#             plot_data = data[~np.isnan(data)]
#
#             # for x_data, x_data_label in zip([trials, time, num_prev_homings_EV, num_prev_homings_HV, prev_edginess, time_exploring, distance_exploring, time_exploring_far, time_exploring_obstacle],
#             # ['Trials', 'Time', 'Edge vector homings', 'Homing vector homings', 'Mean prior trajectory','Time exploring', 'Distance explored', 'Time exploring far side', 'Time exploring obstacle']):
#
#             for x_data, x_data_label in zip([trials, time_exploring], ['trial number']):  # , 'Time exploring']):
#
#                 print('\nCorrelation between ' + data_label + ' and ' + x_data_label)
#
#                 # only plot escapes
#                 data_for_box_plot = data[~np.isnan(data)]
#                 print(len(data_for_box_plot))
#                 x_data = x_data[~np.isnan(data)]
#
#                 # get the correlation
#                 r, p = scipy.stats.pearsonr(x_data, data_for_box_plot)
#                 print('r = ' + str(np.round(r, 3)) + '\np = ' + str(np.round(p, 3)))
#
#                 # initialize figure
#                 plt.title(data_label + ' x ' + x_data_label)
#                 # set up the figure
#                 # if data_label=='Efficiency': ax1.set_ylim([-.03, 1.03])
#                 # elif data_label=='Duration': ax1.set_ylim([-.1, 7])
#
#                 if np.max(x_data) < 5:
#                     ax1.set_xticks(np.unique(x_data).astype(int))
#                 else:
#                     ax1.set_xticks(np.arange(5, 25, 5))
#                     # ax1.set_xlim([5,20])
#
#                 # jitter the axis
#                 scatter_axis = scatter_the_axis_efficiency(plot_data, x_data + c/3 - .2)
#                 # plot each trial
#                 ax1.scatter(scatter_axis, plot_data, color=colors[c], s=15, alpha=1, edgecolor=colors[c], linewidth=1)
#
#                 for x in np.unique(x_data):
#                     # plot kde
#                     kde = fit_kde(plot_data[x_data==x], bw=.02) #.2)  # .04
#                     plot_kde(ax1, kde, plot_data[x_data==x], z=x + c/3 - .15, vertical=True, normto=.15, color=kde_colors[c], violin=False, clip=True)
#
#                     # box and whisker
#                     bp = ax1.boxplot([plot_data[x_data==x], [0, 0]], positions=[x + c / 3 - .2, -10], showfliers=False, widths = [0.05, .05], zorder=99)
#                     plt.setp(bp['boxes'], color=[.5, .5, .5], linewidth=2)
#                     plt.setp(bp['whiskers'], color=[.5, .5, .5], linewidth=2)
#                     plt.setp(bp['medians'], linewidth=2)
#                     ax1.set_xlim(.25, 3.75)
#                     ax1.set_ylim(.5, 1.05)
#                     # ax1.set_ylim(.95, 1.9)
#                     ax1.set_xticks([1,2,3])
#                     ax1.set_xticklabels([1,2,3])
#
#
#
#                 # # for each trial
#                 # for x in np.unique(x_data):
#                 #     # plot kde
#                 #     kde = fit_kde(plot_data[x_data>=0], bw=.02) #.2)  # .04
#                 #     plot_kde(ax1, kde, plot_data[x_data>=0], z=x + c/3 - .15, vertical=True, normto=.15, color=kde_colors[c], violin=False, clip=True)
#                 #
#                 #     # box and whisker
#                 #     bp = ax1.boxplot([plot_data[x_data>=0], [0, 0]], positions=[x + c / 3 - .2, -10], showfliers=False, widths = [0.05, .05], zorder=99)
#                 #     plt.setp(bp['boxes'], color=[.5, .5, .5], linewidth=2)
#                 #     plt.setp(bp['whiskers'], color=[.5, .5, .5], linewidth=2)
#                 #     plt.setp(bp['medians'], linewidth=2)
#                 #     ax1.set_xlim(.25, 3.75)
#                 #     ax1.set_ylim(.5, 1.05)
#                 #     # ax1.set_ylim(.95, 1.9)
#                 #     ax1.set_xticks([1,2,3])
#                 #     ax1.set_xticklabels([1,2,3])
#                 #
#                 #     # jitter the axis
#                 #     scatter_axis = scatter_the_axis_efficiency(plot_data, np.ones_like(plot_data) * (x + c/3 - .2))
#                 #     # plot each trial
#                 #     ax1.scatter(scatter_axis, plot_data, color=colors[c], s=15, alpha=1, edgecolor=colors[c], linewidth=1)
#
#
#
#             ax1.plot([-1, 4], [1, 1], linestyle='--', color=[.5, .5, .5, .5])
#             # save the plot
#             plt.savefig(os.path.join(self.summary_plots_folder, data_label + ' by ' + x_data_label + ' - ' + self.labels[c] + '.png'), format='png')
#             plt.savefig(os.path.join(self.summary_plots_folder, data_label + ' by ' + x_data_label + ' - ' + self.labels[c] + '.eps'), format='eps')
#
#     plt.show()
#     print('done')
#
#
#
def plot_efficiency(self):
    # initialize parameters
    fps = 30
    traj_loc = 40
    escape_duration = 12 #12 #6
    HV_cutoff = .681
    ETD = 10

    # ax2, fig2, ax3, fig3 = initialize_figures_efficiency(self)
    efficiency_data = [[],[],[],[]]
    duration_data = [[],[],[],[]]
    # initialize arrays for stats
    efficiency_data_all = []
    duration_data_all = []
    prev_homings_data_all = []
    all_conditions = []
    mouse_ID = []; m = 1
    # data_condition = ['naive','experienced']
    data_condition = ['escape', 'food']
    # data_condition = ['OR - EV', 'OR - HV', 'OF']
    # data_condition = ['Obstacle removed (no shelter)', 'obstacle removed', 'acute OR', 'obstacle']
    colors = [[0,0,0],[1,0,0]]
    #
    plot_stuff = True
    do_traversals = False

    # loop over experiments and conditions
    for c, (experiment, condition) in enumerate(zip(self.experiments, self.conditions)):
        print(' - - - -- - - - -- - - - - - - -- - - - - - - - - -')
        # extract experiments from nested list
        sub_experiments, sub_conditions = extract_experiments(experiment, condition)
        # get the number of trials
        number_of_trials = get_number_of_trials(sub_experiments, sub_conditions, self.analysis)
        number_of_mice = get_number_of_mice(sub_experiments, sub_conditions, self.analysis)
        # initialize array to fill in with each trial's data
        efficiency, efficiency_RT, end_idx, num_prev_homings_EV, num_prev_homings_front_EV, num_prev_homings_other_EV, num_prev_homings_HV, time_exploring_pre, time_exploring_post, distance_exploring_pre, distance_exploring_post, time_exploring_obstacle_pre,\
                                time_exploring_obstacle_post,time_exploring_far_pre,time_exploring_far_post, time_exploring_edge, time_exploring_other_edge, scaling_factor, time, trial_num, trials, edginess, prev_edginess, dist_to_SH, dist_to_other_SH, RT_all, avg_speed, _ = \
            initialize_variables_efficiency(number_of_trials, self, sub_experiments)
        # loop over each experiment and condition
        for e, (experiment, condition) in enumerate(zip(sub_experiments, sub_conditions)):
            if 'void' in experiment or 'dark' in experiment:
                escape_duration = 12
            if 'food' in experiment: escape_duration = 9
            # else:escape_duration = 9
            # loop over each mouse
            for i, mouse in enumerate(self.analysis[experiment][condition]['start time']):
                print(mouse)
                # initialize arrays for stats
                efficiency_data_mouse = []
                duration_data_mouse = []
                prev_homings_data_mouse = []
                # control analysis
                if self.analysis_options['control'] and not mouse=='control': continue
                if not self.analysis_options['control'] and mouse=='control': continue
                # loop over each trial
                t = 0
                for trial in range(len(self.analysis[experiment][condition]['end time'][mouse])):

                    trial_num += 1
                    if t > 2 and not 'food' in experiment and not 'void' in experiment and not 'dark' in experiment: continue
                    if 'food' in experiment and condition == 'no obstacle' and self.analysis[experiment][condition]['start time'][mouse][trial] < 20: continue

                    if t > 8: continue
                    # if t > 2: continue

                    # if 'on off' in experiment and trial: continue
                    # print(t)
                    # impose coniditions
                    end_idx[trial_num] = self.analysis[experiment][condition]['end time'][mouse][trial]
                    if (end_idx[trial_num] > escape_duration * fps) or np.isnan(end_idx[trial_num]): continue
                    # skip certain trials
                    y_start = self.analysis[experiment][condition]['path'][mouse][trial][1][0] * scaling_factor
                    x_start = self.analysis[experiment][condition]['path'][mouse][trial][0][0] * scaling_factor
                    if y_start > 25: continue
                    if abs(x_start-50) > 30: continue #25

                    # get prev edginess
                    _, _ = get_prev_edginess(ETD, condition, experiment, mouse, prev_edginess, dist_to_SH, dist_to_other_SH,
                                                            scaling_factor, self, traj_loc, trial, trial_num, edginess, [], [])

                    # only do predict edgy:
                    # if c == 0:
                    #     if prev_edginess[trial_num] <= HV_cutoff and 'down' in experiment: continue
                    # elif c == 1:
                    #     if prev_edginess[trial_num] > HV_cutoff and 'down' in experiment: continue

                    # add data
                    fill_in_trial_data_efficiency(ETD, condition, efficiency, efficiency_RT, experiment, mouse, num_prev_homings_EV,num_prev_homings_front_EV, num_prev_homings_other_EV,num_prev_homings_HV,
                                time_exploring_pre, time_exploring_post, distance_exploring_pre, distance_exploring_post, time_exploring_obstacle_pre,
                                time_exploring_obstacle_post, time_exploring_far_pre, time_exploring_far_post,  time_exploring_edge, time_exploring_other_edge,
                                self, time, trial, trial_num, trials, edginess, t)

                    # if edginess[trial_num] < HV_cutoff: continue

                    if do_traversals:
                        traversal = self.analysis[experiment][condition]['back traversal'][mouse]
                        # get the duration of those paths
                        # duration = traversal[t*5+3]
                        if traversal:
                            x_edge = self.analysis[experiment][condition]['x edge'][mouse][trial]
                            # if x_edge==25: x_edge = 75
                            # else: x_edge = 25

                            spont_edge = []
                            for trav in traversal[0 * 5 + 0]:
                                spont_edge.append(trav[0][-1]*scaling_factor)
                            esc_edge = []
                            for trav in traversal[1 * 5 + 0]:
                                esc_edge.append(trav[0][-1]*scaling_factor)

                            num_prev_homings_EV[trial_num] = np.sum((np.array(traversal[0 * 5 + 3]) < 1.5) * (abs(np.array(spont_edge)-x_edge) < 25) * \
                                                            (np.array(traversal[0 * 5 + 2]) > HV_cutoff) * \
                                                            (np.array(traversal[0 * 5 + 1]) < self.analysis[experiment][condition]['start time'][mouse][trial] * 30 * 60) * \
                                                            (np.array(traversal[0 * 5 + 1]) > (self.analysis[experiment][condition]['start time'][mouse][trial]-(15+20*('void' in experiment))) * 30 * 60)) + \
                                                            np.sum((np.array(traversal[1 * 5 + 3]) < 1.5) * (abs(np.array(esc_edge)-x_edge) < 25) * \
                                                                   (np.array(traversal[1 * 5 + 2]) > HV_cutoff) * \
                                                                   (np.array(traversal[1 * 5 + 1]) < self.analysis[experiment][condition]['start time'][mouse][trial] * 30 * 60) * \
                                                                   (np.array(traversal[1 * 5 + 1]) > (self.analysis[experiment][condition]['start time'][mouse][trial] - (15+20*('void' in experiment))) * 30 * 60))

                            num_prev_homings_HV[trial_num] = np.sum((np.array(traversal[0 * 5 + 3]) < 1.5) * (abs(np.array(spont_edge)-x_edge) < 25) * \
                                                            (np.array(traversal[0 * 5 + 2]) < HV_cutoff) * \
                                                            (np.array(traversal[0 * 5 + 1]) < self.analysis[experiment][condition]['start time'][mouse][trial] * 30 * 60) * \
                                                            (np.array(traversal[0 * 5 + 1]) > (self.analysis[experiment][condition]['start time'][mouse][trial]-(15+20*('void' in experiment))) * 30 * 60)) + \
                                                            np.sum((np.array(traversal[1 * 5 + 3]) < 1.5) * (abs(np.array(esc_edge)-x_edge) < 25) * \
                                                                   (np.array(traversal[1 * 5 + 2]) < HV_cutoff) * \
                                                                   (np.array(traversal[1 * 5 + 1]) < self.analysis[experiment][condition]['start time'][mouse][trial] * 30 * 60) * \
                                                                   (np.array(traversal[1 * 5 + 1]) > (self.analysis[experiment][condition]['start time'][mouse][trial] - (15+20*('void' in experiment))) * 30 * 60))

                            eligible_homings = ~((np.array(traversal[0 * 5 + 2]) > HV_cutoff) * (abs(np.array(spont_edge)-x_edge) > 40)) * (np.array(traversal[0 * 5 + 3]) < 3) * \
                                               (np.array(traversal[0 * 5 + 1]) < self.analysis[experiment][condition]['start time'][mouse][trial] * 30 * 60) * \
                                               (np.array(traversal[0 * 5 + 1]) > (self.analysis[experiment][condition]['start time'][mouse][trial] - 15) * 30 * 60)
                            if np.sum(eligible_homings):
                                mean_homing = np.mean(np.array(traversal[0 * 5 + 2])[eligible_homings])
                            else: mean_homing = 0

                            eligible_escapes = ~((np.array(traversal[1 * 5 + 2]) > HV_cutoff) * (abs(np.array(esc_edge) - x_edge) > 40)) * (np.array(traversal[1 * 5 + 3]) < 3) * \
                                               (np.array(traversal[1 * 5 + 1]) < self.analysis[experiment][condition]['start time'][mouse][trial] * 30 * 60) * \
                                               (np.array(traversal[1 * 5 + 1]) > (self.analysis[experiment][condition]['start time'][mouse][trial] - 15) * 30 * 60)

                            if np.sum(eligible_escapes):
                                mean_escape = np.mean(np.array(traversal[1 * 5 + 2])[eligible_escapes])
                            else: mean_escape = 0

                            prev_edginess[trial_num] = ( mean_homing * np.sum(eligible_homings) + mean_escape * np.sum(eligible_escapes) ) / \
                                                        (np.sum(eligible_homings) + np.sum(eligible_escapes))

                        else:
                            num_prev_homings_EV[trial_num] = 0
                            # prev_edginess[trial_num] = 0

                        if np.isnan(prev_edginess[trial_num]):
                            prev_edginess[trial_num] = 0


                        traversal = self.analysis[experiment][condition]['front traversal'][mouse]
                        # get the duration of those paths
                        # duration = traversal[t*5+3]
                        if traversal:
                            x_edge = self.analysis[experiment][condition]['x edge'][mouse][trial]

                            spont_edge = []
                            for trav in traversal[0 * 5 + 0]:
                                spont_edge.append(trav[0][-1]*scaling_factor)
                            esc_edge = []
                            for trav in traversal[1 * 5 + 0]:
                                esc_edge.append(trav[0][-1]*scaling_factor)


                            num_prev_homings_other_EV[trial_num] = np.sum((np.array(traversal[0 * 5 + 3]) < 1.5) * (abs(np.array(spont_edge)-x_edge) < 25) * \
                                                            (np.array(traversal[0 * 5 + 2]) > HV_cutoff) * \
                                                            (np.array(traversal[0 * 5 + 1]) < self.analysis[experiment][condition]['start time'][mouse][trial] * 30 * 60))

                        else:
                            num_prev_homings_other_EV[trial_num] = 0


                        # print(mouse)
                        # print(trial + 1)
                        # print(num_prev_homings_EV[trial_num])
                        # print(num_prev_homings_other_EV[trial_num])
                        # print(edginess[trial_num])
                        # print('')


                        # normalize end idx to
                    RT = self.analysis[experiment][condition]['RT'][mouse][trial]
                    # if not RT:
                    #     print(RT)
                    #     continue
                    RT_all[trial_num] = RT

                    avg_speed[trial_num] = self.analysis[experiment][condition]['RT path length'][mouse][trial] * scaling_factor / ((end_idx[trial_num] - RT) / fps)
                    # avg_speed[trial_num] = self.analysis[experiment][condition]['full path length'][mouse][trial] * scaling_factor / (end_idx[trial_num] / fps)
                    time[trial_num] = self.analysis[experiment][condition]['start time'][mouse][trial]
                    time[trial_num] = (end_idx[trial_num] / fps) / self.analysis[experiment][condition]['optimal path length'][mouse][trial] / scaling_factor * 100

                    time[trial_num] = abs(50 - x_start)

                    end_idx[trial_num] = (end_idx[trial_num] / fps - RT) / self.analysis[experiment][condition]['optimal RT path length'][mouse][trial] / scaling_factor * 100


                    # add data for stats
                    efficiency_data_mouse.append([efficiency_RT[trial_num], trial])
                    duration_data_mouse.append([end_idx[trial_num], trial]) #TEMP COMMENTING #RT
                    # duration_data_mouse.append(num_prev_homings_EV[trial_num])
                    prev_homings_data_mouse.append(num_prev_homings_EV[trial_num])

                    t += 1
                    # print(trial+1)

                #append data for stats
                if efficiency_data_mouse:
                    efficiency_data_all.append(efficiency_data_mouse)
                    duration_data_all.append(duration_data_mouse)
                    prev_homings_data_all.append(prev_homings_data_mouse)
                    all_conditions.append(data_condition[c])
                    mouse_ID.append(m); m+= 1

        # format end ind
        # end_idx = np.array([e/30 for e in end_idx])
        end_idx[np.isnan(efficiency)] = np.nan
        # loop over data to plot
        # for i, (data, data_label) in enumerate(zip([edginess, efficiency_RT, end_idx, RT_all, avg_speed], ['Trajectory'])): #,'Efficiency', 'Duration',  'Reaction Time', 'Speed', 'Trajectory'])): #edginess, 'Trajectory',
        # for i, (data, data_label) in enumerate(zip([edginess], ['Trajectory'])):
        for i, (data, data_label) in enumerate(zip([end_idx], ['RT duration', 'RT duration', 'Efficiency', 'RT'])): # time, , efficiency_RT, RT_all
        # for i, (data, data_label) in enumerate(zip([RT_all], ['Reaction time'])):

        # for i, (data, data_label) in enumerate(zip([edginess, efficiency, end_idx], ['Trajectory', 'Efficiency', 'Duration'])):
        #     for x_data, x_data_label in zip([num_prev_homings], ['Prior homings']):
            plot_data = data[~np.isnan(data)]
            if False or True:
                # for x_data, x_data_label in zip([trials, time, num_prev_homings_EV, num_prev_homings_other_EV, num_prev_homings_HV, prev_edginess, time_exploring, distance_exploring, time_exploring_far, time_exploring_obstacle, time_exploring_edge, time_exploring_other_edge],
                # ['Trials', 'Time', 'Edge vector homings','Other edge vector homings', 'Homing vector homings', 'Mean prior trajectory','Time exploring', 'Distance explored', 'Time exploring far side', 'Time exploring obstacle', 'Time exploring edge', 'Time exploring other edge']):

                # for x_data, x_data_label in zip([trials, time, time_exploring_pre, distance_exploring_pre, time_exploring_post, distance_exploring_post,
                #          time_exploring_far_pre,time_exploring_far_post, time_exploring_obstacle_pre, time_exploring_obstacle_post,  time_exploring_other_edge, time_exploring_edge],
                #         ['Trials', 'Time', 'Time exploring (pre)', 'Distance explored (pre)', 'Time exploring (post)', 'Distance explored (post)',
                #          'Time exploring far side (pre)', 'Time exploring far side (post)', 'Time exploring obstacle (pre)', 'Time exploring obstacle (post)',
                #          'Time exploring other edge (pre)', 'Time exploring edge (pre)']):


                # num_homings_combined = (num_prev_homings_EV>0).astype(int) - (num_prev_homings_HV>0).astype(int)
                # num_homings_combined[num_prev_homings_EV==0] = -1
                #
                # for x_data, x_data_label in zip([time, num_prev_homings_EV>0, num_prev_homings_EV, num_prev_homings_other_EV, num_prev_homings_other_EV>0,
                #                                  num_prev_homings_front_EV, num_prev_homings_front_EV>0, prev_edginess, num_prev_homings_HV, num_prev_homings_HV>2, num_homings_combined],
                #                                         ['Time', '1 Edge vector homings', 'Edge vector homings','Other edge vector homings','1 other edge vector homings',
                #                                          'Front edge vectors','1 front edge vector', 'Mean prior trajectory', 'Homing vector homings', '1 Homing vector homing', 'Combined homings']):

                # for x_data, x_data_label in zip([trials, num_prev_homings_EV>0, num_prev_homings_EV, prev_edginess], ['trial', '1 Edge vector homings', 'Edge vector homings', 'Mean prior trajectory']):
                for x_data, x_data_label in zip([trials], ['trial']):  # ,edginess>HV_cutoff    #, 'edginess'

                    print('\nCorrelation between ' + data_label + ' and ' + x_data_label)

                    # only plot escapes
                    data_for_box_plot = data[~np.isnan(data)]
                    x_data = x_data[~np.isnan(data)]
                    print(np.sum(x_data==0))

                    # get the correlation
                    r, p = scipy.stats.pearsonr(x_data, data_for_box_plot)
                    print('r = ' + str(np.round(r, 3)) + '\np = ' + str(p))
                    if p < .05: print('SIGGY STATDUST')

                    # m = (((np.mean(x_data) * np.mean(data_for_box_plot)) - np.mean(x_data * data_for_box_plot)) /
                    #      ((np.mean(x_data) ** 2) - np.mean(x_data ** 2)))
                    # regressed_data = data_for_box_plot - x_data * m
                    # r, p = scipy.stats.pearsonr(x_data, regressed_data)
                    # print('r = ' + str(np.round(r, 3)) + '\np = ' + str(np.round(p, 3)))

                    if plot_stuff and not np.isnan(r):
                        fig1, ax1 = plt.subplots(figsize=(15, 15))
                        # initialize figure
                        # fig1, ax1 = plt.subplots(figsize=(9, 9))
                        plt.title(data_label + ' x ' + x_data_label)
                        # set up the figure
                        # if data_label=='Efficiency': ax1.set_ylim([-.03, 1.03])
                        # elif data_label=='Duration': ax1.set_ylim([-.1, 7])

                        # if np.max(x_data) < 5:
                        #     ax1.set_xticks(np.unique(x_data).astype(int))
                        # else:
                        #     ax1.set_xticks(np.arange(5, 25, 5))
                            # ax1.set_xlim([5,20])

                        # jitter the axis
                        scatter_axis = scatter_the_axis_efficiency(plot_data, x_data)
                        # plot each trial
                        ax1.scatter(scatter_axis, plot_data, color=colors[0], s=40, alpha=1, edgecolor=colors[0], linewidth=1)
                        # ax1.scatter(scatter_axis[plot_data > HV_cutoff], plot_data[plot_data > HV_cutoff], color=[0,0,0], s=50, alpha=1, edgecolor=[0, 0, 0], linewidth=1)
                        # do a linear regression
                        try:
                            x_data, prediction = do_linear_regression(plot_data, x_data.astype(int))
                        except:
                            print('hi')

                        # # plot kde
                        kde = fit_kde(plot_data, bw=.02) #.2)  # .04
                        plot_kde(ax1, kde, plot_data, z=c + .1, vertical=True, normto=.3, color=[.75, .75, .75], violin=False, clip=True)  # True)

                        # plot the linear regression
                        # ax1.plot(x_data, prediction['Pred'].values, color=colors[0], linewidth=1, linestyle='--', alpha=.7)
                        # ax1.fill_between(x_data, prediction['lower'].values, prediction['upper'].values, color=colors[0], alpha=.05)  # 6
                        # save the plot
                        plt.savefig(os.path.join(self.summary_plots_folder, data_label + ' by ' + x_data_label + ' - ' + self.labels[c] + '.png'), format='png')
                        plt.savefig(os.path.join(self.summary_plots_folder, data_label + ' by ' + x_data_label + ' - ' + self.labels[c] + '.eps'), format='eps')
                        # plt.show()
                        # plt.close()

            # plot the boxplot
            # if data_label == 'Efficiency':
            #     ax, fig = ax2, fig2
            #     efficiency_data[c] = plot_data
            # elif data_label == 'Duration':
            #     ax, fig = ax3, fig3
            #     duration_data[c] = plot_data
            # else: continue
            # scatter_axis = scatter_the_axis_efficiency(plot_data, np.ones_like(plot_data)*c)
            # ax.scatter(scatter_axis, plot_data, color=[0, 0, 0], s=40, alpha=1, edgecolor=[0, 0, 0], linewidth=1)
            # # plot kde
            # kde = fit_kde(plot_data, bw=.02) #.2)  # .04
            # plot_kde(ax, kde, plot_data, z=c + .1, vertical=True, normto=.3, color=[.75, .75, .75], violin=False, clip=True)  # True)
            # # plot errorbar
            # median = np.percentile(plot_data, 50)
            # third_quartile = np.percentile(plot_data, 75)
            # first_quartile = np.percentile(plot_data, 25)
            # ax.errorbar(c - .2, median, yerr=np.array([[median - first_quartile], [third_quartile - median]]), color=[0,0,0], capsize=10, capthick=3, alpha=1, linewidth=3)
            # ax.scatter(c - .2, median, color=[0,0,0], s=175, alpha=1)
            # # save the plot
            # fig.savefig(os.path.join(self.summary_plots_folder, data_label + ' comparison - ' + self.labels[c] + '.png'), format='png')
            # fig.savefig(os.path.join(self.summary_plots_folder, data_label + ' comparison - ' + self.labels[c] + '.eps'), format='eps')

    plt.show()

    # test correlation and stats thru permutation test
    data_x = prev_homings_data_all
    data_y = efficiency_data_all
    # permutation_correlation(data_x, data_y, iterations=10000, two_tailed=False, pool_all=False)
    #
    # # do t test
    # t, p = scipy.stats.ttest_ind(efficiency_data[0], efficiency_data[1], equal_var=False)
    # print('Efficiency: ' + str(p))
    # print(np.mean(efficiency_data[0]))
    # print(np.mean(efficiency_data[1]))
    #
    # t, p = scipy.stats.ttest_ind(duration_data[0], duration_data[1], equal_var=False)
    # print('Duration: ' + str(p))
    # print(np.mean(duration_data[0]))
    # print(np.mean(duration_data[1]))
    #
    efficiency_0 = []
    efficiency_more = []
    for m, mouse_data in enumerate(efficiency_data_all):
        EV_array = np.array(duration_data_all[m])
        efficiency_array = np.array(mouse_data)
        if 0 in EV_array:
            efficiency_0.append(list(efficiency_array[EV_array==0]))
        if np.sum(EV_array==0)<len(EV_array):
            efficiency_more.append(list(efficiency_array[EV_array>0]))

    permutation_test(efficiency_0, efficiency_more, iterations=10000, two_tailed=False)

    # group_A = list(np.array(efficiency_data_all)[np.array(all_conditions) == 'OFn'])
    # group_B = list(np.array(efficiency_data_all)[np.array(all_conditions) == 'ORe'])
    # permutation_test(group_A, group_B, iterations = 10000, two_tailed = False)
    #
    group_A = list(np.array(duration_data_all)[np.array(all_conditions) == 'food'])
    group_B = list(np.array(duration_data_all)[np.array(all_conditions) == 'escape'])
    permutation_test(group_A, group_B, iterations = 10000, two_tailed = True)


    group_A = [[d] for d in end_idx[(trials==1)*(end_idx < 4.5)]]
    group_B = [[d] for d in end_idx[(trials==3)*(end_idx < 4.5)]]

    group_A = [[d] for d in end_idx[(edginess<HV_cutoff)*(end_idx < 4.5)]]
    group_B = [[d] for d in end_idx[(edginess>HV_cutoff)*(end_idx < 4.5)]]

    permutation_test(group_A, group_B, iterations=10000, two_tailed=True)
    #
    # np.percentile(list(flatten(group_A)), 25)
    # np.percentile(list(flatten(group_A)), 50)
    # np.percentile(list(flatten(group_A)), 75)
    # np.percentile(list(flatten(group_B)), 25)
    # np.percentile(list(flatten(group_B)), 50)
    # np.percentile(list(flatten(group_B)), 75)
    #
    # a = np.mean(list(flatten(group_A)))
    # b = np.mean(list(flatten(group_B)))
    # (a - b) / b
    #
    #
    efficiency_data_save = []
    conditions_save = []
    trial_num_save = []
    mouse_ID_save = []
    for m, mouse_data in enumerate(duration_data_all):
        for t, trial_data in enumerate(mouse_data):
            efficiency_data_save.append(trial_data)
            conditions_save.append(all_conditions[m])
            trial_num_save.append(t+1)
            mouse_ID_save.append(mouse_ID[m])

    efficiency_data_save = []
    conditions_save = []
    trial_num_save = []
    mouse_ID_save = []
    for m, mouse_data in enumerate(duration_data_all):
        for t, trial_data in enumerate(mouse_data):
            efficiency_data_save.append(trial_data[0])
            conditions_save.append(all_conditions[m])
            trial_num_save.append(trial_data[1]+1)
            mouse_ID_save.append(mouse_ID[m])


    import pandas
    df = pandas.DataFrame(data={"mouse_id": mouse_ID_save, "condition": conditions_save, "trial": trial_num_save, "efficiency": efficiency_data_save})
    df.to_csv("./Escape Efficiency.csv", sep=',', index=False)
    #
    # import pandas
    # df = pandas.DataFrame(data={"mouse_id": mouse_ID, "condition": all_conditions, "data": efficiency_data_all})
    # df.to_csv("./Escape Duration.csv", sep=',', index=False)





def plot_exploration(self, saturation_percentile = 97, color_dimming = .7, dots_multiplier = .8):

    '''     plot the average exploration heat map       '''
    # loop over experiments and conditions
    for c, (experiment, condition) in enumerate(zip(self.experiments, self.conditions)):
        # extract experiments from nested list
        sub_experiments, sub_conditions = extract_experiments(experiment, condition)
        # get the number of trials
        number_of_mice = get_number_of_mice(sub_experiments, sub_conditions, self.analysis)
        # initialize array to fill in with each mouse's data
        shape = self.analysis[sub_experiments[0]]['obstacle']['shape']
        exploration = np.zeros((shape[0], shape[1], 1))
        mouse_exploration = np.zeros((shape[0], shape[1], 1))
        # loop over each experiment and condition
        for e, (experiment, condition) in enumerate(zip(sub_experiments, sub_conditions)):
            # loop over each mouse
            for mouse in self.analysis[experiment][condition]['exploration']:
                # fill array with each mouse's data
                mouse_explored = np.array(self.analysis[experiment][condition]['exploration'][mouse])
                if mouse_explored.size:
                    mouse_exploration[:,:,0] = mouse_explored
                    exploration = np.concatenate((exploration, mouse_exploration), axis = 2)

        # average all mice data
        exploration_all = np.mean(exploration, 2)

        # make an image out of it
        exploration_image = exploration_all.copy()
        exploration_image = (exploration_image / .00001 * 255)
        exploration_image[exploration_image > 255] = 255


        # gaussian blur
        exploration_blur = cv2.GaussianBlur(exploration_image, ksize=(201, 201), sigmaX=15, sigmaY=15)

        # normalize
        exploration_blur = (exploration_blur / 110 * 255) # np.percentile(exploration_blur, saturation_percentile)
        exploration_blur[exploration_blur > 255] = 255

        # colorize
        exploration_blur = exploration_blur * color_dimming
        exploration_blur = cv2.applyColorMap(255 - exploration_blur.astype(np.uint8), cv2.COLORMAP_HOT)

        # make composite image
        exploration_blur[exploration_all > 0] = exploration_blur[exploration_all > 0] * [dots_multiplier, dots_multiplier, dots_multiplier]

        # present it
        cv2.imshow('heat map blur', exploration_blur)
        cv2.waitKey(100)
        # save results
        scipy.misc.imsave(os.path.join(self.summary_plots_folder, experiment + '_exploration_' + condition + '.png'), exploration_blur[:,:,::-1])

        shape = self.analysis[sub_experiments[0]]['obstacle']['shape']

        '''
        PLOT EXPLORATION AROUND OBSTACLE, DIFFERENT COLOR FOR EACH MOUSE
        '''
        # initialize the arena
        arena, arena_color, scaling_factor, obstacle = initialize_arena(self, sub_experiments, sub_conditions)

        # generate figure
        path_ax, path_fig = get_arena_plot(obstacle, sub_conditions, sub_experiments)

        # get the colors
        num_colors = number_of_mice+3
        cm = plt.get_cmap('gist_rainbow')
        plot_colors = [cm(1. * mouse / num_colors) for mouse in range(num_colors)]
        plot_colors = plot_colors[::-1]
        i = 0

        # loop over each experiment and condition
        for e, (experiment, condition) in enumerate(zip(sub_experiments, sub_conditions)):
            # loop over each mouse
            for mouse in self.analysis[experiment][condition]['exploration']:
                # get color
                plot_color = list(plot_colors[i])
                plot_color[3] = .2
                i+=1
                # get positions
                try:
                    x_idx = self.analysis[experiment][condition]['obstacle exploration'][mouse][1][0] * scaling_factor
                    y_idx = self.analysis[experiment][condition]['obstacle exploration'][mouse][1][1] * scaling_factor
                    time_idx = self.analysis[experiment][condition]['obstacle exploration'][mouse][0]
                except: continue
                # loop over each idx
                for j in range(len(x_idx) - 1):
                    x1, y1 = x_idx[j], y_idx[j]
                    x2, y2 = x_idx[j + 1], y_idx[j + 1]
                    # plot if contiguous
                    if (time_idx[j+1] - time_idx[j]) < 5:
                        path_ax.plot([x1, x2], [98.5 - y1, 98.5 - y2], color=plot_color)

        plt.show()
        print('done')



        # for mouse in range(exploration.shape[2]):
        #     # initialize array
        #     mouse_exploration = np.zeros(exploration_all.shape).astype(np.uint8)
        #
        #     # get mouse's exploration
        #     y_width = 7.5
        #     x_width = 32.5
        #     mouse_exploration[int((50 - y_width-1.5)/scaling_factor):int((50 + y_width-1.5)/scaling_factor), \
        #                                      int((50 - x_width)/scaling_factor):int((50 + x_width)/scaling_factor)] \
        #                             = exploration[int((50 - y_width-1.5)/scaling_factor):int((50 + y_width-1.5)/scaling_factor), \
        #                                      int((50 - x_width)/scaling_factor):int((50 + x_width)/scaling_factor), mouse] > 0
        #
        #     # draw the path
        #     y_idx = np.where(mouse_exploration)[0] * scaling_factor
        #     x_idx = np.where(mouse_exploration)[1] * scaling_factor
        #
        #     path_ax.scatter(x_idx, 98.5-y_idx, s = 1, color = plot_color, edgecolors = 'none')
        #
        #     # loop over each idx
        #     for j in range(len(x_idx) - 1):
        #         x1, y1 = x_idx[j], y_idx[j]
        #         x2, y2 = x_idx[j + 1], y_idx[j + 1]
        #
        #         # plot crap
        #         if (x2 - x1) < 5 and (y2 - y1) < 5:
        #             print(x1, x2)
        #             print(100-y1, 100-y2)
        #             path_ax.plot([x1, x2], [100 - y1, 100 - y2], color=plot_color)

            # present it
            # cv2.imshow('mouse exploration II', mouse_exploration*255)
            # cv2.waitKey(100)




        # obstacle_type = self.analysis[sub_experiments[0]]['obstacle']['type']
        # _, _, shelter_roi = model_arena(shape, False, False, obstacle_type, simulate=False, dark=self.dark_theme)
        # percent_in_shelter = []
        # for m in range( exploration.shape[2]):
        #     mouse_exploration = exploration[:,:,m]
        #     percent_in_shelter.append( np.sum(mouse_exploration*shelter_roi) )



