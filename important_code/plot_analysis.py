import numpy as np
import cv2
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
    types = ['spontaneous'] #, 'evoked']
    fast_color = np.array([.5, 1, .5])
    slow_color = np.array([1, .9, .9])
    edge_vector_color = np.array([1, .95, .85])
    homing_vector_color = np.array([.725, .725, .725])
    non_escape_color = np.array([0,0,0])

    p = 0
    HV_cutoff = .681 #0.75
    # initialize figures
    fig, fig2, ax, ax2 = initialize_figures_traversals(self, types = len(types)+1)
    # initialize lists for stats
    all_data = []
    all_conditions = []
    # loop over spontaneous vs evoked
    for t, type in enumerate(types):
        # loop over experiments and conditions
        for c, (experiment, condition) in enumerate(zip(self.experiments, self.conditions)):
            strategies = [0, 0, 0]
            # extract experiments from nested list
            sub_experiments, sub_conditions = extract_experiments(experiment, condition)
            # initialize the arena
            arena, arena_color, scaling_factor = initialize_arena(self, sub_experiments, sub_conditions)
            # initialize edginess
            all_traversals_edginess = {}
            # proportion_edgy = {}
            for s in sides: all_traversals_edginess[s] = []
            # proportion_edgy[start] = []

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
                            number_of_edge_vectors = np.sum((np.array(traversal[t*5+3]) < 1.5) * \
                                                                   (np.array(traversal[t*5+2]) > HV_cutoff) * \
                                                                   (np.array(traversal[t*5+1]) < 20*30*60) )
                            all_traversals_edginess[start].append( number_of_edge_vectors )
                            print(number_of_edge_vectors)
                            mouse_data.append(number_of_edge_vectors)

                            # proportion_edgy[start].append(np.sum((np.array(traversal[t*5 + 3]) < 2) * \
                            #                                       (np.array(traversal[t*5 + 2]) > HV_cutoff)) / \
                            #                                     traversal[t*5 + 4] * 30 * 60)

                            traversal_coords = np.array(traversal[t*5+0])
                        else:
                            # all_traversals_edginess[start].append(0)
                            continue
                        m += .5

                        # loop over all paths
                        for trial in range(traversal_coords.shape[0]):
                            # apply strategies
                            if traversal[t*5+3][trial] < 2: #1
                                if traversal[t * 5 + 2][trial] < HV_cutoff:
                                    strategies[0] = strategies[0] + 1 #HV
                                else:
                                    strategies[1] = strategies[1] + 1 #EV

                                # mouse_data.append(int(traversal[t * 5 + 2][trial] > HV_cutoff))

                                display_traversal(arena, arena_color, fast_color, scaling_factor, slow_color, traversal_coords, trial, type)

                            # downsample obstacle condition
                            # if 'obstacle' == self.labels[c] and np.random.rand() > 10/19:
                            #     continue
                            # if 'obstacle (dark)' == self.labels[c] and np.random.rand() > 10/14:
                            #     continue
                            # speed requirement
                            # if duration[trial] < 2:

                    if mouse_data:
                        # all_data.append(mouse_data)
                        all_conditions.append(c)
            # save image
            imageio.imwrite(os.path.join(self.summary_plots_folder, 'Traversals - ' + type + ' - ' + self.labels[c] + '.tif'), arena_color[:, :, ::-1])

            # plot the data
            if type == 'spontaneous':
                plot_number_edgy = np.array(all_traversals_edginess['front']).astype(float) + np.array(all_traversals_edginess['back']).astype(float)
                all_data.append(plot_number_edgy)
                # plot_proportion_edgy = (np.array(proportion_edgy['front']).astype(float) + np.array(proportion_edgy['back']).astype(float)) / 2
            else:
                plot_number_edgy = np.array(all_traversals_edginess['back']).astype(float)
                # plot_proportion_edgy = np.array(proportion_edgy['back']).astype(float)

            for i, (plot_data, ax0) in enumerate(zip([plot_number_edgy], [ax])): #, plot_proportion_edgy , ax2
                print(plot_data)
                # plot each trial
                scatter_axis = scatter_the_axis(p*4/3+.5/3, plot_data)
                ax0.scatter(scatter_axis, plot_data, color=[0,0,0], s=25, zorder=99)
                # do kde
                kde = fit_kde(plot_data, bw=.7 - 0.4*i)  # .04
                plot_kde(ax0, kde, plot_data, z=4*p + .8, vertical=True, normto=1.2, color=[.5, .5, .5], violin=False, clip=True)  # True)

            p+=1

            # plot a stacked bar of strategies
            fig3 = plot_strategies(strategies, homing_vector_color, non_escape_color, edge_vector_color)
            fig3.savefig(os.path.join(self.summary_plots_folder, 'Traversal categories - ' + self.labels[c] + '.tif'), format='tif', bbox_inches = 'tight', pad_inches = 0)
            fig3.savefig(os.path.join(self.summary_plots_folder, 'Traversal categories - ' + self.labels[c] + '.eps'), format='eps', bbox_inches = 'tight', pad_inches = 0)

    plt.show()

    # # save the plot
    fig.savefig(os.path.join(self.summary_plots_folder, 'Traversal edginess comparison.tif'), format='tif', bbox_inches='tight', pad_inches=0)
    fig.savefig(os.path.join(self.summary_plots_folder, 'Traversal edginess comparison.eps'), format='eps', bbox_inches='tight', pad_inches=0)

    group_A = list(np.array(all_data)[np.array(all_conditions) == 1])
    group_B = list(np.array(all_data)[np.array(all_conditions) == 2])
    permutation_test(group_A, group_B, iterations = 10000, two_tailed = True)

    # fig2.savefig(os.path.join(self.summary_plots_folder, 'Traversal proportion edgy.tif'), format='tif', bbox_inches='tight', pad_inches=0)
    # fig2.savefig(os.path.join(self.summary_plots_folder, 'Traversal proportion edgy.eps'), format='eps', bbox_inches='tight', pad_inches=0)
    plt.show()


def plot_speed_traces(self, speed = 'absolute'):
    '''     plot the speed traces       '''
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
        colormap = speed_colormap(scaling_factor, n_bins=256, v_min=0, v_max=65)
        # loop over each experiment and condition
        for e, (experiment, condition) in enumerate(zip(sub_experiments, sub_conditions)):
            # loop over each mouse
            for i, mouse in enumerate(self.analysis[experiment][condition]['speed']):
                # control analysis
                if self.analysis_options['control'] and not mouse=='control': continue
                if not self.analysis_options['control'] and mouse=='control': continue
                # loop over each trial
                for trial in range(len(self.analysis[experiment][condition]['speed'][mouse])):
                    trial_num = fill_in_trial_data(RT, condition, end_idx, experiment, mouse, scaling_factor, self,
                                       speed_traces, subgoal_speed_traces, time, trial, trial_num)
        # print some useful metrics
        print_metrics(RT, end_idx, number_of_mice, number_of_trials)
        # put the speed traces on the plot
        fig = show_speed_traces(colormap, condition, end_idx, experiment, number_of_trials, speed, speed_traces, subgoal_speed_traces, time_axis)
        # save the plot
        fig.savefig(os.path.join(self.summary_plots_folder,'Speed traces - ' + self.labels[c] + '.tif'), format='tif', bbox_inches = 'tight', pad_inches = 0)
        fig.savefig(os.path.join(self.summary_plots_folder,'Speed traces - ' + self.labels[c] + '.eps'), format='eps', bbox_inches = 'tight', pad_inches = 0)
    plt.show()




def plot_escape_paths(self):
    '''     plot the escape paths       '''
    # initialize parameters
    edge_vector_color = np.array([1, .95, .85])
    homing_vector_color = np.array([.725, .725, .725])
    non_escape_color = np.array([0,0,0])

    fps = 30
    escape_duration = 12 #6 #12 # #12 #6 #9 for food # 18 for U
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
        arena, arena_color, scaling_factor = initialize_arena(self, sub_experiments, sub_conditions)
        # more arena stuff for this analysis type
        arena_reference = arena_color.copy()
        arena_color[arena_reference == 245] = 255
        get_arena_details(self, experiment=sub_experiments[0])
        shelter_location = [s / scaling_factor / 10 for s in self.shelter_location]
        # initialize strategy array
        strategies = np.array([0,0,0])
        # loop over each experiment and condition
        for e, (experiment, condition) in enumerate(zip(sub_experiments, sub_conditions)):
            # loop over each mouse
            for i, mouse in enumerate(self.analysis[experiment][condition]['speed']):
                print(mouse)
                # control analysis
                if self.analysis_options['control'] and not mouse=='control': continue
                if not self.analysis_options['control'] and mouse=='control': continue

                # color based on visual vs tactile obst avoidance
                # if mouse == 'CA7190' or mouse == 'CA3210' or mouse == 'CA3155' or mouse == 'CA8100':
                #     edge_vector_color = np.array([.6, .4, .99])
                #     homing_vector_color = np.array([.6, .4, .99])
                # else:
                #     edge_vector_color = np.array([.8, .95, 0])
                #     homing_vector_color = np.array([.8, .95, 0])

                # show escape paths
                show_escape_paths(HV_cutoff, arena, arena_color, arena_reference, c, condition, edge_vector_color, escape_duration, experiment, fps,
                                  homing_vector_color, min_distance_to_shelter, mouse, non_escape_color, scaling_factor, self, shelter_location, strategies,
                                  determine_strategy = ('dark' in experiment and condition=='obstacle'))





        # save image
        # scipy.misc.imsave(os.path.join(self.summary_plots_folder, 'Escape paths - ' + self.labels[c] + '.tif'), arena_color[:,:,::-1])
        imageio.imwrite(os.path.join(self.summary_plots_folder, 'Escape paths - ' + self.labels[c] + '.tif'), arena_color[:,:,::-1])
        # plot a stacked bar of strategies
        fig = plot_strategies(strategies, homing_vector_color, non_escape_color, edge_vector_color)
        fig.savefig(os.path.join(self.summary_plots_folder, 'Escape categories - ' + self.labels[c] + '.tif'), format='tif', bbox_inches = 'tight', pad_inches = 0)
        fig.savefig(os.path.join(self.summary_plots_folder, 'Escape categories - ' + self.labels[c] + '.eps'), format='eps', bbox_inches = 'tight', pad_inches = 0)
    plt.show()

# strategies = np.array([4,5,0])
# fig = plot_strategies(strategies, homing_vector_color, non_escape_color, edge_vector_color)
# plt.show()
# fig.savefig(os.path.join(self.summary_plots_folder, 'Trajectory by previous edge-vectors 2.tif'), format='tif', bbox_inches='tight', pad_inches=0)
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
    HV_cutoff = .681 #.7#5 #.7 #.75 #
    ETD = 10#8
    traj_loc = 40

    edge_vector_color = np.array([.98, .9, .6])**5
    # edge_vector_color = np.array([.99, .95, .6]) ** 5
    homing_vector_color = np.array([0, 0, 0])**5


    # colors for diff conditions
    colors = [np.array([.7, 0, .3]), np.array([0, .8, .5])]
    colors = [np.array([.3,.3,.3]), np.array([1, .2, 0]), np.array([0, .8, .4]), np.array([0, .7, .9])]
    colors = [np.array([.3, .3, .3]), np.array([1, .2, 0]), np.array([.7, 0, .7]), np.array([0, .7, .9])]
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

    repetitions = 1
    for rand_select in range(repetitions):
        # loop over experiments and conditions
        for c, (experiment, condition) in enumerate(zip(self.experiments, self.conditions)):
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
                # loop over each mouse
                for i, mouse in enumerate(self.analysis[experiment][condition]['start time']):
                    # initialize mouse data for stats
                    mouse_data = [[],[],[],[]]
                    print(mouse)
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
                            if t > 8: continue
                            if condition == 'no obstacle' and self.analysis[experiment][condition]['start time'][mouse][trial] < 20: continue
                        else:
                            # if trial > 2: continue
                            # if condition == 'obstacle' and trial != 0: continue
                            # if condition == 'obstacle' and c == 1 and not trial: continue
                            # if condition == 'obstacle' and c == 2 and trial: continue

                            if t>2: continue
                            # if not c and trial: continue
                            # if c and (not trial==2 and not trial==3): continue

                        if trial and 'quick' in experiment: continue
                        elif 'Square' in experiment:
                            # only three trials
                            if t > 2: continue


                            # y_idx = self.analysis[experiment][condition]['path'][mouse][trial][1]
                            # x_idx = self.analysis[experiment][condition]['path'][mouse][trial][0]
                            # # get where the mouse is in the center
                            # y_center_idx = np.argmin(abs(y_idx * scaling_factor - 50))
                            # # only do wall-left trials (this captures all of them only for this particular dataset!)
                            # if (x_idx[y_center_idx] * scaling_factor) > 72 and False: continue
                            # # only ones that start to the left of the obstacle edge
                            # if x_idx[0] * scaling_factor > 60 and False: continue
                            # # not the one HV, for the edge-vector-only plot
                            # if y_idx[0] * scaling_factor < 9: continue  # 11
                        else:
                            # skip certain trials
                            y_start = self.analysis[experiment][condition]['path'][mouse][trial][1][0] * scaling_factor
                            x_start = self.analysis[experiment][condition]['path'][mouse][trial][0][0] * scaling_factor
                            if y_start > 25: continue
                            if abs(x_start-50) > 25: continue

                        end_idx[trial_num] = self.analysis[experiment][condition]['end time'][mouse][trial]
                        if np.isnan(end_idx[trial_num]):
                            mouse_data[3].append(0)
                            continue
                        if (end_idx[trial_num] > escape_duration * fps):
                            mouse_data[3].append(0)
                            continue

                        '''     check for previous edgy homings         '''

                        if 'dark' in experiment: # or True:
                            num_prev_edge_vectors, x_edge = get_num_edge_vectors(self, experiment, condition, mouse, trial)
                            # if num_prev_edge_vectors < 2 and c==2:  print('prev edgy homing'); continue
                            # if num_prev_edge_vectors >= 2 and c==1:  print('prev edgy homing'); continue
                            if num_prev_edge_vectors >= 2:  print('prev edgy homing'); continue

                            # if x_edge in x_edges_used: print('prev edgy escape'); continue

                            print('-----------' + mouse + '--------------')

                            if self.analysis[experiment][condition]['edginess'][mouse][trial] <= HV_cutoff:
                                print('       HV        ')
                            else:
                                print('       EDGY        ')
                                # edgy trial has occurred
                                print('EDGY TRIAL ' + str(trial))
                                x_edges_used.append(x_edge)

                            # select only *with* prev homings
                            # if not num_prev_edge_vectors:
                            #     if not x_edge in x_edges_used:
                            #         if self.analysis[experiment][condition]['edginess'][mouse][trial] > HV_cutoff:
                            #             x_edges_used.append(x_edge)
                            #         continue

                        # add data
                        edginess[trial_num] = self.analysis[experiment][condition]['edginess'][mouse][trial]
                        if 'Square' in experiment and edginess[trial_num] <=-.4:
                            edginess[trial_num] = np.nan
                            continue

                        # get previous edginess #TEMPORARY COMMENT
                        if False or True:
                            time_to_shelter, SR = get_prev_edginess(ETD, condition, experiment, mouse, prev_edginess, dist_to_SH, dist_to_other_SH,
                                                                    scaling_factor, self, traj_loc, trial, trial_num, edginess, delta_ICs, delta_x_end)
                            # get time in center
                            time_in_center[trial_num] = self.analysis[experiment][condition]['in center'][mouse][trial] / 30

                            # if prev_edginess[trial_num] >= .68 and c:
                            #     edginess[trial_num] = np.nan
                            #     continue


                        # get time since obstacle removal?
                        # time_since_down[trial_num] = self.analysis[experiment][condition]['start time'][mouse][trial] - self.analysis[experiment]['probe']['start time'][mouse][0]
                        # add data for stats

                        mouse_data[0].append(int(edginess[trial_num] > HV_cutoff))
                        mouse_data[1].append(edginess[trial_num])
                        mouse_data[2].append(prev_edginess[trial_num])
                        mouse_data[3].append(1)
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
                    #     print(mouse)
                        print('~~0 trials~~')
                    # get prev homings
                    time_to_shelter_all.append(time_to_shelter)

            dist_data_EV_other_all = np.append(dist_data_EV_other_all, dist_to_other_SH[edginess > HV_cutoff])

            print(t_total)

            '''     plot edginess by condition     '''
            # get the data
            # data = abs(edginess)
            data = edginess

            plot_data = data[~np.isnan(data)]
            # plot each trial
            scatter_axis = scatter_the_axis(c, plot_data)
            ax.scatter(scatter_axis[plot_data>HV_cutoff], plot_data[plot_data>HV_cutoff], color=edge_vector_color[::-1], s=25, zorder = 99)
            ax.scatter(scatter_axis[plot_data<=HV_cutoff], plot_data[plot_data<=HV_cutoff], color=homing_vector_color[::-1], s=25, zorder = 99)
            #do kde
            try:
                kde = fit_kde(plot_data, bw=.04)  # .04
                if 'Square' in experiment:
                    plot_kde(ax, kde, plot_data, z=3*c + .3, vertical=True, normto=1, color=[.5,.5,.5], violin=False, clip=False, cutoff = HV_cutoff, cutoff_colors = [homing_vector_color[::-1], edge_vector_color[::-1]])
                else:
                    plot_kde(ax, kde, plot_data, z=3*c + .3, vertical=True, normto=1.3, color=[.5,.5,.5], violin=False, clip=True, cutoff = HV_cutoff, cutoff_colors = [homing_vector_color[::-1], edge_vector_color[::-1]])
            except: pass

            # plot the polar plot or initial trajectories
            plt.figure(fig4.number)
            ax4 = plt.subplot(1,4,4 - c, polar=True)
            plt.axis('off')

            mean_value = np.mean(plot_data)
            value_color = mean_value * edge_vector_color[::-1] + (1 - mean_value) * homing_vector_color[::-1]

            # ax4.plot([0, mean_value + 3 * np.pi / 2], [0, 1.3], color=value_color, alpha=1, linewidth=1)
            ax4.arrow(mean_value + 3 * np.pi / 2, 0, 0, 2, color=value_color, alpha=1, width = 0.05, linewidth=2)
            ax4.plot([0, 0 + 3 * np.pi / 2], [0, 2.5], color=[.5,.5,.5], alpha=1, linewidth=1, linestyle = '--')
            ax4.plot([0, 1 + 3 * np.pi / 2], [0, 2.5], color=[.5,.5,.5], alpha=1, linewidth=1, linestyle = '--')

            for point in plot_data:
                if point > HV_cutoff:
                    ax4.plot([0, point + 3 * np.pi/2], [0, 1.5], color=edge_vector_color[::-1], alpha = .8, linewidth=1)
                else:
                    ax4.plot([0, point + 3 * np.pi / 2], [0, 1.5], color=homing_vector_color[::-1], alpha=.8, linewidth=1)

            # compare all conditions on one plot
            # if c!= 1:
            #     plt.figure(fig5.number)
            #     ax5 = plt.subplot(111, polar=True)
            #     plt.axis('off')
            #     ax5.plot([0, np.median(plot_data) + 3 * np.pi/2], [0, 1.475 + offset[c]], color=colors[c], alpha = 1, linewidth=4)
            #     Q1 = np.percentile(plot_data, 25)
            #     Q3 = np.percentile(plot_data, 75)
            #
            #     ax5.bar((Q3 + Q1)/2 + 3 * np.pi/2, 1.5 + offset[c], width=Q3-Q1, bottom=0.0, color=colors[c], alpha=0.35)
            '''         plot the correaltion between dist to ICs and dist to x final        '''
            # plt.ion()

            # fig6, ax6 = plt.subplots(figsize=(9, 9))
            # # ax6.set_xlim([-2, 59])
            # # ax6.set_ylim([-2, 59])
            # # ax6.set_xticks(np.arange(0, 60, 10))
            # # ax6.set_xticks(np.arange(0, 60, 10))
            #
            # x_data = list(flatten(delta_ICs))
            # y_data = list(flatten(delta_x_end))
            #
            # ax6.scatter(x_data, y_data, color=colors[c], s=60, alpha=1, edgecolors=colors[c] / 2, linewidth=1)
            # # do correlation
            # r, p = scipy.stats.pearsonr(x_data, y_data)
            # print(r, p)
            # # do linear regression
            # plot_data_corr, prediction = do_linear_regression(np.array(y_data), np.array(x_data))
            # # plot linear regresssion
            # ax6.plot(plot_data_corr, prediction['Pred'].values, color=colors[c], linewidth=1, linestyle='--', alpha=.7)  # color=[.0, .0, .0]
            # ax6.fill_between(plot_data_corr, prediction['lower'].values, prediction['upper'].values, color=colors[c], alpha=.075)  # color=[.2, .2, .2]


            # plt.draw()
            # plt.pause(0.01)
            #
            # fig6.savefig(os.path.join(self.summary_plots_folder, 'Correlation to ICs.tif'), format='tif', bbox_inches='tight', pad_inches=0)
            # fig6.savefig(os.path.join(self.summary_plots_folder, 'Correlation to ICs.eps'), format='eps', bbox_inches='tight', pad_inches=0)



            if False or True:
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
                    fig_corr.savefig(os.path.join(self.summary_plots_folder, 'Edginess by ' + data_label + ' - ' + self.labels[c] + '.tif'), format='tif')
                    fig_corr.savefig(os.path.join(self.summary_plots_folder, 'Edginess by ' + data_label + ' - ' + self.labels[c] + '.eps'), format='eps')

                # test correlation and stats thru permutation test
                # data_x = list(np.array(all_data[2])[np.array(all_conditions) == c])
                # data_y = list(np.array(all_data[1])[np.array(all_conditions) == c])
                # permutation_correlation(data_x, data_y, iterations=10000, two_tailed=False, pool_all = True)

    '''         plot the dist to prev edge vectors      '''
    # dist_data_EV = dist_to_SH[data>HV_cutoff]
    # dist_data_EV = dist_data_EV[~np.isnan(dist_data_EV)]
    #
    # # dist_data_EV_other = dist_to_other_SH[data>HV_cutoff]
    # dist_data_EV_other_all = dist_data_EV_other_all[~np.isnan(dist_data_EV_other_all)]
    #
    # # plt.ion()
    #
    # fig6, ax6 = plt.subplots(figsize=(9, 9))
    #
    # bin_size = 5
    # bins = np.arange(0,40,bin_size)
    # widths = np.diff(bins)
    # hist_same_side, bins = np.histogram(dist_data_EV, bins=bins, normed=True)
    # hist_other_side, bins = np.histogram(dist_data_EV_other_all, bins=bins, normed=True)
    #
    # plt.bar(bins[:-1] + bin_size / 2, hist_same_side * bin_size, widths, facecolor = edge_vector_color[::-1], alpha = .8)
    # plt.bar(bins[:-1] + bin_size / 2, hist_other_side * bin_size, widths, facecolor= homing_vector_color[::-1], alpha= .35)
    #
    # plt.xlim([-5, 35])
    # plt.ylim([0, .36])
    # plt.yticks([0, .1, .2, .3])
    #
    # permutation_test(list(dist_data_EV), list(dist_data_EV_other_all), iterations=10000, two_tailed=True)

    # save the plot
    # fig6.savefig(os.path.join(self.summary_plots_folder, 'Generalizability.tif'), format='tif', bbox_inches='tight', pad_inches=0)
    # fig6.savefig(os.path.join(self.summary_plots_folder, 'Generalizability.eps'), format='eps', bbox_inches='tight', pad_inches=0)

    # save the plot
    fig.savefig(os.path.join(self.summary_plots_folder, 'Edginess comparison.tif'), format='tif', bbox_inches='tight', pad_inches=0)
    fig.savefig(os.path.join(self.summary_plots_folder, 'Edginess comparison.eps'), format='eps', bbox_inches='tight', pad_inches=0)

    fig4.savefig(os.path.join(self.summary_plots_folder, 'Angle comparison.tif'), format='tif', bbox_inches='tight', pad_inches=0)
    fig4.savefig(os.path.join(self.summary_plots_folder, 'Angle comparison.eps'), format='eps', bbox_inches='tight', pad_inches=0)

    # fig5.savefig(os.path.join(self.summary_plots_folder, 'Angle dist comparison.tif'), format='tif', bbox_inches='tight', pad_inches=0)
    # fig5.savefig(os.path.join(self.summary_plots_folder, 'Angle dist comparison.eps'), format='eps', bbox_inches='tight', pad_inches=0)

    plt.show()
    time_to_shelter_all = np.concatenate(list(flatten(time_to_shelter_all))).astype(float)
    np.percentile(time_to_shelter_all, 25)
    np.percentile(time_to_shelter_all, 75)

    group_A = list(np.array(all_data[0])[np.array(all_conditions) == 0])
    group_B = list(np.array(all_data[0])[np.array(all_conditions) == 1])
    permutation_test(group_A, group_B, iterations = 10000, two_tailed = True)

    group_A = list(np.array(all_data[3])[np.array(all_conditions) == 0])
    group_B = list(np.array(all_data[3])[np.array(all_conditions) == 2])
    permutation_test(group_A, group_B, iterations = 10000, two_tailed = True)

    import pandas
    df = pandas.DataFrame(data={"mouse_id": mouse_ID, "condition": all_conditions, "x-data": all_data[2], "y-data": all_data[1]})
    df.to_csv("./Foraging Path Types.csv", sep=',', index=False)


def plot_metrics_by_strategy(self):
    '''     plot the escape paths       '''
    # initialize parameters
    edge_vector_color = np.array([1, .95, .85])
    homing_vector_color = np.array([.725, .725, .725])
    non_escape_color = np.array([0,0,0])

    ETD = 8
    traj_loc = 40

    fps = 30
    escape_duration = 9 #12 #6 #9 for food
    HV_cutoff = .65 #.65
    edgy_cutoff = .8 #.8

    # loop over experiments and conditions
    for c, (experiment, condition) in enumerate(zip(self.experiments, self.conditions)):
        # extract experiments from nested list
        sub_experiments, sub_conditions = extract_experiments(experiment, condition)
        # get the number of trials
        number_of_trials = get_number_of_trials(sub_experiments, sub_conditions, self.analysis)
        number_of_mice = get_number_of_mice(sub_experiments, sub_conditions, self.analysis)
        # initialize array to fill in with each trial's data
        efficiency, efficiency_RT, duration, num_prev_homings, scaling_factor, _, trial_num, prev_edginess, edginess, \
        duration_RT, avg_speed_RT, peak_speed, RT, escape_speed, strategy = \
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
                for trial in range(len(self.analysis[experiment][condition]['end time'][mouse])):
                    trial_num += 1
                    # impose coniditions - escape duration
                    duration[trial_num] = self.analysis[experiment][condition]['end time'][mouse][trial]
                    if np.isnan(duration[trial_num]) or (duration[trial_num] > (escape_duration * fps)): continue
                    # skip certain trials
                    y_start = self.analysis[experiment][condition]['path'][mouse][trial][1][0] * scaling_factor
                    x_start = self.analysis[experiment][condition]['path'][mouse][trial][0][0] * scaling_factor
                    # needs to start at top
                    if y_start > 25: continue
                    if abs(x_start - 50) > 25: continue
                    # get the strategy used
                    edgy_escape = self.analysis[experiment][condition]['edginess'][mouse][trial] > edgy_cutoff
                    # is it a homing vector
                    strategy_code = 0
                    # TEMPORARY COMMENTING
                    if not edgy_escape:
                        if self.analysis[experiment][condition]['edginess'][mouse][trial] < HV_cutoff: strategy_code = 0 # homing vector
                        else: continue
                    else:
                        # get the strategy used -- NUMBER OF PREVIOUS EDGE VECTOR HOMINGS
                        time_to_shelter, SR = get_prev_edginess(ETD, condition, experiment, mouse, prev_edginess, [], [],
                                                                scaling_factor, self, traj_loc, trial, trial_num, edginess, [], [])

                        if prev_edginess[trial_num] >= .68: strategy_code = 1 # path learning
                        elif prev_edginess[trial_num] < .5: strategy_code = 2 # map-based
                        else: continue


                        # how many prev homings to that edge: if 0, then map-based, if >1, then PL
                        # num_edge_vectors, x_edge = get_num_edge_vectors(self, experiment, condition, mouse, trial)
                        # if num_edge_vectors >= 2:
                        #     strategy_code = 1 # path learning
                        #     print('PATH LEARNING -- ' + mouse + ' - trial ' + str(trial))
                        # elif num_edge_vectors <= 1:
                        #     strategy_code = 2 # map-based
                        #     print('MAP BASED -- ' + mouse + ' - trial ' + str(trial))
                        # else: continue


                    strategy[trial_num] = strategy_code
                    # add data for each metric
                    RT[trial_num] = self.analysis[experiment][condition]['RT'][mouse][trial]
                    # avg_speed[trial_num] = np.mean(self.analysis[experiment][condition]['speed'][mouse][trial][10*fps : 10*fps+int(duration[trial_num])])
                    avg_speed_RT[trial_num] = np.mean(self.analysis[experiment][condition]['speed'][mouse][trial][10*fps + int(RT[trial_num]*30) : 10*fps+int(duration[trial_num])])
                    peak_speed[trial_num] = np.max(self.analysis[experiment][condition]['speed'][mouse][trial][10*fps : 10*fps+int(duration[trial_num])])*fps*scaling_factor
                    escape_speed[trial_num] = self.analysis[experiment][condition]['optimal path length'][mouse][trial] * scaling_factor / (duration[trial_num]/30)
                    efficiency[trial_num] = np.min((1, self.analysis[experiment][condition]['optimal path length'][mouse][trial] / \
                                                    self.analysis[experiment][condition]['full path length'][mouse][trial]))
                    efficiency_RT[trial_num] = np.min((1, self.analysis[experiment][condition]['optimal RT path length'][mouse][trial] / \
                                                       self.analysis[experiment][condition]['RT path length'][mouse][trial]))

                    duration_RT[trial_num] = (duration[trial_num] / fps - RT[trial_num]) / self.analysis[experiment][condition]['optimal path length'][mouse][trial] / scaling_factor * 100
                    duration[trial_num] = duration[trial_num] / fps / self.analysis[experiment][condition]['optimal path length'][mouse][trial] / scaling_factor * 100

                    # add data for stats
                    mouse_id[trial_num] = m
                m+=1

        for metric, data in zip(['Reaction time', 'Peak speed','Path efficiency - RT','Duration', 'Duration - RT'],\
                                [RT, peak_speed, efficiency_RT, duration, duration_RT]):
            # format data
            x_data = strategy[~np.isnan(data)]
            y_data = data[~np.isnan(data)]

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
            elif 'Path efficiency' in metric:
                ax.plot([-.75, 3], [.5,.5], linestyle='--', color=[.5, .5, .5, .5])
                ax.plot([-.75, 3], [.75, .75], linestyle='--', color=[.5, .5, .5, .5])
                ax.plot([-.75, 3], [1, 1], linestyle='--', color=[.5, .5, .5, .5])
            elif 'Duration' in metric:
                ax.plot([-.75, 3], [0, 0], linestyle='--', color=[.5, .5, .5, .5])
                ax.plot([-.75, 3], [10, 10], linestyle='--', color=[.5, .5, .5, .5])
                ax.plot([-.75, 3], [5, 5], linestyle='--', color=[.5, .5, .5, .5])

            #initialize stats array
            stats_data = [[], [], []]

            # go thru each strategy
            for s in [0,1,2]:
                # format data
                plot_data = y_data[x_data==s]
                median = np.percentile(plot_data, 50);
                third_quartile = np.percentile(plot_data, 75);
                first_quartile = np.percentile(plot_data, 25)
                if 'Reaction' in metric: print(str(first_quartile), str(median), str(third_quartile))
                IQR = third_quartile - first_quartile
                # remove outliers
                outliers = abs(plot_data - median) > 2*IQR
                plot_data = plot_data[~outliers]

                # plot all data
                ax.scatter(np.ones_like(plot_data)*s, plot_data, color=[0,0,0], s=30, zorder = 99)
                # plot kde
                if 'efficiency' in metric: bw_factor = .02
                elif 'speed' in metric or 'Duration' in metric or 'efficiency' in metric: bw_factor = .04
                else: bw_factor = .09

                kde = fit_kde(plot_data, bw=np.median(y_data)*bw_factor)  # .04
                plot_kde(ax, kde, plot_data, z= s + .1, vertical=True, normto=.4, color=[.75, .75, .75], violin=False, clip=True)
                # plot errorbar
                ax.errorbar(s - .15, median, yerr=np.array([[median - first_quartile], [third_quartile - median]]), color=[0, 0, 0], capsize=10, capthick=3, alpha=1, linewidth=3)
                ax.scatter(s - .15, median, color=[0, 0, 0], s=175, alpha=1)

                print(len(plot_data))

                # get mouse ids for stats
                mouse_id_stats = mouse_id[~np.isnan(data)]
                mouse_id_stats = mouse_id_stats[x_data==s]
                mouse_id_stats = mouse_id_stats[~outliers]

                for m in np.unique(mouse_id_stats):
                    stats_data[s].append( list(plot_data[mouse_id_stats==m]) )

            print(metric)
            # for ss in [[0,1], [0,2], [1,2]]:
            #     group_A = stats_data[ss[0]]
            #     group_B = stats_data[ss[1]]
            #     permutation_test(group_A, group_B, iterations=10000, two_tailed=True)

            # save figure
            fig.savefig(os.path.join(self.summary_plots_folder, metric + ' - ' + self.labels[c] + '.tif'), format='tif', bbox_inches='tight', pad_inches=0)
            fig.savefig(os.path.join(self.summary_plots_folder, metric + ' - ' + self.labels[c] + '.eps'), format='eps', bbox_inches='tight', pad_inches=0)

    plt.show()

    plt.close('all')



def plot_prediction(self):
    # initialize parameters
    fps = 30
    escape_duration = 6
    ETD = 10 #4
    traj_loc = 40


    # initialize figures
    fig1, ax1, fig2, ax2 = initialize_figures_prediction(self)
    # loop over experiments and conditions
    for c, (experiment, condition) in enumerate(zip(self.experiments, self.conditions)):
        # extract experiments from nested list
        sub_experiments, sub_conditions = extract_experiments(experiment, condition)
        # get the number of trials
        number_of_trials = get_number_of_trials(sub_experiments, sub_conditions, self.analysis)
        number_of_mice = get_number_of_mice(sub_experiments, sub_conditions, self.analysis)
        # initialize array to fill in with each trial's data
        edginess, end_idx, angle_turned, _, _, prev_edginess, scaling_factor, _, trial_num, prev_movement_and_ICs, data_y_for_prev_movement, dist_to_SH, dist_to_other_SH = \
            initialize_variable_edginess(number_of_trials, self, sub_experiments)
        # loop over each experiment and condition
        for e, (experiment, condition) in enumerate(zip(sub_experiments, sub_conditions)):
            # loop over each mouse
            for i, mouse in enumerate(self.analysis[experiment][condition]['start time']):
                if self.analysis_options['control'] and not mouse=='control': continue
                if not self.analysis_options['control'] and mouse=='control': continue
                # loop over each trial
                prev_homings = []
                for trial in range(len(self.analysis[experiment][condition]['end time'][mouse])):
                    trial_num += 1
                    # impose conditions
                    if trial > 2: continue
                    end_idx[trial_num] = self.analysis[experiment][condition]['end time'][mouse][trial]
                    if np.isnan(end_idx[trial_num]): continue
                    if (end_idx[trial_num] > escape_duration * fps): continue
                    elif trial and 'quick' in experiment: continue
                    # skip certain trials
                    y_start = self.analysis[experiment][condition]['path'][mouse][trial][1][0] * scaling_factor
                    x_start = self.analysis[experiment][condition]['path'][mouse][trial][0][0] * scaling_factor
                    if y_start > 25: continue
                    if abs(x_start-50) > 25: continue #22
                    # add data
                    edginess[trial_num] = abs(self.analysis[experiment][condition]['edginess'][mouse][trial])
                    # get previous edginess
                    time_to_shelter, SR = get_prev_edginess(ETD, condition, experiment, mouse, prev_edginess, dist_to_SH, dist_to_other_SH, scaling_factor, self, traj_loc, trial, trial_num, edginess, [], [])
                    _, _, prev_edginess_all, elig_idx = get_all_prev_edginess(ETD, condition, experiment, mouse, prev_edginess, dist_to_SH, dist_to_other_SH, scaling_factor, self, traj_loc, trial, trial_num, edginess, [], [])
                    # prev_edginess[trial_num] = np.mean(self.analysis[experiment][condition]['prev movements'][mouse][trial][3][-ETD:]) #TEMP replace prev edginess w mean angle turned


                    elig_idx = np.ones_like(elig_idx).astype(bool)

                    x_edge = self.analysis[experiment][condition]['x edge'][mouse][trial]
                    if x_edge==25: x_edge = 0
                    else: x_edge = 100
                    x_edge = 0

                    # get the angle turned during the escape
                    # angle_turned[trial_num] = self.analysis[experiment][condition]['movement'][mouse][trial][2] #turn angle
                    angle_turned[trial_num] = abs(self.analysis[experiment][condition]['x escape'][mouse][trial] - x_edge) # X_SH TEMP
                    # angle_turned[trial_num] = np.mean(prev_edginess_all[elig_idx]) #TEMP


                    # get the angle turned, delta x, delta y, and delta phi of previous homings
                    # x_start, y_start, angle_start, turn_angle
                    bout_start_angle = self.analysis[experiment][condition]['movement'][mouse][trial][1]
                    bout_start_position  = self.analysis[experiment][condition]['movement'][mouse][trial][0]
                    start_time = self.analysis[experiment][condition]['start time'][mouse][trial]


                    # get initial conditions and endpoint quantities
                    IC_x = np.array(self.analysis[experiment][condition]['prev movements'][mouse][trial][0][-ETD:])
                    IC_y = np.array(self.analysis[experiment][condition]['prev movements'][mouse][trial][1][-ETD:])
                    IC_angle = np.array(self.analysis[experiment][condition]['prev movements'][mouse][trial][2][-ETD:])
                    IC_time = np.array(self.analysis[experiment][condition]['prev homings'][mouse][trial][3][-ETD:])

                    end_x = abs(np.array(self.analysis[experiment][condition]['prev movements'][mouse][trial][4][-ETD:]) - x_edge)
                    turn_angles = self.analysis[experiment][condition]['prev movements'][mouse][trial][3][-ETD:]

                    # get sorting of output differences
                    # delta_turn = abs(turn_angles - angle_turned[trial_num])
                    # delta_turn[delta_turn > 180] = 360 - delta_turn[delta_turn > 180]

                    delta_turn = abs(end_x - angle_turned[trial_num])
                    diff_idx = np.argsort(np.argsort(delta_turn))



                    # calculate difference in ICs
                    delta_x = abs( np.array(IC_x[elig_idx] - bout_start_position[0]) )
                    delta_y = abs( np.array(IC_y[elig_idx] - bout_start_position[1]) )
                    delta_angle = abs( np.array(IC_angle[elig_idx] - bout_start_angle) )
                    delta_angle[delta_angle > 180] = 360 - delta_angle[delta_angle > 180]
                    delta_time = start_time - np.array(IC_time[elig_idx])

                    prev_movement_and_ICs[0].append(list(np.sort(delta_x)[diff_idx]))
                    prev_movement_and_ICs[1].append(list(np.sort(delta_y)[diff_idx]))
                    prev_movement_and_ICs[2].append(list(np.sort(delta_angle)[diff_idx]))
                    prev_movement_and_ICs[4].append(list(np.sort(delta_time)[diff_idx]))


                    # prev_movement_and_ICs[3].append(turn_angles) # turn angle
                    prev_movement_and_ICs[3].append(list(end_x[elig_idx])) # X_SH movement TEMP
                    # prev_movement_and_ICs[3].append(list(prev_edginess_all[elig_idx]))  # prev edginess TEMP




                    data_y_for_prev_movement[0].append(list(np.ones(len(IC_x[elig_idx])) * edginess[trial_num]))
                    data_y_for_prev_movement[1].append(list(np.ones(len(IC_x[elig_idx])) * angle_turned[trial_num]))


        # format the prev movement and ICs data
        delta_x = scipy.stats.zscore(list(flatten(prev_movement_and_ICs[0])))
        delta_y = scipy.stats.zscore(list(flatten(prev_movement_and_ICs[1])))
        delta_angle = scipy.stats.zscore(list(flatten(prev_movement_and_ICs[2])))
        turn_angle = scipy.stats.zscore(list(flatten(prev_movement_and_ICs[3])))
        delta_time = scipy.stats.zscore(list(flatten(prev_movement_and_ICs[4])))
        #
        # delta_x = list(flatten(prev_movement_and_ICs[0]))
        # delta_y = list(flatten(prev_movement_and_ICs[1]))
        # delta_angle = list(flatten(prev_movement_and_ICs[2]))
        # turn_angle = np.array(list(flatten(prev_movement_and_ICs[3])))
        # delta_time = list(flatten(prev_movement_and_ICs[4]))

        edginess_for_prev_movement   = list(flatten(data_y_for_prev_movement[0]))
        turn_angle_for_prev_movement = list(flatten(data_y_for_prev_movement[1]))

        prev_movements_and_ICs_array = np.ones((len(delta_x), 5)) * 0 #np.nan

        prev_movements_and_ICs_array[:, 0] = turn_angle * delta_x
        prev_movements_and_ICs_array[:, 1] = turn_angle * delta_y
        prev_movements_and_ICs_array[:, 2] = turn_angle * delta_angle
        prev_movements_and_ICs_array[:, 3] = turn_angle
        prev_movements_and_ICs_array[:, 4] = turn_angle * delta_time



        # get the data
        predict_data_y_all = [[  edginess[~np.isnan(edginess)].reshape(-1, 1), # for the mean edginess input data
                                 angle_turned[~np.isnan(edginess)].reshape(-1, 1)], # for the mean edginess input data
                                  [edginess_for_prev_movement, turn_angle_for_prev_movement]] # for the movements input data
        data_y_labels = ['trajectory', 'angle']

        predict_data_x_all = [prev_edginess[~np.isnan(edginess)].reshape(-1, 1),  # mean prev edginess
                              prev_movements_and_ICs_array] # all prev homing movements
                             # np.zeros_like(prev_edginess[~np.isnan(edginess)]).reshape(-1, 1),  # zeros as negative control

        # edgy input colors
        input_colors = [[0, .6, .4], [.6, 0, .4]] #[.5, .5, .5],
        # split the data for cross val
        trials = 1000
        # loop acros angle prediction and traj prediction
        for h, (fig, ax) in enumerate(zip([fig1, fig2],[ax1, ax2])):
            # loop across traj inputs and movement inputs
            for i, predict_data_x in enumerate(predict_data_x_all):
                # get prediction data
                predict_data_y = predict_data_y_all[i][h]
                # get color
                color = input_colors[i]
                # initialize prediction arrays
                prediction_scores = np.zeros(trials)
                for j in range(trials):
                    X_train, X_test, y_train, y_test = train_test_split(predict_data_x, \
                                                predict_data_y, test_size=0.5, random_state=j)
                    # create the model
                    # LR = linear_model.LinearRegression()
                    LR = linear_model.Ridge(alpha = .5)
                    # train the model
                    LR.fit(X_train, y_train)
                    # print(LR.coef_)
                    # get the score
                    prediction_scores[j] = LR.score(X_test, y_test)
                # exclude super negative ones
                prediction_scores = prediction_scores[prediction_scores > np.percentile(prediction_scores, 10)]
                # plot the scores
                # ax.scatter(prediction_scores, np.zeros_like(prediction_scores), color=color, s=20, alpha = .1)
                #do kde
                kde = fit_kde(prediction_scores, bw=.03)  # .04
                plot_kde(ax, kde, prediction_scores, z = 0, vertical=False, color=color, violin=False, clip=False)  # True)
            fig.savefig(os.path.join(self.summary_plots_folder,'Prediction of ' + data_y_labels[h] + ' - ' + self.labels[c] + '.tif'), format='tif')
            fig.savefig(os.path.join(self.summary_plots_folder,'Precition of ' + data_y_labels[h] + ' - ' + self.labels[c] + '.eps'), format='eps')
    plt.show()
    print('hi')


def plot_efficiency(self):
    # initialize parameters
    fps = 30
    traj_loc = 40
    escape_duration = 6
    HV_cutoff = .681
    ETD = np.inf
    ax2, fig2, ax3, fig3 = initialize_figures_efficiency(self)
    efficiency_data = [[],[],[],[]]
    duration_data = [[],[],[],[]]
    # initialize arrays for stats
    efficiency_data_all = []
    duration_data_all = []
    prev_homings_data_all = []
    all_conditions = []
    mouse_ID = []; m = 1
    data_condition = ['naive','experienced']
    data_condition = ['food','escape']
    data_condition = ['OR - EV', 'OR - HV', 'OF']

    # loop over experiments and conditions
    for c, (experiment, condition) in enumerate(zip(self.experiments, self.conditions)):
        # extract experiments from nested list
        sub_experiments, sub_conditions = extract_experiments(experiment, condition)
        # get the number of trials
        number_of_trials = get_number_of_trials(sub_experiments, sub_conditions, self.analysis)
        number_of_mice = get_number_of_mice(sub_experiments, sub_conditions, self.analysis)
        # initialize array to fill in with each trial's data
        efficiency, efficiency_RT, end_idx, num_prev_homings, scaling_factor, time, trial_num, trials, edginess, prev_edginess, dist_to_SH, dist_to_other_SH, _, _, _ = \
            initialize_variables_efficiency(number_of_trials, self, sub_experiments)
        # loop over each experiment and condition
        for e, (experiment, condition) in enumerate(zip(sub_experiments, sub_conditions)):
            # loop over each mouse
            for i, mouse in enumerate(self.analysis[experiment][condition]['full path length']):
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
                    if t > 2 and not 'food' in experiment: continue
                    if t > 8: continue
                    # impose coniditions
                    end_idx[trial_num] = self.analysis[experiment][condition]['end time'][mouse][trial]
                    if (end_idx[trial_num] > escape_duration * fps) or np.isnan(end_idx[trial_num]): continue
                    # skip certain trials
                    y_start = self.analysis[experiment][condition]['path'][mouse][trial][1][0] * scaling_factor
                    x_start = self.analysis[experiment][condition]['path'][mouse][trial][0][0] * scaling_factor
                    if y_start > 25: continue
                    if abs(x_start-50) > 25: continue #22

                    # get prev edginess
                    _, _ = get_prev_edginess(ETD, condition, experiment, mouse, prev_edginess, dist_to_SH, dist_to_other_SH,
                                                            scaling_factor, self, traj_loc, trial, trial_num, edginess, [], [])

                    # only do predict edgy:
                    # if c == 0:
                    #     if prev_edginess[trial_num] <= HV_cutoff and 'down' in experiment: continue
                    # elif c == 1:
                    #     if prev_edginess[trial_num] > HV_cutoff and 'down' in experiment: continue

                    # add data
                    fill_in_trial_data_efficiency(ETD, condition, efficiency, efficiency_RT, experiment, mouse, num_prev_homings, self, time, trial, trial_num, trials, edginess, t)



                    # normalize end idx to
                    RT = self.analysis[experiment][condition]['RT'][mouse][trial]
                    if not RT: continue

                    end_idx[trial_num] = (end_idx[trial_num] / fps - RT) / self.analysis[experiment][condition]['optimal path length'][mouse][trial] / scaling_factor * 100

                    # add data for stats
                    efficiency_data_mouse.append(efficiency[trial_num])
                    # duration_data_mouse.append(end_idx[trial_num]) #TEMP COMMENTING
                    duration_data_mouse.append(RT)
                    prev_homings_data_mouse.append(num_prev_homings[trial_num])

                    t += 1

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
        for i, (data, data_label) in enumerate(zip([efficiency, end_idx, edginess], ['Efficiency', 'Duration', 'Trajectory'])):

        # for i, (data, data_label) in enumerate(zip([edginess, efficiency, end_idx], ['Trajectory', 'Efficiency', 'Duration'])):
        #     for x_data, x_data_label in zip([num_prev_homings], ['Prior homings']):
            plot_data = data[~np.isnan(data)]
            if False or True:
                for x_data, x_data_label in zip([num_prev_homings, trials, time], ['Prior homings', 'Trials', 'Time']):
                    # initialize figure
                    fig1, ax1 = plt.subplots(figsize=(9, 9))
                    # set up the figure
                    if data_label=='Efficiency': ax1.set_ylim([-.03, 1.03])
                    elif data_label=='Duration': ax1.set_ylim([-.1, 6.5])
                    # only plot escapes
                    data_for_box_plot = data[~np.isnan(data)]
                    x_data = x_data[~np.isnan(data)]
                    # format for time and trial differently
                    if np.max(x_data) < 5:
                        plot_data_x, plot_data = x_data, data_for_box_plot
                        ax1.set_xticks(np.unique(x_data).astype(int))
                    else:
                        plot_data_x, plot_data = x_data[x_data < 60], data_for_box_plot[x_data < 60]
                        ax1.set_xticks(np.arange(5,25,5))
                        # ax1.set_xlim([5,20])
                    # get the correlation
                    r, p = scipy.stats.pearsonr(plot_data_x, plot_data)
                    print(r, p)
                    # jitter the axis
                    scatter_axis = scatter_the_axis_efficiency(plot_data, plot_data_x)
                    # plot each trial
                    ax1.scatter(scatter_axis, plot_data, color=[0,0,0], s=30, alpha=1, edgecolor=[0,0,0], linewidth=1)
                    ax1.scatter(scatter_axis[plot_data > HV_cutoff], plot_data[plot_data > HV_cutoff], color=[0,0,0], s=50, alpha=1, edgecolor=[0, 0, 0], linewidth=1)
                    # do a linear regression
                    plot_data_x, prediction = do_linear_regression(plot_data, plot_data_x)
                    # plot the linear regression
                    ax1.plot(plot_data_x, prediction['Pred'].values, color=[.0, .0, .0], linewidth=1, linestyle='--', alpha=.7)
                    ax1.fill_between(plot_data_x, prediction['lower'].values, prediction['upper'].values, color=[.2,.2,.2], alpha=.05)  # 6
                    # save the plot
                    plt.savefig(os.path.join(self.summary_plots_folder, data_label + ' by ' + x_data_label + ' - ' + self.labels[c] + '.tif'), format='tif')
                    plt.savefig(os.path.join(self.summary_plots_folder, data_label + ' by ' + x_data_label + ' - ' + self.labels[c] + '.eps'), format='eps')
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
            # fig.savefig(os.path.join(self.summary_plots_folder, data_label + ' comparison - ' + self.labels[c] + '.tif'), format='tif')
            # fig.savefig(os.path.join(self.summary_plots_folder, data_label + ' comparison - ' + self.labels[c] + '.eps'), format='eps')

    plt.show()

    # test correlation and stats thru permutation test
    data_x = prev_homings_data_all
    data_y = efficiency_data_all
    permutation_correlation(data_x, data_y, iterations=10000, two_tailed=False, pool_all=False)

    # do t test
    t, p = scipy.stats.ttest_ind(efficiency_data[0], efficiency_data[1], equal_var=False)
    print('Efficiency: ' + str(p))
    print(np.mean(efficiency_data[0]))
    print(np.mean(efficiency_data[1]))

    t, p = scipy.stats.ttest_ind(duration_data[0], duration_data[1], equal_var=False)
    print('Duration: ' + str(p))
    print(np.mean(duration_data[0]))
    print(np.mean(duration_data[1]))

    group_A = list(np.array(efficiency_data_all)[np.array(all_conditions) == 'OFn'])
    group_B = list(np.array(efficiency_data_all)[np.array(all_conditions) == 'ORe'])
    permutation_test(group_A, group_B, iterations = 10000, two_tailed = False)

    group_A = list(np.array(duration_data_all)[np.array(all_conditions) == 'food'])
    group_B = list(np.array(duration_data_all)[np.array(all_conditions) == 'escape'])
    permutation_test(group_A, group_B, iterations = 10000, two_tailed = True)

    np.percentile(list(flatten(group_A)), 25)
    np.percentile(list(flatten(group_A)), 50)
    np.percentile(list(flatten(group_A)), 75)
    np.percentile(list(flatten(group_B)), 25)
    np.percentile(list(flatten(group_B)), 50)
    np.percentile(list(flatten(group_B)), 75)

    a = np.mean(list(flatten(group_A)))
    b = np.mean(list(flatten(group_B)))
    (a - b) / b


    import pandas
    df = pandas.DataFrame(data={"mouse_id": mouse_ID, "condition": all_conditions, "data": efficiency_data_all})
    df.to_csv("./Escape Efficiency.csv", sep=',', index=False)

    import pandas
    df = pandas.DataFrame(data={"mouse_id": mouse_ID, "condition": all_conditions, "data": efficiency_data_all})
    df.to_csv("./Escape Duration.csv", sep=',', index=False)





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
        exploration_blur = (exploration_blur / np.percentile(exploration_blur, saturation_percentile) * 255)
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
        scipy.misc.imsave(os.path.join(self.summary_plots_folder, experiment + '_exploration_' + condition + '.tif'), exploration_image) #[:,:,::-1])

        shape = self.analysis[sub_experiments[0]]['obstacle']['shape']


        # obstacle_type = self.analysis[sub_experiments[0]]['obstacle']['type']
        # _, _, shelter_roi = model_arena(shape, False, False, obstacle_type, simulate=False, dark=self.dark_theme)
        # percent_in_shelter = []
        # for m in range( exploration.shape[2]):
        #     mouse_exploration = exploration[:,:,m]
        #     percent_in_shelter.append( np.sum(mouse_exploration*shelter_roi) )



