import numpy as np
import cv2
import matplotlib.pyplot as plt
import os
import scipy
from scipy.ndimage import gaussian_filter1d, gaussian_filter
from matplotlib.colors import ListedColormap
import statsmodels.api as sm
import pandas as pd
from sklearn import linear_model
from helper_code.registration_funcs import model_arena, get_arena_details
from helper_code.processing_funcs import speed_colors
plt.rcParams.update({'font.size': 24})


'''     GENERAL FUNCTIONS       '''

def initialize_arena(self, sub_experiments, sub_conditions):
    '''     initialize arena        '''
    shape = self.analysis[sub_experiments[0]]['obstacle']['shape']
    obstacle_type = self.analysis[sub_experiments[0]]['obstacle']['type']
    obstacle = sub_conditions[0]=='obstacle' or ('left' in sub_experiments[0] and sub_conditions[0]=='no obstacle') or 'lights' in sub_experiments[0]
    print(obstacle)
    arena, _, _ = model_arena(shape, obstacle, False, obstacle_type, simulate=False, dark=self.dark_theme)
    scaling_factor = 100 / arena.shape[0]
    arena_color = cv2.cvtColor(arena, cv2.COLOR_GRAY2RGB)
    return arena, arena_color, scaling_factor

def print_metrics(RT, end_idx, number_of_mice, number_of_trials):
    # print out metrics
    print('Number of sessions: ' + str(number_of_mice))
    print('Number of trials: ' + str(number_of_trials))
    num_escapes = np.sum(~np.isnan(end_idx))
    print('percent escape: ' + str(num_escapes / number_of_trials))
    RT_for_quartiles = RT.copy()
    RT_for_quartiles[np.isnan(RT)] = np.inf
    RT_quartiles = np.percentile(RT_for_quartiles, 25), np.percentile(RT_for_quartiles, 50), np.percentile(RT_for_quartiles, 75)
    print('RT quartiles: ' + str(RT_quartiles))
    end_idx_for_quartiles = end_idx.copy()
    end_idx_for_quartiles[np.isnan(end_idx)] = np.inf
    end_quartiles = np.percentile(end_idx_for_quartiles, 25), np.percentile(end_idx_for_quartiles, 50), np.percentile(end_idx_for_quartiles, 75)
    print('to-shelter quartiles: ' + str(end_quartiles))

def initialize_variables(number_of_trials, self, sub_experiments):
    '''     initialize variables for speed traces analysis       '''
    # initialize array to fill in with each trial's data
    time_axis = np.arange(-10, 15, 1 / 30)
    speed_traces = np.zeros((25 * 30, number_of_trials)) * np.nan
    subgoal_speed_traces = np.zeros((25 * 30, number_of_trials)) * np.nan
    time = np.zeros(number_of_trials)
    end_idx = np.zeros(number_of_trials)
    RT = np.zeros(number_of_trials)
    trial_num = 0
    shape = self.analysis[sub_experiments[0]]['obstacle']['shape']
    scaling_factor = 100 / shape[0]
    return RT, end_idx, scaling_factor, speed_traces, subgoal_speed_traces, time, time_axis, trial_num


def initialize_variable_edginess(number_of_trials, self, sub_experiments):
    '''     initialize variables for edginess analysis       '''
    edginess = np.ones(number_of_trials) * np.nan
    prev_edginess = np.ones(number_of_trials) * np.nan
    time_in_center = np.ones(number_of_trials) * np.nan
    end_idx = np.ones(number_of_trials) * np.nan
    time_since_down = np.ones(number_of_trials) * np.nan
    dist_to_SH = np.ones(number_of_trials) * np.nan
    dist_to_other_SH = np.ones(number_of_trials) * np.nan
    previous_movement_and_ICs = [[],[],[],[],[]]
    data_y_for_prev_movement = [[],[]]
    trial_num = -1
    shape = self.analysis[sub_experiments[0]]['obstacle']['shape']
    scaling_factor = 100 / shape[0]
    time_to_shelter = []
    time_to_shelter_all = []
    SR_all = []
    return edginess, end_idx, time_since_down, time_to_shelter,time_to_shelter_all, prev_edginess, \
           scaling_factor, time_in_center, trial_num, previous_movement_and_ICs, data_y_for_prev_movement, dist_to_SH, dist_to_other_SH


def initialize_variables_efficiency(number_of_trials, self, sub_experiments):
    '''     initialize variables for efficiency analysis       '''
    efficiency = np.ones(number_of_trials) * np.nan
    efficiency_RT = np.ones(number_of_trials) * np.nan
    edginess = np.ones(number_of_trials) * np.nan
    num_prev_homings = np.ones(number_of_trials) * np.nan
    time = np.ones(number_of_trials) * np.nan
    trials = np.ones(number_of_trials) * np.nan
    end_idx = np.ones(number_of_trials) * np.nan
    avg_speed = np.ones(number_of_trials) * np.nan
    avg_speed_RT = np.ones(number_of_trials) * np.nan
    peak_speed = np.ones(number_of_trials) * np.nan
    RT = np.ones(number_of_trials) * np.nan
    escape_speed = np.ones(number_of_trials) * np.nan
    strategy = np.ones(number_of_trials) * np.nan
    trial_num = -1
    shape = self.analysis[sub_experiments[0]]['obstacle']['shape']
    scaling_factor = 100 / shape[0]
    return efficiency, efficiency_RT, end_idx, num_prev_homings, scaling_factor, time, trial_num, trials, edginess, avg_speed, avg_speed_RT, peak_speed, RT, escape_speed, strategy


def fill_in_trial_data(RT, condition, end_idx, experiment, mouse, scaling_factor, self, speed_traces, subgoal_speed_traces, time, trial, trial_num):
    '''     fill in trial data for speed traces analysis     '''
    # extract data
    trial_speed = [s * scaling_factor * 30 for s in self.analysis[experiment][condition]['speed'][mouse][trial]]
    trial_subgoal_speed = [s * scaling_factor * 30 for s in self.analysis[experiment][condition]['geo speed'][mouse][trial]]
    # filter and add data to the arrays
    speed_traces[:len(speed_traces), trial_num] = gaussian_filter1d(trial_speed, 2)[:len(speed_traces)]
    subgoal_speed_traces[:len(speed_traces), trial_num] = gaussian_filter1d(trial_subgoal_speed, 2)[:len(speed_traces)]
    # add additional data to the arrays
    time[trial_num] = self.analysis[experiment][condition]['start time'][mouse][trial]
    end_idx[trial_num] = self.analysis[experiment][condition]['end time'][mouse][trial]
    RT[trial_num] = self.analysis[experiment][condition]['RT'][mouse][trial]
    trial_num += 1
    return trial_num

def fill_in_trial_data_efficiency(ETD, condition, efficiency, efficiency_RT, experiment, mouse, num_prev_homings,
                                    self, time, trial, trial_num, trials, edginess, t):
    '''     initialize variables for efficiency analysis       '''
    efficiency[trial_num] = np.min((1,self.analysis[experiment][condition]['optimal path length'][mouse][trial] / \
                            self.analysis[experiment][condition]['full path length'][mouse][trial]))
    efficiency_RT[trial_num] = np.min((1,self.analysis[experiment][condition]['optimal RT path length'][mouse][trial] / \
                               self.analysis[experiment][condition]['RT path length'][mouse][trial]))
    time[trial_num] = self.analysis[experiment][condition]['start time'][mouse][trial]
    if time[trial_num] > 20: time[trial_num] = np.nan
    # trials[trial_num] = int(trial)
    trials[trial_num] = t+1
    # compute SHs
    # get the stored exploration plot - proportion of time at each location


    # SH_data = self.analysis[experiment][condition]['prev homings'][mouse][trial]
    # SR = np.array(SH_data[0])
    # x_edge = SH_data[4]
    # # only use recent escapes
    # if len(SR) >= (ETD + 1): SR = SR[-ETD:]
    # # num_prev_homings[trial_num] = int(np.min((4, np.sum(abs(SR - x_edge) < 6))))
    # num_prev_homings[trial_num] = int(np.sum(abs(SR - x_edge) < 10))
    # num_prev_homings[trial_num] = int(np.sum(abs(SR - x_edge) < 5))

    num_edge_vectors, x_edge = get_num_edge_vectors(self, experiment, condition, mouse, trial, ETD=ETD)
    num_prev_homings[trial_num] = num_edge_vectors
    print(mouse + ' -- trial ' + str(trial+1) )
    print(num_prev_homings[trial_num])

    # num_prev_homings[trial_num] = 0

    # get edginess
    edginess[trial_num] = self.analysis[experiment][condition]['edginess'][mouse][trial]

class LRPI:
    '''     linear regression you can get prediction interval from      '''
    def __init__(self, normalize=False, n_jobs=1, t_value=2.13144955):
        self.normalize = normalize
        self.n_jobs = n_jobs
        self.LR = linear_model.LinearRegression(normalize=self.normalize, n_jobs=self.n_jobs)
        self.t_value = t_value

    def fit(self, X_train, y_train):
        self.X_train = pd.DataFrame(X_train)
        self.y_train = pd.DataFrame(y_train)

        self.LR.fit(self.X_train, self.y_train)
        X_train_fit = self.LR.predict(self.X_train)
        self.MSE = np.power(self.y_train.subtract(X_train_fit), 2).sum(axis=0) / (self.X_train.shape[0] - self.X_train.shape[1] - 1)
        self.X_train.loc[:, 'const_one'] = 1
        self.XTX_inv = np.linalg.inv(np.dot(np.transpose(self.X_train.values), self.X_train.values))

    def predict(self, X_test):
        self.X_test = pd.DataFrame(X_test)
        self.pred = self.LR.predict(self.X_test)
        self.X_test.loc[:, 'const_one'] = 1
        SE = [np.dot(np.transpose(self.X_test.values[i]), np.dot(self.XTX_inv, self.X_test.values[i])) for i in range(len(self.X_test))]
        results = pd.DataFrame(self.pred, columns=['Pred'])

        results.loc[:, "lower"] = results['Pred'].subtract((self.t_value) * (np.sqrt(self.MSE.values + np.multiply(SE, self.MSE.values))), axis=0)
        results.loc[:, "upper"] = results['Pred'].add((self.t_value) * (np.sqrt(self.MSE.values + np.multiply(SE, self.MSE.values))), axis=0)

        return results


def extract_experiments(experiment, condition):
    '''     extract experiments from nested list     '''
    if type(experiment) == list:
        sub_experiments = experiment
        sub_conditions = condition
    else:
        sub_experiments = [experiment]
        sub_conditions = [condition]

    return sub_experiments, sub_conditions

def get_number_of_trials(sub_experiments, sub_conditions, analysis):
    '''     find out how many trials in each condition, for data initialization     '''
    # initialize the number of trials
    number_of_trials = 0
    max_number_of_trials = 99999
    # loop over each experiment/condition
    for e, (experiment, condition) in enumerate(zip(sub_experiments, sub_conditions)):
        # loop over each mouse
        for i, mouse in enumerate(analysis[experiment][condition]['start time']):
            # and count up the number of trials
            number_of_trials += np.min((max_number_of_trials, len(analysis[experiment][condition]['start time'][mouse])))
        if not number_of_trials: continue
    return number_of_trials

def get_number_of_mice(sub_experiments, sub_conditions, analysis):
    '''     find out how many mice in each condition, for data initialization     '''
    # initialize the number of mice
    number_of_mice = 0
    max_number_of_mice = 99999
    list_of_mice = []
    # loop over each experiment/condition
    for e, (experiment, condition) in enumerate(zip(sub_experiments, sub_conditions)):
        # and count up the number of mice
        key = list(analysis[experiment][condition].keys())[-1]
        number_of_mice += len(analysis[experiment][condition][key])
        list_of_mice.append(list(analysis[experiment][condition][key].keys()))
    print('number of mice: ' + str(len(np.unique(list(flatten(list_of_mice))))))

    return number_of_mice

def speed_colormap(scaling_factor, n_bins=256, v_min = 0, v_max = 65):
    '''     create a custom, gray-blue-green colormap       '''
    # create an empty colormap
    new_colors = np.ones((n_bins, 4))
    # loop over each color entry
    for c in range(new_colors.shape[0]):
        # get speed
        speed = (c * (v_max-v_min) / new_colors.shape[0] + v_min) / 30 / scaling_factor
        # get speed color
        _, speed_color = speed_colors(speed, plotting = True)
        # insert into color array
        new_colors[c, :3] = speed_color
    # insert into colormap
    colormap = ListedColormap(new_colors)
    return colormap

def flatten(iterable):
    '''       flatten a nested list       '''
    it = iter(iterable)
    for e in it:
        if isinstance(e, (list, tuple)):
            for f in flatten(e):
                yield f
        else:
            yield e

def fit_kde(x, **kwargs):
    """ Fit a KDE using StatsModels.
        kwargs is useful to pass stuff to the fit, e.g. the binwidth (bw)"""
    x = np.array(x).astype(np.float)
    kde = sm.nonparametric.KDEUnivariate(x)
    kde.fit(**kwargs)  # Estimate the densities
    return kde

def plot_shaded_withline(ax, x, y, z=None, label=None, violin=True, vertical = True, **kwargs):
    """[Plots a curve with shaded area and the line of the curve clearly visible]

    Arguments:
        ax {[type]} -- [matplotlib axis]
        x {[np.array, list]} -- [x data]
        y {[np.array, list]} -- [y data]

    Keyword Arguments:
        z {[type]} -- [description] (default: {None})
        label {[type]} -- [description] (default: {None})
        alpha {float} -- [description] (default: {.15})
    """
    # if z is not None:
    fill_alpha = .8
    line_alpha = .8

    redness = kwargs['color'][0]
    if type(redness) != str:
        if redness > .5: fill_alpha += .1

    if vertical: ax.fill_betweenx(y, z+x, z, alpha=fill_alpha, **kwargs)
    else: ax.fill_between(x, y+z, z, alpha=fill_alpha, **kwargs)
    ax.plot(z + x, y, alpha=line_alpha, label=label, **kwargs)

    if violin:
        ax.fill_betweenx(y, z - x, z, alpha=fill_alpha, **kwargs)
        ax.plot(z - x, y, alpha=line_alpha, label=label, **kwargs)
    # else:
    #     ax.fill_between(x, y, alpha=alpha, **kwargs)

def plot_kde(ax, kde, data, z, vertical=False, normto=None, label=None, violin=True, clip=False, cutoff = False, cutoff_colors = None, **kwargs):
    """[Plots a KDE distribution. Plots first the shaded area and then the outline.
       KDE can be oriented vertically, inverted, normalised...]

    Arguments:
        ax {[plt.axis]} -- [ax onto which to plot]
        kde {[type]} -- [KDE fitted with statsmodels]
        z {[type]} -- [value used to shift the plotted curve. e.g for a horizzontal KDE if z=0 the plot will lay on the X axis]

    Keyword Arguments:
        invert {bool} -- [mirror the KDE plot relative to the X or Y axis, depending on ortentation] (default: {False})
        vertical {bool} -- [plot KDE vertically] (default: {False})
        normto {[float]} -- [normalise the KDE so that the peak of the distribution is at a certain value] (default: {None})
        label {[string]} -- [label for the legend] (default: {None})

    Returns:
        ax, kde
    """
    if vertical:
        x, y = kde.density, kde.support
    else:
        x, y = kde.support, kde.density

    if clip:
        if np.max(data) > .95: #1: #0.8
            x = x[(y > 0)]  # * (y < 1)]
            y = y[(y > 0)]  # * (y < 1)]
        else:
            x = x[(y > 0) * (y < 1)]
            y = y[(y > 0) * (y < 1)]

    if normto is not None:
        if not vertical:
            y = y / np.max(y) * normto
        else:
            x = x / np.max(x) * normto

    if cutoff:
        plot_shaded_withline(ax, x[y < cutoff], y[y < cutoff], z=z, violin=violin, vertical=vertical, color = cutoff_colors[0])
        plot_shaded_withline(ax, x[y > cutoff], y[y > cutoff], z=z, violin=violin, vertical=vertical, color = cutoff_colors[1])
    else:
        plot_shaded_withline(ax, x, y, z=z, violin=violin, vertical=vertical, **kwargs)

    return ax, kde


def scatter_the_axis(c, data_for_plot):
    '''     scatter the axis so points aren't overlapping       '''
    scatter_axis = np.ones_like(data_for_plot) * 3 * c - .2
    for i in range(len(data_for_plot)):
        difference = abs(data_for_plot[i] - data_for_plot)
        difference[i] = np.inf
        # if np.min(difference) == 0:
        #     scatter_axis[i] = np.random.normal(3 * c - .15, 0.25)
        if np.min(difference) < .01:
            scatter_axis[i] = np.random.normal(3 * c - .2, 0.1) #.15
    return scatter_axis

def scatter_the_axis_efficiency(plot_data, plot_data_x):
    '''     scatter the axis for efficiency so points aren't overlapping       '''
    scatter_axis = plot_data_x.copy()
    for j in range(len(scatter_axis)):
        difference = abs(plot_data[j] - plot_data)
        difference[j] = np.inf
        if np.min(difference) < (.01 * np.max(plot_data)): # .001
            scatter_axis[j] = scatter_axis[j] + np.random.normal(0, 0.02)
    return scatter_axis

def do_linear_regression(plot_data, plot_data_x):
    ' do linear regression'
    order = np.argsort(plot_data_x)
    plot_data_x = plot_data_x[order]
    plot_data = plot_data[order]
    LR = LRPI(t_value=1)
    LR.fit(plot_data_x, plot_data)
    prediction = LR.predict(plot_data_x)
    return plot_data_x, prediction


'''     ANALYSIS-TYPE-SPECIFIC FUNCTIONS        '''

def get_prev_edginess(ETD, condition, experiment, mouse, prev_edginess, dist_to_SH, dist_to_other_SH, scaling_factor, self, traj_loc, trial, trial_num, edginess, delta_ICs, delta_x_end):
    SH_data = self.analysis[experiment][condition]['prev homings'][mouse][trial]
    SH_start_data = self.analysis[experiment][condition]['prev movements'][mouse][trial]
    SH_x_movements = np.array(SH_start_data[4])
    SH_angle_start = np.array(SH_start_data[2])
    # escape_start_angle = self.analysis[experiment][condition]['movement'][mouse][trial][1]

    SH_x = np.array(SH_data[0])
    x_edge = self.analysis[experiment][condition]['x edge'][mouse][trial]
    SR_time = np.array(SH_data[3])
    time_to_shelter = np.array(SH_data[2])
    RT = self.analysis[experiment][condition]['RT'][mouse][trial]

    # get previous homing initial conditions
    SH_start_x =  np.array(SH_start_data[0])[abs(SH_x_movements - x_edge) < 10]
    SH_start_y = np.array(SH_start_data[1])[abs(SH_x_movements - x_edge) < 10]
    SH_all_start_x =  np.array(SH_start_data[0])
    SH_all_start_y = np.array(SH_start_data[1])

    # RT = 0
    # get position for the trial
    x_pos = self.analysis[experiment][condition]['path'][mouse][trial][0][int(RT * 30):] * scaling_factor
    y_pos = self.analysis[experiment][condition]['path'][mouse][trial][1][int(RT * 30):] * scaling_factor

    # get the sidedness of the escape
    mouse_at_center = np.argmin(abs(y_pos - traj_loc))
    # get line to the closest edge
    y_edge = 50
    # only use spontaneous
    # stim_evoked = np.array(SH_data[5])
    # SH_x = SH_x[~stim_evoked]
    # SH_x = SH_x[time_to_shelter < 10]

    # only use recent escapes
    if len(SH_x) >= (ETD + 1): SH_x = SH_x[-ETD:]


    # get line to the closest edge, exclude escapes to other edge
    MOE = 10.2  # 20 #10.2
    if x_edge > 50:
        SH_x = SH_x[SH_x > 25 + MOE]  # 35
    else:
        SH_x = SH_x[SH_x < 75 - MOE]  # 65
    # take the mean of the prev homings
    if SH_x.size:
        x_repetition = np.mean(SH_x)
        # NOW DO THE EDGINESS ANALYSIS, WITH REPETITION AS THE REAL DATA
        # do line from starting position to shelter
        y_pos_end = 86.5
        x_pos_end = 50
        x_pos_start = x_pos[0]
        y_pos_start = y_pos[0]
        slope = (y_pos_end - y_pos_start) / (x_pos_end - x_pos_start)
        intercept = y_pos_start - x_pos_start * slope
        if SH_x.size: distance_to_line = abs(traj_loc - slope * x_repetition - intercept) / np.sqrt((-slope) ** 2 + (1) ** 2)
        homing_vector_at_center = (traj_loc - intercept) / slope
        # do line from starting position to edge position
        slope = (y_edge - y_pos_start) / (x_edge - x_pos_start)
        intercept = y_pos_start - x_pos_start * slope
        distance_to_edge = abs(traj_loc - slope * x_repetition - intercept) / np.sqrt((-slope) ** 2 + (1) ** 2)
        # compute the max possible deviation
        edge_vector_at_center = (traj_loc - intercept) / slope
        line_to_edge_offset = abs(homing_vector_at_center - edge_vector_at_center)  # + 5
        # get index at center point (wall location)
        prev_edginess[trial_num] = min(1, (distance_to_line - distance_to_edge + line_to_edge_offset) / (2 * line_to_edge_offset))
        # print(prev_edginess[trial_num])
        # num_edge_vectors, x_edge = get_num_edge_vectors(self, experiment, condition, mouse, trial, ETD = ETD)
        # prev_edginess[trial_num] = num_edge_vectors

        try:
            if SH_start_x.size:
                dist_to_SH[trial_num] = np.min(np.linalg.norm( [x_pos_start - SH_start_x, y_pos_start - SH_start_y], axis = 0))

                random_idx = np.random.choice(SH_all_start_x.size, SH_start_x.size)
                dist_to_other_SH[trial_num] = np.min(np.linalg.norm([x_pos_start - SH_all_start_x[random_idx], y_pos_start - SH_all_start_y[random_idx]], axis=0))

            else:
                dist_to_SH[trial_num] = np.nan

            # difference in initial conditions
            # delta_angle = abs(escape_start_angle - SH_angle_start)
            # delta_angle[delta_angle > 180] = 360 - delta_angle[delta_angle > 180]

            delta_ICs.append( list( np.linalg.norm( [x_pos_start - SH_all_start_x, y_pos_start - SH_all_start_y], axis = 0) )) #, delta_angle

            # difference in final x position
            center_y_idx = np.argmin(abs(y_pos - 45))
            x_at_center_y = x_pos[center_y_idx]
            delta_x_end.append( list(abs(SH_x_movements - x_at_center_y) ))
        except:
            print('ICs not done')

    else: prev_edginess[trial_num] = 0



    return time_to_shelter, SH_x



def get_all_prev_edginess(ETD, condition, experiment, mouse, prev_edginess, dist_to_SH, dist_to_other_SH, scaling_factor, self, traj_loc, trial, trial_num, edginess, delta_ICs, delta_x_end):
    SH_data = self.analysis[experiment][condition]['prev homings'][mouse][trial]
    SH_x = np.array(SH_data[0])
    x_edge = self.analysis[experiment][condition]['x edge'][mouse][trial]
    SR_time = np.array(SH_data[3])
    time_to_shelter = np.array(SH_data[2])
    RT = self.analysis[experiment][condition]['RT'][mouse][trial]


    # RT = 0
    # get position for the trial
    x_pos = self.analysis[experiment][condition]['path'][mouse][trial][0][int(RT * 30):] * scaling_factor
    y_pos = self.analysis[experiment][condition]['path'][mouse][trial][1][int(RT * 30):] * scaling_factor
    # get the sidedness of the escape
    mouse_at_center = np.argmin(abs(y_pos - traj_loc))
    # get line to the closest edge
    y_edge = 50


    # only use recent escapes
    SH_x = SH_x[-ETD:]

    # # get line to the closest edge, exclude escapes to other edge
    MOE = 10.2  # 20 #10.2
    if x_edge > 50:
        elig_idx = SH_x > 25 + MOE
        # SH_x = SH_x[elig_idx]  # 35
    else:
        elig_idx = SH_x < 75 - MOE
        # SH_x = SH_x[elig_idx]  # 65
    # take the edginess of all the prev homings

    prev_edginess_all = []
    for move in range(SH_x.size):

        x_repetition = SH_x[move]
        # NOW DO THE EDGINESS ANALYSIS, WITH REPETITION AS THE REAL DATA
        # do line from starting position to shelter
        y_pos_end = 86.5
        x_pos_end = 50
        x_pos_start = x_pos[0]
        y_pos_start = y_pos[0]
        slope = (y_pos_end - y_pos_start) / (x_pos_end - x_pos_start)
        intercept = y_pos_start - x_pos_start * slope
        if SH_x.size: distance_to_line = abs(traj_loc - slope * x_repetition - intercept) / np.sqrt((-slope) ** 2 + (1) ** 2)
        homing_vector_at_center = (traj_loc - intercept) / slope
        # do line from starting position to edge position
        slope = (y_edge - y_pos_start) / (x_edge - x_pos_start)
        intercept = y_pos_start - x_pos_start * slope
        distance_to_edge = abs(traj_loc - slope * x_repetition - intercept) / np.sqrt((-slope) ** 2 + (1) ** 2)
        # compute the max possible deviation
        edge_vector_at_center = (traj_loc - intercept) / slope
        line_to_edge_offset = abs(homing_vector_at_center - edge_vector_at_center)  # + 5
        # get index at center point (wall location)
        prev_edginess_all.append( (distance_to_line - distance_to_edge + line_to_edge_offset) / (2 * line_to_edge_offset) )

    prev_edginess_all = np.array(prev_edginess_all)

    return time_to_shelter, SH_x, prev_edginess_all, elig_idx


def get_num_edge_vectors(self, experiment, condition, mouse, trial, ETD = False):
    SH_data = self.analysis[experiment][condition]['prev homings'][mouse][trial]
    SR = np.array(SH_data[0])
    if ETD:
        if len(SR) >= (ETD + 1): SR = SR[-ETD:]

    x_edge = self.analysis[experiment][condition]['x edge'][mouse][trial]

    homings_dist_from_edge = abs(SR - x_edge)
    num_edge_vectors = np.sum(homings_dist_from_edge < 10) #5 #10 #4

    return num_edge_vectors, x_edge


def initialize_figures(self, traversals = 1):

    if 'Square' in self.experiments[0]:
        fig, ax = plt.subplots(figsize=(6 * traversals + 3+4, 6+3))
        ax.set_xlim([-1, traversals * 3 * len(self.experiments) - 1])
        ax.set_ylim([-.7, 1.7]) #1.15
    else:
        if 'food' in self.experiments[0] or 'food' in self.experiments[0][0]:
            fig, ax = plt.subplots(figsize=(9, 9))
        elif 'down' in self.experiments[0]:
            fig, ax = plt.subplots(figsize=(6, 6))
        else:
            fig, ax = plt.subplots(figsize=(9, 6))

        # fig, ax = plt.subplots(figsize=(6 * traversals + 3, 6 + 3))

        ax.set_xlim([-1, traversals * 3 * len(self.experiments) - 1])
        ax.set_ylim([-.1, 1.15])
        # ax.set_ylim([-.1, 1.3])
    # plot dotted lines
    ax.plot([-1, traversals * 3 * len(self.experiments) - 1], [0, 0], linestyle='--', color=[.5, .5, .5, .5])
    ax.plot([-1, traversals * 3 * len(self.experiments) - 1], [1, 1], linestyle='--', color=[.5, .5, .5, .5])
    plt.axis('off')
    ax.margins(0, 0)
    ax.xaxis.set_major_locator(plt.NullLocator())
    ax.yaxis.set_major_locator(plt.NullLocator())

    fig2, ax2 = plt.subplots(figsize=(9, 9))
    ax2.set_ylim([-.05, 1.19])
    fig3, ax3 = plt.subplots(figsize=(9, 9))
    ax3.set_ylim([-.05, 1.19])

    fig4 = plt.figure(figsize=(12,3))
    fig5 = None #plt.figure(figsize=(6,6))

    return fig, fig2, fig3, fig4, fig5, ax, ax2, ax3

def initialize_figures_traversals(self, types = 1):
    # fig, ax = plt.subplots(figsize=(9, 6))
    fig, ax = plt.subplots(figsize=(6, 6))
    # ax.set_xlim([-1, types * 4 * len(self.experiments) - 1])
    ax.set_ylim([-1, 25])
    # plot dotted lines
    ax.plot([-1, 4 * len(self.experiments) - 1], [0, 0], linestyle='--', color=[.5, .5, .5, .5])
    # ax.plot([-1, 4 * len(self.experiments) - 1], [5, 5], linestyle='--', color=[.5, .5, .5, .5])
    ax.plot([-1, 4 * len(self.experiments) - 1], [10, 10], linestyle='--', color=[.5, .5, .5, .5])
    # ax.plot([-1, 4 * len(self.experiments) - 1], [15, 15], linestyle='--', color=[.5, .5, .5, .5])
    ax.plot([-1, 4 * len(self.experiments) - 1], [20, 20], linestyle='--', color=[.5, .5, .5, .5])
    ax.plot([-1, 4 * len(self.experiments) - 1], [30, 30], linestyle='--', color=[.5, .5, .5, .5])

    plt.axis('off')
    ax.margins(0, 0)
    ax.xaxis.set_major_locator(plt.NullLocator())
    ax.yaxis.set_major_locator(plt.NullLocator())

    fig2, ax2 = plt.subplots(figsize=(6, 6))
    # ax2.set_xlim([-1, types * 4 * len(self.experiments) - 1])
    ax2.set_ylim([-.1, 5])
    # plot dotted lines
    ax2.plot([0, types *4 * len(self.experiments) - 6], [0, 0], linestyle='--', color=[.5, .5, .5, .5])
    ax2.plot([0, types *4 * len(self.experiments) - 6], [2, 2], linestyle='--', color=[.5, .5, .5, .5])
    ax2.plot([0, types *4 * len(self.experiments) - 6], [4, 4], linestyle='--', color=[.5, .5, .5, .5])

    plt.axis('off')
    ax2.margins(0, 0)
    ax2.xaxis.set_major_locator(plt.NullLocator())
    ax2.yaxis.set_major_locator(plt.NullLocator())

    return fig, fig2, ax, ax2

def initialize_figures_prediction(self):
    fig1, ax1 = plt.subplots(figsize=(12, 6))
    ax1.set_xlim([-.85, 1.05])
    ax1.set_ylim([-.05, 12.05])
    ax1.spines['right'].set_visible(False)
    ax1.spines['top'].set_visible(False)
    ax1.plot([0,0],[-1,10], color = [.5,.5,.5], linestyle = '--')

    fig2, ax2 = plt.subplots(figsize=(12, 6))
    ax2.set_xlim([-.85, 1.05])
    ax2.set_ylim([-.05, 12.05])
    ax2.spines['right'].set_visible(False)
    ax2.spines['top'].set_visible(False)
    ax2.plot([0,0],[-1,10], color = [.5,.5,.5], linestyle = '--')

    return fig1, ax1, fig2, ax2

def initialize_figures_efficiency(self):
    fig2, ax2 = plt.subplots(figsize=(9, 9))
    ax2.set_ylim([.43, 1.03])
    ax2.set_xlim([-.5, len(self.experiments)-.5])
    ax2.plot([-10,10],[1,1], color = [.5,.5,.5], linestyle = '--')
    ax2.plot([-10, 10], [.5, .5], color=[.5, .5, .5], linestyle='--')
    ax2.xaxis.set_major_locator(plt.NullLocator())
    ax2.yaxis.set_major_locator(plt.NullLocator())
    plt.axis('off')
    fig3, ax3 = plt.subplots(figsize=(9, 9))
    # ax3.set_ylim([-.1, 6.5])
    ax3.set_xlim([-.5, len(self.experiments)-.5])
    ax3.plot([-10,10],[6,6], color = [.5,.5,.5], linestyle = '--')
    ax3.plot([-10, 10], [3, 3], color=[.5, .5, .5], linestyle='--')
    ax3.plot([-10, 10], [0, 0], color=[.5, .5, .5], linestyle='--')
    ax3.xaxis.set_major_locator(plt.NullLocator())
    ax3.yaxis.set_major_locator(plt.NullLocator())
    plt.axis('off')

    return ax2, fig2, ax3, fig3


def show_speed_traces(colormap, condition, end_idx, experiment, number_of_trials, speed, speed_traces, subgoal_speed_traces, time_axis, colorbar = False):
    '''     plot speed traces       '''
    # order the data chronologically or by RT
    order = np.argsort(end_idx)
    order = order[::-1]
    # format speed data
    if speed == 'geodesic':
        # speed toward shelter
        z = -subgoal_speed_traces[:, order].T
    else:
        # speed in general
        z = speed_traces[:, order].T
    # separate out the escapes and non-escapes (here, under 6 seconds)
    gap_size = 2
    num_non_escapes = np.sum(np.isnan(end_idx)) + np.sum(end_idx > 9 * 30)
    z_with_gap = np.ones((z.shape[0] + gap_size, z.shape[1])) * np.nan
    z_with_gap[:num_non_escapes, :] = z[:num_non_escapes, :]
    z_with_gap[num_non_escapes + gap_size:, :] = z[num_non_escapes:, :]
    # generate 2 2d grids for the x & y bounds
    if colorbar: fig, ax = plt.subplots(figsize=(12, 5))
    else:
        fig, ax = plt.subplots(figsize=(9, 5))
        ax.margins(0, 0)
        ax.xaxis.set_major_locator(plt.NullLocator())
        ax.yaxis.set_major_locator(plt.NullLocator())

    x, y = np.meshgrid(time_axis, np.arange(0, number_of_trials + gap_size))
    # ax.set_xlabel('time since stimulus onset (s)')
    # ax.set_ylabel('escape trial')
    plt.axis('off')
    # plot speed data
    c = ax.pcolormesh(x, y, z_with_gap, cmap=colormap, vmin=0, vmax=65)
    # show the colorbar
    if colorbar: fig.colorbar(c, ax=ax)
    # plot timing ticks for each trial
    ax.plot([0, 0], [0, number_of_trials + gap_size - 1], color='white', linewidth=2, linestyle='--')
    ax.plot([6, 6], [0, number_of_trials + gap_size - 1], color='gray', linewidth=2, linestyle='--')
    return fig

def display_traversal(arena, arena_color, fast_color, scaling_factor, slow_color, traversal, trial, type):
    '''     plot traversals     '''
    # get the x and y coordinates of the path
    x_idx = traversal[trial][0].astype(int);
    y_idx = traversal[trial][1].astype(int)
    # get the duration and distance traveled, to compute net speed
    time = len(x_idx) / 30
    dist = np.sum(np.sqrt(np.diff(x_idx) ** 2 + np.diff(y_idx) ** 2)) * scaling_factor
    speed = np.min((40, dist / time))
    # choose a color accordingly
    speed_color = ((40 - speed) * slow_color + speed * fast_color) / 40
    # initialize a mask array
    mask_arena = np.zeros_like(arena)
    # loop over each point, drawing line segments on the mask array
    for j in range(len(x_idx) - 1):
        x1, y1 = x_idx[j], y_idx[j]
        x2, y2 = x_idx[j + 1], y_idx[j + 1]
        cv2.line(mask_arena, (x1, y1), (x2, y2), 1, thickness=1 + 1 * (speed > 15) + 2 * (speed > 25) + 1 * (speed > 35))
    # draw on the actual array
    arena_color[mask_arena.astype(bool)] = arena_color[mask_arena.astype(bool)] * speed_color
    # display the traversals
    cv2.imshow('traversals - ' + type, arena_color)
    cv2.waitKey(1)

    return arena_color


def show_escape_paths(HV_cutoff, arena, arena_color, arena_reference, c, condition, edge_vector_color, escape_duration, experiment, fps, homing_vector_color,
        min_distance_to_shelter, mouse, non_escape_color, scaling_factor, self, shelter_location, strategies, determine_strategy = False):
    '''     show escape paths       '''
    # find all the paths across the arena
    paths = self.analysis[experiment][condition]['path'][mouse]
    # loop over all paths
    t = 0
    x_edges_used = []
    for trial in range(len(paths)):
        if 'food' in experiment:
            # select the number of successful trials
            if t > 8: continue
            # select the timing
            if self.analysis[experiment][condition]['start time'][mouse][trial] < 20 and condition == 'no obstacle': continue
        else:
            # select the number of trials
            # if trial > 2 and not 'Square' in experiment: continue
            # if trial != 1 and trial !=2: continue
            if t > 2: continue
            # if t: continue
            # if not trial: continue

            # if trial != 2 and condition == 'no obstacle': continue

            # if trial !=3 and trial !=4: continue
            # if not trial: continue
            # if condition == 'obstacle' and trial != 2: continue

            # if 'dark' in experiment and condition == 'obstacle':
            #     if t>999: continue
            # pass

        # t += 1
        # get the x and y coordinates of the path
        x_idx = paths[trial][0][:fps * escape_duration].astype(int)
        y_idx = paths[trial][1][:fps * escape_duration].astype(int)
        #
        if 'side' in experiment:
            if (x_idx[0] * scaling_factor) > 50: x_idx = (100 / scaling_factor - x_idx).astype(int)
            # if abs(x_idx[0] * scaling_factor - 50) < 25: continue
            # if abs(y_idx[0] * scaling_factor - 50) > 25: continue
        elif 'Square' in experiment:
            if x_idx[0] * scaling_factor > 75: continue
            # not the one HV, for the edge-vector-only plot
            # if y_idx[0] * scaling_factor < 9: continue #11
        else:
            # needs to start at top
            if y_idx[0] * scaling_factor > 25: continue
            if abs(x_idx[0] * scaling_factor - 50) > 25: continue
            # if abs(self.analysis[experiment][condition]['start angle'][mouse][trial]) < 45:
            #     print(abs(self.analysis[experiment][condition]['start angle'][mouse][trial])); continue
        # t+=1
        # if abs(x_idx[0] * scaling_factor-50) < 25: continue
        # if abs(y_idx[0] * scaling_factor-50) > 35: continue
        # categorize the escape
        time_to_shelter = self.analysis[experiment][condition]['end time'][mouse][trial]
        # determine prev edgy homings or escapes

        if 'dark' in experiment: # and condition == 'obstacle':
            num_prev_edge_vectors, x_edge = get_num_edge_vectors(self, experiment, condition, mouse, trial)
            if num_prev_edge_vectors >= 2 and False: # and condition == 'obstacle': # and not 'U shaped' in experiment:
                print('prev edgy homing')
                continue

            # if x_edge in x_edges_used: print('prev edgy escape'); continue

        print('-----------' + mouse + '--------------')

        # non-escapes
        if np.isnan(time_to_shelter) or time_to_shelter > (escape_duration * fps):
            path_color = non_escape_color
            strategies[2] = strategies[2] + 1
            print('       NONESCAPE        ')
            continue
        else:
            # determine if edge vector or HV
            # for u-shaped obstacle
            if 'U shaped' in experiment:
                # if descends into the cup, then HV, otherwise EV
                if np.sum( (abs(x_idx* scaling_factor - 50) < 25) * (abs(y_idx* scaling_factor - 40) < 10) ):
                    self.analysis[experiment][condition]['edginess'][mouse][trial] = 0
                else: self.analysis[experiment][condition]['edginess'][mouse][trial] = 1

            # print(abs(self.analysis[experiment][condition]['edginess'][mouse][trial]))
            if 'Square' in experiment:
                if self.analysis[experiment][condition]['edginess'][mouse][trial] <= -.4:
                    print('HV')
                    continue

            if self.analysis[experiment][condition]['edginess'][mouse][trial] <= HV_cutoff:
                path_color = homing_vector_color
                strategies[0] = strategies[0] + 1
                print('       HV        ')
            else:
                path_color = edge_vector_color
                strategies[1] = strategies[1] + 1
                print('       EDGY        ')
                # edgy trial has occurred
                if determine_strategy:
                    if 'dark' in experiment: # and condition == 'obstacle': # and not 'U shaped' in experiment:
                        print('EDGY TRIAL ' + str(trial))
                        x_edges_used.append(x_edge)

        # get reaction time for starting the traces
        if 'Square' in experiment:
            RT = int(self.analysis[experiment][condition]['RT'][mouse][trial] * 30 / 3)
            # RT = 0
        elif 'food' in experiment:
            RT = int(self.analysis[experiment][condition]['RT'][mouse][trial] * 30 / 3)
        else:
            RT = 0

        # initialize a mask array
        mask_arena_for_blur = np.zeros_like(arena)
        mask_arena = np.zeros_like(arena)
        # loop over each point, drawing line segments on the mask array
        for j in range(RT, len(x_idx) - 1):
            x1, y1 = x_idx[j], y_idx[j]
            x2, y2 = x_idx[j + 1], y_idx[j + 1]
            cv2.line(mask_arena, (x1, y1), (x2, y2), 1, thickness=6, lineType=16) #5
            cv2.line(mask_arena_for_blur, (x1, y1), (x2, y2), 2, thickness=1, lineType=16) #1
            # end if close to shelter
            distance_to_shelter = np.sqrt((x1 - shelter_location[0]) ** 2 + (y1 - shelter_location[1]) ** 2)
            if distance_to_shelter < min_distance_to_shelter: break
            elif distance_to_shelter < 2*min_distance_to_shelter:
                next_distance_to_shelter = np.sqrt((x2 - shelter_location[0]) ** 2 + (y2 - shelter_location[1]) ** 2)
                if next_distance_to_shelter > distance_to_shelter: break
        # blur line
        mask_arena_blur = np.ones(arena_color.shape)
        for i in range(3):
            if path_color[i] < 1:
                mask_arena_blur[:, :, i] = (.5 * - gaussian_filter(mask_arena_for_blur.astype(float), sigma=.5)) + 1
            elif i == 1:
                mask_arena_blur[:, :, i] = (.2 * - gaussian_filter(mask_arena_for_blur.astype(float), sigma=.2)) + 1
        arena_color[mask_arena.astype(bool)] = arena_color[mask_arena.astype(bool)] * mask_arena_blur[mask_arena.astype(bool)] * path_color
        # shelter like new
        arena_color[arena_reference < 245] = arena_reference[arena_reference < 245]
        # display the traversals
        cv2.imshow('escapes ' + str(c), arena_color)
        cv2.waitKey(1)

        t += 1
    print(t)

def plot_strategies(strategies, homing_vector_color, non_escape_color, edge_vector_color):
    # format plot bar of strategies
    fig, ax = plt.subplots(figsize=(3, 6))
    plt.axis('off')
    ax.margins(0, 0)
    ax.xaxis.set_major_locator(plt.NullLocator())
    ax.yaxis.set_major_locator(plt.NullLocator())
    # format data
    normed_strategies = strategies / np.sum(strategies)
    # plot strategies
    ax.bar(1, normed_strategies[0], .5, color=homing_vector_color[::-1]**5)
    ax.bar(1, normed_strategies[1], .5, bottom=normed_strategies[0], color=edge_vector_color[::-1]**5)
    ax.bar(1, normed_strategies[2], .5, bottom=normed_strategies[1] + normed_strategies[0], color=non_escape_color[::-1]**5)
    # print data
    print(np.sum(strategies))
    print(strategies)
    print(normed_strategies)
    return fig
