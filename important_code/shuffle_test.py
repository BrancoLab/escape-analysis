import numpy as np
import random
from scipy.stats import percentileofscore, ttest_ind, mannwhitneyu, pearsonr
def flatten(iterable):
    '''       flatten a nested list       '''
    it = iter(iterable)
    for e in it:
        if isinstance(e, (list, tuple)):
            for f in flatten(e):
                yield f
        else:
            yield e



def permutation_test(group_A, group_B, iterations = 1000, two_tailed = True):
    '''    POOL -> SHUFFLE MOUSE IDENTITY    '''
    # pool trials for the test statistic
    group_A_pooled = np.array(list(flatten(group_A)))
    group_B_pooled = np.array(list(flatten(group_B)))
    # create identity array
    group_A_mouse_ID = np.zeros(len(group_A))
    group_B_mouse_ID = np.ones(len(group_B))
    # create 2-D data / ID array
    data = np.array(group_A + group_B)
    # create a copy to be shuffled
    shuffled_labels = np.concatenate((group_A_mouse_ID, group_B_mouse_ID))
    # initialize test statistic
    null_distribution = np.ones(iterations) * np.nan
    # iterate over shuffles
    for i in range(iterations):
        # shuffle the labels
        random.shuffle(shuffled_labels)
        # get back the data
        group_A_mean = np.mean(list(flatten(data[shuffled_labels == 0])))
        group_B_mean = np.mean(list(flatten(data[shuffled_labels == 1])))
        # get the test statistic for the null distribution
        if two_tailed:
            null_distribution[i] = abs(group_A_mean - group_B_mean)
        else:
            null_distribution[i] = group_B_mean - group_A_mean
    # get the test statistic from the actual data
    if two_tailed:
        test_statistic = abs(np.mean(group_A_pooled) - np.mean(group_B_pooled))
    else:
        test_statistic = np.mean(group_B_pooled) - np.mean(group_A_pooled)
    # get the p value
    p = 1 - percentileofscore(null_distribution, test_statistic, kind='mean') / 100
    if p > 0.0001: p = np.round(p, 4)

    print('\nPooled stats, shuffled by mouse: p = ' + str(p))
    if p < 0.05:
        print('SIGNIFICANT.')
    else:
        print('not significant')
#
# group_A = [[28,25,19,1,4,24], [27,22,14], [4,21,5,0,19,17,1], [10,2,5,27,31,2,5,0,77,30,4,9,0], [1,4,0,3,0,3,97,3,5,0,10,0], [0,22,23,0,28,12,0,36,19,5]]
# group_B = [[27,46,26,9,2,28,24,51,23,175,28,11,39,12],[70, 49, 0,11,12,9,0,2,29,9,1,64,9,66],[4,22,121,8,88,106,158,152,7,6,],[14,0,7,112,3,9,1,43,31,175,34,25,18],[21,9,0,6],\
#            [82,0,154,7,115,88,99,82,82,20,23,169],[32,9,23,4,42,25],[32,8,7,23,4,29,15],[60,27,88,33,10,35,13,49,18,14,28,18,37,],[24,0,26,2,14,0,128,169,6,19,18],[4]]
#
# permutation_test(group_A, group_B, iterations = 10000, two_tailed = True)


def permutation_correlation(data_x, data_y, iterations = 1000, two_tailed = False, pool_all = True):
    # get number of mice
    num_mice = len(data_x)
    # initialize test statistics
    null_distribution = np.ones(iterations) * np.nan
    data_distribution = np.ones(iterations) * np.nan
    # iterate over shuffles
    for i in range(iterations):
        if pool_all:
            # pool trials
            data_x_pooled = np.array(list(flatten(data_x)))
            data_y_pooled = np.array(list(flatten(data_y)))
        else:
            # initialize pooled list
            data_x_pooled = []
            data_y_pooled = []
            # pick one datum at random from each mouse
            for m in range(num_mice):
                # if the mouse did any trials
                trials = len(data_x[m])
                if trials:
                    # pick a trial at random
                    t = np.random.randint(0, trials)
                    # add that datum to the pooled list
                    data_x_pooled.append(data_x[m][t])
                    data_y_pooled.append(data_y[m][t])
        # get the corr coeff of the actual data
        r, _ = pearsonr(data_x_pooled, data_y_pooled)
        data_distribution[i] = r
        # shuffle the data
        random.shuffle(data_x_pooled)
        # get the corr coeff of the shuffled data
        r, _ = pearsonr(data_x_pooled, data_y_pooled)
        null_distribution[i] = r
    # get the test statistic
    if two_tailed:
        test_statistic = np.mean(abs(data_distribution))
    else:
        test_statistic = np.mean(data_distribution)
    # get the p value
    if two_tailed:
        p = np.round(1 - percentileofscore(abs(null_distribution), test_statistic) / 100, 5)
    else:
        p = np.round(1 - percentileofscore(null_distribution, test_statistic) / 100, 5)
    print('\ncorrelation 1 trial per mouse: r = ' + str(np.round(test_statistic,2)) + '  p = ' + str(p))
    if p < 0.05:
        print('SIGNIFICANT.')
    else:
        print('not significant')





# group_A = [[],[28,25,19,1,4,24], [],[],[27,22,14], [],[],[4,21,5,0,19,17,1], [10,2,5,27,31,2,5,0,77,30,4,9,0], [1,4,0,3,0,3,97,3,5,0,10,0], [0,22,23,0,28,12,0,36,19,5]]
# group_B = [[27,46,26,9,2,28,24,51,23,175,28,11,39,12],[70, 49, 0,11,12,9,0,2,29,9,1,64,9,66],[4,22,121,8,88,106,158,152,7,6,],[14,0,7,112,3,9,1,43,31,175,34,25,18],[21,9,0,6],\
#            [82,0,154,7,115,88,99,82,82,20,23,169],[32,9,23,4,42,25],[32,8,7,23,4,29,15],[60,27,88,33,10,35,13,49,18,14,28,18,37,],[24,0,26,2,14,0,128,169,6,19,18],[4]]
#
# group_A = [[28,25,19,1,4,24], [27,22,14], [4,21,5,0,19,17,1], [10,2,5,27,31,2,5,0,77,30,4,9,0], [1,4,0,3,0,3,97,3,5,0,10,0], [0,22,23,0,28,12,0,36,19,5]]
# group_B = [[70, 49, 0,11,12,9,0,2,29,9,1,64,9,66],[21,9,0,6],[32,8,7,23,4,29,15],[60,27,88,33,10,35,13,49,18,14,28,18,37,],[24,0,26,2,14,0,128,169,6,19,18],[4]]
# permutation_test_paired(group_A, group_B, iterations = 10000, two_tailed = True)

def permutation_test_paired(group_A, group_B, iterations = 1000, two_tailed = True):
    '''    POOL -> SHUFFLE MOUSE IDENTITY    '''
    # pool trials
    group_A_means = [np.mean(session) for session in group_A]
    group_B_means = [np.mean(session) for session in group_B]
    # create identity array
    group_A_mouse_ID = np.zeros(len(group_A))
    group_B_mouse_ID = np.ones(len(group_B))
    # create 2-D data / ID array
    data = np.array(group_A + group_B)
    # create a copy to be shuffled
    shuffle_data = np.concatenate((group_A_mouse_ID, group_B_mouse_ID))
    # initialize test statistic
    null_distribution = np.ones(iterations) * np.nan
    # iterate over shuffles
    for i in range(iterations):
        # shuffle the labels
        idx_0 = (np.random.random(len(group_A_means)) > .5).astype(int)
        idx_1 = 1 - idx_0
        # create new groups
        group_A_mean = group_A_means * idx_0 + group_B_means * idx_1
        group_B_mean = group_B_means * idx_0 + group_A_means * idx_1
        # get the test statistic for the null distribution
        if two_tailed:
            null_distribution[i] = abs(np.mean([a - b for a, b in zip(group_A_mean, group_B_mean)] )) #abs(group_A_mean - group_B_mean)
        else:
            null_distribution[i] = np.mean([a - b for a, b in zip(group_A_mean, group_B_mean)])
    # get the test statistic from the actual data
    if two_tailed:
        test_statistic = abs(np.mean([a - b for a, b in zip(group_A_means, group_B_means)] ))
    else:
        test_statistic = np.mean([a - b for a, b in zip(group_A_means, group_B_means)])
    # get the p value
    p = np.round(1 - percentileofscore(null_distribution, test_statistic, kind='mean') / 100, 4)
    print('\nPooled stats, shuffled by mouse: p = ' + str(p))
    if p < 0.05:
        print('SIGNIFICANT.')
    else:
        print('not significant')




# def permutation_test_paired(group_A, group_B, iterations = 1000, two_tailed = True):
#     '''    POOL -> SHUFFLE MOUSE IDENTITY    '''
#     # pool trials
#     group_A_means = [np.mean(session) for session in group_A]
#     group_B_means = [np.mean(session) for session in group_B]
#     # create identity array
#     group_A_mouse_ID = np.zeros(len(group_A))
#     group_B_mouse_ID = np.ones(len(group_B))
#     # create 2-D data / ID array
#     data = np.array(group_A + group_B)
#     # create a copy to be shuffled
#     shuffle_data = np.concatenate((group_A_mouse_ID, group_B_mouse_ID))
#     # initialize test statistic
#     null_distribution = np.ones(iterations) * np.nan
#     # iterate over shuffles
#     for i in range(iterations):
#         # shuffle the labels
#         random.shuffle(shuffle_data)
#         # get back the data
#         group_A_mean = np.mean(list(flatten(data[shuffle_data == 0])))
#         group_B_mean = np.mean(list(flatten(data[shuffle_data == 1])))
#         # get the test statistic for the null distribution
#         if two_tailed:
#             null_distribution[i] = abs(np.mean([a - b for a, b in zip(group_A_mean, group_B_mean)] )) #abs(group_A_mean - group_B_mean)
#         else:
#             null_distribution[i] = group_B_mean - group_A_mean
#     # get the test statistic from the actual data
#     if two_tailed:
#         test_statistic = abs(np.mean([a - b for a, b in zip(group_A_means, group_B_means)] ))
#     else:
#         test_statistic = np.mean([a - b for a, b in zip(group_A_means, group_B_means)])
#     # get the p value
#     p = np.round(1 - percentileofscore(null_distribution, test_statistic, kind='mean') / 100, 4)
#     print('\nPooled stats, shuffled by mouse: p = ' + str(p))
#     if p < 0.05:
#         print('SIGNIFICANT.')
#     else:
#         print('not significant')

#
#
# def permutation_correlation(data_x, data_y, iterations = 1000, two_tailed = False, pool_all = True):
#     # get number of mice
#     num_mice = len(data_x)
#     # initialize test statistics
#     null_distribution = np.ones(iterations) * np.nan
#     data_distribution = np.ones(iterations) * np.nan
#     # iterate over shuffles
#     for i in range(iterations):
#         if pool_all:
#             # pool trials
#             data_x_pooled = np.array(list(flatten(data_x)))
#             data_y_pooled = np.array(list(flatten(data_y)))
#         else:
#             # initialize pooled list
#             data_x_pooled = []
#             data_y_pooled = []
#             # pick one datum at random from each mouse
#             for m in range(num_mice):
#                 # if the mouse did any trials
#                 trials = len(data_x[m])
#                 if trials:
#                     # pick a trial at random
#                     t = np.random.randint(0, trials)
#                     # add that datum to the pooled list
#                     data_x_pooled.append(data_x[m][t])
#                     data_y_pooled.append(data_y[m][t])
#         # get the corr coeff of the actual data
#         r, _ = pearsonr(data_x_pooled, data_y_pooled)
#         data_distribution[i] = r
#         # shuffle the data
#         random.shuffle(data_x_pooled)
#         # get the corr coeff of the shuffled data
#         r, _ = pearsonr(data_x_pooled, data_y_pooled)
#         null_distribution[i] = r
#     # get the test statistic
#     if two_tailed:
#         test_statistic = np.mean(abs(data_distribution))
#     else:
#         test_statistic = np.mean(data_distribution)
#     # get the p value
#     if two_tailed:
#         p = np.round(1 - percentileofscore(abs(null_distribution), test_statistic) / 100, 5)
#     else:
#         p = np.round(1 - percentileofscore(null_distribution, test_statistic) / 100, 5)
#     print('\ncorrelation 1 trial per mouse: r = ' + str(np.round(test_statistic,2)) + '  p = ' + str(p))
#     if p < 0.05:
#         print('SIGNIFICANT.')
#     else:
#         print('not significant')

# iterations = 1000


# situation = 'lots of trials few mice'
# situation = 'number of trials correlated with effect'
# situation = 'few trials few mice'
# situation = 'few trials lots of mice'
# situation = 'more trials makes more reliable'
# situation = 'discrete'
#
# print(situation + '\n')
#
#
# if situation == 'lots of trials few mice':
#     group_A = [[50, 50], [75, 75, 75, 75, 75, 75]]
#
#     group_B = [[75, 75], [50, 50, 50, 50, 50, 50]]
#
# elif situation == 'number of trials correlated with effect':
#     group_A = [ [75,75], [75,75], [75,75], [75,75], [75,75], [75,75], [75,75], [75,75], [75,75], [75,75] ]
#
#     group_B = [ [100], [100], [100], [100], [100],
#                 [50, 50, 50], [50, 50, 50], [50, 50, 50], [50, 50, 50], [50, 50, 50] ]
#
# elif situation == 'few trials few mice':
#     group_A = [ [75,75], [75,75] ]
#
#     group_B = [ [50,50], [50,50] ]
#
# elif situation == 'few trials lots of mice':
#     group_A = [ [50,75], [50,75],  [50,75], [50,75],  [50,75], [50,75] ]
#
#     group_B = [ [50], [50], [50], [50], [50], [50] ]
#
# elif situation == 'more trials makes more reliable':
#     group_A = [ [75,75], [75,75],  [75,75], [75,75],  [75,75], [75,75] ]
#
#     group_B = [ [100], [100], [100], [0,0,50, 50], [100,0,50, 50], [50,100,0,50,0] ]
#
# elif situation == 'discrete':
#     group_A = [ [0,1], [0,0],  [0,0], [0,0],  [0,0], [0,0] ]
#
#     group_B = [ [1], [0], [0], [1,1,1], [0,1,1], [0,1] ]
#
#
#
# '''
# SHUFFLE ALL TRIALS
# '''
# # pool trials
# group_A_pooled = np.array(list(flatten(group_A)))
# group_B_pooled = np.array(list(flatten(group_B)))
# # do man-whitney
# s, p = mannwhitneyu(group_A_pooled, group_B_pooled)
# print('Mann Whitney pooled, p = ' + str(p))
#
#
# # create identity array
# group_A_trial_ID = np.zeros_like(group_A_pooled)
# group_B_trial_ID = np.ones_like(group_B_pooled)
# # create 2-D data / ID array
# data = np.stack( (np.concatenate((group_A_trial_ID, group_B_trial_ID)),
#                   np.concatenate((group_A_pooled, group_B_pooled))), 0)
# # create a copy to be shuffled
# shuffle_data = data.copy()
# #initialize test statistic
# null_distribution = np.ones(iterations) * np.nan
# # iterate over shuffles
# for i in range(iterations):
#     # shuffle the labels
#     random.shuffle(shuffle_data[0, :])
#     # get back the data
#     group_A_mean = np.mean(data[1, shuffle_data[0, :] == 0])
#     group_B_mean = np.mean(data[1, shuffle_data[0, :] == 1])
#     # get the test statistic for the null distribution
#     null_distribution[i] = abs(group_A_mean - group_B_mean)
# # get the test statistic from the actual data
# test_statistic = abs(np.mean(group_A_pooled) - np.mean(group_B_pooled))
# # get the p value
# p = np.round(1 - percentileofscore(null_distribution, test_statistic)/100, 2)
# print('\nPooled data: p = ' + str(p))
# if p < 0.05: print('SIGNIFICANT.')
# else: print('not significant')
#
#
#
# '''
# AVERAGE -> SHUFFLE MOUSE IDENTITY
# '''
# # get each mouse's mean value
# group_A_averaged = np.array([np.mean(x) for x in group_A])
# group_B_averaged = np.array([np.mean(x) for x in group_B])
# # create identity array
# group_A_mouse_ID = np.zeros_like(group_A_averaged)
# group_B_mouse_ID = np.ones_like(group_B_averaged)
# # create 2-D data / ID array
# data = np.stack( (np.concatenate((group_A_mouse_ID, group_B_mouse_ID)),
#                   np.concatenate((group_A_averaged, group_B_averaged))), 0)
# # create a copy to be shuffled
# shuffle_data = data.copy()
# #initialize test statistic
# null_distribution = np.ones(iterations) * np.nan
# # iterate over shuffles
# for i in range(iterations):
#     # shuffle the labels
#     random.shuffle(shuffle_data[0, :])
#     # get back the data
#     group_A_mean = np.mean(data[1, shuffle_data[0, :] == 0])
#     group_B_mean = np.mean(data[1, shuffle_data[0, :] == 1])
#     # get the test statistic for the null distribution
#     null_distribution[i] = abs(group_A_mean - group_B_mean)
# # get the test statistic from the actual data
# test_statistic = abs(np.mean(group_A_averaged) - np.mean(group_B_averaged))
# # get the p value
# p = np.round(1 - percentileofscore(null_distribution, test_statistic)/100, 2)
# print('\nAveraged data: p = ' + str(p))
# if p < 0.05: print('SIGNIFICANT.')
# else: print('not significant')
#
# #
#
#
# '''
# POOL -> SHUFFLE MOUSE IDENTITY
# '''
# # pool trials
# group_A_pooled = np.array(list(flatten(group_A)))
# group_B_pooled = np.array(list(flatten(group_B)))
# # create identity array
# group_A_mouse_ID = np.zeros(len(group_A))
# group_B_mouse_ID = np.ones(len(group_B))
# # create 2-D data / ID array
# data = np.array(group_A + group_B)
# # create a copy to be shuffled
# shuffle_data = np.concatenate((group_A_mouse_ID, group_B_mouse_ID))
# #initialize test statistic
# null_distribution = np.ones(iterations) * np.nan
# # iterate over shuffles
# for i in range(iterations):
#     # shuffle the labels
#     random.shuffle(shuffle_data)
#     # get back the data
#     group_A_mean = np.mean(list(flatten(data[shuffle_data == 0])))
#     group_B_mean = np.mean(list(flatten(data[shuffle_data == 1])))
#     # get the test statistic for the null distribution
#     null_distribution[i] = abs(group_A_mean - group_B_mean)
# # get the test statistic from the actual data
# test_statistic = abs(np.mean(group_A_pooled) - np.mean(group_B_pooled))
# # get the p value
# p = np.round(1 - percentileofscore(null_distribution, test_statistic)/100, 2)
# print('\nPooled stats, shuffled by mouse: p = ' + str(p))
# if p < 0.05: print('SIGNIFICANT.')
# else: print('not significant')
#
# print('')


#
# '''
# SHUFFLE TRIAL IDENTITY WITHIN GROUP -> AVERAGE
# '''
# # pool trials
# group_A_pooled = np.array(list(flatten(group_A)))
# group_B_pooled = np.array(list(flatten(group_B)))
# # initialize average list
# group_A_averaged = []
# group_B_averaged = []
# # create identity array
# group_A_trial_ID = np.zeros_like(group_A_pooled)
# group_B_trial_ID = np.ones_like(group_B_pooled)
# # get number of trials per mouse
# group_A_trials = [len(x) for x in group_A]
# group_B_trials = [len(x) for x in group_B]
# #initialize test statistic
# null_distribution = np.ones(iterations) * np.nan
# # iterate over shuffles
# for i in range(iterations):
#     # shuffle each group
#     random.shuffle(group_A_pooled)
#     random.shuffle(group_B_pooled)
#     # create 2-D data / ID array
#     data = np.stack((np.concatenate((group_A_trial_ID, group_B_trial_ID)),
#                      np.concatenate((group_A_pooled, group_B_pooled))), 0)
#     # initialize group array
#     trials_added = 0
#     group_A_data = []
#     group_B_data = []
#     # fill in each mouse with shuffled data
#     for mouse in range(len(group_A)):
#         group_A_data.append(np.mean(data[1, trials_added: trials_added + group_A_trials[mouse]]))
#         trials_added += group_A_trials[mouse]
#     for mouse in range(len(group_B)):
#         group_B_data.append(np.mean(data[1, trials_added: trials_added + group_B_trials[mouse]]))
#         trials_added += group_B_trials[mouse]
#
#     # create identity array
#     group_A_mouse_ID = np.zeros_like(group_A_data)
#     group_B_mouse_ID = np.ones_like(group_B_data)
#     # create 2-D data / ID array
#     data = np.stack((np.concatenate((group_A_mouse_ID, group_B_mouse_ID)),
#                      np.concatenate((group_A_data, group_B_data))), 0)
#     # get back the data
#     group_A_averaged.append( np.mean(data[1, data[0, :] == 0]) )
#     group_B_averaged.append( np.mean(data[1, data[0, :] == 1]) )
#     # create a copy to be shuffled
#     shuffle_data = data.copy()
#     # shuffle the labels
#     random.shuffle(shuffle_data[0, :])
#     # get back the data
#     group_A_mean = np.mean(data[1, shuffle_data[0, :] == 0])
#     group_B_mean = np.mean(data[1, shuffle_data[0, :] == 1])
#     # get the test statistic for the null distribution
#     null_distribution[i] = abs(group_A_mean - group_B_mean)
# # get the test statistic from the actual data
# test_statistic = abs(np.mean(group_A_averaged) - np.mean(group_B_averaged))
# # get the p value
# p = np.round(1 - percentileofscore(null_distribution, test_statistic) / 100, 2)
# print('\nShuffled within group then averaged data: p = ' + str(p))
# if p < 0.05:
#     print('SIGNIFICANT.')
# else:
#     print('not significant')
#
#
# print('')




#
#
# '''
# SHUFFLE TRIAL IDENTITY -> AVERAGE
# '''
# # pool trials
# group_A_pooled = np.array(list(flatten(group_A)))
# group_B_pooled = np.array(list(flatten(group_B)))
# # create identity array
# group_A_trial_ID = np.zeros_like(group_A_pooled)
# group_B_trial_ID = np.ones_like(group_B_pooled)
# # get number of trials per mouse
# group_A_trials = [len(x) for x in group_A]
# group_B_trials = [len(x) for x in group_B]
# # create 2-D data / ID array
# data = np.stack( (np.concatenate((group_A_trial_ID, group_B_trial_ID)),
#                   np.concatenate((group_A_pooled, group_B_pooled))), 0)
# # create a copy to be shuffled
# shuffle_data = data.copy()
# #initialize test statistic
# null_distribution = np.ones(iterations) * np.nan
# # iterate over shuffles
# for i in range(iterations):
#     # shuffle the data
#     random.shuffle(shuffle_data[1, :])
#     # initialize group array
#     trials_added = 0
#     group_A_data = []
#     group_B_data = []
#     # fill in each mouse with shuffled data
#     for mouse in range(len(group_A)):
#         group_A_data.append(np.mean(shuffle_data[1, trials_added: trials_added + group_A_trials[mouse]]))
#         trials_added += group_A_trials[mouse]
#     for mouse in range(len(group_B)):
#         group_B_data.append(np.mean(shuffle_data[1, trials_added: trials_added + group_B_trials[mouse]]))
#         trials_added += group_B_trials[mouse]
#     # get back the data
#     group_A_mean = np.mean(group_A_data)
#     group_B_mean = np.mean(group_B_data)
#     # get the test statistic for the null distribution
#     null_distribution[i] = abs(group_A_mean - group_B_mean)
# # get the test statistic from the actual data
# test_statistic = abs(np.mean(group_A_pooled) - np.mean(group_B_pooled))
# # get the p value
# p = np.round(1 - percentileofscore(null_distribution, test_statistic)/100, 2)
# print('\nPooled then averaged data: p = ' + str(p))
# if p < 0.05: print('SIGNIFICANT.')
# else: print('not significant')
#







