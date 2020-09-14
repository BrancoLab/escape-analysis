
def setup(object):
    '''     setup the parameters for tracking, registration, processing, and analysis      '''

    '''    LOADING THE DATA    '''
    # Add recently added sessions or metadata to database
    object.file_name = 'project_obstacles_dataframe'
    object.update_database = False
    # select which sessions to analyze ('all', 'experiment', 'number')
    object.selector_type = 'number'
    # object.selector = ['Circle lights on off (baseline)','Circle wall down']
    object.selector = [70] #[1,6,3,47]# [47]#[190,192,193] #[190,192,191]#[55,56,82,156,169]#[1,3,4,5,6,45]#[32,33,230,231] # [202, 239, 241, 164] #[13,8, 70,72]#[4,3,5,2]  # [44,148,160,4,5,3]  #[10,13,70,72]
    # the most important parameter.
    object.dark_theme = False


    '''    WHAT ARE WE DOING    '''
    # Use DLC to analyze the raw videos
    object.do_tracking = False
    # Register arena to model arena
    object.do_registration = False
    # Do initial visualizations of the experiments
    object.do_processing =  True
    # Analyze data and test hypotheses
    object.do_analysis = False


    '''    PROCESSING OPTIONS    '''
    # Do initial visualizations of the experiments
    object.processing_options = {
        # Re-process the DLC tracking data
        'process tracking': False,
        # Save raw videos of the peri-stimulus period
        'raw clips': False,
        # Save both registered videos and videos of DLC model mouse in model arena
        'visualize escapes': False,
        # Capture spontaneous homings
        'spontaneous homings': False,
        # Decompose homings into piecewise linear bouts
        'decompose homings': False,
        # Capture spontaneous anti-homings
        'spontaneous anti-homings': False,
        # Decompose anti homings into piecewise linear bouts
        'decompose anti-homings': False,
        # Analyze exploration
        'exploration': False,
        # Simulate each strategy and get its likelihood
        'simulate': False,
        # Simulate each strategy and get its likelihood
        'control': False,
        # Make the moon-like summary plots
        'summary': False }
    object.processing_parameters = {
        # Number of sessions to analyze simultaneously
        'parallel processes': 1,
        # Number of sessions to analyze simultaneously
        'dark theme': True,
        # When to start the trial video
        'seconds pre stimulus':   2,
        # When to end the trial video
        'seconds post stimulus': 12 }


    '''    ANALYSIS OPTIONS    '''
    # Analyze data and test hypotheses
    # What type of analysis to do
    object.analysis_options = {
        # Process and analyze the data (one analysis type at a time...)
        'analyze data': False,
        # Analyze non-escape control epochs
        'control': False}
    object.analysis_types = {
        # Make an exploration heat map
        'exploration': False,
        # Get escape paths
        'escape paths': False,
        # Get speed traces
        'edginess': False,
        # Predict escape behavior
        'prediction': False,
        # Get efficiency correlations
        'efficiency': True,
        # Get metrics by strategy
        'metrics': False,
        # Get speed traces
        'speed traces': False,
        # Plot all traversals across the arena
        'traversals': False }

    object.analysis_experiments= {
        # '''     many-condition edginess comparison (naive)    '''
        # Which experiments to analyze
        # 'experiments': [['Circle wall up'], 'Circle wall down (no shelter)', ['Circle wall down', 'Circle wall down (no baseline)'], ['Circle wall down', 'Circle lights on off (baseline)']], #
        # # # Which conditions to analyze
        # 'conditions': [['no obstacle'],  'no obstacle', ['no obstacle', 'no obstacle'],['obstacle', 'obstacle']],
        # # # what to call each condition for plots
        # 'labels': ['Open field', 'Obstacle removed (no shelter)', 'obstacle removed', 'Obstacle']}

        # '''     many-condition edginess comparison (naive)    '''
        # Which experiments to analyze
        # 'experiments': [['Circle wall up']],
        # # # Which conditions to analyze
        # 'conditions': [['no obstacle']],
        # # # what to call each condition for plots
        # 'labels': ['Open field']}

        # '''     many-condition edginess comparison (naive)    '''
        # Which experiments to analyze
        # 'experiments': [['Circle wall down', 'Circle lights on off (baseline)']],#['Circle wall down', 'Circle lights on off (baseline)']],#,'Circle wall up'],  #
        # # # Which conditions to analyze
        # 'conditions': [['obstacle', 'obstacle']],#['obstacle', 'obstacle']],#, 'no obstacle'],
        # # # what to call each condition for plots
        # 'labels': [ 'obstacle',  'obstacle23']}

        # # Which experiments to analyze
        # 'experiments': [['Circle wall up', 'Circle wall down (no shelter)', 'Circle wall down', 'Circle wall down (no baseline)', 'Circle wall down', 'Circle lights on off (baseline)'],
        #                 ['Circle food wall down', 'Circle food', 'Circle food wall up']], #
        # # # # Which conditions to analyze
        # 'conditions': [['no obstacle',  'no obstacle', 'no obstacle', 'no obstacle','obstacle', 'obstacle'],['no obstacle', 'no obstacle', 'no obstacle']],
        # # # # what to call each condition for plots
        # 'labels': ['Escape', 'Food']}

        # # '''     many-condition edginess comparison (naive)    '''
        # Which experiments to analyze
        # 'experiments': [['Circle wall down', 'Circle lights on off (baseline)'], ['Circle wall down', 'Circle lights on off (baseline)'],'Circle wall up'],  #
        # # # Which conditions to analyze
        # 'conditions': [['obstacle', 'obstacle'],['obstacle', 'obstacle'], 'no obstacle'],
        # # # what to call each condition for plots
        # 'labels': ['obstacle', 'obstacle 3', 'OF']}

        # # '''     many-condition edginess comparison (naive)    '''
        # # # Which experiments to analyze
        # 'experiments': [ 'Circle wall up',['Circle wall down', 'Circle wall down (no baseline)', 'Circle wall down (trial 1)',
        #                   'Circle wall down'], ['Circle wall down', 'Circle lights on off (baseline)']],
        # # # Which conditions to analyze
        # 'conditions': [ 'no obstacle',['no obstacle', 'no obstacle','probe', 'probe'], ['obstacle', 'obstacle']],
        # # # what to call each condition for plots
        # 'labels': [ 'open field','obstacle removed + acute OR', 'obstacle']}

        # Which experiments to analyze
        # 'experiments': [['Circle wall down', 'Circle wall down (no baseline)']],
        # # # Which conditions to analyze
        # 'conditions': [['no obstacle', 'no obstacle']],
        # # # what to call each condition for plots
        # 'labels': ['obstacle removed']}

        # Which experiments to analyze
        # 'experiments': ['Circle wall down (no baseline)'],
        # # # Which conditions to analyze
        # 'conditions': ['no obstacle'],
        # # # what to call each condition for plots
        # 'labels': ['Obstacle removed (no baseline)']}

        # # '''     many-condition edginess comparison (naive)    '''
        # Which experiments to analyze
        # 'experiments': [ ['Circle wall down', 'Circle wall down (trial 1)'],  ['Circle wall down', 'Circle wall down (trial 1)']],  #
        # # # Which conditions to analyze
        # 'conditions': [ ['probe','probe'], ['probe','probe']],
        # # # what to call each condition for plots
        # 'labels': ['acute OR 1','acute OR 0']}

        # 'experiments': [ [ 'Circle wall down (trial 1)'],  [ 'Circle wall down (trial 1)']],  #
        # # # # Which conditions to analyze
        # 'conditions': [ ['probe'], ['probe']],
        # # # # what to call each condition for plots
        # 'labels': ['acute OR 1','acute OR 0']}

        # # '''     many-condition edginess comparison (naive)    '''
        # Which experiments to analyze
        # 'experiments': [ 'Circle wall down'],  #
        # # # Which conditions to analyze
        # 'conditions': [ 'probe'],
        # # # what to call each condition for plots
        # 'labels': ['acute OR']}

        # # '''     many-condition edginess comparison (naive)    '''
        # # Which experiments to analyze
        # 'experiments': ['Circle wall down (no baseline)', 'Circle wall down', 'Circle wall up', 'Circle wall down light (U shaped)'],  #
        # # # Which conditions to analyze
        # 'conditions': [ 'no obstacle', 'no obstacle' , 'no obstacle', 'no obstacle'],
        # # # what to call each condition for plots
        # 'labels': ['obstacle removed 0 BLT', 'obstacle removed 3 BLT', 'open field', 'U']}

        # # '''     many-condition edginess comparison (naive)    '''
        # Which experiments to analyze
        # 'experiments': ['Circle wall down',['Circle wall down', 'Circle wall down (no baseline)'],'Circle wall up'],  #
        # # # Which conditions to analyze
        # 'conditions': ['probe',['no obstacle', 'no obstacle'],'no obstacle'],
        # # # what to call each condition for plots
        # 'labels': ['open field','obstacle removed', 'obstacle removed']}


        # Which experiments to analyze
        # 'experiments': [['Circle wall down', 'Circle wall down (no baseline)'], 'Circle void up'],  #
        # # # Which conditions to analyze
        # 'conditions': [['no obstacle', 'no obstacle'], 'obstacle'],
        # # # what to call each condition for plots
        # 'labels': [ 'obstacle removed', 'hole vs wall removed']}

        # Which experiments to analyze
        # 'experiments': ['Circle void up'],  #
        # # # Which conditions to analyze
        # 'conditions': ['obstacle'],
        # # # what to call each condition for plots
        # 'labels': ['hole']}


        # '''     many-condition edginess comparison (naive)    '''
        # # Which experiments to analyze
        # 'experiments': ['Circle wall up'],  #
        # # # Which conditions to analyze
        # 'conditions': [ 'obstacle'],
        # # # what to call each condition for plots
        # 'labels': ['Open field']}


        # Which experiments to analyze
        # 'experiments': [ 'Circle wall down (no shelter)','Circle wall down (no shelter)','Circle wall down (no shelter)','Circle wall down (no shelter)'], #
        # # # Which conditions to analyze
        # 'conditions': ['no obstacle','no obstacle','no obstacle','no obstacle'],
        # # # what to call each condition for plots
        # 'labels': ['Obstacle removed (no shelter)', 'Obstacle removed (no shelter)', 'Obstacle removed (no shelter)', 'Obstacle removed (no shelter)']}


        # # Which experiments to analyze
        # 'experiments': [['Circle wall down', 'Circle wall down (no baseline)','Circle wall down', 'Circle wall down (trial 1)'], ['Circle wall down', 'Circle wall down (no baseline)','Circle wall down', 'Circle wall down (trial 1)'],
        #                 ['Circle wall down', 'Circle wall down (no baseline)','Circle wall down', 'Circle wall down (trial 1)'], ['Circle wall down', 'Circle wall down (no baseline)','Circle wall down', 'Circle wall down (trial 1)']],  #
        # # # Which conditions to analyze
        # 'conditions': [['no obstacle', 'no obstacle','probe','probe'], ['no obstacle', 'no obstacle','probe','probe'], ['no obstacle', 'no obstacle','probe','probe'], ['no obstacle', 'no obstacle','probe','probe']],
        # # # what to call each condition for plots
        # 'labels': ['obstacle removed 4+', 'obstacle removed 2-3', 'obstacle removed 1', 'obstacle removed 0']}

    #     # # Which experiments to analyze
    #     'experiments': [['Circle wall down', 'Circle wall down (no baseline)'], ['Circle wall down', 'Circle wall down (no baseline)'],
    #                     ['Circle wall down', 'Circle wall down (no baseline)'], ['Circle wall down', 'Circle wall down (no baseline)']],  #
    #     # # Which conditions to analyze
    #     'conditions': [['no obstacle', 'no obstacle'], ['no obstacle', 'no obstacle'], ['no obstacle', 'no obstacle'], ['no obstacle', 'no obstacle']],
    #     # # what to call each condition for plots
    #     'labels': ['obstacle removed 4+', 'obstacle removed 2-3', 'obstacle removed 1', 'obstacle removed 0']}


    # # '''     many-condition edginess comparison (naive)    '''
    #     Which experiments to analyze
    #     'experiments': [['Circle wall down', 'Circle lights on off (baseline)'],['Circle wall down', 'Circle lights on off (baseline)']],  #
    #     # # Which conditions to analyze
    #     'conditions': [['obstacle', 'obstacle'], ['obstacle', 'obstacle']],
    #     # # what to call each condition for plots
    #     'labels': ['Obstacle 1', 'Obstacle 2-3']}

        # '''     many-condition initial traj comparison (naive)    '''
        # Which experiments to analyze
        # 'experiments': [['Circle wall down', 'Circle lights on off (baseline)'], ['Circle wall down', 'Circle lights on off (baseline)'],['Circle wall up', 'Circle wall up (2)'],'Circle wall up'], #
        # # # Which conditions to analyze
        # 'conditions': [ ['obstacle','obstacle'], ['obstacle','obstacle'], ['probe', 'probe'],'no obstacle'],
        # # # what to call each condition for plots
        # 'labels': ['Obstacle trial 2-3', 'Obstacle trial 1', 'Unexpected obstacle','Open field']}

        # 'experiments': [['Circle wall down', 'Circle lights on off (baseline)'],['Circle wall down', 'Circle lights on off (baseline)'], ['Circle wall down', 'Circle lights on off (baseline)'],
        #                 ['Circle wall up', 'Circle wall up (2)'], 'Circle wall up'],  #
        # # # Which conditions to analyze
        # 'conditions': [['obstacle', 'obstacle'], ['obstacle', 'obstacle'], ['obstacle', 'obstacle'], ['probe', 'probe'], 'no obstacle'],
        # # # what to call each condition for plots
        # 'labels': ['Obstacle trial 3','Obstacle trial 2', 'Obstacle trial 1', 'Unexpected obstacle', 'Open field']}


        # '''     osbtacle    '''
        # # Which experiments to analyze
        # 'experiments': [['Circle wall down', 'Circle lights on off (baseline)']], #
        # # # Which conditions to analyze
        # 'conditions': [['obstacle', 'obstacle']],
        # # # what to call each condition for plots
        # 'labels': ['Obstacle 1']}#, 'Obstacle']}

        # '''     many-condition edginess comparison (experienced)    '''
        # # # Which experiments to analyze
        # 'experiments': ['Circle wall up (2)', 'Circle lights on off (baseline)', 'Circle wall down (no baseline no naive)', 'Circle lights on off (baseline)'],  #
        # # # Which conditions to analyze
        # 'conditions': ['no obstacle', 'probe', 'no obstacle','probe'],
        # # # what to call each condition for plots
        # 'labels': ['Open field', 'null', 'obstacle removed','null']}

        # '''     many-condition edginess comparison (experienced)    '''
        # # # Which experiments to analyze
        # 'experiments': ['Circle wall up (2)', 'Circle wall down (no baseline no naive)'],  #
        # # # Which conditions to analyze
        # 'conditions': ['no obstacle', 'no obstacle'],
        # # # what to call each condition for plots
        # 'labels': ['Open field', 'obstacle removed']}

        # # # '''     light, obstacle     '''
        # Which experiments to analyze
        # 'experiments': [ 'Circle lights on off (baseline)', 'Circle (dark)'], #
        # # # Which conditions to analyze
        # 'conditions': ['no obstacle', 'no obstacle'],
        # # # what to call each condition for plots
        # 'labels': ['obstacle light on off baseline', 'OF lights']}

        # '''     dark, obstacle     '''
        # Which experiments to analyze
        # 'experiments': ['Circle wall up',['Circle wall down', 'Circle lights on off (baseline)'],'Circle wall down dark', 'Circle (dark)'],
        # # # Which conditions to analyze
        # 'conditions': ['no obstacle',['obstacle','obstacle'],'obstacle', 'no obstacle'],
        # # # what to call each condition for plots
        # 'labels': ['open field light','obstacle light','obstacle dark', 'open field dark']} #'obstacle dark exp',  'U shaped dark'

        # '''     dark, obstacle     '''
        # Which experiments to analyze
        # 'experiments': ['Circle (dark)'],#, 'Circle wall lights on off quick'],
        # # # Which conditions to analyze
        # 'conditions': ['no obstacle'],#, 'obstacle'],
        # # # what to call each condition for plots
        # 'labels': ['no obstacle dark']}#, 'obstacle acute dark']} #'obstacle dark exp',  'U shaped dark'

        # '''     dark, obstacle     '''
        # Which experiments to analyze
        # 'experiments': ['Circle wall up'], #'Circle wall lights on off quick'],#
        # # # Which conditions to analyze
        # 'conditions': ['no obstacle'], #'obstacle'],#
        # # # what to call each condition for plots
        # 'labels': ['open field']}#,'light on off']}  # , 'obstacle acute dark']} #'obstacle dark exp',  'U shaped dark'

        # '''     dark, obstacle     '''
        # Which experiments to analyze
        # 'experiments': ['Circle wall down (dark non naive)', 'Circle (dark)'], #
        # # # Which conditions to analyze
        # 'conditions': ['obstacle', 'no obstacle'], #
        # # # what to call each condition for plots
        # 'labels': [ 'obstacle dark', 'no obstacle dark']}  #

        # '''     dark, obstacle     '''
        # Which experiments to analyze
        # 'experiments': ['Circle lights on off (baseline)','Circle wall lights on off quick','Circle (dark)'],
        # # # Which conditions to analyze
        # 'conditions': [ 'no obstacle', 'obstacle','no obstacle'],
        # # # what to call each condition for plots
        # 'labels': ['light on off', 'light on off quick', 'open field dark']}

        # '''     dark, obstacle     '''
        # Which experiments to analyze
        # 'experiments': ['Circle (dark)','Circle lights on off (baseline)','Circle wall lights on off quick','Circle wall down dark','Circle wall down (dark non naive)'],  #
        # # # Which conditions to analyze
        # 'conditions': ['no obstacle', 'no obstacle', 'obstacle', 'obstacle', 'obstacle'],  #
        # # # what to call each condition for plots
        # 'labels': ['no obstacle dark','light on off', 'light on off quick', 'obstacle dark' , 'obstacle dark NN']}  #

        # '''     dark, obstacle     '''
        # Which experiments to analyze
        # 'experiments': ['Circle (dark)', 'Circle wall lights on off quick', 'Circle wall down dark', 'Circle wall down (dark non naive)'],  #
        # # # Which conditions to analyze
        # 'conditions': ['no obstacle', 'obstacle', 'obstacle', 'obstacle'],  #
        # # # what to call each condition for plots
        # 'labels': ['no obstacle dark', 'light on off quick', 'obstacle dark', 'obstacle dark NN']}  #

        # # # '''     dark, U-shaped obstacle     '''
        # # Which experiments to analyze
        # 'experiments': ['Circle wall down dark (U shaped)'],  #
        # # # Which conditions to analyze
        # 'conditions': ['obstacle'],
        # # # what to call each condition for plots
        # 'labels': ['U shaped dark']}

        # '''     shelter on side    '''
        # # Which experiments to analyze
        # 'experiments': ['Circle wall (shelter on side)',], #'Circle void (shelter on side)',
        # # # Which conditions to analyze
        # 'conditions': ['obstacle'], #'obstacle',
        # # # what to call each condition for plots
        # 'labels': [ 'wall SOS']} #'hole SOS',

        # #'''     obstacle present        '''
        # Which experiments to analyze
        # 'experiments': [['Circle wall down', 'Circle lights on off (baseline)'] , ['Circle wall down', 'Circle wall down (no baseline)']],
        # # Which conditions to analyze
        # 'conditions': [['obstacle', 'obstacle'], ['no obstacle', 'no obstacle']],
        # # what to call each condition for plots
        # 'labels': ['Obstacle', 'OR']}

        # '''     obstacle removed        '''
        # Which experiments to analyze
        # 'experiments': ['Circle wall down','Circle wall down (no baseline)'],
        # # Which conditions to analyze
        # 'conditions': ['no obstacle','no obstacle'], #,['obstacle', 'obstacle'], 'no obstacle'],
        # # what to call each condition for plots
        # 'labels': ['Obstacle removed', 'OR no baseline']} #, 'obstacle','obstacle long']}

        # '''     obstacle removed00        '''
        # Which experiments to analyze
        # 'experiments': ['Circle wall down (trial 1)'],
        # # Which conditions to analyze
        # 'conditions': ['no obstacle'],  # ,['obstacle', 'obstacle'], 'no obstacle'],
        # # what to call each condition for plots
        # 'labels': ['or tr 1']}  # , 'obstacle','obstacle long']}


        # # '''     obstacle removed for efficiency plot        '''
        # # Which experiments to analyze
        # 'experiments': [['Circle wall down', 'Circle wall down (no baseline)', 'Circle wall down (no baseline no naive)'], ['Circle wall down', 'Circle wall down (no baseline)', 'Circle wall down (no baseline no naive)'], ['Circle wall up', 'Circle wall up (2)']],
        # # Which conditions to analyze
        # 'conditions': [['no obstacle', 'no obstacle', 'no obstacle'], ['no obstacle', 'no obstacle', 'no obstacle'], ['no obstacle', 'no obstacle']],
        # # what to call each condition for plots
        # 'labels': ['Obstacle removed', 'Open field' , 'OR (exp)', 'OF (exp)']}

        # # # '''     obstacle removed (experienced)        '''
        # # Which experiments to analyze
        # 'experiments': ['Circle wall down (no baseline no naive)', 'Circle wall up (2)'], #'Circle wall down light (U shaped)', 'Circle wall (11 shaped)',
        # # Which conditions to analyze
        # 'conditions': ['no obstacle', 'no obstacle'], #'no obstacle', 'obstacle',
        # # what to call each condition for plots
        # 'labels': ['Obstacle removed (exp)', 'Open field (exp)']} #'U-shaped', '11-shaped',

        # # '''     all obstacle present vs open field      '''
        # # Which experiments to analyze
        # 'experiments': [['Circle wall up', 'Circle wall up (2)', 'Circle wall down'], ['Circle wall down', 'Circle lights on off (baseline)','Circle wall up', 'Circle wall up (2)']], #
        # # Which conditions to analyze
        # 'conditions': [['no obstacle', 'no obstacle', 'no obstacle'], ['obstacle', 'obstacle','obstacle', 'obstacle']],
        # # what to call each condition for plots
        # 'labels': ['Open field', 'Obstacle']}

        # # '''     obstacle present vs open field      '''
        # # Which experiments to analyze
        # 'experiments': [ ['Circle wall down', 'Circle lights on off (baseline)'], 'Circle wall up'],  #
        # # Which conditions to analyze
        # 'conditions': [['obstacle', 'obstacle'],'no obstacle'],
        # # what to call each condition for plots
        # 'labels': ['Open field', 'Obstacle']}

        # # '''     U-shaped and 11-shaped      '''
        # # Which experiments to analyze
        # 'experiments': ['Circle wall down light (U shaped)'],  #
        # # Which conditions to analyze
        # 'conditions': ['no obstacle'],
        # # what to call each condition for plots
        # 'labels': ['U shaped']}

        # # # '''     U-shaped and 11-shaped      '''
        # # # Which experiments to analyze
        # 'experiments': ['Circle wall down light (U shaped)', ['Circle wall (11 shaped)', 'Circle wall down light (U shaped)']],  #
        # # Which conditions to analyze
        # 'conditions': ['obstacle', ['no obstacle', 'no obstacle']],
        # # what to call each condition for plots
        # 'labels': ['U shaped', '11 shaped']}

        # # '''    11-shaped      '''
        # Which experiments to analyze
        # 'experiments': ['Circle wall (11 shaped)'],  #
        # # Which conditions to analyze
        # 'conditions': ['obstacle'],
        # # what to call each condition for plots
        # 'labels': ['11 shaped']}

        # # '''     exploration comparison      '''
        # # Which experiments to analyze
        # 'experiments': [['Circle wall up', 'Circle wall up (2)', 'Circle wall down'], ['Circle wall down', 'Circle lights on off (baseline)','Circle wall up', 'Circle wall up (2)'],
        #                 'Circle void up', 'Circle wall (no shelter)', 'Circle void (no shelter)', 'Circle (no shelter)'],
        # # Which conditions to analyze
        # 'conditions': [['no obstacle', 'no obstacle', 'no obstacle'], ['obstacle', 'obstacle','obstacle', 'obstacle'],'obstacle','obstacle','obstacle', 'no obstacle'],
        # # what to call each condition for plots
        # 'labels': ['Open field', 'Obstacle', 'Hole obstacle', 'Wall no shelter','Hole no shelter','Empty']}

        # # '''     hole obstacle       '''
        # Which experiments to analyze
        # 'experiments': [['Circle void up'], ['Circle wall down', 'Circle lights on off (baseline)']], #
        # # Which conditions to analyze
        # 'conditions': [['obstacle'], ['obstacle', 'obstacle']],
        # # what to call each condition for plots
        # 'labels': ['Hole obstacle', 'Wall obstacle']}

        # # '''     Lights on -> off       '''
        # # # Which experiments to analyze
        # 'experiments': ['Circle wall lights on off quick', 'Circle lights on off (baseline)'],
        # # Which conditions to analyze
        # 'conditions': ['obstacle', 'no obstacle'],
        # # what to call each condition for plots
        # 'labels': ['lights on off (no baseline)','lights on off (baseline)']}

        # # '''     Lights on -> off       '''
        # # # Which experiments to analyze
        # 'experiments': ['Circle wall up', 'Circle wall lights on off quick', ['Circle wall down', 'Circle lights on off (baseline)']],
        # # Which conditions to analyze
        # 'conditions': ['no obstacle','obstacle',  ['obstacle', 'obstacle']],
        # # what to call each condition for plots
        # 'labels': ['open field','lights on off (baseline)',  'obstacle']}

        # # '''     no obstacle/wall up       '''
        # # Which experiments to analyze
        # 'experiments': [['Circle wall down (trial 1)', 'Circle wall down'],['Circle wall down (trial 1)', 'Circle wall down']], #
        # # Which conditions to analyze
        # 'conditions': [['probe', 'probe'],['probe', 'probe']],
        # # # what to call each condition for plots
        # 'labels': ['no evs','evs']}

        # # '''     no obstacle/wall up       '''
        # # Which experiments to analyze
        # 'experiments': [['Circle wall up', 'Circle wall up (2)']], #
        # # # Which conditions to analyze
        # 'conditions': [['probe', 'probe']],
        # # # # what to call each condition for plots
        # 'labels': ['wall up']}


        # # '''     no obstacle/wall up       '''
        # # Which experiments to analyze
        # 'experiments': [['Circle wall up', 'Circle wall up (2)'],['Circle wall up', 'Circle wall up (2)'],['Circle wall up', 'Circle wall up (2)']], #
        # # Which conditions to analyze
        # 'conditions': [['probe', 'probe'],['obstacle', 'obstacle'], ['no obstacle', 'no obstacle']],
        # # # what to call each condition for plots
        # 'labels': ['wall up', 'obstacle added', 'open field']} #''Open field (naive)', 'Open field (exp)']}

        # '''     square wall moves left       '''
        # Which experiments to analyze
        # 'experiments': [ ['Square wall moves right', 'Square wall left'],'Square wall moves left',  'Square wall moves left'], #, 'Square wall moves right'], #
        # # Which conditions to analyze
        # 'conditions': [['no obstacle', 'no obstacle'],'obstacle',  'no obstacle'], #, 'obstacle'], #
        # # what to call each condition for plots
        # 'labels': ['obstacle short', 'wall shortened', 'obstacle long']} #, 'wall lengthened']} #

        # '''     square wall moves left       '''
        # Which experiments to analyze
        # 'experiments': ['Square wall moves right', ['Square wall moves right', 'Square wall left'], 'Square wall moves left'  ], #
        # # Which conditions to analyze
        # 'conditions': ['obstacle', ['no obstacle', 'no obstacle'],  'obstacle'], #
        # # what to call each condition for plots
        # 'labels': ['wall lengthened', 'obstacle short',  'wall shortened']} #

        # '''     square wall moves left       '''
        # Which experiments to analyze
        # 'experiments': [ 'Square wall moves left', ['Square wall moves right', 'Square wall left', 'Square wall moves left'], 'Square wall moves right'],
        # # Which conditions to analyze
        # 'conditions': [ 'obstacle', ['no obstacle', 'no obstacle', 'no obstacle'], 'obstacle'],  #
        # # what to call each condition for plots
        # 'labels': [ 'wall shortened','obstacle short', 'wall lengthened']}  #

        # # '''     square wall moves left       '''
        # # Which experiments to analyze
        # 'experiments': [['Square wall moves left', 'Square wall moves right'], ['Square wall moves right', 'Square wall left', 'Square wall moves left']],
        # # Which conditions to analyze
        # 'conditions': [['obstacle', 'obstacle'], ['no obstacle', 'no obstacle', 'no obstacle']],  #
        # # what to call each condition for plots
        # 'labels': ['wall shortened', 'consistent', 'wall lengthened']}  #

        # '''     square wall moves left       '''
        # Which experiments to analyze
        # 'experiments': ['Square wall moves right'],
        # # Which conditions to analyze
        # 'conditions': ['obstacle', ],  #
        # # what to call each condition for plots
        # 'labels': ['wall lengthened']}  #

        # '''     square wall moves left       '''
        # Which experiments to analyze
        # 'experiments': ['Square wall moves right', 'Square wall moves left'  ], #
        # # Which conditions to analyze
        # 'conditions': ['obstacle', 'no obstacle'], #
        # # what to call each condition for plots
        # 'labels': ['wall lengthened', 'obstacle long']} #


        # # '''     no obstacle/wall up       '''
        # # Which experiments to analyze
        # 'experiments': ['Circle wall (no shelter)'], #
        # # # # Which conditions to analyze
        # 'conditions': ['obstacle'],
        # # # # # what to call each condition for plots
        # 'labels': ['wall no shelter']}



        # '''     food expts        '''
        # Which expfferiments to analyze
        # 'experiments': ['Circle food wall down',  ['Circle food', 'Circle food wall up']], #'Circle food wall down'], #
        # # Which conditions to analyze
        # 'conditions': [ 'no obstacle', ['no obstacle', 'no obstacle']], #'obstacle'], #
        # # what to call each condition for plots
        # 'labels': ['Food wall down',  'Food open field']} #'Food obstacle']} #

        # # Which experiments to analyze
        # 'experiments': ['Circle food wall down'],
        # # Which conditions to analyze
        # 'conditions': ['obstacle'],
        # # what to call each condition for plots
        # 'labels': ['Food obstacle']}

        # # # '''     food wall down expts        '''
        # Which experiments to analyze
        # 'experiments': ['Circle food wall down'],# ['Circle wall down', 'Circle wall down (no baseline)']],
        # # Which conditions to analyze
        # 'conditions': ['no obstacle'],# ['no obstacle','no obstacle']],
        # # what to call each condition for plots
        # 'labels': ['Food wall down', 'Escape wall down']}

        # # # '''     food vs escape        '''
        # # Which experiments to analyze
        'experiments': [['Circle food', 'Circle food wall up', 'Circle food wall down', 'Circle food wall down'],
                        ['Circle wall up', 'Circle wall down (no shelter)', 'Circle wall down', 'Circle wall down (no baseline)',
                        'Circle wall down', 'Circle lights on off (baseline)', 'Circle wall up (2)', 'Circle wall down (no baseline no naive)']],
        # Which conditions to analyze
        'conditions': [['no obstacle', 'no obstacle', 'no obstacle', 'obstacle'],
                       ['no obstacle',  'no obstacle', 'no obstacle', 'no obstacle','obstacle', 'obstacle', 'no obstacle', 'no obstacle']],
        # what to call each condition for plots
        'labels': ['Food', 'Escape']}

        # '''     many-condition traversal comparison     '''
        # # # Which experiments to analyze
        # 'experiments': ['Circle food', 'Circle food wall down', 'Circle food wall down', ['Circle wall up', 'Circle wall up (2)'], ['Circle wall down', 'Circle wall down (no baseline)'],
        #                  ['Circle wall down', 'Circle lights on off (baseline)']],
        # # # Which conditions to analyze
        # 'conditions': ['no obstacle', 'no obstacle', 'obstacle', ['no obstacle', 'no obstacle'],  ['no obstacle', 'no obstacle'],
        #                 ['obstacle', 'obstacle']],
        # # # what to call each condition for plots
        # 'labels': ['Open field food', 'Obstacle removed food', 'Obstacle food', 'Open field escape',  'Obstacle removed escape',
        #             'Obstacle escape']}

        # '''     3-condition traversal comparison     '''
        # # # Which experiments to analyze
        # 'experiments': [['Circle wall up'], ['Circle wall down', 'Circle wall down (no baseline)'], ['Circle wall down', 'Circle lights on off (baseline)']],
        # # # # Which conditions to analyze
        # 'conditions': [['no obstacle'],  ['no obstacle', 'no obstacle'], ['obstacle', 'obstacle']],
        # # # what to call each condition for plots
        # 'labels': ['Open field escape',  'Obstacle removed escape', 'Obstacle escape']}

        # # '''     spontaneous edge vector comparison     '''
        # # # Which experiments to analyze
        # 'experiments': ['Circle wall up', 'Circle (no shelter)', 'Circle void up','Circle void (no shelter)',
        #                 ['Circle wall down', 'Circle lights on off (baseline)'], ['Circle wall down (no shelter)', 'Circle wall (no shelter)'] ],
        # # # Which conditions to analyze
        # 'conditions': ['no obstacle', 'no obstacle','obstacle', 'obstacle',  ['obstacle', 'obstacle'], ['obstacle', 'obstacle']],
        # # # what to call each condition for plots
        # 'labels': ['open field', 'open field (no shelter)', 'hole obstacle', 'hole obstacle (no shelter)','obstacle', 'obstacle (no shelter)']}

        # # '''     spontaneous edge vector comparison     '''
        # # # Which experiments to analyze
        # 'experiments': [['Circle wall up', 'Circle (no shelter)'], ['Circle void up', 'Circle void (no shelter)'],
        #                 ['Circle wall down', 'Circle lights on off (baseline)', 'Circle wall down (no shelter)', 'Circle wall (no shelter)']],
        # # # Which conditions to analyze
        # 'conditions': [['no obstacle', 'no obstacle'], ['obstacle', 'obstacle'], ['obstacle', 'obstacle', 'obstacle', 'obstacle']],
        # # # what to call each condition for plots
        # 'labels': ['open field',  'hole obstacle', 'obstacle']}


    # # '''     spontaneous edge vector comparison     '''
        # # # Which experiments to analyze
        # 'experiments': [['Circle wall down', 'Circle lights on off (baseline)'] ],
        # # # Which conditions to analyze
        # 'conditions': [['obstacle', 'obstacle']],
        # # # what to call each condition for plots
        # 'labels': ['obstacle']}
        #

        # # '''     spontaneous edge vector comparison     '''
        # # Which experiments to analyze
        # 'experiments': ['Circle void up'],  #
        # # # Which conditions to analyze
        # 'conditions': [ 'obstacle'],
        # # # what to call each condition for plots
        # 'labels': ['hole obstacle']}

        # # '''     spontaneous edge vector comparison II    '''
        # # Which experiments to analyze
        # 'experiments': ['Circle wall down (no shelter)'],  #
        # # # Which conditions to analyze
        # 'conditions': ['obstacle'],
        # # # what to call each condition for plots
        # 'labels': ['obstacle down no shelter']}


        # '''     obstacle no shelter     '''
        # # Which experiments to analyze
        # 'experiments': ['Circle wall (no shelter)'],  #
        # # # Which conditions to analyze
        # 'conditions': [ 'obstacle'],
        # # # what to call each condition for plots
        # 'labels': ['obstacle (no shelter)']}f

        # '''     goal directed     '''
        # Which experiments to analyze
        # 'experiments': ['Circle wall down shelter move', ['Circle wall down', 'Circle wall down (no baseline)']],  #
        # # # Which conditions to analyze
        # 'conditions': [ 'no obstacle', ['no obstacle', 'no obstacle']],
        # # # what to call each condition for plots
        # 'labels': ['obstacle removed (shelter move)', 'obstacle removed']}




    '''    MISC    '''
    # DeepLabCut options
    object.dlc_settings = {
        'clips_folder': 'D:\\data\\Paper',
        'dlc_network_posecfg': 'D:\\data\\DLC_nets\\Barnes-Philip-2018-11-22\\dlc-models\\iteration-9\\Barnes2018-11-22-trainset95shuffle1\\test',
        'dlc_network_snapshot': 'D:\\data\\DLC_nets\\Barnes-Philip-2018-11-22\\dlc-models\\iteration-9\\Barnes2018-11-22-trainset95shuffle1\\train\\snapshot-1000000',
        'scorer': 'DeepCut_resnet101_Philip_50000',
        'config_file': 'D:\\data\\DLC_nets\\Barnes-Philip-2018-11-22\\config.yaml',
        'body parts': ['nose','L eye','R eye','L ear','neck','R ear','L shoulder','upper back','R shoulder','L hind limb','Lower back','R hind limb','derriere'],
        'inverse_fisheye_map_location': 'C:\\Drive\\DLC\\arena_files\\inverse_fisheye_maps.npy' }
    # folders where things are saved
    object.folders = {
    'excel_path': 'D:\\data\\experiments_paper.xlsx',
    'save_folder': 'D:\\data',
    'DLC_folder': 'C:\\Drive\\Behaviour\\DeepLabCut',
    'fisheye_map_location': 'C:\\Drive\\DLC\\arena_files\\fisheye_maps.npy' }
