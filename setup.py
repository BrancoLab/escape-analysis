
def setup(object):
    '''     setup the parameters for tracking, registration, processing, and analysis      '''

    '''    LOADING THE DATA    '''
    # Add recently added sessions or metadata to database
    object.file_name = 'project_obstacles_dataframe'
    object.update_database = False
    # select which sessions to analyze ('all', 'experiment', 'number')
    object.selector_type = 'number'
    # object.selector = ['Circle wall down', 'Circle lights on off (baseline)']
    object.selector = [5]
    # the most important parameter.
    object.dark_theme = False


    '''    WHAT ARE WE DOING    '''
    # Use DLC to analyze the raw videos
    object.do_tracking = False
    # Register arena to model arena
    object.do_registration = False
    # Do initial visualizations of the experiments
    object.do_processing = True
    # Analyze data and test hypotheses
    object.do_analysis = True


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
        'decompose homings': True,
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
        'parallel processes': 6,
        # Number of sessions to analyze simultaneously
        'dark theme': True,
        # When to start the trial video
        'seconds pre stimulus': 3,
        # When to end the trial video
        'seconds post stimulus': 12 }


    '''    ANALYSIS OPTIONS    '''
    # Analyze data and test hypotheses
    object.analysis_options = {
        # Process and analyze the data (one analysis type at a time...)
        'analyze data': True,
        # Analyze non-escape control epochs
        'control': False}
    # What type of analysis to do
    object.analysis_types = {
        # Make an exploration heat map
        'exploration': False,
        # Get speed traces
        'escape paths': False,
        # Get speed traces
        'edginess': False,
        # Plot all traversals across the arena
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
        # # '''     many-condition edginess comparison (naive)    '''
        # # Which experiments to analyze
        # 'experiments': [['Circle wall up'], 'Circle wall down (no shelter)', ['Circle wall down', 'Circle wall down (no baseline)'], ['Circle wall down', 'Circle lights on off (baseline)']], #
        # # # Which conditions to analyze
        # 'conditions': [['no obstacle'],  'no obstacle', ['no obstacle', 'no obstacle'],['obstacle', 'obstacle']],
        # # # what to call each condition for plots
        # 'labels': ['Open field', 'Obstacle removed (no shelter)', 'obstacle removed', 'Obstacle']}

        # # '''     many-condition edginess comparison (naive)    '''
        # # Which experiments to analyze
        # 'experiments': [['Circle wall down', 'Circle lights on off (baseline)'],['Circle wall down', 'Circle lights on off (baseline)']],  #
        # # # Which conditions to analyze
        # 'conditions': [['obstacle', 'obstacle'], ['obstacle', 'obstacle']],
        # # # what to call each condition for plots
        # 'labels': ['Obstacle 1', 'Obstacle 2-3']}

        # '''     many-condition initial traj comparison (naive)    '''
        # # Which experiments to analyze
        # 'experiments': [['Circle wall up', 'Circle wall up (2)'],['Circle wall down', 'Circle lights on off (baseline)'], ['Circle wall down', 'Circle lights on off (baseline)'],'Circle wall up'], #
        # # # Which conditions to analyze
        # 'conditions': [['probe', 'probe'], ['obstacle','obstacle'], ['obstacle','obstacle'], 'no obstacle'],
        # # # what to call each condition for plots
        # 'labels': ['Unexpected obstacle','Obstacle trial 2-3', 'Obstacle trial 1', 'Open field']}

        # # '''     many-condition edginess comparison (naive)    '''
        # # Which experiments to analyze
        # 'experiments': [['Circle wall down', 'Circle lights on off (baseline)']], #
        # # # Which conditions to analyze
        # 'conditions': [['obstacle', 'obstacle']],
        # # # what to call each condition for plots
        # 'labels': ['obstacle']}#, 'Obstacle']}

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

        # '''     light, obstacle     '''
        # Which experiments to analyze
        'experiments': [ ['Circle wall down', 'Circle lights on off (baseline)']], #
        # # Which conditions to analyze
        'conditions': [['obstacle', 'obstacle']],
        # # what to call each condition for plots
        'labels': ['obstacle light']}

        # '''     dark, obstacle     '''
        # # Which experiments to analyze
        # 'experiments': ['Circle (dark)', 'Circle wall down dark', 'Circle wall down dark'],
        # # # Which conditions to analyze
        # 'conditions': ['no obstacle','obstacle', 'obstacle'],
        # # # what to call each condition for plots
        # 'labels': ['open field dark','obstacle dark', 'obstacle dark', ]} #'obstacle dark exp',  'U shaped dark'

        # # # '''     dark, U-shaped obstacle     '''
        # # Which experiments to analyze
        # 'experiments': ['Circle wall down dark (U shaped)'],  #
        # # # Which conditions to analyze
        # 'conditions': ['obstacle'],
        # # # what to call each condition for plots
        # 'labels': ['U shaped dark']}

        # '''     shelter on side    '''
        # # Which experiments to analyze
        # 'experiments': ['Circle void (shelter on side)', 'Circle wall (shelter on side)',],
        # # # Which conditions to analyze
        # 'conditions': ['obstacle', 'obstacle'],
        # # # what to call each condition for plots
        # 'labels': ['hole SOS', 'wall SOS']}

        # #'''     obstacle present        '''
        # # Which experiments to analyze
        # 'experiments': [['Circle wall down', 'Circle lights on off (baseline)']], #
        # # Which conditions to analyze
        # 'conditions': [['obstacle', 'obstacle']], #
        # # what to call each condition for plots
        # 'labels': ['Obstacle']}

        # # '''     obstacle removed        '''
        # # Which experiments to analyze
        # 'experiments': [['Circle wall down','Circle wall down (no baseline)']], #, ['Circle wall down', 'Circle lights on off (baseline)'],  'Square wall moves left'],
        # # Which conditions to analyze
        # 'conditions': [['no obstacle','no obstacle']], #,['obstacle', 'obstacle'], 'no obstacle'],
        # # what to call each condition for plots
        # 'labels': ['Obstacle removed']} #, 'obstacle','obstacle long']}

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
        # 'experiments': ['Circle wall up', ['Circle wall down', 'Circle lights on off (baseline)']],  #
        # # Which conditions to analyze
        # 'conditions': ['no obstacle', ['obstacle', 'obstacle']],
        # # what to call each condition for plots
        # 'labels': ['Open field', 'Obstacle']}

        # # '''     U-shaped and 11-shaped      '''
        # # Which experiments to analyze
        # 'experiments': ['Circle wall down light (U shaped)', ['Circle wall (11 shaped)', 'Circle wall down light (U shaped)']],  #
        # # Which conditions to analyze
        # 'conditions': ['obstacle', ['no obstacle', 'no obstacle']],
        # # what to call each condition for plots
        # 'labels': ['U shaped', '11 shaped']}

        # # '''    11-shaped      '''
        # Which experiments to analyze
        # 'experiments': ['Circle wall (11 shaped)', 'Circle wall up'],  #
        # # Which conditions to analyze
        # 'conditions': ['obstacle', 'no obstacle'],
        # # what to call each condition for plots
        # 'labels': ['11 shaped', 'open']}

        # # '''     exploration comparison      '''
        # # Which experiments to analyze
        # 'experiments': [['Circle wall up', 'Circle wall up (2)', 'Circle wall down'], ['Circle wall down', 'Circle lights on off (baseline)','Circle wall up', 'Circle wall up (2)'],
        #                 'Circle void up', 'Circle wall (no shelter)', 'Circle void (no shelter)', 'Circle (no shelter)'],
        # # Which conditions to analyze
        # 'conditions': [['no obstacle', 'no obstacle', 'no obstacle'], ['obstacle', 'obstacle','obstacle', 'obstacle'],'obstacle','obstacle','obstacle', 'no obstacle'],
        # # what to call each condition for plots
        # 'labels': ['Open field', 'Obstacle', 'Hole obstacle', 'Wall no shelter','Hole no shelter','Empty']}

        # # # '''     hole obstacle       '''
        # # Which experiments to analyze
        # 'experiments': ['Circle void up'], #['Circle wall down', 'Circle lights on off (baseline)']], #
        # # Which conditions to analyze
        # 'conditions': ['obstacle'], #['obstacle', 'obstacle']],
        # # what to call each condition for plots
        # 'labels': ['Hole obstacle']}#, 'Wall obstacle']}

        # # '''     Lights on -> off       '''
        # # Which experiments to analyze
        # 'experiments': ['Circle wall lights on off quick', 'Circle lights on off (baseline)'],
        # # Which conditions to analyze
        # 'conditions': ['obstacle', 'no obstacle'],
        # # what to call each condition for plots
        # 'labels': ['lights on off (no baseline)','lights on off (baseline)']}

        # # '''     no obstacle/wall up       '''
        # # Which experiments to analyze
        # 'experiments': [['Circle wall up', 'Circle wall up (2)'],['Circle wall up', 'Circle wall up (2)'],['Circle wall up', 'Circle wall up (2)']], #
        # # Which conditions to analyze
        # 'conditions': [['probe', 'probe'],['obstacle', 'obstacle'], ['no obstacle', 'no obstacle']],
        # # # what to call each condition for plots
        # 'labels': ['wall up', 'obstacle added', 'open field']} #''Open field (naive)', 'Open field (exp)']}

        # '''     square wall moves left       '''
        # # Which experiments to analyze
        # 'experiments': [ 'Square wall moves left', ['Square wall moves right', 'Square wall left'], 'Square wall moves left', 'Square wall moves right'], #
        # # Which conditions to analyze
        # 'conditions': ['obstacle', ['no obstacle', 'no obstacle'], 'no obstacle', 'obstacle'], #
        # # what to call each condition for plots
        # 'labels': ['wall shortened', 'obstacle short', 'obstacle long', 'wall lengthened']} #

        # '''     food expts        '''
        # Which experiments to analyze
        # 'experiments': [['Circle food', 'Circle food wall up'],'Circle food wall down'], #,  'Circle food wall down'],
        # # Which conditions to analyze
        # 'conditions': [['no obstacle', 'no obstacle'], 'no obstacle'], #, 'obstacle'],
        # # what to call each condition for plots
        # 'labels': ['Food open field', 'Food wall down']} #,  'Food obstacle']}

        # # # '''     food wall down expts        '''
        # # Which experiments to analyze
        # 'experiments': ['Circle food wall down'],
        # # Which conditions to analyze
        # 'conditions': ['no obstacle'],
        # # what to call each condition for plots
        # 'labels': ['Food wall down']}

        # # # '''     food vs escape        '''
        # # Which experiments to analyze
        # 'experiments': [['Circle food', 'Circle food wall up', 'Circle food wall down', 'Circle food wall down'],
        #                 ['Circle wall up', 'Circle wall down (no shelter)', 'Circle wall down', 'Circle wall down (no baseline)',
        #                 'Circle wall down', 'Circle lights on off (baseline)', 'Circle wall up (2)', 'Circle wall down (no baseline no naive)']],
        # # Which conditions to analyze
        # 'conditions': [['no obstacle', 'no obstacle', 'no obstacle', 'obstacle'],
        #                ['no obstacle',  'no obstacle', 'no obstacle', 'no obstacle','obstacle', 'obstacle', 'no obstacle', 'no obstacle']],
        # # what to call each condition for plots
        # 'labels': ['Food', 'Escape']}

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
        # 'experiments': [['Circle wall up'], ['Circle wall down', 'Circle wall down (no baseline)'], ['Circle wall down', 'Circle lights on off (baseline)']],  #
        # # # Which conditions to analyze
        # 'conditions': [['no obstacle'],  ['no obstacle', 'no obstacle'], ['obstacle', 'obstacle']], #, 'no obstacle'
        # # # what to call each condition for plots
        # 'labels': ['Open field escape',  'Obstacle removed escape', 'Obstacle escape']}

        # # '''     spontaneous edge vector comparison     '''
        # # Which experiments to analyze
        # 'experiments': ['Circle wall up', ['Circle wall down (no shelter)', 'Circle wall (no shelter)'],
        #                 ['Circle wall down', 'Circle lights on off (baseline)']],  #
        # # # Which conditions to analyze
        # 'conditions': ['no obstacle', ['obstacle', 'obstacle'], ['obstacle', 'obstacle']],
        # # # what to call each condition for plots
        # 'labels': ['open field', 'obstacle (no shelter)', 'obstacle']}

        # # '''     spontaneous edge vector comparison II    '''
        # # Which experiments to analyze
        # 'experiments': ['Circle wall up', 'Circle void up', ['Circle wall down', 'Circle lights on off (baseline)']],  #
        # # # Which conditions to analyze
        # 'conditions': ['no obstacle', 'obstacle', ['obstacle', 'obstacle']],
        # # # what to call each condition for plots
        # 'labels': ['open field', 'hole obstacle', 'obstacle']}


        # '''     obstacle no shelter     '''
        # # Which experiments to analyze
        # 'experiments': ['Circle wall (no shelter)'],  #
        # # # Which conditions to analyze
        # 'conditions': [ 'obstacle'],
        # # # what to call each condition for plots
        # 'labels': ['obstacle (no shelter)']}




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
