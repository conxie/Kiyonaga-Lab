#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
This experiment was created using PsychoPy3 Experiment Builder (v2024.2.4),
    on May 27, 2025, at 19:30
If you publish work using this script the most relevant publication is:

    Peirce J, Gray JR, Simpson S, MacAskill M, Höchenberger R, Sogo H, Kastman E, Lindeløv JK. (2019) 
        PsychoPy2: Experiments in behavior made easy Behav Res 51: 195. 
        https://doi.org/10.3758/s13428-018-01193-y

"""

# --- Import packages ---
from psychopy import locale_setup
from psychopy import prefs
from psychopy import plugins
plugins.activatePlugins()
from psychopy import sound, gui, visual, core, data, event, logging, clock, colors, layout, hardware
from psychopy.tools import environmenttools
from psychopy.constants import (NOT_STARTED, STARTED, PLAYING, PAUSED,
                                STOPPED, FINISHED, PRESSED, RELEASED, FOREVER, priority)

import numpy as np  # whole numpy lib is available, prepend 'np.'
from numpy import (sin, cos, tan, log, log10, pi, average,
                   sqrt, std, deg2rad, rad2deg, linspace, asarray)
from numpy.random import random, randint, normal, shuffle, choice as randchoice
import os  # handy system and path functions
import sys  # to get file system encoding

import psychopy.iohub as io
from psychopy.hardware import keyboard

# Run 'Before Experiment' code from stimParams
import numpy as np  # whole numpy lib is available, prepend 'np.'
import pandas as pd
import random
import os  # handy system and path functions
import sys  # to get file system encoding
import glob


# Run 'Before Experiment' code from connectTracker
import platform
from PIL import Image  # for preparing the Host backdrop image
from string import ascii_letters
import time

# import eyelink libs
import pylink
from EyeLinkCoreGraphicsPsychoPy import EyeLinkCoreGraphicsPsychoPy


# --- Setup global variables (available in all functions) ---
# create a device manager to handle hardware (keyboards, mice, mirophones, speakers, etc.)
deviceManager = hardware.DeviceManager()
# ensure that relative paths start from the same directory as this script
_thisDir = os.path.dirname(os.path.abspath(__file__))
# store info about the experiment session
psychopyVersion = '2024.2.4'
expName = 'psychedeLights_temp'  # from the Builder filename that created this script
# information about this experiment
expInfo = {
    'participant': '',
    'session': '001',
    'date|hid': data.getDateStr(),
    'expName|hid': expName,
    'psychopyVersion|hid': psychopyVersion,
}

# --- Define some variables which will change depending on pilot mode ---
'''
To run in pilot mode, either use the run/pilot toggle in Builder, Coder and Runner, 
or run the experiment with `--pilot` as an argument. To change what pilot 
#mode does, check out the 'Pilot mode' tab in preferences.
'''
# work out from system args whether we are running in pilot mode
PILOTING = core.setPilotModeFromArgs()
# start off with values from experiment settings
_fullScr = True
_winSize = [1920, 1080]
# if in pilot mode, apply overrides according to preferences
if PILOTING:
    # force windowed mode
    if prefs.piloting['forceWindowed']:
        _fullScr = False
        # set window size
        _winSize = prefs.piloting['forcedWindowSize']

def showExpInfoDlg(expInfo):
    """
    Show participant info dialog.
    Parameters
    ==========
    expInfo : dict
        Information about this experiment.
    
    Returns
    ==========
    dict
        Information about this experiment.
    """
    # show participant info dialog
    dlg = gui.DlgFromDict(
        dictionary=expInfo, sortKeys=False, title=expName, alwaysOnTop=True
    )
    if dlg.OK == False:
        core.quit()  # user pressed cancel
    # return expInfo
    return expInfo


def setupData(expInfo, dataDir=None):
    """
    Make an ExperimentHandler to handle trials and saving.
    
    Parameters
    ==========
    expInfo : dict
        Information about this experiment, created by the `setupExpInfo` function.
    dataDir : Path, str or None
        Folder to save the data to, leave as None to create a folder in the current directory.    
    Returns
    ==========
    psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    """
    # remove dialog-specific syntax from expInfo
    for key, val in expInfo.copy().items():
        newKey, _ = data.utils.parsePipeSyntax(key)
        expInfo[newKey] = expInfo.pop(key)
    
    # data file name stem = absolute path + name; later add .psyexp, .csv, .log, etc
    if dataDir is None:
        dataDir = _thisDir
    filename = u'data/%s_%s_%s' % (expInfo['participant'], expName, expInfo['date'])
    # make sure filename is relative to dataDir
    if os.path.isabs(filename):
        dataDir = os.path.commonprefix([dataDir, filename])
        filename = os.path.relpath(filename, dataDir)
    
    # an ExperimentHandler isn't essential but helps with data saving
    thisExp = data.ExperimentHandler(
        name=expName, version='',
        extraInfo=expInfo, runtimeInfo=None,
        originPath='C:\\Users\\cocon\\OneDrive\\Documents\\GitHub\\Kiyonaga-Lab\\PNR Project\\psychedeLights_abstractness_updatedStim_lastrun.py',
        savePickle=True, saveWideText=True,
        dataFileName=dataDir + os.sep + filename, sortColumns='time'
    )
    thisExp.setPriority('thisRow.t', priority.CRITICAL)
    thisExp.setPriority('expName', priority.LOW)
    # return experiment handler
    return thisExp


def setupLogging(filename):
    """
    Setup a log file and tell it what level to log at.
    
    Parameters
    ==========
    filename : str or pathlib.Path
        Filename to save log file and data files as, doesn't need an extension.
    
    Returns
    ==========
    psychopy.logging.LogFile
        Text stream to receive inputs from the logging system.
    """
    # set how much information should be printed to the console / app
    if PILOTING:
        logging.console.setLevel(
            prefs.piloting['pilotConsoleLoggingLevel']
        )
    else:
        logging.console.setLevel('warning')
    # save a log file for detail verbose info
    logFile = logging.LogFile(filename+'.log')
    if PILOTING:
        logFile.setLevel(
            prefs.piloting['pilotLoggingLevel']
        )
    else:
        logFile.setLevel(
            logging.getLevel('error')
        )
    
    return logFile


def setupWindow(expInfo=None, win=None):
    """
    Setup the Window
    
    Parameters
    ==========
    expInfo : dict
        Information about this experiment, created by the `setupExpInfo` function.
    win : psychopy.visual.Window
        Window to setup - leave as None to create a new window.
    
    Returns
    ==========
    psychopy.visual.Window
        Window in which to run this experiment.
    """
    if PILOTING:
        logging.debug('Fullscreen settings ignored as running in pilot mode.')
    
    if win is None:
        # if not given a window to setup, make one
        win = visual.Window(
            size=_winSize, fullscr=_fullScr, screen=0,
            winType='pyglet', allowGUI=False, allowStencil=False,
            monitor='testMonitor', color=[0,0,0], colorSpace='rgb',
            backgroundImage='', backgroundFit='none',
            blendMode='avg', useFBO=True,
            units='height',
            checkTiming=False  # we're going to do this ourselves in a moment
        )
    else:
        # if we have a window, just set the attributes which are safe to set
        win.color = [0,0,0]
        win.colorSpace = 'rgb'
        win.backgroundImage = ''
        win.backgroundFit = 'none'
        win.units = 'height'
    if expInfo is not None:
        # get/measure frame rate if not already in expInfo
        if win._monitorFrameRate is None:
            win._monitorFrameRate = win.getActualFrameRate(infoMsg='Attempting to measure frame rate of screen, please wait...')
        expInfo['frameRate'] = win._monitorFrameRate
    win.hideMessage()
    # show a visual indicator if we're in piloting mode
    if PILOTING and prefs.piloting['showPilotingIndicator']:
        win.showPilotingIndicator()
    
    return win


def setupDevices(expInfo, thisExp, win):
    """
    Setup whatever devices are available (mouse, keyboard, speaker, eyetracker, etc.) and add them to 
    the device manager (deviceManager)
    
    Parameters
    ==========
    expInfo : dict
        Information about this experiment, created by the `setupExpInfo` function.
    thisExp : psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    win : psychopy.visual.Window
        Window in which to run this experiment.
    Returns
    ==========
    bool
        True if completed successfully.
    """
    # --- Setup input devices ---
    ioConfig = {}
    
    # Setup iohub keyboard
    ioConfig['Keyboard'] = dict(use_keymap='psychopy')
    
    # Setup iohub experiment
    ioConfig['Experiment'] = dict(filename=thisExp.dataFileName)
    
    # Start ioHub server
    ioServer = io.launchHubServer(window=win, **ioConfig)
    
    # store ioServer object in the device manager
    deviceManager.ioServer = ioServer
    
    # create a default keyboard (e.g. to check for escape)
    if deviceManager.getDevice('defaultKeyboard') is None:
        deviceManager.addDevice(
            deviceClass='keyboard', deviceName='defaultKeyboard', backend='iohub'
        )
    if deviceManager.getDevice('ibiSpace') is None:
        # initialise ibiSpace
        ibiSpace = deviceManager.addDevice(
            deviceClass='keyboard',
            deviceName='ibiSpace',
        )
    # return True if completed successfully
    return True

def pauseExperiment(thisExp, win=None, timers=[], playbackComponents=[]):
    """
    Pause this experiment, preventing the flow from advancing to the next routine until resumed.
    
    Parameters
    ==========
    thisExp : psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    win : psychopy.visual.Window
        Window for this experiment.
    timers : list, tuple
        List of timers to reset once pausing is finished.
    playbackComponents : list, tuple
        List of any components with a `pause` method which need to be paused.
    """
    # if we are not paused, do nothing
    if thisExp.status != PAUSED:
        return
    
    # start a timer to figure out how long we're paused for
    pauseTimer = core.Clock()
    # pause any playback components
    for comp in playbackComponents:
        comp.pause()
    # make sure we have a keyboard
    defaultKeyboard = deviceManager.getDevice('defaultKeyboard')
    if defaultKeyboard is None:
        defaultKeyboard = deviceManager.addKeyboard(
            deviceClass='keyboard',
            deviceName='defaultKeyboard',
            backend='ioHub',
        )
    # run a while loop while we wait to unpause
    while thisExp.status == PAUSED:
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=['escape']):
            endExperiment(thisExp, win=win)
        # sleep 1ms so other threads can execute
        clock.time.sleep(0.001)
    # if stop was requested while paused, quit
    if thisExp.status == FINISHED:
        endExperiment(thisExp, win=win)
    # resume any playback components
    for comp in playbackComponents:
        comp.play()
    # reset any timers
    for timer in timers:
        timer.addTime(-pauseTimer.getTime())


def run(expInfo, thisExp, win, globalClock=None, thisSession=None):
    """
    Run the experiment flow.
    
    Parameters
    ==========
    expInfo : dict
        Information about this experiment, created by the `setupExpInfo` function.
    thisExp : psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    psychopy.visual.Window
        Window in which to run this experiment.
    globalClock : psychopy.core.clock.Clock or None
        Clock to get global time from - supply None to make a new one.
    thisSession : psychopy.session.Session or None
        Handle of the Session object this experiment is being run from, if any.
    """
    # mark experiment as started
    thisExp.status = STARTED
    # make sure window is set to foreground to prevent losing focus
    win.winHandle.activate()
    # make sure variables created by exec are available globally
    exec = environmenttools.setExecEnvironment(globals())
    # get device handles from dict of input devices
    ioServer = deviceManager.ioServer
    # get/create a default keyboard (e.g. to check for escape)
    defaultKeyboard = deviceManager.getDevice('defaultKeyboard')
    if defaultKeyboard is None:
        deviceManager.addDevice(
            deviceClass='keyboard', deviceName='defaultKeyboard', backend='ioHub'
        )
    eyetracker = deviceManager.getDevice('eyetracker')
    # make sure we're running in the directory for this experiment
    os.chdir(_thisDir)
    # get filename from ExperimentHandler for convenience
    filename = thisExp.dataFileName
    frameTolerance = 0.001  # how close to onset before 'same' frame
    endExpNow = False  # flag for 'escape' or other condition => quit the exp
    # get frame duration from frame rate in expInfo
    if 'frameRate' in expInfo and expInfo['frameRate'] is not None:
        frameDur = 1.0 / round(expInfo['frameRate'])
    else:
        frameDur = 1.0 / 60.0  # could not measure, so guess
    
    # Start Code - component code to be run after the window creation
    
    # --- Initialize components for Routine "loadExpVar" ---
    # Run 'Begin Experiment' code from stimParams
    eyetracking = 0
    
    imageFilePath = ''
    
    """trial countdown related variables"""
    #category information
    uniqueCategories = ['shoppingcart', 'ferriswheel', 'circus', 'house', 'playground',
                        'bench', 'gasstation', 'schoolbus', 'airport']
    uniqueCategoriesCnt = len(uniqueCategories)
    
    imEachCat = 12 #how many images are in 1 category
    trlTotal = uniqueCategoriesCnt*imEachCat*2 #total trial = 9 categories * #images in each category *2 repetition
    trlEachBlk = 12 #how many trials are in each block
    blkTotal = int(trlTotal/trlEachBlk)
    
    #distractor trials perc
    distractorPerc = 0.3
    
    #example label
    distractorLabels = ['shopica', 'ferriwee', 'cicu', 'huse', 'paygund',
                        'benc', 'gatation', 'scolbu', 'irpo']
    distractorMapping = dict(zip(uniqueCategories,distractorLabels))
    
    #init counter
    trlCntTotal = 0
    
    """timing related params"""
    fixationT = 0.5
    stimT = 3
    retrocueT = 0.5
    noisePatchT = 0.25
    preCueRestT = 0.5
    delayT = 3
    noDistractorDelayT = 2
    distractorT = 1.5
    distractorRspT = 1.5
    probeT = 5
    itiT = 0.5
    
    """display relayed variables"""
    #init position/size related params
    leftImPos = [-0.2,0]
    rightImPos = [0.2,0]
    fixSize = np.array([0.12,0.12])#np.array([0.16,0.16])
    probeSize = [0.18,0.18]
    probePosVarX = 0.25
    probePosVarY = 0.12
    
    #other variables
    probePosArr = np.asarray([[-1*probePosVarX,probePosVarY],[0,probePosVarY],[probePosVarX,probePosVarY],
                              [-1*probePosVarX,-1*probePosVarY],[0,-1*probePosVarY],[probePosVarX,-1*probePosVarY],])
    
    textSize = 0.03
    visualProbeSizeScalar = 1 #np.sqrt(2)
    
    #Cue colors
    cueColArr = [[0,128,0],[128,0,128]] #green,purple
    cueColDefault = [128,115,96]
    
    
    #
    clock = core.Clock()
    kb = keyboard.Keyboard()
    
    # Run 'Begin Experiment' code from loadImages
    
    import newDistrTempGenConditionFile
    ouch = True
    while ouch:
        try:
            # Call the main function
            newDistrTempGenConditionFile.main()
            ouch = False
        except ValueError:
            print('b')
            continue
    
    df = pd.read_csv('updatedProbeTempCondFile.csv')
    
    
    # --- Initialize components for Routine "connectEL" ---
    # Run 'Begin Experiment' code from connectTracker
    if eyetracking == 0:
        # this is adapted from the SR research's eyelink tutorial code
        #parameters to change
        calib_style = 13 #9 for head fixed, 13 for remote 
        samprate = 1000 #250, 500, 1000, or 2000
        calib_tar_size = 24 #size for the calibration target
    
        """change this"""
        width_param = 53.0 
        distance_param =70.0
    
    
        #create a folder to store all edf files, call this folder 'results'
        edf_folder = 'C:/Users/yhung/Documents/elRaw_psychedeLights'
        if not os.path.exists(edf_folder):
            os.makedirs(edf_folder)
    
        # We download EDF data file from the EyeLink Host PC to the local hard
        # drive at the end of each testing session, here we rename the EDF to
        # include session start date/time
        time_str = time.strftime("_%Y_%m_%d_%H_%M", time.localtime())
        session_identifier = str(expInfo['participant']) + time_str
    
        # create a folder for the current testing session in the "results" folder
        session_folder = os.path.join(edf_folder, session_identifier)
        if not os.path.exists(session_folder):
            os.makedirs(session_folder)
    
    
        #helper function for displaying text
        def clear_screen(win):
            """ clear up the PsychoPy window"""
    
            win.fillColor = genv.getBackgroundColor()
            win.flip()
        def show_msg(win, text, wait_for_keypress=True):
            """ Show task instructions on screen"""
    
            msg = visual.TextStim(win, text,
                                  color=genv.getForegroundColor(),
                                  wrapWidth=scn_width/2)
            clear_screen(win)
            msg.draw()
            win.flip()
    
            # wait indefinitely, terminates upon any key press
            if wait_for_keypress:
                event.waitKeys(keyList = ['space','escape'],maxWait = 60)
                clear_screen(win)
             
        #function to terminate task and retrieve the EDF data file from the host PC and 
        #download to the display pc
        def terminate_task():
            el_tracker = pylink.getEYELINK()
    
            if el_tracker.isConnected():
    
                # Put tracker in Offline mode
                el_tracker.setOfflineMode()
    
                # Clear the Host PC screen and wait for 500 ms
                el_tracker.sendCommand('clear_screen 0')
                pylink.msecDelay(500)
    
                # Close the edf data file on the Host
                el_tracker.closeDataFile()
    
                # Show a file transfer message on the screen
                msg = 'EDF data is transferring from EyeLink Host PC...'
                show_msg(win, msg, wait_for_keypress=False)
    
                # Download the EDF data file from the Host PC to a local data folder
                # parameters: source_file_on_the_host, destination_file_on_local_drive
                local_edf = os.path.join(session_folder, session_identifier + '.EDF')
                try:
                    el_tracker.receiveDataFile(edf_file, local_edf)
                except RuntimeError as error:
                    print('ERROR:', error)
    
                # Close the link to the tracker.
                el_tracker.close()
    
            # close the PsychoPy window
            win.close()
    
            # quit PsychoPy
            core.quit()
            sys.exit()
            
        def abort_trial():
            """Ends recording """
    
            el_tracker = pylink.getEYELINK()
    
            # Stop recording
            if el_tracker.isRecording():
                # add 100 ms to catch final trial events
                pylink.pumpDelay(100)
                el_tracker.stopRecording()
    
            # clear the screen
            clear_screen(win)
            # Send a message to clear the Data Viewer screen
            bgcolor_RGB = (116, 116, 116)
            el_tracker.sendMessage('!V CLEAR %d %d %d' % bgcolor_RGB)
    
            # send a message to mark trial end
            el_tracker.sendMessage('TRIAL_RESULT %d' % pylink.TRIAL_ERROR)
    
    
    # --- Initialize components for Routine "blkInit" ---
    
    # --- Initialize components for Routine "drift_check" ---
    
    # --- Initialize components for Routine "trlInit" ---
    
    # --- Initialize components for Routine "fixationRest" ---
    
    # --- Initialize components for Routine "stim" ---
    
    # --- Initialize components for Routine "preCueRest" ---
    
    # --- Initialize components for Routine "retrocue" ---
    
    # --- Initialize components for Routine "delayNoDistractor" ---
    
    # --- Initialize components for Routine "distractor" ---
    
    # --- Initialize components for Routine "distractorResponse" ---
    
    # --- Initialize components for Routine "probe" ---
    transparentPlaceHolder = visual.ShapeStim(
        win=win, name='transparentPlaceHolder',
        size=(0.5, 0.5), vertices='triangle',
        ori=0.0, pos=(0, 0), draggable=False, anchor='center',
        lineWidth=1.0,
        colorSpace='rgb', lineColor='white', fillColor=[0.0000, 0.0000, 0.0000],
        opacity=0.0, depth=-1.0, interpolate=True)
    # Run 'Begin Experiment' code from drawProbe
    probeMouse = event.Mouse(newPos = (0,0))
    
    
    # --- Initialize components for Routine "ITI" ---
    
    # --- Initialize components for Routine "IBI" ---
    text = visual.TextStim(win=win, name='text',
        text="you've reached the end of this block, press space when you're ready to continue",
        font='Open Sans',
        pos=(0, 0), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    ibiSpace = keyboard.Keyboard(deviceName='ibiSpace')
    
    # --- Initialize components for Routine "terminateExp" ---
    
    # create some handy timers
    
    # global clock to track the time since experiment started
    if globalClock is None:
        # create a clock if not given one
        globalClock = core.Clock()
    if isinstance(globalClock, str):
        # if given a string, make a clock accoridng to it
        if globalClock == 'float':
            # get timestamps as a simple value
            globalClock = core.Clock(format='float')
        elif globalClock == 'iso':
            # get timestamps in ISO format
            globalClock = core.Clock(format='%Y-%m-%d_%H:%M:%S.%f%z')
        else:
            # get timestamps in a custom format
            globalClock = core.Clock(format=globalClock)
    if ioServer is not None:
        ioServer.syncClock(globalClock)
    logging.setDefaultClock(globalClock)
    # routine timer to track time remaining of each (possibly non-slip) routine
    routineTimer = core.Clock()
    win.flip()  # flip window to reset last flip timer
    # store the exact time the global clock started
    expInfo['expStart'] = data.getDateStr(
        format='%Y-%m-%d %Hh%M.%S.%f %z', fractionalSecondDigits=6
    )
    
    # --- Prepare to start Routine "loadExpVar" ---
    # create an object to store info about Routine loadExpVar
    loadExpVar = data.Routine(
        name='loadExpVar',
        components=[],
    )
    loadExpVar.status = NOT_STARTED
    continueRoutine = True
    # update component parameters for each repeat
    # Run 'Begin Routine' code from loadImages
    """Create visual objects"""
    #create fixation circle and retrocue
    #fixation cross
    fixCircMask = visual.ImageStim(win,image = imageFilePath + 'visualObj/fixCirc.png',size= fixSize)
    retrocueMask = visual.ImageStim(win,image = imageFilePath + 'visualObj/retrocue.png',size= fixSize)
    #color
    cueCircColor = visual.Polygon(win,edges = 4,radius=1,ori = 45,colorSpace = 'rgb255',fillColor=cueColDefault,size= fixSize/2)
    #create memory items
    imLeft = visual.ImageStim(win, )
    imRight = visual.ImageStim(win, )
    imRight.setPos(rightImPos)
    imLeft.setPos(leftImPos)
    
    #create noise patch
    noisePatchLeft = visual.ImageStim(win,image = imageFilePath + 'visualObj/noiseGauss50.png')
    noisePatchRight = visual.ImageStim(win,image = imageFilePath + 'visualObj/noiseGauss50.png')
    noisePatchLeft.setPos(leftImPos)
    noisePatchRight.setPos(rightImPos)
    
    #create the probestims
    #visual condition
    probeArr = np.asarray([visual.ImageStim(win,image = None,size = probeSize) for i in range(6)])
    
    #create distractor
    visualDistractor = visual.ImageStim(win,image = None)
    verbalDistractor = visual.TextStim(win,text = '',height = textSize, bold=True)
    instrTxt = visual.TextStim(win=win, name='instrTxt',
            text="",
            font='Open Sans',
            pos=(0, 0.2), height=0.03, wrapWidth=None, ori=0.0, 
            color='white', colorSpace='rgb', opacity=0.6, 
            languageStyle='LTR',
            depth=0.0);
    # store start times for loadExpVar
    loadExpVar.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
    loadExpVar.tStart = globalClock.getTime(format='float')
    loadExpVar.status = STARTED
    thisExp.addData('loadExpVar.started', loadExpVar.tStart)
    loadExpVar.maxDuration = None
    # keep track of which components have finished
    loadExpVarComponents = loadExpVar.components
    for thisComponent in loadExpVar.components:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "loadExpVar" ---
    loadExpVar.forceEnded = routineForceEnded = not continueRoutine
    while continueRoutine:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, win=win)
            return
        # pause experiment here if requested
        if thisExp.status == PAUSED:
            pauseExperiment(
                thisExp=thisExp, 
                win=win, 
                timers=[routineTimer], 
                playbackComponents=[]
            )
            # skip the frame we paused on
            continue
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            loadExpVar.forceEnded = routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in loadExpVar.components:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "loadExpVar" ---
    for thisComponent in loadExpVar.components:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    # store stop times for loadExpVar
    loadExpVar.tStop = globalClock.getTime(format='float')
    loadExpVar.tStopRefresh = tThisFlipGlobal
    thisExp.addData('loadExpVar.stopped', loadExpVar.tStop)
    thisExp.nextEntry()
    # the Routine "loadExpVar" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # --- Prepare to start Routine "connectEL" ---
    # create an object to store info about Routine connectEL
    connectEL = data.Routine(
        name='connectEL',
        components=[],
    )
    connectEL.status = NOT_STARTED
    continueRoutine = True
    # update component parameters for each repeat
    # Run 'Begin Routine' code from connectTracker
    # Step 1: Connect to the EyeLink Host PC
    host_ip = "100.1.1.1"
    if eyetracking == 1:
        try:
            el_tracker = pylink.EyeLink(host_ip)
        except RuntimeError as error:
            print('ERROR:', error)
            core.quit()
            sys.exit()
        
        # Step 2: Open an EDF data file on the Host PC
        edf_file = str(expInfo['participant']) + ".EDF"
        try:
            el_tracker.openDataFile(edf_file)
        except RuntimeError as err:
            print('ERROR:', err)
            # close the link if we have one open
            if el_tracker.isConnected():
                el_tracker.close()
            core.quit()
            sys.exit()
    
        # Step 3: Configure the tracker
        # Put the tracker in offline mode before we change tracking parameters
        el_tracker.setOfflineMode()
        # File and Link data control
        # what eye events to save in the EDF file, include everything by default
        file_event_flags = 'LEFT,RIGHT,FIXATION,SACCADE,BLINK,MESSAGE,BUTTON,INPUT'
        file_sample_flags = 'LEFT,RIGHT,GAZE,GAZERES,HREF,RAW,PUPIL,AREA,HTARGET,STATUS,INPUT'
        # what eye events to make available over the link, include everything by default
        link_event_flags = 'LEFT,RIGHT,FIXATION,SACCADE,BLINK,BUTTON,FIXUPDATE,INPUT'
        link_sample_flags = 'LEFT,RIGHT,GAZE,GAZERES,AREA,HTARGET,STATUS,INPUT'
        el_tracker.sendCommand("file_event_filter = %s" % file_event_flags)
        el_tracker.sendCommand("file_sample_data = %s" % file_sample_flags)
        el_tracker.sendCommand("link_event_filter = %s" % link_event_flags)
        el_tracker.sendCommand("link_sample_data = %s" % link_sample_flags)
        #set calibration style
        el_tracker.sendCommand("calibration_type = HV%s" % str(calib_style))
        #set sampling rate
        el_tracker.sendCommand("sample_rate %s" % str(samprate))
     
        # Step 4: set up a graphics environment for calibration
        # get the native screen resolution used by PsychoPy
        scn_width, scn_height = win.size
        # resolution fix for Mac retina displays
        if 'Darwin' in platform.system():
            if use_retina:
                scn_width = int(scn_width/2.0)
                scn_height = int(scn_height/2.0)
        # Pass the display pixel coordinates (left, top, right, bottom) to the tracker
        el_coords = "screen_pixel_coords = 0 0 %d %d" % (scn_width - 1, scn_height - 1)
        el_tracker.sendCommand(el_coords)
        # Write a DISPLAY_COORDS message to the EDF file
        # Data Viewer needs this piece of info for proper visualization
        dv_coords = "DISPLAY_COORDS  0 0 %d %d" % (scn_width - 1, scn_height - 1)
        el_tracker.sendMessage(dv_coords)
        # Configure a graphics environment (genv) for tracker calibration
        genv = EyeLinkCoreGraphicsPsychoPy(el_tracker, win)
        # Set background and foreground colors for the calibration target
        # in PsychoPy, (-1, -1, -1)=black, (1, 1, 1)=white, (0, 0, 0)=mid-gray
        foreground_color = (-1, -1, -1)
        background_color = win.color
        genv.setCalibrationColors(foreground_color, background_color)
        genv.setTargetSize(calib_tar_size)
        # Request Pylink to use the PsychoPy window we opened above for calibration
        pylink.openGraphicsEx(genv)
        task_msg = 'Press <space>, then <enter> to start calibration'
        show_msg(win, task_msg)
        print('line69')
        
        try:
            el_tracker.doTrackerSetup()
        except RuntimeError as err:
            print('ERROR:', err)
            el_tracker.exitCalibration()
    else:
        continueRoutine = False
    
    # store start times for connectEL
    connectEL.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
    connectEL.tStart = globalClock.getTime(format='float')
    connectEL.status = STARTED
    thisExp.addData('connectEL.started', connectEL.tStart)
    connectEL.maxDuration = None
    # keep track of which components have finished
    connectELComponents = connectEL.components
    for thisComponent in connectEL.components:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "connectEL" ---
    connectEL.forceEnded = routineForceEnded = not continueRoutine
    while continueRoutine:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, win=win)
            return
        # pause experiment here if requested
        if thisExp.status == PAUSED:
            pauseExperiment(
                thisExp=thisExp, 
                win=win, 
                timers=[routineTimer], 
                playbackComponents=[]
            )
            # skip the frame we paused on
            continue
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            connectEL.forceEnded = routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in connectEL.components:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "connectEL" ---
    for thisComponent in connectEL.components:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    # store stop times for connectEL
    connectEL.tStop = globalClock.getTime(format='float')
    connectEL.tStopRefresh = tThisFlipGlobal
    thisExp.addData('connectEL.stopped', connectEL.tStop)
    # Run 'End Routine' code from connectTracker
    #dont record yet
    if eyetracking == 1:
        el_tracker.setOfflineMode()
    win.mouseVisible = True
    
    thisExp.nextEntry()
    # the Routine "connectEL" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # set up handler to look after randomisation of conditions etc
    expBlk = data.TrialHandler2(
        name='expBlk',
        nReps=blkTotal, 
        method='random', 
        extraInfo=expInfo, 
        originPath=-1, 
        trialList=[None], 
        seed=None, 
    )
    thisExp.addLoop(expBlk)  # add the loop to the experiment
    thisExpBlk = expBlk.trialList[0]  # so we can initialise stimuli with some values
    # abbreviate parameter names if possible (e.g. rgb = thisExpBlk.rgb)
    if thisExpBlk != None:
        for paramName in thisExpBlk:
            globals()[paramName] = thisExpBlk[paramName]
    
    for thisExpBlk in expBlk:
        currentLoop = expBlk
        thisExp.timestampOnFlip(win, 'thisRow.t', format=globalClock.format)
        # abbreviate parameter names if possible (e.g. rgb = thisExpBlk.rgb)
        if thisExpBlk != None:
            for paramName in thisExpBlk:
                globals()[paramName] = thisExpBlk[paramName]
        
        # --- Prepare to start Routine "blkInit" ---
        # create an object to store info about Routine blkInit
        blkInit = data.Routine(
            name='blkInit',
            components=[],
        )
        blkInit.status = NOT_STARTED
        continueRoutine = True
        # update component parameters for each repeat
        # Run 'Begin Routine' code from initBlkParams
        trlCntThisBlk = 0
        # store start times for blkInit
        blkInit.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
        blkInit.tStart = globalClock.getTime(format='float')
        blkInit.status = STARTED
        thisExp.addData('blkInit.started', blkInit.tStart)
        blkInit.maxDuration = None
        # keep track of which components have finished
        blkInitComponents = blkInit.components
        for thisComponent in blkInit.components:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "blkInit" ---
        # if trial has changed, end Routine now
        if isinstance(expBlk, data.TrialHandler2) and thisExpBlk.thisN != expBlk.thisTrial.thisN:
            continueRoutine = False
        blkInit.forceEnded = routineForceEnded = not continueRoutine
        while continueRoutine:
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, win=win)
                return
            # pause experiment here if requested
            if thisExp.status == PAUSED:
                pauseExperiment(
                    thisExp=thisExp, 
                    win=win, 
                    timers=[routineTimer], 
                    playbackComponents=[]
                )
                # skip the frame we paused on
                continue
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                blkInit.forceEnded = routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in blkInit.components:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "blkInit" ---
        for thisComponent in blkInit.components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        # store stop times for blkInit
        blkInit.tStop = globalClock.getTime(format='float')
        blkInit.tStopRefresh = tThisFlipGlobal
        thisExp.addData('blkInit.stopped', blkInit.tStop)
        # the Routine "blkInit" was not non-slip safe, so reset the non-slip timer
        routineTimer.reset()
        
        # --- Prepare to start Routine "drift_check" ---
        # create an object to store info about Routine drift_check
        drift_check = data.Routine(
            name='drift_check',
            components=[],
        )
        drift_check.status = NOT_STARTED
        continueRoutine = True
        # update component parameters for each repeat
        # Run 'Begin Routine' code from drift_check_code
        if eyetracking == 1:
            # drift-check and re-do camera setup if ESCAPE is pressed
            try:
                error = el_tracker.doDriftCorrect(int(scn_width/2.0),
                                                  int(scn_height/2.0), 1, 1)
                # break following a success drift-check
            except:
                continue
            # put tracker in idle/offline mode before recording
            el_tracker.setOfflineMode()
        # store start times for drift_check
        drift_check.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
        drift_check.tStart = globalClock.getTime(format='float')
        drift_check.status = STARTED
        thisExp.addData('drift_check.started', drift_check.tStart)
        drift_check.maxDuration = None
        # keep track of which components have finished
        drift_checkComponents = drift_check.components
        for thisComponent in drift_check.components:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "drift_check" ---
        # if trial has changed, end Routine now
        if isinstance(expBlk, data.TrialHandler2) and thisExpBlk.thisN != expBlk.thisTrial.thisN:
            continueRoutine = False
        drift_check.forceEnded = routineForceEnded = not continueRoutine
        while continueRoutine:
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, win=win)
                return
            # pause experiment here if requested
            if thisExp.status == PAUSED:
                pauseExperiment(
                    thisExp=thisExp, 
                    win=win, 
                    timers=[routineTimer], 
                    playbackComponents=[]
                )
                # skip the frame we paused on
                continue
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                drift_check.forceEnded = routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in drift_check.components:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "drift_check" ---
        for thisComponent in drift_check.components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        # store stop times for drift_check
        drift_check.tStop = globalClock.getTime(format='float')
        drift_check.tStopRefresh = tThisFlipGlobal
        thisExp.addData('drift_check.stopped', drift_check.tStop)
        # the Routine "drift_check" was not non-slip safe, so reset the non-slip timer
        routineTimer.reset()
        
        # set up handler to look after randomisation of conditions etc
        trials = data.TrialHandler2(
            name='trials',
            nReps=trlEachBlk, 
            method='random', 
            extraInfo=expInfo, 
            originPath=-1, 
            trialList=[None], 
            seed=None, 
        )
        thisExp.addLoop(trials)  # add the loop to the experiment
        thisTrial = trials.trialList[0]  # so we can initialise stimuli with some values
        # abbreviate parameter names if possible (e.g. rgb = thisTrial.rgb)
        if thisTrial != None:
            for paramName in thisTrial:
                globals()[paramName] = thisTrial[paramName]
        if thisSession is not None:
            # if running in a Session with a Liaison client, send data up to now
            thisSession.sendExperimentData()
        
        for thisTrial in trials:
            currentLoop = trials
            thisExp.timestampOnFlip(win, 'thisRow.t', format=globalClock.format)
            if thisSession is not None:
                # if running in a Session with a Liaison client, send data up to now
                thisSession.sendExperimentData()
            # abbreviate parameter names if possible (e.g. rgb = thisTrial.rgb)
            if thisTrial != None:
                for paramName in thisTrial:
                    globals()[paramName] = thisTrial[paramName]
            
            # --- Prepare to start Routine "trlInit" ---
            # create an object to store info about Routine trlInit
            trlInit = data.Routine(
                name='trlInit',
                components=[],
            )
            trlInit.status = NOT_STARTED
            continueRoutine = True
            # update component parameters for each repeat
            # Run 'Begin Routine' code from initTrlParams
            """create trial id"""
            trlId = (expInfo['participant'],trlCntThisBlk,expBlk.thisN)
            
            """cue related params"""
            cueCircColor.fillColor = cueColDefault
            if df['trlType'][trlCntTotal] == 'visual':
                cueCorThis = cueColArr[0] #green
            else:
                cueCorThis = cueColArr[1] #purple
            
            
            """distractor related params"""
            distractorType = None
            distractorCtrl = 1 #default add distractor
            visualDistractor.opacity = 0
            verbalDistractor.setOpacity(0)
            
            if df['distractorType'][trlCntTotal] == 'noDistractor': #no distactor
                distractorCtrl = 0
            else: #visual
                distractorCtrl = 1
                if df['trlType'][trlCntTotal] == 'visual' and  df['distractorType'][trlCntTotal] == 'congruent': #visual distractor:
                    instrTxt.setText('Scrambled?')
                    visualDistractor.setImage(df['distractorPath'][trlCntTotal].replace(os.sep, "/"))
                    visualDistractor.opacity = 1
                elif df['trlType'][trlCntTotal] == 'verbal' and  df['distractorType'][trlCntTotal] == 'incongruent': #visual distractor:
                    instrTxt.setText('Scrambled?')
                    visualDistractor.setImage(df['distractorPath'][trlCntTotal].replace(os.sep, "/"))
                    visualDistractor.opacity = 1
                elif df['trlType'][trlCntTotal] == 'visual' and  df['distractorType'][trlCntTotal] == 'incongruent': #verbal distractor:
                    instrTxt.setText('Manmade?')
                    verbalDistractor.setText(df['distractorPath'][trlCntTotal])
                    verbalDistractor.setOpacity(1)
                else:
                    instrTxt.setText('Manmade?')
                    verbalDistractor.setText(df['distractorPath'][trlCntTotal])
                    verbalDistractor.setOpacity(1)
                    
            """probe related params"""
            probePathThis = df[['probe1Path', 'probe2Path', 'probe3Path', 'probe4Path',
                   'probe5Path', 'probe6Path']].iloc[trlCntTotal].values
            clickableLabel = df[['probe1', 'probe2', 'probe3', 'probe4',
                   'probe5', 'probe6']].iloc[trlCntTotal].values
            for i in range(6):
               probeArr[i].setImage(probePathThis[i].replace(os.sep, "/"))
               probeArr[i].setPos(probePosArr[i])
            
            # store start times for trlInit
            trlInit.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
            trlInit.tStart = globalClock.getTime(format='float')
            trlInit.status = STARTED
            thisExp.addData('trlInit.started', trlInit.tStart)
            trlInit.maxDuration = None
            # keep track of which components have finished
            trlInitComponents = trlInit.components
            for thisComponent in trlInit.components:
                thisComponent.tStart = None
                thisComponent.tStop = None
                thisComponent.tStartRefresh = None
                thisComponent.tStopRefresh = None
                if hasattr(thisComponent, 'status'):
                    thisComponent.status = NOT_STARTED
            # reset timers
            t = 0
            _timeToFirstFrame = win.getFutureFlipTime(clock="now")
            frameN = -1
            
            # --- Run Routine "trlInit" ---
            # if trial has changed, end Routine now
            if isinstance(trials, data.TrialHandler2) and thisTrial.thisN != trials.thisTrial.thisN:
                continueRoutine = False
            trlInit.forceEnded = routineForceEnded = not continueRoutine
            while continueRoutine:
                # get current time
                t = routineTimer.getTime()
                tThisFlip = win.getFutureFlipTime(clock=routineTimer)
                tThisFlipGlobal = win.getFutureFlipTime(clock=None)
                frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
                # update/draw components on each frame
                
                # check for quit (typically the Esc key)
                if defaultKeyboard.getKeys(keyList=["escape"]):
                    thisExp.status = FINISHED
                if thisExp.status == FINISHED or endExpNow:
                    endExperiment(thisExp, win=win)
                    return
                # pause experiment here if requested
                if thisExp.status == PAUSED:
                    pauseExperiment(
                        thisExp=thisExp, 
                        win=win, 
                        timers=[routineTimer], 
                        playbackComponents=[]
                    )
                    # skip the frame we paused on
                    continue
                
                # check if all components have finished
                if not continueRoutine:  # a component has requested a forced-end of Routine
                    trlInit.forceEnded = routineForceEnded = True
                    break
                continueRoutine = False  # will revert to True if at least one component still running
                for thisComponent in trlInit.components:
                    if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                        continueRoutine = True
                        break  # at least one component has not yet finished
                
                # refresh the screen
                if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                    win.flip()
            
            # --- Ending Routine "trlInit" ---
            for thisComponent in trlInit.components:
                if hasattr(thisComponent, "setAutoDraw"):
                    thisComponent.setAutoDraw(False)
            # store stop times for trlInit
            trlInit.tStop = globalClock.getTime(format='float')
            trlInit.tStopRefresh = tThisFlipGlobal
            thisExp.addData('trlInit.stopped', trlInit.tStop)
            # Run 'End Routine' code from initTrlParams
            thisExp.addData('TRIALID',trlId)
            for col,val in zip(df.iloc[trlCntTotal].index,df.iloc[trlCntTotal]):
                thisExp.addData(col,val) 
            
            # the Routine "trlInit" was not non-slip safe, so reset the non-slip timer
            routineTimer.reset()
            
            # --- Prepare to start Routine "fixationRest" ---
            # create an object to store info about Routine fixationRest
            fixationRest = data.Routine(
                name='fixationRest',
                components=[],
            )
            fixationRest.status = NOT_STARTED
            continueRoutine = True
            # update component parameters for each repeat
            # Run 'Begin Routine' code from drawFixation
            cueCircColor.setAutoDraw(True)
            fixCircMask.setAutoDraw(True)
            
            clock.reset()
            kb.clock.reset()
            kb.clearEvents()
            
            win.flip()
            # Run 'Begin Routine' code from elRecord_fixation
            this_epoch = 'fixation'
            aaa = core.monotonicClock.getTime
            thisExp.addData('fixationStart',str(aaa()))
            if eyetracking == 1:
                # get a reference to the currently active EyeLink connection
                el_tracker = pylink.getEYELINK()
            
                try:
                    #start recording
                    el_tracker.startRecording(1, 1, 1, 1) 
                    #send message to tracker to count trial number
                    el_tracker.sendMessage('TRIALID %s' % str(trlId))
                    el_tracker.sendMessage('fixationRest')
                except RuntimeError as error:
                    print("ERROR:", error)
                    abort_trial()
            
                
            
            # store start times for fixationRest
            fixationRest.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
            fixationRest.tStart = globalClock.getTime(format='float')
            fixationRest.status = STARTED
            thisExp.addData('fixationRest.started', fixationRest.tStart)
            fixationRest.maxDuration = None
            # keep track of which components have finished
            fixationRestComponents = fixationRest.components
            for thisComponent in fixationRest.components:
                thisComponent.tStart = None
                thisComponent.tStop = None
                thisComponent.tStartRefresh = None
                thisComponent.tStopRefresh = None
                if hasattr(thisComponent, 'status'):
                    thisComponent.status = NOT_STARTED
            # reset timers
            t = 0
            _timeToFirstFrame = win.getFutureFlipTime(clock="now")
            frameN = -1
            
            # --- Run Routine "fixationRest" ---
            # if trial has changed, end Routine now
            if isinstance(trials, data.TrialHandler2) and thisTrial.thisN != trials.thisTrial.thisN:
                continueRoutine = False
            fixationRest.forceEnded = routineForceEnded = not continueRoutine
            while continueRoutine:
                # get current time
                t = routineTimer.getTime()
                tThisFlip = win.getFutureFlipTime(clock=routineTimer)
                tThisFlipGlobal = win.getFutureFlipTime(clock=None)
                frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
                # update/draw components on each frame
                # Run 'Each Frame' code from drawFixation
                continuing = True
                while clock.getTime() <= fixationT:
                    key = kb.getKeys(['space', 'escape'], waitRelease=False)
                    if 'escape' in key:
                          core.quit()
                
                # check for quit (typically the Esc key)
                if defaultKeyboard.getKeys(keyList=["escape"]):
                    thisExp.status = FINISHED
                if thisExp.status == FINISHED or endExpNow:
                    endExperiment(thisExp, win=win)
                    return
                # pause experiment here if requested
                if thisExp.status == PAUSED:
                    pauseExperiment(
                        thisExp=thisExp, 
                        win=win, 
                        timers=[routineTimer], 
                        playbackComponents=[]
                    )
                    # skip the frame we paused on
                    continue
                
                # check if all components have finished
                if not continueRoutine:  # a component has requested a forced-end of Routine
                    fixationRest.forceEnded = routineForceEnded = True
                    break
                continueRoutine = False  # will revert to True if at least one component still running
                for thisComponent in fixationRest.components:
                    if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                        continueRoutine = True
                        break  # at least one component has not yet finished
                
                # refresh the screen
                if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                    win.flip()
            
            # --- Ending Routine "fixationRest" ---
            for thisComponent in fixationRest.components:
                if hasattr(thisComponent, "setAutoDraw"):
                    thisComponent.setAutoDraw(False)
            # store stop times for fixationRest
            fixationRest.tStop = globalClock.getTime(format='float')
            fixationRest.tStopRefresh = tThisFlipGlobal
            thisExp.addData('fixationRest.stopped', fixationRest.tStop)
            # Run 'End Routine' code from elRecord_fixation
            aaa = core.monotonicClock.getTime
            thisExp.addData('fixationEnd',str(aaa()))
            
            # the Routine "fixationRest" was not non-slip safe, so reset the non-slip timer
            routineTimer.reset()
            
            # --- Prepare to start Routine "stim" ---
            # create an object to store info about Routine stim
            stim = data.Routine(
                name='stim',
                components=[],
            )
            stim.status = NOT_STARTED
            continueRoutine = True
            # update component parameters for each repeat
            # Run 'Begin Routine' code from drawVisualStim
            imLeft.setImage(df['leftImagePath'][trlCntTotal].replace(os.sep, "/"))
            imRight.setImage(df['rightImagePath'][trlCntTotal].replace(os.sep, "/"),)
            
            clock.reset()
            kb.clock.reset()
            kb.clearEvents()
            
            imLeft.setAutoDraw(True)
            imRight.setAutoDraw(True)
            
            
            
            
            win.flip()
            # Run 'Begin Routine' code from elRecord_stim
            this_epoch = 'stim'
            aaa = core.monotonicClock.getTime
            thisExp.addData(this_epoch+'Start',str(aaa()))
            
            if eyetracking == 1:
                el_tracker.sendMessage(this_epoch)
            
            # store start times for stim
            stim.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
            stim.tStart = globalClock.getTime(format='float')
            stim.status = STARTED
            thisExp.addData('stim.started', stim.tStart)
            stim.maxDuration = None
            # keep track of which components have finished
            stimComponents = stim.components
            for thisComponent in stim.components:
                thisComponent.tStart = None
                thisComponent.tStop = None
                thisComponent.tStartRefresh = None
                thisComponent.tStopRefresh = None
                if hasattr(thisComponent, 'status'):
                    thisComponent.status = NOT_STARTED
            # reset timers
            t = 0
            _timeToFirstFrame = win.getFutureFlipTime(clock="now")
            frameN = -1
            
            # --- Run Routine "stim" ---
            # if trial has changed, end Routine now
            if isinstance(trials, data.TrialHandler2) and thisTrial.thisN != trials.thisTrial.thisN:
                continueRoutine = False
            stim.forceEnded = routineForceEnded = not continueRoutine
            while continueRoutine:
                # get current time
                t = routineTimer.getTime()
                tThisFlip = win.getFutureFlipTime(clock=routineTimer)
                tThisFlipGlobal = win.getFutureFlipTime(clock=None)
                frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
                # update/draw components on each frame
                # Run 'Each Frame' code from drawVisualStim
                continuing = True
                while clock.getTime() <= stimT:
                    pass
                
                # check for quit (typically the Esc key)
                if defaultKeyboard.getKeys(keyList=["escape"]):
                    thisExp.status = FINISHED
                if thisExp.status == FINISHED or endExpNow:
                    endExperiment(thisExp, win=win)
                    return
                # pause experiment here if requested
                if thisExp.status == PAUSED:
                    pauseExperiment(
                        thisExp=thisExp, 
                        win=win, 
                        timers=[routineTimer], 
                        playbackComponents=[]
                    )
                    # skip the frame we paused on
                    continue
                
                # check if all components have finished
                if not continueRoutine:  # a component has requested a forced-end of Routine
                    stim.forceEnded = routineForceEnded = True
                    break
                continueRoutine = False  # will revert to True if at least one component still running
                for thisComponent in stim.components:
                    if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                        continueRoutine = True
                        break  # at least one component has not yet finished
                
                # refresh the screen
                if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                    win.flip()
            
            # --- Ending Routine "stim" ---
            for thisComponent in stim.components:
                if hasattr(thisComponent, "setAutoDraw"):
                    thisComponent.setAutoDraw(False)
            # store stop times for stim
            stim.tStop = globalClock.getTime(format='float')
            stim.tStopRefresh = tThisFlipGlobal
            thisExp.addData('stim.stopped', stim.tStop)
            # Run 'End Routine' code from drawVisualStim
            imLeft.setAutoDraw(False)
            imRight.setAutoDraw(False)
            
            #add other info
            thisExp.addData('stimT', stimT)
            # Run 'End Routine' code from elRecord_stim
            aaa = core.monotonicClock.getTime
            thisExp.addData(this_epoch+'End',str(aaa()))
            
            # the Routine "stim" was not non-slip safe, so reset the non-slip timer
            routineTimer.reset()
            
            # --- Prepare to start Routine "preCueRest" ---
            # create an object to store info about Routine preCueRest
            preCueRest = data.Routine(
                name='preCueRest',
                components=[],
            )
            preCueRest.status = NOT_STARTED
            continueRoutine = True
            # update component parameters for each repeat
            # Run 'Begin Routine' code from drawNoise
            noisePatchRight.setAutoDraw(True)
            noisePatchLeft.setAutoDraw(True)
            
            clock.reset()
            kb.clearEvents()
            
            win.flip()
            # Run 'Begin Routine' code from elRecord_preCue
            this_epoch = 'preCueRest'
            aaa = core.monotonicClock.getTime
            thisExp.addData(this_epoch+'Start',str(aaa()))
            
            if eyetracking == 1:
                #el_tracker.startRecording(1, 1, 1, 1)
                el_tracker.sendMessage(this_epoch)
            
            # store start times for preCueRest
            preCueRest.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
            preCueRest.tStart = globalClock.getTime(format='float')
            preCueRest.status = STARTED
            thisExp.addData('preCueRest.started', preCueRest.tStart)
            preCueRest.maxDuration = None
            # keep track of which components have finished
            preCueRestComponents = preCueRest.components
            for thisComponent in preCueRest.components:
                thisComponent.tStart = None
                thisComponent.tStop = None
                thisComponent.tStartRefresh = None
                thisComponent.tStopRefresh = None
                if hasattr(thisComponent, 'status'):
                    thisComponent.status = NOT_STARTED
            # reset timers
            t = 0
            _timeToFirstFrame = win.getFutureFlipTime(clock="now")
            frameN = -1
            
            # --- Run Routine "preCueRest" ---
            # if trial has changed, end Routine now
            if isinstance(trials, data.TrialHandler2) and thisTrial.thisN != trials.thisTrial.thisN:
                continueRoutine = False
            preCueRest.forceEnded = routineForceEnded = not continueRoutine
            while continueRoutine:
                # get current time
                t = routineTimer.getTime()
                tThisFlip = win.getFutureFlipTime(clock=routineTimer)
                tThisFlipGlobal = win.getFutureFlipTime(clock=None)
                frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
                # update/draw components on each frame
                # Run 'Each Frame' code from drawNoise
                continuing = True
                while clock.getTime() <= noisePatchT:
                    key = kb.getKeys(['escape'], waitRelease=False)
                    if 'escape' in key:
                          core.quit()
                noisePatchRight.setAutoDraw(False)
                noisePatchLeft.setAutoDraw(False)
                win.flip()
                
                while clock.getTime() <= preCueRestT:
                    pass
                
                
                # check for quit (typically the Esc key)
                if defaultKeyboard.getKeys(keyList=["escape"]):
                    thisExp.status = FINISHED
                if thisExp.status == FINISHED or endExpNow:
                    endExperiment(thisExp, win=win)
                    return
                # pause experiment here if requested
                if thisExp.status == PAUSED:
                    pauseExperiment(
                        thisExp=thisExp, 
                        win=win, 
                        timers=[routineTimer], 
                        playbackComponents=[]
                    )
                    # skip the frame we paused on
                    continue
                
                # check if all components have finished
                if not continueRoutine:  # a component has requested a forced-end of Routine
                    preCueRest.forceEnded = routineForceEnded = True
                    break
                continueRoutine = False  # will revert to True if at least one component still running
                for thisComponent in preCueRest.components:
                    if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                        continueRoutine = True
                        break  # at least one component has not yet finished
                
                # refresh the screen
                if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                    win.flip()
            
            # --- Ending Routine "preCueRest" ---
            for thisComponent in preCueRest.components:
                if hasattr(thisComponent, "setAutoDraw"):
                    thisComponent.setAutoDraw(False)
            # store stop times for preCueRest
            preCueRest.tStop = globalClock.getTime(format='float')
            preCueRest.tStopRefresh = tThisFlipGlobal
            thisExp.addData('preCueRest.stopped', preCueRest.tStop)
            # Run 'End Routine' code from drawNoise
            cueCircColor.setAutoDraw(False)
            fixCircMask.setAutoDraw(False)
            
            
            # Run 'End Routine' code from elRecord_preCue
            aaa = core.monotonicClock.getTime
            thisExp.addData(this_epoch+'End',str(aaa()))
            
            # the Routine "preCueRest" was not non-slip safe, so reset the non-slip timer
            routineTimer.reset()
            
            # --- Prepare to start Routine "retrocue" ---
            # create an object to store info about Routine retrocue
            retrocue = data.Routine(
                name='retrocue',
                components=[],
            )
            retrocue.status = NOT_STARTED
            continueRoutine = True
            # update component parameters for each repeat
            # Run 'Begin Routine' code from drawRetrocue
            if df['cuedItem'][trlCntTotal] == 'left':
                thisOri = 180
            else:
                thisOri = 0
            
            retrocueMask.ori = thisOri
            
            cueCircColor.colorSpace='rgb255'
            cueCircColor.color = cueCorThis
            cueCircColor.setAutoDraw(True)
            retrocueMask.setAutoDraw(True) #draw retrocue
            
            
            clock.reset()
            kb.clock.reset()
            kb.clearEvents()
            
            win.flip()
            # Run 'Begin Routine' code from elRecord_retrocue
            this_epoch = 'retrocue'
            aaa = core.monotonicClock.getTime
            thisExp.addData(this_epoch+'Start',str(aaa()))
            
            if eyetracking == 1:
                #el_tracker.startRecording(1, 1, 1, 1)
                el_tracker.sendMessage(this_epoch)
            
            # store start times for retrocue
            retrocue.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
            retrocue.tStart = globalClock.getTime(format='float')
            retrocue.status = STARTED
            thisExp.addData('retrocue.started', retrocue.tStart)
            retrocue.maxDuration = None
            # keep track of which components have finished
            retrocueComponents = retrocue.components
            for thisComponent in retrocue.components:
                thisComponent.tStart = None
                thisComponent.tStop = None
                thisComponent.tStartRefresh = None
                thisComponent.tStopRefresh = None
                if hasattr(thisComponent, 'status'):
                    thisComponent.status = NOT_STARTED
            # reset timers
            t = 0
            _timeToFirstFrame = win.getFutureFlipTime(clock="now")
            frameN = -1
            
            # --- Run Routine "retrocue" ---
            # if trial has changed, end Routine now
            if isinstance(trials, data.TrialHandler2) and thisTrial.thisN != trials.thisTrial.thisN:
                continueRoutine = False
            retrocue.forceEnded = routineForceEnded = not continueRoutine
            while continueRoutine:
                # get current time
                t = routineTimer.getTime()
                tThisFlip = win.getFutureFlipTime(clock=routineTimer)
                tThisFlipGlobal = win.getFutureFlipTime(clock=None)
                frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
                # update/draw components on each frame
                # Run 'Each Frame' code from drawRetrocue
                continuing = True
                while clock.getTime() <= retrocueT:
                    key = kb.getKeys(['space', 'escape'], waitRelease=False)
                    if 'escape' in key:
                          core.quit()
                
                # check for quit (typically the Esc key)
                if defaultKeyboard.getKeys(keyList=["escape"]):
                    thisExp.status = FINISHED
                if thisExp.status == FINISHED or endExpNow:
                    endExperiment(thisExp, win=win)
                    return
                # pause experiment here if requested
                if thisExp.status == PAUSED:
                    pauseExperiment(
                        thisExp=thisExp, 
                        win=win, 
                        timers=[routineTimer], 
                        playbackComponents=[]
                    )
                    # skip the frame we paused on
                    continue
                
                # check if all components have finished
                if not continueRoutine:  # a component has requested a forced-end of Routine
                    retrocue.forceEnded = routineForceEnded = True
                    break
                continueRoutine = False  # will revert to True if at least one component still running
                for thisComponent in retrocue.components:
                    if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                        continueRoutine = True
                        break  # at least one component has not yet finished
                
                # refresh the screen
                if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                    win.flip()
            
            # --- Ending Routine "retrocue" ---
            for thisComponent in retrocue.components:
                if hasattr(thisComponent, "setAutoDraw"):
                    thisComponent.setAutoDraw(False)
            # store stop times for retrocue
            retrocue.tStop = globalClock.getTime(format='float')
            retrocue.tStopRefresh = tThisFlipGlobal
            thisExp.addData('retrocue.stopped', retrocue.tStop)
            # Run 'End Routine' code from drawRetrocue
            cueCircColor.setAutoDraw(False)
            retrocueMask.setAutoDraw(False)
            
            
            
            # Run 'End Routine' code from elRecord_retrocue
            aaa = core.monotonicClock.getTime
            thisExp.addData(this_epoch+'End',str(aaa()))
            
            # the Routine "retrocue" was not non-slip safe, so reset the non-slip timer
            routineTimer.reset()
            
            # --- Prepare to start Routine "delayNoDistractor" ---
            # create an object to store info about Routine delayNoDistractor
            delayNoDistractor = data.Routine(
                name='delayNoDistractor',
                components=[],
            )
            delayNoDistractor.status = NOT_STARTED
            continueRoutine = True
            # update component parameters for each repeat
            # Run 'Begin Routine' code from delayCode_noDistractor
            cueCircColor.color = cueColDefault
            
            cueCircColor.setAutoDraw(True)
            fixCircMask.setAutoDraw(True)
            noisePatchRight.setAutoDraw(False)
            noisePatchLeft.setAutoDraw(False)
            
            clock.reset()
            kb.clearEvents()
            
            win.flip()
            # Run 'Begin Routine' code from elRecord_delay_noDistractor
            this_epoch = 'delay'
            aaa = core.monotonicClock.getTime
            thisExp.addData(this_epoch+'Start',str(aaa()))
            
            if eyetracking == 1:
                #el_tracker.startRecording(1, 1, 1, 1)
                el_tracker.sendMessage(this_epoch)
            
            # store start times for delayNoDistractor
            delayNoDistractor.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
            delayNoDistractor.tStart = globalClock.getTime(format='float')
            delayNoDistractor.status = STARTED
            thisExp.addData('delayNoDistractor.started', delayNoDistractor.tStart)
            delayNoDistractor.maxDuration = None
            # keep track of which components have finished
            delayNoDistractorComponents = delayNoDistractor.components
            for thisComponent in delayNoDistractor.components:
                thisComponent.tStart = None
                thisComponent.tStop = None
                thisComponent.tStartRefresh = None
                thisComponent.tStopRefresh = None
                if hasattr(thisComponent, 'status'):
                    thisComponent.status = NOT_STARTED
            # reset timers
            t = 0
            _timeToFirstFrame = win.getFutureFlipTime(clock="now")
            frameN = -1
            
            # --- Run Routine "delayNoDistractor" ---
            # if trial has changed, end Routine now
            if isinstance(trials, data.TrialHandler2) and thisTrial.thisN != trials.thisTrial.thisN:
                continueRoutine = False
            delayNoDistractor.forceEnded = routineForceEnded = not continueRoutine
            while continueRoutine:
                # get current time
                t = routineTimer.getTime()
                tThisFlip = win.getFutureFlipTime(clock=routineTimer)
                tThisFlipGlobal = win.getFutureFlipTime(clock=None)
                frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
                # update/draw components on each frame
                # Run 'Each Frame' code from delayCode_noDistractor
                while (clock.getTime() < delayT):
                    pass
                
                # check for quit (typically the Esc key)
                if defaultKeyboard.getKeys(keyList=["escape"]):
                    thisExp.status = FINISHED
                if thisExp.status == FINISHED or endExpNow:
                    endExperiment(thisExp, win=win)
                    return
                # pause experiment here if requested
                if thisExp.status == PAUSED:
                    pauseExperiment(
                        thisExp=thisExp, 
                        win=win, 
                        timers=[routineTimer], 
                        playbackComponents=[]
                    )
                    # skip the frame we paused on
                    continue
                
                # check if all components have finished
                if not continueRoutine:  # a component has requested a forced-end of Routine
                    delayNoDistractor.forceEnded = routineForceEnded = True
                    break
                continueRoutine = False  # will revert to True if at least one component still running
                for thisComponent in delayNoDistractor.components:
                    if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                        continueRoutine = True
                        break  # at least one component has not yet finished
                
                # refresh the screen
                if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                    win.flip()
            
            # --- Ending Routine "delayNoDistractor" ---
            for thisComponent in delayNoDistractor.components:
                if hasattr(thisComponent, "setAutoDraw"):
                    thisComponent.setAutoDraw(False)
            # store stop times for delayNoDistractor
            delayNoDistractor.tStop = globalClock.getTime(format='float')
            delayNoDistractor.tStopRefresh = tThisFlipGlobal
            thisExp.addData('delayNoDistractor.stopped', delayNoDistractor.tStop)
            # Run 'End Routine' code from delayCode_noDistractor
            noisePatchRight.setAutoDraw(False)
            noisePatchLeft.setAutoDraw(False)
            cueCircColor.setAutoDraw(False)
            fixCircMask.setAutoDraw(False)
            
            win.flip()
            # Run 'End Routine' code from elRecord_delay_noDistractor
            aaa = core.monotonicClock.getTime
            thisExp.addData(this_epoch+'End',str(aaa()))
            
            # the Routine "delayNoDistractor" was not non-slip safe, so reset the non-slip timer
            routineTimer.reset()
            
            # set up handler to look after randomisation of conditions etc
            distractorCtrl = data.TrialHandler2(
                name='distractorCtrl',
                nReps=distractorCtrl, 
                method='random', 
                extraInfo=expInfo, 
                originPath=-1, 
                trialList=[None], 
                seed=None, 
            )
            thisExp.addLoop(distractorCtrl)  # add the loop to the experiment
            thisDistractorCtrl = distractorCtrl.trialList[0]  # so we can initialise stimuli with some values
            # abbreviate parameter names if possible (e.g. rgb = thisDistractorCtrl.rgb)
            if thisDistractorCtrl != None:
                for paramName in thisDistractorCtrl:
                    globals()[paramName] = thisDistractorCtrl[paramName]
            if thisSession is not None:
                # if running in a Session with a Liaison client, send data up to now
                thisSession.sendExperimentData()
            
            for thisDistractorCtrl in distractorCtrl:
                currentLoop = distractorCtrl
                thisExp.timestampOnFlip(win, 'thisRow.t', format=globalClock.format)
                if thisSession is not None:
                    # if running in a Session with a Liaison client, send data up to now
                    thisSession.sendExperimentData()
                # abbreviate parameter names if possible (e.g. rgb = thisDistractorCtrl.rgb)
                if thisDistractorCtrl != None:
                    for paramName in thisDistractorCtrl:
                        globals()[paramName] = thisDistractorCtrl[paramName]
                
                # --- Prepare to start Routine "distractor" ---
                # create an object to store info about Routine distractor
                distractor = data.Routine(
                    name='distractor',
                    components=[],
                )
                distractor.status = NOT_STARTED
                continueRoutine = True
                # update component parameters for each repeat
                # Run 'Begin Routine' code from drawDistractor
                visualDistractor.setAutoDraw(True)
                verbalDistractor.setAutoDraw(True)
                instrTxt.setAutoDraw(True)
                clock.reset()
                kb.clock.reset()
                kb.clearEvents()
                
                win.flip()
                # Run 'Begin Routine' code from elRecord_distractor
                this_epoch = 'distractor'
                aaa = core.monotonicClock.getTime
                thisExp.addData(this_epoch+'Start',str(aaa()))
                
                if eyetracking == 1:
                    #el_tracker.startRecording(1, 1, 1, 1)
                    el_tracker.sendMessage(this_epoch)
                
                # store start times for distractor
                distractor.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
                distractor.tStart = globalClock.getTime(format='float')
                distractor.status = STARTED
                thisExp.addData('distractor.started', distractor.tStart)
                distractor.maxDuration = None
                # keep track of which components have finished
                distractorComponents = distractor.components
                for thisComponent in distractor.components:
                    thisComponent.tStart = None
                    thisComponent.tStop = None
                    thisComponent.tStartRefresh = None
                    thisComponent.tStopRefresh = None
                    if hasattr(thisComponent, 'status'):
                        thisComponent.status = NOT_STARTED
                # reset timers
                t = 0
                _timeToFirstFrame = win.getFutureFlipTime(clock="now")
                frameN = -1
                
                # --- Run Routine "distractor" ---
                # if trial has changed, end Routine now
                if isinstance(distractorCtrl, data.TrialHandler2) and thisDistractorCtrl.thisN != distractorCtrl.thisTrial.thisN:
                    continueRoutine = False
                distractor.forceEnded = routineForceEnded = not continueRoutine
                while continueRoutine:
                    # get current time
                    t = routineTimer.getTime()
                    tThisFlip = win.getFutureFlipTime(clock=routineTimer)
                    tThisFlipGlobal = win.getFutureFlipTime(clock=None)
                    frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
                    # update/draw components on each frame
                    # Run 'Each Frame' code from drawDistractor
                    continuing = True
                    while clock.getTime() <= distractorT:
                        key = kb.getKeys(['w','d','escape'], waitRelease=False)
                        
                        if 'escape' in key:
                              core.quit()
                        elif ('w' in key):
                            if df['trlType'][trlCntTotal] == 'visual' and  df['distractorType'][trlCntTotal] == 'congruent':
                                rsp = 'scrambledImage'
                            elif df['trlType'][trlCntTotal] == 'verbal' and  df['distractorType'][trlCntTotal] == 'incongruent': #visual distractor:
                                rsp = 'scrambledImage'
                            else: #verbal distractor:
                                rsp = 'manmadeImage'
                            if df['distractorRealness'][trlCntTotal] == rsp:
                                correct = True
                            else:
                                correct = False
                        elif ('d' in key):
                            if df['trlType'][trlCntTotal] == 'visual' and  df['distractorType'][trlCntTotal] == 'congruent':
                                rsp = 'nonScrambledImage'
                            elif df['trlType'][trlCntTotal] == 'verbal' and  df['distractorType'][trlCntTotal] == 'incongruent': #visual distractor:
                                rsp = 'nonScrambledImage'
                            else: #verbal distractor:
                                rsp = 'nonManmadeImage'
                            if df['distractorRealness'][trlCntTotal] == rsp:
                                correct = True
                            else:
                                correct = False
                        else:
                            rsp = None
                            correct = None
                    
                    
                    # check for quit (typically the Esc key)
                    if defaultKeyboard.getKeys(keyList=["escape"]):
                        thisExp.status = FINISHED
                    if thisExp.status == FINISHED or endExpNow:
                        endExperiment(thisExp, win=win)
                        return
                    # pause experiment here if requested
                    if thisExp.status == PAUSED:
                        pauseExperiment(
                            thisExp=thisExp, 
                            win=win, 
                            timers=[routineTimer], 
                            playbackComponents=[]
                        )
                        # skip the frame we paused on
                        continue
                    
                    # check if all components have finished
                    if not continueRoutine:  # a component has requested a forced-end of Routine
                        distractor.forceEnded = routineForceEnded = True
                        break
                    continueRoutine = False  # will revert to True if at least one component still running
                    for thisComponent in distractor.components:
                        if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                            continueRoutine = True
                            break  # at least one component has not yet finished
                    
                    # refresh the screen
                    if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                        win.flip()
                
                # --- Ending Routine "distractor" ---
                for thisComponent in distractor.components:
                    if hasattr(thisComponent, "setAutoDraw"):
                        thisComponent.setAutoDraw(False)
                # store stop times for distractor
                distractor.tStop = globalClock.getTime(format='float')
                distractor.tStopRefresh = tThisFlipGlobal
                thisExp.addData('distractor.stopped', distractor.tStop)
                # Run 'End Routine' code from drawDistractor
                visualDistractor.setAutoDraw(False)
                verbalDistractor.setAutoDraw(False)
                instrTxt.setAutoDraw(False)
                win.flip()
                if not rsp == None:
                    thisExp.addData('distractorRsp',rsp)
                    thisExp.addData('distractorRspRealness',correct)
                # Run 'End Routine' code from elRecord_distractor
                aaa = core.monotonicClock.getTime
                thisExp.addData(this_epoch+'End',str(aaa()))
                
                # the Routine "distractor" was not non-slip safe, so reset the non-slip timer
                routineTimer.reset()
                
                # --- Prepare to start Routine "distractorResponse" ---
                # create an object to store info about Routine distractorResponse
                distractorResponse = data.Routine(
                    name='distractorResponse',
                    components=[],
                )
                distractorResponse.status = NOT_STARTED
                continueRoutine = True
                # update component parameters for each repeat
                # Run 'Begin Routine' code from distractorResponse_code
                cueCircColor.color = cueColDefault
                
                cueCircColor.setAutoDraw(True)
                fixCircMask.setAutoDraw(True)
                noisePatchRight.setAutoDraw(False)
                noisePatchLeft.setAutoDraw(False)
                
                clock.reset()
                kb.clearEvents()
                
                win.flip()
                # Run 'Begin Routine' code from elDistractorResponse_code
                this_epoch = 'distractorResponse'
                aaa = core.monotonicClock.getTime
                thisExp.addData(this_epoch+'Start',str(aaa()))
                
                if eyetracking == 1:
                    #el_tracker.startRecording(1, 1, 1, 1)
                    el_tracker.sendMessage(this_epoch)
                
                # store start times for distractorResponse
                distractorResponse.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
                distractorResponse.tStart = globalClock.getTime(format='float')
                distractorResponse.status = STARTED
                thisExp.addData('distractorResponse.started', distractorResponse.tStart)
                distractorResponse.maxDuration = None
                # keep track of which components have finished
                distractorResponseComponents = distractorResponse.components
                for thisComponent in distractorResponse.components:
                    thisComponent.tStart = None
                    thisComponent.tStop = None
                    thisComponent.tStartRefresh = None
                    thisComponent.tStopRefresh = None
                    if hasattr(thisComponent, 'status'):
                        thisComponent.status = NOT_STARTED
                # reset timers
                t = 0
                _timeToFirstFrame = win.getFutureFlipTime(clock="now")
                frameN = -1
                
                # --- Run Routine "distractorResponse" ---
                # if trial has changed, end Routine now
                if isinstance(distractorCtrl, data.TrialHandler2) and thisDistractorCtrl.thisN != distractorCtrl.thisTrial.thisN:
                    continueRoutine = False
                distractorResponse.forceEnded = routineForceEnded = not continueRoutine
                while continueRoutine:
                    # get current time
                    t = routineTimer.getTime()
                    tThisFlip = win.getFutureFlipTime(clock=routineTimer)
                    tThisFlipGlobal = win.getFutureFlipTime(clock=None)
                    frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
                    # update/draw components on each frame
                    # Run 'Each Frame' code from distractorResponse_code
                    continuing = True
                    while clock.getTime() <= distractorRspT:
                        key = kb.getKeys(['w','d','escape'], waitRelease=False)
                        
                        if 'escape' in key:
                              core.quit()
                        elif ('w' in key):
                            if df['trlType'][trlCntTotal] == 'visual' and  df['distractorType'][trlCntTotal] == 'congruent':
                                rsp = 'scrambledImage'
                            elif df['trlType'][trlCntTotal] == 'verbal' and  df['distractorType'][trlCntTotal] == 'incongruent': #visual distractor:
                                rsp = 'scrambledImage'
                            else: #verbal distractor:
                                rsp = 'manmadeImage'
                            if df['distractorRealness'][trlCntTotal] == rsp:
                                correct = True
                            else:
                                correct = False
                            break
                        elif ('d' in key):
                            if df['trlType'][trlCntTotal] == 'visual' and  df['distractorType'][trlCntTotal] == 'congruent':
                                rsp = 'nonScrambledImage'
                            elif df['trlType'][trlCntTotal] == 'verbal' and  df['distractorType'][trlCntTotal] == 'incongruent': #visual distractor:
                                rsp = 'nonScrambledImage'
                            else: #verbal distractor:
                                rsp = 'nonManmadeImage'
                            if df['distractorRealness'][trlCntTotal] == rsp:
                                correct = True
                            else:
                                correct = False
                            break
                        else:
                            rsp = None
                            correct = None
                        
                    continueRoutine  = False
                    
                    # check for quit (typically the Esc key)
                    if defaultKeyboard.getKeys(keyList=["escape"]):
                        thisExp.status = FINISHED
                    if thisExp.status == FINISHED or endExpNow:
                        endExperiment(thisExp, win=win)
                        return
                    # pause experiment here if requested
                    if thisExp.status == PAUSED:
                        pauseExperiment(
                            thisExp=thisExp, 
                            win=win, 
                            timers=[routineTimer], 
                            playbackComponents=[]
                        )
                        # skip the frame we paused on
                        continue
                    
                    # check if all components have finished
                    if not continueRoutine:  # a component has requested a forced-end of Routine
                        distractorResponse.forceEnded = routineForceEnded = True
                        break
                    continueRoutine = False  # will revert to True if at least one component still running
                    for thisComponent in distractorResponse.components:
                        if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                            continueRoutine = True
                            break  # at least one component has not yet finished
                    
                    # refresh the screen
                    if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                        win.flip()
                
                # --- Ending Routine "distractorResponse" ---
                for thisComponent in distractorResponse.components:
                    if hasattr(thisComponent, "setAutoDraw"):
                        thisComponent.setAutoDraw(False)
                # store stop times for distractorResponse
                distractorResponse.tStop = globalClock.getTime(format='float')
                distractorResponse.tStopRefresh = tThisFlipGlobal
                thisExp.addData('distractorResponse.stopped', distractorResponse.tStop)
                # Run 'End Routine' code from distractorResponse_code
                noisePatchRight.setAutoDraw(False)
                noisePatchLeft.setAutoDraw(False)
                cueCircColor.setAutoDraw(False)
                fixCircMask.setAutoDraw(False)
                
                win.flip()
                thisExp.addData('distractorRsp',rsp)
                thisExp.addData('distractorRspRealness',correct)
                # Run 'End Routine' code from elDistractorResponse_code
                aaa = core.monotonicClock.getTime
                thisExp.addData(this_epoch+'End',str(aaa()))
                
                # the Routine "distractorResponse" was not non-slip safe, so reset the non-slip timer
                routineTimer.reset()
                thisExp.nextEntry()
                
            # completed distractorCtrl repeats of 'distractorCtrl'
            
            if thisSession is not None:
                # if running in a Session with a Liaison client, send data up to now
                thisSession.sendExperimentData()
            
            # --- Prepare to start Routine "probe" ---
            # create an object to store info about Routine probe
            probe = data.Routine(
                name='probe',
                components=[transparentPlaceHolder],
            )
            probe.status = NOT_STARTED
            continueRoutine = True
            # update component parameters for each repeat
            # Run 'Begin Routine' code from elRecord_probe
            this_epoch = 'probe'
            aaa = core.monotonicClock.getTime
            thisExp.addData(this_epoch+'Start',str(aaa()))
            
            if eyetracking == 1:
                el_tracker.sendMessage(this_epoch)
            
            # Run 'Begin Routine' code from drawProbe
            mouseIsDown = False
            
            #record continuous mouse activity
            probeMouse.x = []
            probeMouse.y = []
            probeMouse.leftButton = []
            probeMouse.midButton = []
            probeMouse.rightButton = []
            probeMouse.time = []
            probeMouse.clickOn = []
            
            #other mouse related variables
            eachClickTime = []
            eachClickItem = []
            
            clicksTotal = np.zeros(len(probeArr))
            clickOnClickable = np.zeros(len(probeArr))
            
            #prepare to draw
            for i in probeArr:
                i.setAutoDraw(True)
                
            #resets
            clock.reset()
            kb.clearEvents()
            probeMouse.clickReset(buttons=(0, 1, 2))
            event.clearEvents('mouse')
            probeMouse.mouseClock.reset()
            
            #set mouse starting position
            probeMouse.setPos((0,0))
            probeMouse.setVisible(1)
            
            win.flip()
            
            # store start times for probe
            probe.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
            probe.tStart = globalClock.getTime(format='float')
            probe.status = STARTED
            thisExp.addData('probe.started', probe.tStart)
            probe.maxDuration = None
            # keep track of which components have finished
            probeComponents = probe.components
            for thisComponent in probe.components:
                thisComponent.tStart = None
                thisComponent.tStop = None
                thisComponent.tStartRefresh = None
                thisComponent.tStopRefresh = None
                if hasattr(thisComponent, 'status'):
                    thisComponent.status = NOT_STARTED
            # reset timers
            t = 0
            _timeToFirstFrame = win.getFutureFlipTime(clock="now")
            frameN = -1
            
            # --- Run Routine "probe" ---
            # if trial has changed, end Routine now
            if isinstance(trials, data.TrialHandler2) and thisTrial.thisN != trials.thisTrial.thisN:
                continueRoutine = False
            probe.forceEnded = routineForceEnded = not continueRoutine
            while continueRoutine:
                # get current time
                t = routineTimer.getTime()
                tThisFlip = win.getFutureFlipTime(clock=routineTimer)
                tThisFlipGlobal = win.getFutureFlipTime(clock=None)
                frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
                # update/draw components on each frame
                
                # *transparentPlaceHolder* updates
                
                # if transparentPlaceHolder is starting this frame...
                if transparentPlaceHolder.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    transparentPlaceHolder.frameNStart = frameN  # exact frame index
                    transparentPlaceHolder.tStart = t  # local t and not account for scr refresh
                    transparentPlaceHolder.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(transparentPlaceHolder, 'tStartRefresh')  # time at next scr refresh
                    # update status
                    transparentPlaceHolder.status = STARTED
                    transparentPlaceHolder.setAutoDraw(True)
                
                # if transparentPlaceHolder is active this frame...
                if transparentPlaceHolder.status == STARTED:
                    # update params
                    pass
                
                # if transparentPlaceHolder is stopping this frame...
                if transparentPlaceHolder.status == STARTED:
                    # is it time to stop? (based on global clock, using actual start)
                    if tThisFlipGlobal > transparentPlaceHolder.tStartRefresh + probeT-frameTolerance:
                        # keep track of stop time/frame for later
                        transparentPlaceHolder.tStop = t  # not accounting for scr refresh
                        transparentPlaceHolder.tStopRefresh = tThisFlipGlobal  # on global time
                        transparentPlaceHolder.frameNStop = frameN  # exact frame index
                        # update status
                        transparentPlaceHolder.status = FINISHED
                        transparentPlaceHolder.setAutoDraw(False)
                # Run 'Each Frame' code from drawProbe
                
                x, y = probeMouse.getPos()
                probeMouse.x.append(x)
                probeMouse.y.append(y)
                buttons = probeMouse.getPressed()
                
                probeMouse.leftButton.append(buttons[0])
                probeMouse.midButton.append(buttons[1])
                probeMouse.rightButton.append(buttons[2])
                probeMouse.time.append(probeMouse.mouseClock.getTime())
                
                hoverOnClickable = np.asarray([i.contains(probeMouse) for i in probeArr])
                probeMouse.clickOn.append(hoverOnClickable)
                
                
                key = kb.getKeys(['escape'], waitRelease=False)
                if 'escape' in key:
                    core.quit()
                #check for mouse press
                if sum(buttons) and mouseIsDown == False and sum(hoverOnClickable):
                    clickOnClickable = np.where(hoverOnClickable,1,0)
                    
                    eachClickTime.append(probeMouse.mouseClock.getTime())
                    eachClickItem.append(clickableLabel[np.where(clickOnClickable == 1)[0]][0])
                    #mouse is pressing right now
                    mouseIsDown = True
                #check for mouse release    
                if sum(buttons) == 0 and mouseIsDown:
                    #mouse released
                    clicksTotal += clickOnClickable
                    
                    #change opacity
                    clicking = probeArr[clickOnClickable ==1][0]
                    
                    #set opacity
                    for n,i in enumerate(probeArr):
                        i.opacity = 1
                    clicking.opacity = 0.25
                    
                    mouseIsDown = False
                    continueRoutine  = False
                
                # check for quit (typically the Esc key)
                if defaultKeyboard.getKeys(keyList=["escape"]):
                    thisExp.status = FINISHED
                if thisExp.status == FINISHED or endExpNow:
                    endExperiment(thisExp, win=win)
                    return
                # pause experiment here if requested
                if thisExp.status == PAUSED:
                    pauseExperiment(
                        thisExp=thisExp, 
                        win=win, 
                        timers=[routineTimer], 
                        playbackComponents=[]
                    )
                    # skip the frame we paused on
                    continue
                
                # check if all components have finished
                if not continueRoutine:  # a component has requested a forced-end of Routine
                    probe.forceEnded = routineForceEnded = True
                    break
                continueRoutine = False  # will revert to True if at least one component still running
                for thisComponent in probe.components:
                    if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                        continueRoutine = True
                        break  # at least one component has not yet finished
                
                # refresh the screen
                if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                    win.flip()
            
            # --- Ending Routine "probe" ---
            for thisComponent in probe.components:
                if hasattr(thisComponent, "setAutoDraw"):
                    thisComponent.setAutoDraw(False)
            # store stop times for probe
            probe.tStop = globalClock.getTime(format='float')
            probe.tStopRefresh = tThisFlipGlobal
            thisExp.addData('probe.stopped', probe.tStop)
            # Run 'End Routine' code from elRecord_probe
            aaa = core.monotonicClock.getTime
            thisExp.addData(this_epoch+'End',str(aaa()))
            
            # Run 'End Routine' code from drawProbe
            #undraw,reset opacity
            for i in probeArr:
                i.setAutoDraw(False)
                i.opacity = 1
                if df['trlType'][trlCntTotal] == 'visual':
                    i.size = i.size/visualProbeSizeScalar
                else:
                    i.size = i.size*visualProbeSizeScalar
            win.flip()
            
            #resets
            kb.clearEvents()
            probeMouse.setVisible(0)
            
            try:
                #add mouse activity
                trials.addData('probeMouse.x', probeMouse.x)
                trials.addData('probeMouse.y', probeMouse.y)
                trials.addData('probeMouse.leftButton', probeMouse.leftButton)
                trials.addData('probeMouse.midButton', probeMouse.midButton)
                trials.addData('probeMouse.rightButton', probeMouse.rightButton)
                trials.addData('probeMouse.time', probeMouse.time)
                trials.addData('probeMouse.mouseOnProbes', probeMouse.clickOn)
                    
                #record position for each probe item
                thisExp.addData('probeLocationArr',probePosArr)
                #record the final response
                thisExp.addData('rspIndex',clickOnClickable)
                thisExp.addData('rsp',clickableLabel[np.where(clickOnClickable == 1)[0]][0])
                thisExp.addData('rspPath',probeArr[np.where(clickOnClickable == 1)[0]][0]._imName)
            
                #record RT
                thisExp.addData('rt',eachClickTime[0])
                thisExp.addData('timeEachClick', eachClickTime)
                thisExp.addData('itemEachClick', eachClickItem)
                thisExp.addData('missingRsp',False)
            except:
                thisExp.addData('missingRsp',True)
            # the Routine "probe" was not non-slip safe, so reset the non-slip timer
            routineTimer.reset()
            
            # --- Prepare to start Routine "ITI" ---
            # create an object to store info about Routine ITI
            ITI = data.Routine(
                name='ITI',
                components=[],
            )
            ITI.status = NOT_STARTED
            continueRoutine = True
            # update component parameters for each repeat
            # Run 'Begin Routine' code from elRecord_iti
            aaa = core.monotonicClock.getTime
            thisExp.addData('itiStart',str(aaa()))
            this_epoch = 'ITI'
            
            if eyetracking == 1:
                el_tracker.sendMessage('ITI')
            # Run 'Begin Routine' code from resets
            #add 1 to total trl count
            trlCntTotal +=1
            trlCntThisBlk +=1
            
            clock.reset()
            kb.clock.reset()
            kb.clearEvents()
            
            # store start times for ITI
            ITI.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
            ITI.tStart = globalClock.getTime(format='float')
            ITI.status = STARTED
            thisExp.addData('ITI.started', ITI.tStart)
            ITI.maxDuration = None
            # keep track of which components have finished
            ITIComponents = ITI.components
            for thisComponent in ITI.components:
                thisComponent.tStart = None
                thisComponent.tStop = None
                thisComponent.tStartRefresh = None
                thisComponent.tStopRefresh = None
                if hasattr(thisComponent, 'status'):
                    thisComponent.status = NOT_STARTED
            # reset timers
            t = 0
            _timeToFirstFrame = win.getFutureFlipTime(clock="now")
            frameN = -1
            
            # --- Run Routine "ITI" ---
            # if trial has changed, end Routine now
            if isinstance(trials, data.TrialHandler2) and thisTrial.thisN != trials.thisTrial.thisN:
                continueRoutine = False
            ITI.forceEnded = routineForceEnded = not continueRoutine
            while continueRoutine:
                # get current time
                t = routineTimer.getTime()
                tThisFlip = win.getFutureFlipTime(clock=routineTimer)
                tThisFlipGlobal = win.getFutureFlipTime(clock=None)
                frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
                # update/draw components on each frame
                # Run 'Each Frame' code from resets
                continuing = True
                while clock.getTime() <= itiT:
                    key = kb.getKeys([ 'escape'], waitRelease=False)
                    if 'escape' in key:
                          core.quit()
                
                # check for quit (typically the Esc key)
                if defaultKeyboard.getKeys(keyList=["escape"]):
                    thisExp.status = FINISHED
                if thisExp.status == FINISHED or endExpNow:
                    endExperiment(thisExp, win=win)
                    return
                # pause experiment here if requested
                if thisExp.status == PAUSED:
                    pauseExperiment(
                        thisExp=thisExp, 
                        win=win, 
                        timers=[routineTimer], 
                        playbackComponents=[]
                    )
                    # skip the frame we paused on
                    continue
                
                # check if all components have finished
                if not continueRoutine:  # a component has requested a forced-end of Routine
                    ITI.forceEnded = routineForceEnded = True
                    break
                continueRoutine = False  # will revert to True if at least one component still running
                for thisComponent in ITI.components:
                    if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                        continueRoutine = True
                        break  # at least one component has not yet finished
                
                # refresh the screen
                if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                    win.flip()
            
            # --- Ending Routine "ITI" ---
            for thisComponent in ITI.components:
                if hasattr(thisComponent, "setAutoDraw"):
                    thisComponent.setAutoDraw(False)
            # store stop times for ITI
            ITI.tStop = globalClock.getTime(format='float')
            ITI.tStopRefresh = tThisFlipGlobal
            thisExp.addData('ITI.stopped', ITI.tStop)
            # Run 'End Routine' code from elRecord_iti
            aaa = core.monotonicClock.getTime
            thisExp.addData('itiEnd',str(aaa()))
            if eyetracking == 1:
               
                el_tracker.sendMessage('trialEnd')
                el_tracker.sendMessage('!V TRIAL_VAR TRIALID %s'% str(trlId))
                el_tracker.stopRecording()
            
            # Run 'End Routine' code from resets
            clock.reset()
            kb.clock.reset()
            kb.clearEvents()
            # the Routine "ITI" was not non-slip safe, so reset the non-slip timer
            routineTimer.reset()
            thisExp.nextEntry()
            
        # completed trlEachBlk repeats of 'trials'
        
        if thisSession is not None:
            # if running in a Session with a Liaison client, send data up to now
            thisSession.sendExperimentData()
        
        # --- Prepare to start Routine "IBI" ---
        # create an object to store info about Routine IBI
        IBI = data.Routine(
            name='IBI',
            components=[text, ibiSpace],
        )
        IBI.status = NOT_STARTED
        continueRoutine = True
        # update component parameters for each repeat
        # create starting attributes for ibiSpace
        ibiSpace.keys = []
        ibiSpace.rt = []
        _ibiSpace_allKeys = []
        # store start times for IBI
        IBI.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
        IBI.tStart = globalClock.getTime(format='float')
        IBI.status = STARTED
        thisExp.addData('IBI.started', IBI.tStart)
        IBI.maxDuration = None
        # keep track of which components have finished
        IBIComponents = IBI.components
        for thisComponent in IBI.components:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "IBI" ---
        # if trial has changed, end Routine now
        if isinstance(expBlk, data.TrialHandler2) and thisExpBlk.thisN != expBlk.thisTrial.thisN:
            continueRoutine = False
        IBI.forceEnded = routineForceEnded = not continueRoutine
        while continueRoutine:
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *text* updates
            
            # if text is starting this frame...
            if text.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                text.frameNStart = frameN  # exact frame index
                text.tStart = t  # local t and not account for scr refresh
                text.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(text, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'text.started')
                # update status
                text.status = STARTED
                text.setAutoDraw(True)
            
            # if text is active this frame...
            if text.status == STARTED:
                # update params
                pass
            
            # *ibiSpace* updates
            waitOnFlip = False
            
            # if ibiSpace is starting this frame...
            if ibiSpace.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                ibiSpace.frameNStart = frameN  # exact frame index
                ibiSpace.tStart = t  # local t and not account for scr refresh
                ibiSpace.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(ibiSpace, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'ibiSpace.started')
                # update status
                ibiSpace.status = STARTED
                # keyboard checking is just starting
                waitOnFlip = True
                win.callOnFlip(ibiSpace.clock.reset)  # t=0 on next screen flip
                win.callOnFlip(ibiSpace.clearEvents, eventType='keyboard')  # clear events on next screen flip
            if ibiSpace.status == STARTED and not waitOnFlip:
                theseKeys = ibiSpace.getKeys(keyList=['space'], ignoreKeys=["escape"], waitRelease=False)
                _ibiSpace_allKeys.extend(theseKeys)
                if len(_ibiSpace_allKeys):
                    ibiSpace.keys = _ibiSpace_allKeys[-1].name  # just the last key pressed
                    ibiSpace.rt = _ibiSpace_allKeys[-1].rt
                    ibiSpace.duration = _ibiSpace_allKeys[-1].duration
                    # a response ends the routine
                    continueRoutine = False
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, win=win)
                return
            # pause experiment here if requested
            if thisExp.status == PAUSED:
                pauseExperiment(
                    thisExp=thisExp, 
                    win=win, 
                    timers=[routineTimer], 
                    playbackComponents=[]
                )
                # skip the frame we paused on
                continue
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                IBI.forceEnded = routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in IBI.components:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "IBI" ---
        for thisComponent in IBI.components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        # store stop times for IBI
        IBI.tStop = globalClock.getTime(format='float')
        IBI.tStopRefresh = tThisFlipGlobal
        thisExp.addData('IBI.stopped', IBI.tStop)
        # check responses
        if ibiSpace.keys in ['', [], None]:  # No response was made
            ibiSpace.keys = None
        expBlk.addData('ibiSpace.keys',ibiSpace.keys)
        if ibiSpace.keys != None:  # we had a response
            expBlk.addData('ibiSpace.rt', ibiSpace.rt)
            expBlk.addData('ibiSpace.duration', ibiSpace.duration)
        # the Routine "IBI" was not non-slip safe, so reset the non-slip timer
        routineTimer.reset()
    # completed blkTotal repeats of 'expBlk'
    
    
    # --- Prepare to start Routine "terminateExp" ---
    # create an object to store info about Routine terminateExp
    terminateExp = data.Routine(
        name='terminateExp',
        components=[],
    )
    terminateExp.status = NOT_STARTED
    continueRoutine = True
    # update component parameters for each repeat
    # Run 'Begin Routine' code from closeEl
    if eyetracking == 1:
    
        # Step 7: disconnect, download the EDF file, then terminate the task
        terminate_task()
    
    # store start times for terminateExp
    terminateExp.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
    terminateExp.tStart = globalClock.getTime(format='float')
    terminateExp.status = STARTED
    thisExp.addData('terminateExp.started', terminateExp.tStart)
    terminateExp.maxDuration = None
    # keep track of which components have finished
    terminateExpComponents = terminateExp.components
    for thisComponent in terminateExp.components:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "terminateExp" ---
    terminateExp.forceEnded = routineForceEnded = not continueRoutine
    while continueRoutine:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, win=win)
            return
        # pause experiment here if requested
        if thisExp.status == PAUSED:
            pauseExperiment(
                thisExp=thisExp, 
                win=win, 
                timers=[routineTimer], 
                playbackComponents=[]
            )
            # skip the frame we paused on
            continue
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            terminateExp.forceEnded = routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in terminateExp.components:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "terminateExp" ---
    for thisComponent in terminateExp.components:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    # store stop times for terminateExp
    terminateExp.tStop = globalClock.getTime(format='float')
    terminateExp.tStopRefresh = tThisFlipGlobal
    thisExp.addData('terminateExp.stopped', terminateExp.tStop)
    thisExp.nextEntry()
    # the Routine "terminateExp" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # mark experiment as finished
    endExperiment(thisExp, win=win)


def saveData(thisExp):
    """
    Save data from this experiment
    
    Parameters
    ==========
    thisExp : psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    """
    filename = thisExp.dataFileName
    # these shouldn't be strictly necessary (should auto-save)
    thisExp.saveAsWideText(filename + '.csv', delim='auto')
    thisExp.saveAsPickle(filename)


def endExperiment(thisExp, win=None):
    """
    End this experiment, performing final shut down operations.
    
    This function does NOT close the window or end the Python process - use `quit` for this.
    
    Parameters
    ==========
    thisExp : psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    win : psychopy.visual.Window
        Window for this experiment.
    """
    if win is not None:
        # remove autodraw from all current components
        win.clearAutoDraw()
        # Flip one final time so any remaining win.callOnFlip() 
        # and win.timeOnFlip() tasks get executed
        win.flip()
    # return console logger level to WARNING
    logging.console.setLevel(logging.WARNING)
    # mark experiment handler as finished
    thisExp.status = FINISHED
    logging.flush()


def quit(thisExp, win=None, thisSession=None):
    """
    Fully quit, closing the window and ending the Python process.
    
    Parameters
    ==========
    win : psychopy.visual.Window
        Window to close.
    thisSession : psychopy.session.Session or None
        Handle of the Session object this experiment is being run from, if any.
    """
    thisExp.abort()  # or data files will save again on exit
    # make sure everything is closed down
    if win is not None:
        # Flip one final time so any remaining win.callOnFlip() 
        # and win.timeOnFlip() tasks get executed before quitting
        win.flip()
        win.close()
    logging.flush()
    if thisSession is not None:
        thisSession.stop()
    # terminate Python process
    core.quit()


# if running this experiment as a script...
if __name__ == '__main__':
    # call all functions in order
    expInfo = showExpInfoDlg(expInfo=expInfo)
    thisExp = setupData(expInfo=expInfo)
    logFile = setupLogging(filename=thisExp.dataFileName)
    win = setupWindow(expInfo=expInfo)
    setupDevices(expInfo=expInfo, thisExp=thisExp, win=win)
    run(
        expInfo=expInfo, 
        thisExp=thisExp, 
        win=win,
        globalClock='float'
    )
    saveData(thisExp=thisExp)
    quit(thisExp=thisExp, win=win)
