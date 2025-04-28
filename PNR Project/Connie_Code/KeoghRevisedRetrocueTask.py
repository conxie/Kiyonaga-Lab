#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
This experiment was created using PsychoPy3 Experiment Builder (v2024.2.4),
    on April 27, 2025, at 18:16
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
prefs.hardware['audioLib'] = 'ptb'
prefs.hardware['audioLatencyMode'] = '3'
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

# Run 'Before Experiment' code from loadImages
#libraries 
import numpy as np  # whole numpy lib is available, prepend 'np.'
import pandas as pd
import random
import os  # handy system and path functions
import sys  # to get file system encoding
import glob
import itertools

# --- Setup global variables (available in all functions) ---
# create a device manager to handle hardware (keyboards, mice, mirophones, speakers, etc.)
deviceManager = hardware.DeviceManager()
# ensure that relative paths start from the same directory as this script
_thisDir = os.path.dirname(os.path.abspath(__file__))
# store info about the experiment session
psychopyVersion = '2024.2.4'
expName = 'KeoghRevisedRetrocueTask'  # from the Builder filename that created this script
# information about this experiment
expInfo = {
    'participant': f"{randint(0, 999999):06.0f}",
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
_winSize = [2560, 1440]
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
        originPath='C:\\Users\\cocon\\OneDrive\\Documents\\GitHub\\Kiyonaga-Lab\\PNR Project\\Connie_Code\\KeoghRevisedRetrocueTask.py',
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
            logging.getLevel('warning')
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
    # Run 'Begin Experiment' code from loadImages
    
    imageFilePath = "C:/Users/cocon/OneDrive/Documents/GitHub/Kiyonaga-Lab/PNR Project/stimuli"
    
    strMapping = {'cu':'Mycah/cube/',
                  'cy':'Mycah/cylinder/',
                  'sp':'Mycah/sphere/'}
    strMapping_naturalistic = {'cu':'Connie/cube/',
                  'cy':'Connie/cylinder/',
                  'sp':'Connie/sphere/'}
    
    def fix_slashes(path_list):
        return [path.replace('\\', '/') for path in path_list]
    
    
    clock = core.Clock()
    kb = keyboard.Keyboard()
    
    #Cue colors
    cueColArr = [[64,128,72],[128,64,128]]
    cueColDefault = [128,115,96]
    cueCorThis = cueColArr[0]
    
    # Load stimuli (jpg + png)
    MycahCube = fix_slashes(glob.glob(imageFilePath+'/Mycah/cube/*.jpg') + glob.glob(imageFilePath+'/Mycah/cube/*.png'))
    MycahCylinder = fix_slashes(glob.glob(imageFilePath+'/Mycah/cylinder/*.jpg') + glob.glob(imageFilePath+'/Mycah/cylinder/*.png'))
    MycahSphere = fix_slashes(glob.glob(imageFilePath+'/Mycah/sphere/*.jpg') + glob.glob(imageFilePath+'/Mycah/sphere/*.png'))
    
    ConnieCube = fix_slashes(glob.glob(imageFilePath+'/Connie/cube/*.jpg') + glob.glob(imageFilePath+'/Connie/cube/*.png'))
    ConnieCylinder = fix_slashes(glob.glob(imageFilePath+'/Connie/cylinder/*.jpg') + glob.glob(imageFilePath+'/Connie/cylinder/*.png'))
    ConnieSphere = fix_slashes(glob.glob(imageFilePath+'/Connie/sphere/*.jpg') + glob.glob(imageFilePath+'/Connie/sphere/*.png'))
    
    # Combine and shuffle
    MycahConnieCube = MycahCube + ConnieCube
    random.shuffle(MycahConnieCube)
    
    MycahConnieCylinder = MycahCylinder + ConnieCylinder
    random.shuffle(MycahConnieCylinder)
    
    MycahConnieSphere = MycahSphere + ConnieSphere
    random.shuffle(MycahConnieSphere)
    
    all_images = MycahConnieCube + MycahConnieCylinder + MycahConnieSphere 
    
    #info for the images
    fixSize = np.array([0.12,0.12])#np.array([0.16,0.16])
    
    #other images  
    retrocueMask = visual.ImageStim(
        win,
        image='c:/Users/cocon/OneDrive/Documents/GitHub/Kiyonaga-Lab/PNR Project/visualObj/retroCue.png',  # <- your uploaded image!
        size=fixSize  # adjust size if needed
    )
    #fixationCross
    fixCircMask = visual.ImageStim(
        win,
        image='c:/Users/cocon/OneDrive/Documents/GitHub/Kiyonaga-Lab/PNR Project/visualObj/fixCirc.png',  # <- your uploaded image!
        size=fixSize  # adjust size if needed
    )
    #cueCircle
    cueCircColor = visual.Polygon(win,edges = 4,radius=1,ori = 45,colorSpace = 'rgb255',fillColor=cueColDefault,size= fixSize/2)
    
    #pathways 
    saved_image_paths =[]
    probeTypePath = ""
    
    #trial info 
    blkTotal = 10
    trlEachBlk = 20
    trlTotal = len(all_images)
    trlCnt = 0
    trlTotal = blkTotal*trlEachBlk
    repsEachCat = int(trlTotal/2)
    distractorArr = np.asarray([0]*trlTotal)
    
    # Run 'Begin Experiment' code from trialInfo
    trlTotal = len(all_images)
    
    # --- Initialize components for Routine "fixation" ---
    # Run 'Begin Experiment' code from fixationCircle
    # Create the window
    win = visual.Window([800, 600], units="pix")
    
    # --- Initialize components for Routine "showStimuli" ---
    
    # --- Initialize components for Routine "retrocue" ---
    
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
    # Run 'Begin Routine' code from trialInfo
    trlCnt = 0
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
    
    # --- Prepare to start Routine "fixation" ---
    # create an object to store info about Routine fixation
    fixation = data.Routine(
        name='fixation',
        components=[],
    )
    fixation.status = NOT_STARTED
    continueRoutine = True
    # update component parameters for each repeat
    # Run 'Begin Routine' code from fixationCircle
    cueCircColor.setAutoDraw(True)
    fixCircMask.setAutoDraw(True)
    
    clock.reset()
    kb.clock.reset()
    kb.clearEvents()
    
    win.flip()
    
    core.wait(0.5) 
    # store start times for fixation
    fixation.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
    fixation.tStart = globalClock.getTime(format='float')
    fixation.status = STARTED
    thisExp.addData('fixation.started', fixation.tStart)
    fixation.maxDuration = None
    # keep track of which components have finished
    fixationComponents = fixation.components
    for thisComponent in fixation.components:
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
    
    # --- Run Routine "fixation" ---
    fixation.forceEnded = routineForceEnded = not continueRoutine
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
            fixation.forceEnded = routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in fixation.components:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "fixation" ---
    for thisComponent in fixation.components:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    # store stop times for fixation
    fixation.tStop = globalClock.getTime(format='float')
    fixation.tStopRefresh = tThisFlipGlobal
    thisExp.addData('fixation.stopped', fixation.tStop)
    thisExp.nextEntry()
    # the Routine "fixation" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # --- Prepare to start Routine "showStimuli" ---
    # create an object to store info about Routine showStimuli
    showStimuli = data.Routine(
        name='showStimuli',
        components=[],
    )
    showStimuli.status = NOT_STARTED
    continueRoutine = True
    # update component parameters for each repeat
    # Run 'Begin Routine' code from stimuliShown
    
    
    # Setup your left and right images with adjusted positions
    leftImage = visual.ImageStim(win, pos=(-250, 0))  # Adjust position to leave some margin
    rightImage = visual.ImageStim(win, pos=(250, 0))  # Adjust position to leave some margin
    
    # Select images for this trial
    selected_images = random.sample(all_images, 2)
    leftImage.image = selected_images[0]
    rightImage.image = selected_images[1]
    
    # Load the first image and get its original size
    image = visual.ImageStim(win, image=selected_images[0])
    image_size = image.size  # Get the original size of the image
    aspect_ratio = image_size[0] / image_size[1]  # Aspect ratio (width / height)
    
    # Define the new size you want for the images (e.g., 350x350), but keep the aspect ratio
    new_width = 350  # Smaller width to leave a margin
    new_height = new_width / aspect_ratio  # Calculate the new height based on aspect ratio
    
    # Set the size of both images based on the aspect ratio
    leftImage.size = (new_width, new_height)
    rightImage.size = (new_width, new_height)
    
    # Set images to auto-draw
    leftImage.setAutoDraw(True)
    rightImage.setAutoDraw(True)
    # Now your fixation and images are all ready to display
    cueCircColor.setAutoDraw(True)
    fixCircMask.setAutoDraw(True)
    
    clock.reset()
    kb.clock.reset()
    kb.clearEvents()
    
    win.flip()  # flip once to show everything together
    
    # --- wait or collect responses here ---
    core.wait(2.0)  # e.g., wait for 1 second
    
    # --- Now turn off images, but KEEP fixation ---
    leftImage.setAutoDraw(False)
    rightImage.setAutoDraw(False)
    
    win.flip()  # flip to update (now only fixation will be on screen)
    
    core.wait(2.0)  # show just the fixation for 500 ms
    
    # --- Turn everything off before next trial ---
    cueCircColor.setAutoDraw(False)
    fixCircMask.setAutoDraw(False)
    
    # Flip to clear everything from the screen
    win.flip()
    # store start times for showStimuli
    showStimuli.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
    showStimuli.tStart = globalClock.getTime(format='float')
    showStimuli.status = STARTED
    thisExp.addData('showStimuli.started', showStimuli.tStart)
    showStimuli.maxDuration = None
    # keep track of which components have finished
    showStimuliComponents = showStimuli.components
    for thisComponent in showStimuli.components:
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
    
    # --- Run Routine "showStimuli" ---
    showStimuli.forceEnded = routineForceEnded = not continueRoutine
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
            showStimuli.forceEnded = routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in showStimuli.components:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "showStimuli" ---
    for thisComponent in showStimuli.components:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    # store stop times for showStimuli
    showStimuli.tStop = globalClock.getTime(format='float')
    showStimuli.tStopRefresh = tThisFlipGlobal
    thisExp.addData('showStimuli.stopped', showStimuli.tStop)
    thisExp.nextEntry()
    # the Routine "showStimuli" was not non-slip safe, so reset the non-slip timer
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
    # Run 'Begin Routine' code from retrocue
    '''
    # Randomly set retrocue direction (left or right)
    retrocue_direction = random.choice([0, 180])  # 0 = right, 180 = left
    retrocueMask.ori = retrocue_direction
    
    cueCircColor.colorSpace = 'rgb255'
    cueCircColor.color = cueCorThis
    cueCircColor.setAutoDraw(True)
    retrocueMask.setAutoDraw(True)
    
    clock.reset()
    kb.clock.reset()
    kb.clearEvents()
    
    win.flip()
    '''
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
    retrocue.forceEnded = routineForceEnded = not continueRoutine
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
    thisExp.nextEntry()
    # the Routine "retrocue" was not non-slip safe, so reset the non-slip timer
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
