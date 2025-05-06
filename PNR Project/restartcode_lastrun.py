#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
This experiment was created using PsychoPy3 Experiment Builder (v2024.2.4),
    on May 05, 2025, at 18:01
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

from psychopy import visual, event, core
import random
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
        originPath='C:\\Users\\cocon\\OneDrive\\Documents\\GitHub\\Kiyonaga-Lab\\PNR Project\\restartcode_lastrun.py',
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
    
    df = pd.read_csv('connie.csv')
    # Run 'Begin Experiment' code from stimParams
    imageFilePath = ''
    
    """trial countdown related variables"""
    #category information
    uniqueCategories = ['cylinder', 'cube']
    uniqueCategoriesCnt = len(uniqueCategories)
    
    imEachCat = 12 #how many images are in 1 category
    trlTotal = uniqueCategoriesCnt*imEachCat*2 #total trial = 9 categories * #images in each category *2 repetition
    trlEachBlk = 12 #how many trials are in each block
    blkTotal = int(trlTotal/trlEachBlk)
    
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
    itiT = 0.5
    
    """display relayed variables"""
    #init position/size related params
    leftImPos = [-0.2,0]
    rightImPos = [0.2,0]
    fixSize = np.array([0.12,0.12])#np.array([0.16,0.16])
    probeSize = [0.18,0.18]
    probePosVarX = 0.25
    probePosVarY = 0.12
    
    textSize = 0.03
    visualProbeSizeScalar = 1 #np.sqrt(2)
    
    #Cue colors
    cueColArr = [[0,128,0],[128,0,128]] #green,purple
    cueColDefault = [128,115,96]
    
    
    #
    clock = core.Clock()
    kb = keyboard.Keyboard()
    
    
    
    # --- Initialize components for Routine "blockInit" ---
    # Run 'Begin Experiment' code from initBlockParams
    trlCntThisBlk = 0
    
    # --- Initialize components for Routine "trlInit" ---
    
    # --- Initialize components for Routine "fixationCircle" ---
    
    # --- Initialize components for Routine "stim" ---
    
    # --- Initialize components for Routine "preCueRest" ---
    
    # --- Initialize components for Routine "retrocue" ---
    
    # --- Initialize components for Routine "delayCode_2" ---
    
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
    if thisSession is not None:
        # if running in a Session with a Liaison client, send data up to now
        thisSession.sendExperimentData()
    
    for thisExpBlk in expBlk:
        currentLoop = expBlk
        thisExp.timestampOnFlip(win, 'thisRow.t', format=globalClock.format)
        if thisSession is not None:
            # if running in a Session with a Liaison client, send data up to now
            thisSession.sendExperimentData()
        # abbreviate parameter names if possible (e.g. rgb = thisExpBlk.rgb)
        if thisExpBlk != None:
            for paramName in thisExpBlk:
                globals()[paramName] = thisExpBlk[paramName]
        
        # --- Prepare to start Routine "blockInit" ---
        # create an object to store info about Routine blockInit
        blockInit = data.Routine(
            name='blockInit',
            components=[],
        )
        blockInit.status = NOT_STARTED
        continueRoutine = True
        # update component parameters for each repeat
        # store start times for blockInit
        blockInit.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
        blockInit.tStart = globalClock.getTime(format='float')
        blockInit.status = STARTED
        thisExp.addData('blockInit.started', blockInit.tStart)
        blockInit.maxDuration = None
        # keep track of which components have finished
        blockInitComponents = blockInit.components
        for thisComponent in blockInit.components:
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
        
        # --- Run Routine "blockInit" ---
        # if trial has changed, end Routine now
        if isinstance(expBlk, data.TrialHandler2) and thisExpBlk.thisN != expBlk.thisTrial.thisN:
            continueRoutine = False
        blockInit.forceEnded = routineForceEnded = not continueRoutine
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
                blockInit.forceEnded = routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in blockInit.components:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "blockInit" ---
        for thisComponent in blockInit.components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        # store stop times for blockInit
        blockInit.tStop = globalClock.getTime(format='float')
        blockInit.tStopRefresh = tThisFlipGlobal
        thisExp.addData('blockInit.stopped', blockInit.tStop)
        # the Routine "blockInit" was not non-slip safe, so reset the non-slip timer
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
            
            """cue related params""" #CX modified this since I dont have a trlType col in df
            cueCircColor.fillColor = cueColDefault
            if df['cuedItem'][trlCntTotal] == 'left':
                cueCorThis = cueColArr[0] #green
            elif df['cuedItem'][trlCntTotal] == 'right':
                cueCorThis = cueColArr[0] #green
            else:
                cueCorThis = cueColArr[1] #purple
            
            
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
            # the Routine "trlInit" was not non-slip safe, so reset the non-slip timer
            routineTimer.reset()
            
            # --- Prepare to start Routine "fixationCircle" ---
            # create an object to store info about Routine fixationCircle
            fixationCircle = data.Routine(
                name='fixationCircle',
                components=[],
            )
            fixationCircle.status = NOT_STARTED
            continueRoutine = True
            # update component parameters for each repeat
            # Run 'Begin Routine' code from fixationCir
            cueCircColor.setAutoDraw(True)
            fixCircMask.setAutoDraw(True)
            
            clock.reset()
            kb.clock.reset()
            kb.clearEvents()
            
            win.flip()
            '''fixCircMask = visual.ImageStim(win, image=fixCircPath, size=fixSize)
            
            cueCircColor.color = cueColDefault
            cueCircColor.setAutoDraw(True)
            fixCircMask.setAutoDraw(True)
            
            clock.reset()
            kb.clock.reset()
            kb.clearEvents()
            
            win.flip()
            
            core.wait(0.5)'''
            # store start times for fixationCircle
            fixationCircle.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
            fixationCircle.tStart = globalClock.getTime(format='float')
            fixationCircle.status = STARTED
            thisExp.addData('fixationCircle.started', fixationCircle.tStart)
            fixationCircle.maxDuration = None
            # keep track of which components have finished
            fixationCircleComponents = fixationCircle.components
            for thisComponent in fixationCircle.components:
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
            
            # --- Run Routine "fixationCircle" ---
            # if trial has changed, end Routine now
            if isinstance(trials, data.TrialHandler2) and thisTrial.thisN != trials.thisTrial.thisN:
                continueRoutine = False
            fixationCircle.forceEnded = routineForceEnded = not continueRoutine
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
                    fixationCircle.forceEnded = routineForceEnded = True
                    break
                continueRoutine = False  # will revert to True if at least one component still running
                for thisComponent in fixationCircle.components:
                    if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                        continueRoutine = True
                        break  # at least one component has not yet finished
                
                # refresh the screen
                if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                    win.flip()
            
            # --- Ending Routine "fixationCircle" ---
            for thisComponent in fixationCircle.components:
                if hasattr(thisComponent, "setAutoDraw"):
                    thisComponent.setAutoDraw(False)
            # store stop times for fixationCircle
            fixationCircle.tStop = globalClock.getTime(format='float')
            fixationCircle.tStopRefresh = tThisFlipGlobal
            thisExp.addData('fixationCircle.stopped', fixationCircle.tStop)
            # Run 'End Routine' code from fixationCir
            cueCircColor.setAutoDraw(False)
            fixCircMask.setAutoDraw(False)
            win.flip()
            
            # the Routine "fixationCircle" was not non-slip safe, so reset the non-slip timer
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
            
            '''
            retrocueMask = visual.ImageStim(win, image=retrocuePath, size=fixSize) 
            # Randomly set retrocue direction (left or right)
            retrocue_direction = random.choice([0, 180])  # 0 = right, 180 = left
            retrocueMask.ori = retrocue_direction
            
            
            cueCircColor.colorSpace = 'rgb255'
            cueCircColor.color = cueCorThis
            cueCircColor.setAutoDraw(True)
            retrocueMask.setAutoDraw(True)
            
            # Update window to draw the stimuli
            win.flip()
            
            # Wait to make sure the image stays on screen
            core.wait(1.0)
            
            # Turn off auto-drawing after display time
            retrocueMask.setAutoDraw(False)
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
            
            
            '''cueCircColor.setAutoDraw(False)
            fixCircMask.setAutoDraw(False)
            win.flip()
            
            clock.reset()
            kb.clock.reset()
            kb.clearEvents()
            
            
            trlCntTotal += 1'''
            # the Routine "retrocue" was not non-slip safe, so reset the non-slip timer
            routineTimer.reset()
            
            # --- Prepare to start Routine "delayCode_2" ---
            # create an object to store info about Routine delayCode_2
            delayCode_2 = data.Routine(
                name='delayCode_2',
                components=[],
            )
            delayCode_2.status = NOT_STARTED
            continueRoutine = True
            # update component parameters for each repeat
            # Run 'Begin Routine' code from delayCode
            cueCircColor.color = cueColDefault
            
            cueCircColor.setAutoDraw(True)
            fixCircMask.setAutoDraw(True)
            noisePatchRight.setAutoDraw(False)
            noisePatchLeft.setAutoDraw(False)
            
            clock.reset()
            kb.clearEvents()
            
            win.flip()
            # store start times for delayCode_2
            delayCode_2.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
            delayCode_2.tStart = globalClock.getTime(format='float')
            delayCode_2.status = STARTED
            thisExp.addData('delayCode_2.started', delayCode_2.tStart)
            delayCode_2.maxDuration = None
            # keep track of which components have finished
            delayCode_2Components = delayCode_2.components
            for thisComponent in delayCode_2.components:
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
            
            # --- Run Routine "delayCode_2" ---
            # if trial has changed, end Routine now
            if isinstance(trials, data.TrialHandler2) and thisTrial.thisN != trials.thisTrial.thisN:
                continueRoutine = False
            delayCode_2.forceEnded = routineForceEnded = not continueRoutine
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
                    delayCode_2.forceEnded = routineForceEnded = True
                    break
                continueRoutine = False  # will revert to True if at least one component still running
                for thisComponent in delayCode_2.components:
                    if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                        continueRoutine = True
                        break  # at least one component has not yet finished
                
                # refresh the screen
                if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                    win.flip()
            
            # --- Ending Routine "delayCode_2" ---
            for thisComponent in delayCode_2.components:
                if hasattr(thisComponent, "setAutoDraw"):
                    thisComponent.setAutoDraw(False)
            # store stop times for delayCode_2
            delayCode_2.tStop = globalClock.getTime(format='float')
            delayCode_2.tStopRefresh = tThisFlipGlobal
            thisExp.addData('delayCode_2.stopped', delayCode_2.tStop)
            # the Routine "delayCode_2" was not non-slip safe, so reset the non-slip timer
            routineTimer.reset()
            thisExp.nextEntry()
            
        # completed trlEachBlk repeats of 'trials'
        
        if thisSession is not None:
            # if running in a Session with a Liaison client, send data up to now
            thisSession.sendExperimentData()
        thisExp.nextEntry()
        
    # completed blkTotal repeats of 'expBlk'
    
    if thisSession is not None:
        # if running in a Session with a Liaison client, send data up to now
        thisSession.sendExperimentData()
    
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
