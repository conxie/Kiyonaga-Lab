/********************************* 
 * Keoghrevisedretrocuetask *
 *********************************/


// store info about the experiment session:
let expName = 'KeoghRevisedRetrocueTask';  // from the Builder filename that created this script
let expInfo = {
    'participant': `${util.pad(Number.parseFloat(util.randint(0, 999999)).toFixed(0), 6)}`,
    'session': '001',
};

// Start code blocks for 'Before Experiment'
// init psychoJS:
const psychoJS = new PsychoJS({
  debug: true
});

// open window:
psychoJS.openWindow({
  fullscr: true,
  color: new util.Color([0,0,0]),
  units: 'height',
  waitBlanking: true,
  backgroundImage: '',
  backgroundFit: 'none',
});
// schedule the experiment:
psychoJS.schedule(psychoJS.gui.DlgFromDict({
  dictionary: expInfo,
  title: expName
}));

const flowScheduler = new Scheduler(psychoJS);
const dialogCancelScheduler = new Scheduler(psychoJS);
psychoJS.scheduleCondition(function() { return (psychoJS.gui.dialogComponent.button === 'OK'); },flowScheduler, dialogCancelScheduler);

// flowScheduler gets run if the participants presses OK
flowScheduler.add(updateInfo); // add timeStamp
flowScheduler.add(experimentInit);
flowScheduler.add(loadExpVarRoutineBegin());
flowScheduler.add(loadExpVarRoutineEachFrame());
flowScheduler.add(loadExpVarRoutineEnd());
const expBlkLoopScheduler = new Scheduler(psychoJS);
flowScheduler.add(expBlkLoopBegin(expBlkLoopScheduler));
flowScheduler.add(expBlkLoopScheduler);
flowScheduler.add(expBlkLoopEnd);










flowScheduler.add(quitPsychoJS, '', true);

// quit if user presses Cancel in dialog box:
dialogCancelScheduler.add(quitPsychoJS, '', false);

psychoJS.start({
  expName: expName,
  expInfo: expInfo,
  resources: [
    // resources:
  ]
});

psychoJS.experimentLogger.setLevel(core.Logger.ServerLevel.WARNING);


var currentLoop;
var frameDur;
async function updateInfo() {
  currentLoop = psychoJS.experiment;  // right now there are no loops
  expInfo['date'] = util.MonotonicClock.getDateStr();  // add a simple timestamp
  expInfo['expName'] = expName;
  expInfo['psychopyVersion'] = '2024.2.4';
  expInfo['OS'] = window.navigator.platform;


  // store frame rate of monitor if we can measure it successfully
  expInfo['frameRate'] = psychoJS.window.getActualFrameRate();
  if (typeof expInfo['frameRate'] !== 'undefined')
    frameDur = 1.0 / Math.round(expInfo['frameRate']);
  else
    frameDur = 1.0 / 60.0; // couldn't get a reliable measure so guess

  // add info from the URL:
  util.addInfoFromUrl(expInfo);
  

  
  psychoJS.experiment.dataFileName = (("." + "/") + `data/${expInfo["participant"]}_${expName}_${expInfo["date"]}`);
  psychoJS.experiment.field_separator = '\t';


  return Scheduler.Event.NEXT;
}


var loadExpVarClock;
var blockInitClock;
var trlInitClock;
var fixationCircleClock;
var stimClock;
var preCueRestClock;
var retrocueClock;
var delayCode_2Clock;
var globalClock;
var routineTimer;
async function experimentInit() {
  // Initialize components for Routine "loadExpVar"
  loadExpVarClock = new util.Clock();
  // Initialize components for Routine "blockInit"
  blockInitClock = new util.Clock();
  // Initialize components for Routine "trlInit"
  trlInitClock = new util.Clock();
  // Initialize components for Routine "fixationCircle"
  fixationCircleClock = new util.Clock();
  // Initialize components for Routine "stim"
  stimClock = new util.Clock();
  // Initialize components for Routine "preCueRest"
  preCueRestClock = new util.Clock();
  // Initialize components for Routine "retrocue"
  retrocueClock = new util.Clock();
  // Initialize components for Routine "delayCode_2"
  delayCode_2Clock = new util.Clock();
  // Create some handy timers
  globalClock = new util.Clock();  // to track the time since experiment started
  routineTimer = new util.CountdownTimer();  // to track time remaining of each (non-slip) routine
  
  return Scheduler.Event.NEXT;
}


var t;
var frameN;
var continueRoutine;
var loadExpVarMaxDurationReached;
var loadExpVarMaxDuration;
var loadExpVarComponents;
function loadExpVarRoutineBegin(snapshot) {
  return async function () {
    TrialHandler.fromSnapshot(snapshot); // ensure that .thisN vals are up to date
    
    //--- Prepare to start Routine 'loadExpVar' ---
    t = 0;
    frameN = -1;
    continueRoutine = true; // until we're told otherwise
    loadExpVarClock.reset();
    routineTimer.reset();
    loadExpVarMaxDurationReached = false;
    // update component parameters for each repeat
    psychoJS.experiment.addData('loadExpVar.started', globalClock.getTime());
    loadExpVarMaxDuration = null
    // keep track of which components have finished
    loadExpVarComponents = [];
    
    loadExpVarComponents.forEach( function(thisComponent) {
      if ('status' in thisComponent)
        thisComponent.status = PsychoJS.Status.NOT_STARTED;
       });
    return Scheduler.Event.NEXT;
  }
}


function loadExpVarRoutineEachFrame() {
  return async function () {
    //--- Loop for each frame of Routine 'loadExpVar' ---
    // get current time
    t = loadExpVarClock.getTime();
    frameN = frameN + 1;// number of completed frames (so 0 is the first frame)
    // update/draw components on each frame
    // check for quit (typically the Esc key)
    if (psychoJS.experiment.experimentEnded || psychoJS.eventManager.getKeys({keyList:['escape']}).length > 0) {
      return quitPsychoJS('The [Escape] key was pressed. Goodbye!', false);
    }
    
    // check if the Routine should terminate
    if (!continueRoutine) {  // a component has requested a forced-end of Routine
      return Scheduler.Event.NEXT;
    }
    
    continueRoutine = false;  // reverts to True if at least one component still running
    loadExpVarComponents.forEach( function(thisComponent) {
      if ('status' in thisComponent && thisComponent.status !== PsychoJS.Status.FINISHED) {
        continueRoutine = true;
      }
    });
    
    // refresh the screen if continuing
    if (continueRoutine) {
      return Scheduler.Event.FLIP_REPEAT;
    } else {
      return Scheduler.Event.NEXT;
    }
  };
}


function loadExpVarRoutineEnd(snapshot) {
  return async function () {
    //--- Ending Routine 'loadExpVar' ---
    loadExpVarComponents.forEach( function(thisComponent) {
      if (typeof thisComponent.setAutoDraw === 'function') {
        thisComponent.setAutoDraw(false);
      }
    });
    psychoJS.experiment.addData('loadExpVar.stopped', globalClock.getTime());
    // the Routine "loadExpVar" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset();
    
    // Routines running outside a loop should always advance the datafile row
    if (currentLoop === psychoJS.experiment) {
      psychoJS.experiment.nextEntry(snapshot);
    }
    return Scheduler.Event.NEXT;
  }
}


var expBlk;
function expBlkLoopBegin(expBlkLoopScheduler, snapshot) {
  return async function() {
    TrialHandler.fromSnapshot(snapshot); // update internal variables (.thisN etc) of the loop
    
    // set up handler to look after randomisation of conditions etc
    expBlk = new TrialHandler({
      psychoJS: psychoJS,
      nReps: blkTotal, method: TrialHandler.Method.RANDOM,
      extraInfo: expInfo, originPath: undefined,
      trialList: undefined,
      seed: undefined, name: 'expBlk'
    });
    psychoJS.experiment.addLoop(expBlk); // add the loop to the experiment
    currentLoop = expBlk;  // we're now the current loop
    
    // Schedule all the trials in the trialList:
    expBlk.forEach(function() {
      snapshot = expBlk.getSnapshot();
    
      expBlkLoopScheduler.add(importConditions(snapshot));
      expBlkLoopScheduler.add(blockInitRoutineBegin(snapshot));
      expBlkLoopScheduler.add(blockInitRoutineEachFrame());
      expBlkLoopScheduler.add(blockInitRoutineEnd(snapshot));
      const trialsLoopScheduler = new Scheduler(psychoJS);
      expBlkLoopScheduler.add(trialsLoopBegin(trialsLoopScheduler, snapshot));
      expBlkLoopScheduler.add(trialsLoopScheduler);
      expBlkLoopScheduler.add(trialsLoopEnd);
      expBlkLoopScheduler.add(expBlkLoopEndIteration(expBlkLoopScheduler, snapshot));
    });
    
    return Scheduler.Event.NEXT;
  }
}


var trials;
function trialsLoopBegin(trialsLoopScheduler, snapshot) {
  return async function() {
    TrialHandler.fromSnapshot(snapshot); // update internal variables (.thisN etc) of the loop
    
    // set up handler to look after randomisation of conditions etc
    trials = new TrialHandler({
      psychoJS: psychoJS,
      nReps: trlEachBlk, method: TrialHandler.Method.RANDOM,
      extraInfo: expInfo, originPath: undefined,
      trialList: undefined,
      seed: undefined, name: 'trials'
    });
    psychoJS.experiment.addLoop(trials); // add the loop to the experiment
    currentLoop = trials;  // we're now the current loop
    
    // Schedule all the trials in the trialList:
    trials.forEach(function() {
      snapshot = trials.getSnapshot();
    
      trialsLoopScheduler.add(importConditions(snapshot));
      trialsLoopScheduler.add(trlInitRoutineBegin(snapshot));
      trialsLoopScheduler.add(trlInitRoutineEachFrame());
      trialsLoopScheduler.add(trlInitRoutineEnd(snapshot));
      trialsLoopScheduler.add(fixationCircleRoutineBegin(snapshot));
      trialsLoopScheduler.add(fixationCircleRoutineEachFrame());
      trialsLoopScheduler.add(fixationCircleRoutineEnd(snapshot));
      trialsLoopScheduler.add(stimRoutineBegin(snapshot));
      trialsLoopScheduler.add(stimRoutineEachFrame());
      trialsLoopScheduler.add(stimRoutineEnd(snapshot));
      trialsLoopScheduler.add(preCueRestRoutineBegin(snapshot));
      trialsLoopScheduler.add(preCueRestRoutineEachFrame());
      trialsLoopScheduler.add(preCueRestRoutineEnd(snapshot));
      trialsLoopScheduler.add(retrocueRoutineBegin(snapshot));
      trialsLoopScheduler.add(retrocueRoutineEachFrame());
      trialsLoopScheduler.add(retrocueRoutineEnd(snapshot));
      trialsLoopScheduler.add(delayCode_2RoutineBegin(snapshot));
      trialsLoopScheduler.add(delayCode_2RoutineEachFrame());
      trialsLoopScheduler.add(delayCode_2RoutineEnd(snapshot));
      trialsLoopScheduler.add(trialsLoopEndIteration(trialsLoopScheduler, snapshot));
    });
    
    return Scheduler.Event.NEXT;
  }
}


async function trialsLoopEnd() {
  // terminate loop
  psychoJS.experiment.removeLoop(trials);
  // update the current loop from the ExperimentHandler
  if (psychoJS.experiment._unfinishedLoops.length>0)
    currentLoop = psychoJS.experiment._unfinishedLoops.at(-1);
  else
    currentLoop = psychoJS.experiment;  // so we use addData from the experiment
  return Scheduler.Event.NEXT;
}


function trialsLoopEndIteration(scheduler, snapshot) {
  // ------Prepare for next entry------
  return async function () {
    if (typeof snapshot !== 'undefined') {
      // ------Check if user ended loop early------
      if (snapshot.finished) {
        // Check for and save orphaned data
        if (psychoJS.experiment.isEntryEmpty()) {
          psychoJS.experiment.nextEntry(snapshot);
        }
        scheduler.stop();
      } else {
        psychoJS.experiment.nextEntry(snapshot);
      }
    return Scheduler.Event.NEXT;
    }
  };
}


async function expBlkLoopEnd() {
  // terminate loop
  psychoJS.experiment.removeLoop(expBlk);
  // update the current loop from the ExperimentHandler
  if (psychoJS.experiment._unfinishedLoops.length>0)
    currentLoop = psychoJS.experiment._unfinishedLoops.at(-1);
  else
    currentLoop = psychoJS.experiment;  // so we use addData from the experiment
  return Scheduler.Event.NEXT;
}


function expBlkLoopEndIteration(scheduler, snapshot) {
  // ------Prepare for next entry------
  return async function () {
    if (typeof snapshot !== 'undefined') {
      // ------Check if user ended loop early------
      if (snapshot.finished) {
        // Check for and save orphaned data
        if (psychoJS.experiment.isEntryEmpty()) {
          psychoJS.experiment.nextEntry(snapshot);
        }
        scheduler.stop();
      } else {
        psychoJS.experiment.nextEntry(snapshot);
      }
    return Scheduler.Event.NEXT;
    }
  };
}


var blockInitMaxDurationReached;
var blockInitMaxDuration;
var blockInitComponents;
function blockInitRoutineBegin(snapshot) {
  return async function () {
    TrialHandler.fromSnapshot(snapshot); // ensure that .thisN vals are up to date
    
    //--- Prepare to start Routine 'blockInit' ---
    t = 0;
    frameN = -1;
    continueRoutine = true; // until we're told otherwise
    blockInitClock.reset();
    routineTimer.reset();
    blockInitMaxDurationReached = false;
    // update component parameters for each repeat
    psychoJS.experiment.addData('blockInit.started', globalClock.getTime());
    blockInitMaxDuration = null
    // keep track of which components have finished
    blockInitComponents = [];
    
    blockInitComponents.forEach( function(thisComponent) {
      if ('status' in thisComponent)
        thisComponent.status = PsychoJS.Status.NOT_STARTED;
       });
    return Scheduler.Event.NEXT;
  }
}


function blockInitRoutineEachFrame() {
  return async function () {
    //--- Loop for each frame of Routine 'blockInit' ---
    // get current time
    t = blockInitClock.getTime();
    frameN = frameN + 1;// number of completed frames (so 0 is the first frame)
    // update/draw components on each frame
    // check for quit (typically the Esc key)
    if (psychoJS.experiment.experimentEnded || psychoJS.eventManager.getKeys({keyList:['escape']}).length > 0) {
      return quitPsychoJS('The [Escape] key was pressed. Goodbye!', false);
    }
    
    // check if the Routine should terminate
    if (!continueRoutine) {  // a component has requested a forced-end of Routine
      return Scheduler.Event.NEXT;
    }
    
    continueRoutine = false;  // reverts to True if at least one component still running
    blockInitComponents.forEach( function(thisComponent) {
      if ('status' in thisComponent && thisComponent.status !== PsychoJS.Status.FINISHED) {
        continueRoutine = true;
      }
    });
    
    // refresh the screen if continuing
    if (continueRoutine) {
      return Scheduler.Event.FLIP_REPEAT;
    } else {
      return Scheduler.Event.NEXT;
    }
  };
}


function blockInitRoutineEnd(snapshot) {
  return async function () {
    //--- Ending Routine 'blockInit' ---
    blockInitComponents.forEach( function(thisComponent) {
      if (typeof thisComponent.setAutoDraw === 'function') {
        thisComponent.setAutoDraw(false);
      }
    });
    psychoJS.experiment.addData('blockInit.stopped', globalClock.getTime());
    // the Routine "blockInit" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset();
    
    // Routines running outside a loop should always advance the datafile row
    if (currentLoop === psychoJS.experiment) {
      psychoJS.experiment.nextEntry(snapshot);
    }
    return Scheduler.Event.NEXT;
  }
}


var trlInitMaxDurationReached;
var trlInitMaxDuration;
var trlInitComponents;
function trlInitRoutineBegin(snapshot) {
  return async function () {
    TrialHandler.fromSnapshot(snapshot); // ensure that .thisN vals are up to date
    
    //--- Prepare to start Routine 'trlInit' ---
    t = 0;
    frameN = -1;
    continueRoutine = true; // until we're told otherwise
    trlInitClock.reset();
    routineTimer.reset();
    trlInitMaxDurationReached = false;
    // update component parameters for each repeat
    psychoJS.experiment.addData('trlInit.started', globalClock.getTime());
    trlInitMaxDuration = null
    // keep track of which components have finished
    trlInitComponents = [];
    
    trlInitComponents.forEach( function(thisComponent) {
      if ('status' in thisComponent)
        thisComponent.status = PsychoJS.Status.NOT_STARTED;
       });
    return Scheduler.Event.NEXT;
  }
}


function trlInitRoutineEachFrame() {
  return async function () {
    //--- Loop for each frame of Routine 'trlInit' ---
    // get current time
    t = trlInitClock.getTime();
    frameN = frameN + 1;// number of completed frames (so 0 is the first frame)
    // update/draw components on each frame
    // check for quit (typically the Esc key)
    if (psychoJS.experiment.experimentEnded || psychoJS.eventManager.getKeys({keyList:['escape']}).length > 0) {
      return quitPsychoJS('The [Escape] key was pressed. Goodbye!', false);
    }
    
    // check if the Routine should terminate
    if (!continueRoutine) {  // a component has requested a forced-end of Routine
      return Scheduler.Event.NEXT;
    }
    
    continueRoutine = false;  // reverts to True if at least one component still running
    trlInitComponents.forEach( function(thisComponent) {
      if ('status' in thisComponent && thisComponent.status !== PsychoJS.Status.FINISHED) {
        continueRoutine = true;
      }
    });
    
    // refresh the screen if continuing
    if (continueRoutine) {
      return Scheduler.Event.FLIP_REPEAT;
    } else {
      return Scheduler.Event.NEXT;
    }
  };
}


function trlInitRoutineEnd(snapshot) {
  return async function () {
    //--- Ending Routine 'trlInit' ---
    trlInitComponents.forEach( function(thisComponent) {
      if (typeof thisComponent.setAutoDraw === 'function') {
        thisComponent.setAutoDraw(false);
      }
    });
    psychoJS.experiment.addData('trlInit.stopped', globalClock.getTime());
    // the Routine "trlInit" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset();
    
    // Routines running outside a loop should always advance the datafile row
    if (currentLoop === psychoJS.experiment) {
      psychoJS.experiment.nextEntry(snapshot);
    }
    return Scheduler.Event.NEXT;
  }
}


var fixationCircleMaxDurationReached;
var fixationCircleMaxDuration;
var fixationCircleComponents;
function fixationCircleRoutineBegin(snapshot) {
  return async function () {
    TrialHandler.fromSnapshot(snapshot); // ensure that .thisN vals are up to date
    
    //--- Prepare to start Routine 'fixationCircle' ---
    t = 0;
    frameN = -1;
    continueRoutine = true; // until we're told otherwise
    fixationCircleClock.reset();
    routineTimer.reset();
    fixationCircleMaxDurationReached = false;
    // update component parameters for each repeat
    psychoJS.experiment.addData('fixationCircle.started', globalClock.getTime());
    fixationCircleMaxDuration = null
    // keep track of which components have finished
    fixationCircleComponents = [];
    
    fixationCircleComponents.forEach( function(thisComponent) {
      if ('status' in thisComponent)
        thisComponent.status = PsychoJS.Status.NOT_STARTED;
       });
    return Scheduler.Event.NEXT;
  }
}


function fixationCircleRoutineEachFrame() {
  return async function () {
    //--- Loop for each frame of Routine 'fixationCircle' ---
    // get current time
    t = fixationCircleClock.getTime();
    frameN = frameN + 1;// number of completed frames (so 0 is the first frame)
    // update/draw components on each frame
    // check for quit (typically the Esc key)
    if (psychoJS.experiment.experimentEnded || psychoJS.eventManager.getKeys({keyList:['escape']}).length > 0) {
      return quitPsychoJS('The [Escape] key was pressed. Goodbye!', false);
    }
    
    // check if the Routine should terminate
    if (!continueRoutine) {  // a component has requested a forced-end of Routine
      return Scheduler.Event.NEXT;
    }
    
    continueRoutine = false;  // reverts to True if at least one component still running
    fixationCircleComponents.forEach( function(thisComponent) {
      if ('status' in thisComponent && thisComponent.status !== PsychoJS.Status.FINISHED) {
        continueRoutine = true;
      }
    });
    
    // refresh the screen if continuing
    if (continueRoutine) {
      return Scheduler.Event.FLIP_REPEAT;
    } else {
      return Scheduler.Event.NEXT;
    }
  };
}


function fixationCircleRoutineEnd(snapshot) {
  return async function () {
    //--- Ending Routine 'fixationCircle' ---
    fixationCircleComponents.forEach( function(thisComponent) {
      if (typeof thisComponent.setAutoDraw === 'function') {
        thisComponent.setAutoDraw(false);
      }
    });
    psychoJS.experiment.addData('fixationCircle.stopped', globalClock.getTime());
    // the Routine "fixationCircle" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset();
    
    // Routines running outside a loop should always advance the datafile row
    if (currentLoop === psychoJS.experiment) {
      psychoJS.experiment.nextEntry(snapshot);
    }
    return Scheduler.Event.NEXT;
  }
}


var stimMaxDurationReached;
var stimMaxDuration;
var stimComponents;
function stimRoutineBegin(snapshot) {
  return async function () {
    TrialHandler.fromSnapshot(snapshot); // ensure that .thisN vals are up to date
    
    //--- Prepare to start Routine 'stim' ---
    t = 0;
    frameN = -1;
    continueRoutine = true; // until we're told otherwise
    stimClock.reset();
    routineTimer.reset();
    stimMaxDurationReached = false;
    // update component parameters for each repeat
    psychoJS.experiment.addData('stim.started', globalClock.getTime());
    stimMaxDuration = null
    // keep track of which components have finished
    stimComponents = [];
    
    stimComponents.forEach( function(thisComponent) {
      if ('status' in thisComponent)
        thisComponent.status = PsychoJS.Status.NOT_STARTED;
       });
    return Scheduler.Event.NEXT;
  }
}


function stimRoutineEachFrame() {
  return async function () {
    //--- Loop for each frame of Routine 'stim' ---
    // get current time
    t = stimClock.getTime();
    frameN = frameN + 1;// number of completed frames (so 0 is the first frame)
    // update/draw components on each frame
    // check for quit (typically the Esc key)
    if (psychoJS.experiment.experimentEnded || psychoJS.eventManager.getKeys({keyList:['escape']}).length > 0) {
      return quitPsychoJS('The [Escape] key was pressed. Goodbye!', false);
    }
    
    // check if the Routine should terminate
    if (!continueRoutine) {  // a component has requested a forced-end of Routine
      return Scheduler.Event.NEXT;
    }
    
    continueRoutine = false;  // reverts to True if at least one component still running
    stimComponents.forEach( function(thisComponent) {
      if ('status' in thisComponent && thisComponent.status !== PsychoJS.Status.FINISHED) {
        continueRoutine = true;
      }
    });
    
    // refresh the screen if continuing
    if (continueRoutine) {
      return Scheduler.Event.FLIP_REPEAT;
    } else {
      return Scheduler.Event.NEXT;
    }
  };
}


function stimRoutineEnd(snapshot) {
  return async function () {
    //--- Ending Routine 'stim' ---
    stimComponents.forEach( function(thisComponent) {
      if (typeof thisComponent.setAutoDraw === 'function') {
        thisComponent.setAutoDraw(false);
      }
    });
    psychoJS.experiment.addData('stim.stopped', globalClock.getTime());
    // the Routine "stim" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset();
    
    // Routines running outside a loop should always advance the datafile row
    if (currentLoop === psychoJS.experiment) {
      psychoJS.experiment.nextEntry(snapshot);
    }
    return Scheduler.Event.NEXT;
  }
}


var preCueRestMaxDurationReached;
var preCueRestMaxDuration;
var preCueRestComponents;
function preCueRestRoutineBegin(snapshot) {
  return async function () {
    TrialHandler.fromSnapshot(snapshot); // ensure that .thisN vals are up to date
    
    //--- Prepare to start Routine 'preCueRest' ---
    t = 0;
    frameN = -1;
    continueRoutine = true; // until we're told otherwise
    preCueRestClock.reset();
    routineTimer.reset();
    preCueRestMaxDurationReached = false;
    // update component parameters for each repeat
    psychoJS.experiment.addData('preCueRest.started', globalClock.getTime());
    preCueRestMaxDuration = null
    // keep track of which components have finished
    preCueRestComponents = [];
    
    preCueRestComponents.forEach( function(thisComponent) {
      if ('status' in thisComponent)
        thisComponent.status = PsychoJS.Status.NOT_STARTED;
       });
    return Scheduler.Event.NEXT;
  }
}


function preCueRestRoutineEachFrame() {
  return async function () {
    //--- Loop for each frame of Routine 'preCueRest' ---
    // get current time
    t = preCueRestClock.getTime();
    frameN = frameN + 1;// number of completed frames (so 0 is the first frame)
    // update/draw components on each frame
    // check for quit (typically the Esc key)
    if (psychoJS.experiment.experimentEnded || psychoJS.eventManager.getKeys({keyList:['escape']}).length > 0) {
      return quitPsychoJS('The [Escape] key was pressed. Goodbye!', false);
    }
    
    // check if the Routine should terminate
    if (!continueRoutine) {  // a component has requested a forced-end of Routine
      return Scheduler.Event.NEXT;
    }
    
    continueRoutine = false;  // reverts to True if at least one component still running
    preCueRestComponents.forEach( function(thisComponent) {
      if ('status' in thisComponent && thisComponent.status !== PsychoJS.Status.FINISHED) {
        continueRoutine = true;
      }
    });
    
    // refresh the screen if continuing
    if (continueRoutine) {
      return Scheduler.Event.FLIP_REPEAT;
    } else {
      return Scheduler.Event.NEXT;
    }
  };
}


function preCueRestRoutineEnd(snapshot) {
  return async function () {
    //--- Ending Routine 'preCueRest' ---
    preCueRestComponents.forEach( function(thisComponent) {
      if (typeof thisComponent.setAutoDraw === 'function') {
        thisComponent.setAutoDraw(false);
      }
    });
    psychoJS.experiment.addData('preCueRest.stopped', globalClock.getTime());
    // the Routine "preCueRest" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset();
    
    // Routines running outside a loop should always advance the datafile row
    if (currentLoop === psychoJS.experiment) {
      psychoJS.experiment.nextEntry(snapshot);
    }
    return Scheduler.Event.NEXT;
  }
}


var retrocueMaxDurationReached;
var retrocueMaxDuration;
var retrocueComponents;
function retrocueRoutineBegin(snapshot) {
  return async function () {
    TrialHandler.fromSnapshot(snapshot); // ensure that .thisN vals are up to date
    
    //--- Prepare to start Routine 'retrocue' ---
    t = 0;
    frameN = -1;
    continueRoutine = true; // until we're told otherwise
    retrocueClock.reset();
    routineTimer.reset();
    retrocueMaxDurationReached = false;
    // update component parameters for each repeat
    psychoJS.experiment.addData('retrocue.started', globalClock.getTime());
    retrocueMaxDuration = null
    // keep track of which components have finished
    retrocueComponents = [];
    
    retrocueComponents.forEach( function(thisComponent) {
      if ('status' in thisComponent)
        thisComponent.status = PsychoJS.Status.NOT_STARTED;
       });
    return Scheduler.Event.NEXT;
  }
}


function retrocueRoutineEachFrame() {
  return async function () {
    //--- Loop for each frame of Routine 'retrocue' ---
    // get current time
    t = retrocueClock.getTime();
    frameN = frameN + 1;// number of completed frames (so 0 is the first frame)
    // update/draw components on each frame
    // check for quit (typically the Esc key)
    if (psychoJS.experiment.experimentEnded || psychoJS.eventManager.getKeys({keyList:['escape']}).length > 0) {
      return quitPsychoJS('The [Escape] key was pressed. Goodbye!', false);
    }
    
    // check if the Routine should terminate
    if (!continueRoutine) {  // a component has requested a forced-end of Routine
      return Scheduler.Event.NEXT;
    }
    
    continueRoutine = false;  // reverts to True if at least one component still running
    retrocueComponents.forEach( function(thisComponent) {
      if ('status' in thisComponent && thisComponent.status !== PsychoJS.Status.FINISHED) {
        continueRoutine = true;
      }
    });
    
    // refresh the screen if continuing
    if (continueRoutine) {
      return Scheduler.Event.FLIP_REPEAT;
    } else {
      return Scheduler.Event.NEXT;
    }
  };
}


function retrocueRoutineEnd(snapshot) {
  return async function () {
    //--- Ending Routine 'retrocue' ---
    retrocueComponents.forEach( function(thisComponent) {
      if (typeof thisComponent.setAutoDraw === 'function') {
        thisComponent.setAutoDraw(false);
      }
    });
    psychoJS.experiment.addData('retrocue.stopped', globalClock.getTime());
    // the Routine "retrocue" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset();
    
    // Routines running outside a loop should always advance the datafile row
    if (currentLoop === psychoJS.experiment) {
      psychoJS.experiment.nextEntry(snapshot);
    }
    return Scheduler.Event.NEXT;
  }
}


var delayCode_2MaxDurationReached;
var delayCode_2MaxDuration;
var delayCode_2Components;
function delayCode_2RoutineBegin(snapshot) {
  return async function () {
    TrialHandler.fromSnapshot(snapshot); // ensure that .thisN vals are up to date
    
    //--- Prepare to start Routine 'delayCode_2' ---
    t = 0;
    frameN = -1;
    continueRoutine = true; // until we're told otherwise
    delayCode_2Clock.reset();
    routineTimer.reset();
    delayCode_2MaxDurationReached = false;
    // update component parameters for each repeat
    psychoJS.experiment.addData('delayCode_2.started', globalClock.getTime());
    delayCode_2MaxDuration = null
    // keep track of which components have finished
    delayCode_2Components = [];
    
    delayCode_2Components.forEach( function(thisComponent) {
      if ('status' in thisComponent)
        thisComponent.status = PsychoJS.Status.NOT_STARTED;
       });
    return Scheduler.Event.NEXT;
  }
}


function delayCode_2RoutineEachFrame() {
  return async function () {
    //--- Loop for each frame of Routine 'delayCode_2' ---
    // get current time
    t = delayCode_2Clock.getTime();
    frameN = frameN + 1;// number of completed frames (so 0 is the first frame)
    // update/draw components on each frame
    // check for quit (typically the Esc key)
    if (psychoJS.experiment.experimentEnded || psychoJS.eventManager.getKeys({keyList:['escape']}).length > 0) {
      return quitPsychoJS('The [Escape] key was pressed. Goodbye!', false);
    }
    
    // check if the Routine should terminate
    if (!continueRoutine) {  // a component has requested a forced-end of Routine
      return Scheduler.Event.NEXT;
    }
    
    continueRoutine = false;  // reverts to True if at least one component still running
    delayCode_2Components.forEach( function(thisComponent) {
      if ('status' in thisComponent && thisComponent.status !== PsychoJS.Status.FINISHED) {
        continueRoutine = true;
      }
    });
    
    // refresh the screen if continuing
    if (continueRoutine) {
      return Scheduler.Event.FLIP_REPEAT;
    } else {
      return Scheduler.Event.NEXT;
    }
  };
}


function delayCode_2RoutineEnd(snapshot) {
  return async function () {
    //--- Ending Routine 'delayCode_2' ---
    delayCode_2Components.forEach( function(thisComponent) {
      if (typeof thisComponent.setAutoDraw === 'function') {
        thisComponent.setAutoDraw(false);
      }
    });
    psychoJS.experiment.addData('delayCode_2.stopped', globalClock.getTime());
    // the Routine "delayCode_2" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset();
    
    // Routines running outside a loop should always advance the datafile row
    if (currentLoop === psychoJS.experiment) {
      psychoJS.experiment.nextEntry(snapshot);
    }
    return Scheduler.Event.NEXT;
  }
}


function importConditions(currentLoop) {
  return async function () {
    psychoJS.importAttributes(currentLoop.getCurrentTrial());
    return Scheduler.Event.NEXT;
    };
}


async function quitPsychoJS(message, isCompleted) {
  // Check for and save orphaned data
  if (psychoJS.experiment.isEntryEmpty()) {
    psychoJS.experiment.nextEntry();
  }
  psychoJS.window.close();
  psychoJS.quit({message: message, isCompleted: isCompleted});
  
  return Scheduler.Event.QUIT;
}
