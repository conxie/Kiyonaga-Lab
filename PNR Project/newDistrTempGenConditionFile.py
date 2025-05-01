import numpy as np
import pandas as pd
import glob
import random
import os


"""trial countdown related variables"""
#category information
uniqueCategories = ['cylinder', 'cube']


prevVisualDistractors = []
uniqueCategoriesCnt = len(uniqueCategories)


def drawImageFromCategory(categoryToDraw,exceptThese = None,
                          pathroot = 'PNR Project\stimuli'):
    #draw an stimuli from a category categoryToDraw
    #categoryToDraw: cylinder or cube
    pathRight = imageFilePath+pathroot+f'\{categoryToDraw}\*.jpg'
    
    if exceptThese is None:
        return np.random.choice(glob.glob(pathRight))
    else:
        return np.random.choice([i for i in glob.glob(pathRight) if not exceptThese in i])
    

def shuffleDict(dictIn):
    keys =  list(dictIn.keys())      
    random.shuffle(keys)
    return [(key, dictIn[key]) for key in keys]

def genProbes(thisTrl):

    probeArr = []

    #extract some column info
    if thisTrl.cuedItem == 'left':
        cuedCategory = thisTrl.leftCategory
        cuedImageID = thisTrl.leftImageID


    else:
        cuedCategory = thisTrl.rightCategory
        cuedImageID = thisTrl.rightImageID

def main():
    # Declare global for use in other functions
    global imageFilePath  
    global uniqueCategories
    global imEachCat
    global distractorPerc
    global distractorLabels
    global foilCategories

    """trial countdown related variables"""
    #category information
    uniqueCategories = ['cylinder', 'cube']
    #foilCategories = ['wheelbarrow', 'swingrides', 'tent', 'barn', 'inflatableplayground',
    #                    'patioswing', 'tollplaza', 'ambulance', 'helicopter']
    
    uniqueCategoriesCnt = len(uniqueCategories)

    imEachCat = 12 #how many images are in 1 category NOTE: MUST be divisible by 4
    trlTotal = uniqueCategoriesCnt*imEachCat #total trial = 5 categories * #images in each category *2 repetition
    #trlEachBlk = 12 #how many trials are in each block
    #blkTotal = int(trlTotal/trlEachBlk)


    #init counter
    trlCntTotal = 0

    """timing related params"""
    fixationT = 0.5
    stimT = 3
    retrocueT = 0.5
    noisePatchT = 0.25
    preCueRestT = 0.5
    delayT = 4
    noDistractorDelayT = 2
    distractorT = 0.5
    probeT = 20
    itiT = 0.5

    """display relayed variables"""
    #init position/size related params
    leftImPos = [-0.2,0]
    rightImPos = [0.2,0]
    fixSize = np.array([0.12,0.12])#np.array([0.16,0.16])

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

    """initialize stimuli variables"""
    #imageFilePath
    imageFilePath =''

    #load neutral images
    #neutralGreyImages = glob.glob(imageFilePath + 'sceneSet2/step3_probes/*.jpg')

    #load stims
    cubeShape = glob.glob('stimuli\cubes\*.jpg')
    cylinderShape = glob.glob('stimuli\cylinders\*.jpg')

    #shuffle stimuli
    random.shuffle(cylinderShape)
    random.shuffle(cubeShape)

    leftImagePath = []
    rightImagePath = []

    #create stim brightness array, 0 = day, 1 = night;
    #leftStimBri = [0]*int(trlTotal/2) + [1]*int(trlTotal/2) 
    #random.shuffle(leftStimBri)
    #leftStimBri = np.array(leftStimBri)
    #rightStimBri = 1- np.array(leftStimBri)

    """determine trial type visual or categorical"""
    # create trial condtion (0 = visual; 1 = abstract)
    #trlType = ['visual']*int(trlTotal/2) + ['verbal']*int(trlTotal/2) 
    #random.shuffle(trlType)

    """determine cued item"""
    cuedItem = ['left']*int(trlTotal/2) + ['right']*int(trlTotal/2) 
    random.shuffle(cuedItem)
    
    """create stimuli"""
    #draw an equal number of images from each category for cube images
    #temp = pd.DataFrame(zip(['cube' for i in cubeShape],cubeShape)).rename(columns={0:'category'})
   # stim_cubeDf = temp.groupby('category').sample(n = imEachCat*2)
    """random_idx = np.random.choice(len(cubeShape), imEachCat, replace=True)
    stim_cubeDf = [cubeShape[i] for i in random_idx]
    halfImEach = int(imEachCat // 2)
    
    leftStim_cube = np.random.choice(stim_cubeDf,size=halfImEach)
    print(len(stim_cubeDf), len(leftStim_cube))
    right_cube_options = np.setdiff1d(stim_cubeDf, leftStim_cube)
    print(len(right_cube_options))

    if len(right_cube_options) < imEachCat:
        raise ValueError(f"Not enough cube images available. Needed {imEachCat}, have {len(right_cube_options)}")
    rightStim_cube = np.random.choice(right_cube_options, size=halfImEach) """
    stim_cubeDf = np.random.choice(cubeShape, size=imEachCat, replace=True)
    halfImEach = imEachCat // 2

    leftStim_cube = np.random.choice(stim_cubeDf, size=halfImEach, replace=True)
    rightStim_cube = []

    for left_img in leftStim_cube:
    # Sample until different
        while True:
            candidate = np.random.choice(stim_cubeDf)
            if candidate != left_img:
                rightStim_cube.append(candidate)
            break

    #rightStim_cube = np.random.choice(np.setdiff1d(stim_cubeDf[1].values, leftStim_cube),size=imEachCat, replace=False) <- doesn't work cuz replace is weird

    stim_cylinderDf = np.random.choice(cylinderShape, size=imEachCat, replace=True)
    leftStim_cylinder = np.random.choice(stim_cylinderDf, size=halfImEach, replace=True)
    rightStim_cylinder = []

    for left_img in leftStim_cylinder:
        while True:
            candidate = np.random.choice(stim_cylinderDf)
            if candidate != left_img:
                rightStim_cylinder.append(candidate)
            break

    #add all the cube images to the dataframe
    stimTemp_cube1 = pd.DataFrame(zip(['cube'for i in leftStim_cube],leftStim_cube)).rename(columns={0:'leftCategory',1:'leftImagePath'})
    stimTemp_cube2 = pd.DataFrame(zip(['cube' for i in rightStim_cube],rightStim_cube)).rename(columns={0:'rightCategory',1:'rightImagePath'})
    cubeDf = pd.merge(stimTemp_cube1,stimTemp_cube2,left_index=True, right_index=True, how='outer')
    #draw an equal number of images from each category for cube images
    #temp = pd.DataFrame(zip(['cylinder' for i in cylinderShape],cylinderShape)).rename(columns={0:'category'})
    #stim_cylinderDf = temp.groupby('category').sample(n = imEachCat*2)

    """random_idx = np.random.choice(len(cylinderShape), imEachCat, replace=True)
    stim_cylinderDf = [cylinderShape[i] for i in random_idx]
    leftStim_cylinder = np.random.choice(stim_cylinderDf,size=halfImEach, replace=False)
    right_cylinder_options = np.setdiff1d(stim_cylinderDf, leftStim_cylinder)
    if len(right_cylinder_options) < imEachCat:
        raise ValueError(f"Not enough cube images available. Needed {imEachCat}, have {len(right_cylinder_options)}")
    rightStim_cylinder = np.random.choice(right_cylinder_options, size=halfImEach, replace=False)"""
    #rightStim_cylinder = np.random.choice(np.setdiff1d(stim_cylinderDf[1].values, leftStim_cylinder),size=imEachCat, replace=False)

    #add all the cube images to the dataframe
    stimTemp_cy1 = pd.DataFrame(zip(['cylinder'for i in leftStim_cylinder],leftStim_cylinder)).rename(columns={0:'leftCategory',1:'leftImagePath'})
    stimTemp_cy2 = pd.DataFrame(zip(['cylinder' for i in rightStim_cylinder],rightStim_cylinder)).rename(columns={0:'rightCategory',1:'rightImagePath'})
    cylinderDf = pd.merge(stimTemp_cy1,stimTemp_cy2,left_index=True, right_index=True, how='outer')

    #now combine the above
    #print(cubeDf.shape, cylinderDf.shape)
    df = pd.concat([cubeDf, cylinderDf], ignore_index=True)#.reset_index(drop=True)
    #add trial conditions
    #df['trlType'] = trlType
   

    # Check: total number of trials
    num_trials = len(df)

# Randomly shuffle trial indices
    all_indices = np.arange(num_trials)
    np.random.shuffle(all_indices)

# Choose half for 'left', rest for 'right'
    half = num_trials // 2
    left_indices = all_indices[:half]
    right_indices = all_indices[half:]

# Initialize cuedItem column with default
    df['cuedItem'] = 'right'
    df.loc[left_indices, 'cuedItem'] = 'left'

    df.to_csv('connie.csv')
    


if __name__ == "__main__":
    main()
