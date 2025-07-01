import numpy as np
import pandas as pd
import glob
import random
import os


"""trial countdown related variables"""
#category information
uniqueCategories = ['cube']


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

def get_distance_label(filename):
    """Extract distance label from filename."""
    basename = os.path.basename(filename).lower()
    if 'near' in basename:
        return 'near'
    elif 'middle' in basename:
        return 'middle'
    elif 'far' in basename:
        return 'far'
    else:
        return 'unknown'

# equal groups of far, near, middle


def main():
    global imageFilePath  

    uniqueCategories = ['cube']
    total_images = 60  
    num_pairs = total_images // 2 # total trials
    distances = ['near', 'middle', 'far']
    num_cat = len(distances)
    cat_assignments = (distances * (num_pairs // num_cat))

    sides = ['right', 'left']
    side_assignments = (sides * (num_pairs // 2))
    # Shuffle the assignments randomly
    random.shuffle(cat_assignments)
    random.shuffle(side_assignments)


    # Load cube images
    cubeShape = glob.glob('stimuli\\cubes_resized\\*.jpg')
    distance_groups = {'near': [], 'middle': [], 'far': []}

    for img in cubeShape:
        label = get_distance_label(img)
        if label in distance_groups:
            distance_groups[label].append(img)

    leftImagePath = []
    rightImagePath = []
    leftDistance = []
    rightDistance = []


    for _ in range(num_pairs):
        dist_left, dist_right = random.sample(distances, 2)

        left_img = random.choice(distance_groups[dist_left])
        right_img = random.choice(distance_groups[dist_right])

    # Randomly assign left and right images
        if random.random() < 0.5:
            left_img, right_img = left_img, right_img
            l_dist, r_dist = dist_left, dist_right
        else:
            left_img, right_img = right_img, left_img
            l_dist, r_dist = dist_right, dist_left

        leftImagePath.append(left_img)
        rightImagePath.append(right_img)
        leftDistance.append(l_dist)
        rightDistance.append(r_dist)

    cubeDf = pd.DataFrame({
        'leftCategory': ['cube'] * num_pairs, #placeholder
        'leftImagePath': leftImagePath,
        'rightCategory': ['cube'] * num_pairs, #placeholder
        'rightImagePath': rightImagePath,
        'cuedItem': side_assignments,
        'leftDistance': leftDistance,
        'rightDistance': rightDistance 
    })

    # Save
    cubeDf.to_csv('Cubeconnie_distance_matched.csv')
    print(f"Saved {len(cubeDf)} trials to Cubeconnie_distance_matched.csv")

if __name__ == "__main__":
    main()
