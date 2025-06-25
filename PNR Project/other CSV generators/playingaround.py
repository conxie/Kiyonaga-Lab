import pandas as pd
import numpy as np
import glob
import random
import os

def main():

    # imageFilePath is empty because you're using relative paths
    imageFilePath = ''

    # Load and shuffle images
    cubeShape = glob.glob('stimuli/cubes/*.jpg')
    cylinderShape = glob.glob('stimuli/cylinders/*.jpg')
    random.shuffle(cubeShape)
    random.shuffle(cylinderShape)

    # Extract distance from filenames
    def get_distance(image_path):
        try:
            return os.path.basename(image_path).split('_')[-1].replace('.jpg', '')
        except:
            return 'Unknown'

    # Select and pair images with same category and same distance
    def pair_images(images, category, halfImEach):
        pairs = []
        distance_dict = {}
        for img in images:
            dist = get_distance(img)
            distance_dict.setdefault(dist, []).append(img)

        for dist, imgs in distance_dict.items():
            if len(imgs) < 2:
                continue
            random.shuffle(imgs)
            for i in range(0, len(imgs)-1, 2):
                if len(pairs) < halfImEach:
                    pairs.append((imgs[i], imgs[i+1], category, dist))
        return pairs

    imEachCat = 12
    halfImEach = imEachCat // 2

    cube_pairs = pair_images(cubeShape, 'cube', halfImEach)
    cyl_pairs = pair_images(cylinderShape, 'cylinder', halfImEach)

    if len(cube_pairs) < halfImEach or len(cyl_pairs) < halfImEach:
        raise ValueError("Not enough matched pairs in cube or cylinder")

    # Convert to DataFrames
    cubeDf = pd.DataFrame(cube_pairs, columns=['leftImagePath', 'rightImagePath', 'category', 'distance'])
    cylDf = pd.DataFrame(cyl_pairs, columns=['leftImagePath', 'rightImagePath', 'category', 'distance'])

    # Rename columns appropriately
    cubeDf = cubeDf.rename(columns={'category': 'leftCategory'})
    cubeDf['rightCategory'] = cubeDf['leftCategory']
    cubeDf['leftDistance'] = cubeDf['distance']
    cubeDf['rightDistance'] = cubeDf['distance']
    cubeDf = cubeDf.drop(columns=['distance'])

    cylDf = cylDf.rename(columns={'category': 'leftCategory'})
    cylDf['rightCategory'] = cylDf['leftCategory']
    cylDf['leftDistance'] = cylDf['distance']
    cylDf['rightDistance'] = cylDf['distance']
    cylDf = cylDf.drop(columns=['distance'])

    # Combine and shuffle
    df = pd.concat([cubeDf, cylDf], ignore_index=True)

    # Shuffle while avoiding >3 same category repeats
    def shuffle_no_streaks(data, col, max_repeat=3):
        for _ in range(1000):
            shuffled = data.sample(frac=1).reset_index(drop=True)
            streak = 1
            for i in range(1, len(shuffled)):
                if shuffled.loc[i, col] == shuffled.loc[i - 1, col]:
                    streak += 1
                    if streak > max_repeat:
                        break
                else:
                    streak = 1
            else:
                return shuffled
        raise ValueError(f"Couldn't shuffle to avoid >{max_repeat} repeats in '{col}'")

    df = shuffle_no_streaks(df, 'leftCategory', max_repeat=3)

    # Assign cuedItem randomly, half left, half right
    df['cuedItem'] = 'right'
    num_trials = len(df)
    indices = np.arange(num_trials)
    np.random.shuffle(indices)
    df.iloc[indices[:num_trials // 2], df.columns.get_loc('cuedItem')] = 'left'

    # Save final result
    df.to_csv('connie.csv', index=False)

