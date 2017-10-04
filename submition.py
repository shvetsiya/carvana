import numpy as np
import pandas as pd

import cv2
from config import *
from tqdm import tqdm

# https://www.kaggle.com/stainsby/fast-tested-rle
def run_length_encode(mask):
    '''
    mask: binary numpy array, 1 - mask, 0 - background
    Returns run length as string formated
    We save only coordinates of 1 and then the number of its repetitions
    '''
    
    fmask = mask.flatten()
    fmask[0] = 0
    fmask[-1] = 0
    #coordinates where the current pixel is different from the previous one starting from 1 (which is always 0)
    runs = np.where(fmask[1:] != fmask[:-1])[0] + 2
    runs[1::2] = runs[1::2] - runs[:-1:2] # every second pisition contains the number of repetition betwen previous and current positions
    rle = ' '.join([str(r) for r in runs])
    return rle

def main():
    df_test = pd.read_csv(DATA_ROOT / 'sample_submission.csv')
    ids_test = df_test['img'].map(lambda s: s.split('.')[0])
    
    rles = []
    for ids in tqdm(ids_test.values, total=ids_test.shape[0], unit=" masks"):
        mask = cv2.imread('{}/{}_pred_mask.png'.format(str(PREDICTIONS_DIR), ids), cv2.IMREAD_GRAYSCALE)
        mask = (mask.astype(np.float16)/255 > THRESHOLD).astype(np.uint8)

        rle = run_length_encode(mask)
        rles.append(rle)        

    print("Generating submission file...")
    names = []
    for img_id in ids_test:
        names.append('{}.jpg'.format(img_id))

    df = pd.DataFrame({'img': names, 'rle_mask': rles})
    df.to_csv('submits/submission_0_5.csv.gz', index=False, compression='gzip')

if __name__ == '__main__':
    main()
