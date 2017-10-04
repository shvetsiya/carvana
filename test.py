import numpy as np
import pandas as pd

from torch.utils.data import DataLoader

from model.unet import UNet
from train_util import CarvanaSegmenationTest 
from datasets import CarvanaTestDataset
from config import *

def main():
    df_test = pd.read_csv(DATA_ROOT / 'sample_submission.csv')
    ids_test = df_test['img'].map(lambda s: s.split('.')[0])

    test_dataset = CarvanaTestDataset(ids_test.values)
    test_loader = DataLoader(test_dataset, shuffle=False, batch_size=TEST_BATCH_SIZE)

    classifier = CarvanaSegmenationTest(net = UNet(), pred_folder = str(PREDICTIONS_DIR))
    classifier.predict(test_loader)


if __name__ == '__main__':
    main()
