import pandas as pd
from sklearn.model_selection import train_test_split

from torch.utils.data import DataLoader

from train_util import CarvanaSegmenationTrain
from model.unet import UNet
from datasets import CarvanaTrainDataset
from callbacks import TensorBoardVisualizerCallback, TensorBoardLoggerCallback, ModelSaverCallback, SimpleLoggerCallback
from config import *





def main():
    df_train = pd.read_csv(DATA_ROOT / 'train_masks.csv')
    ids_train = df_train['img'].map(lambda s: s.split('.')[0])

    ids_train_split, ids_valid_split = train_test_split(ids_train, test_size=0.2, random_state=SEED)

    print('Training on {} samples'.format(len(ids_train_split)))
    print('Validating on {} samples'.format(len(ids_valid_split)))

    train_dataset = CarvanaTrainDataset(ids_train_split.values)
    valid_dataset = CarvanaTrainDataset(ids_valid_split.values)

    train_loader = DataLoader(train_dataset, shuffle=True, batch_size=TRAIN_BATCH_SIZE)
    valid_loader = DataLoader(valid_dataset, batch_size=TEST_BATCH_SIZE)

    tb_viz_cb = TensorBoardVisualizerCallback(str(LOG_TB_VZ))
    tb_logs_cb = TensorBoardLoggerCallback(str(LOG_TB_L))
    model_saver_cb = ModelSaverCallback(str(SAVED_MODEL), str(BEST_MODEL))
    logger = SimpleLoggerCallback(str(LOG_FILE))

    logs2 = [tb_viz_cb, tb_logs_cb, model_saver_cb, logger]
    classifier = CarvanaSegmenationTrain(net=UNet(),
                                         num_epochs=NUM_EPOCHS,
                                         learning_rate=LEARNING_RATE,
                                         load_model=LOAD_MODEL)
    classifier.train(train_loader, valid_loader, callbacks=logs2)



if __name__ == '__main__':
    main()
