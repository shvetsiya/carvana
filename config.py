from pathlib import Path
import os

import random
import numpy as np
import torch

SEED = 235202

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)


GPU_IDS = [0, 1]
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]=','.join(str(g) for g in GPU_IDS)

DATA_ROOT = Path('./input')

TRAIN_DIR = DATA_ROOT / 'train_hq'
MASKS_DIR = DATA_ROOT / 'train_masks'
TEST_DIR = DATA_ROOT / 'test_hq'

IM_SIZE_H = 1024
IM_SIZE_W = 1024#1024
THRESHOLD = 0.5

NUM_EPOCHS = 70
TRAIN_BATCH_SIZE = 16
TEST_BATCH_SIZE = 32
LEARNING_RATE = 0.0003

ORIGINAL_HEIGHT = 1280
ORIGINAL_WIDTH = 1918

MODELS_ROOT = Path('./output')
SAVED_MODEL = MODELS_ROOT / 'model_checkpoint_1024.pt'
BEST_MODEL = MODELS_ROOT / 'model_best_1024.pt'
LOG_ROOT = Path('./logs')
LOG_TB_VZ = LOG_ROOT / 'tb_viz'
LOG_TB_L = LOG_ROOT / 'tb_logs'
LOG_FILE = LOG_ROOT / 'training.log'


PREDICTIONS_DIR = Path('./predictions/')
SUBMISSIONS_DIR = Path('./submits/')

LOAD_MODEL = False
