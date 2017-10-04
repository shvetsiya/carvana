import os

logs = './logs/'
#models = './models/'
submissions = './submissions/'
data = './input/'
validations = './validations/'
predictions = './predictions/'
#thresholds = './thresholds/'
#ensemble_weights = './ensemble_weights/'
#xgb_configurations = './xgb_configurations/'

train = data+'train/'
train_mask = data+'train_masks'
train_mask_csv = data + 'train_masks.csv'

test = data+'test/'
sample_sub_scv = data + 'sample_submission.csv'

dirs = [
    logs,
    models,
    submissions,
    data,
    validations,
    predictions
]

data = [train, train_mask, test]
files = [train_mask_csv, sample_sub_scv]

for sub_dir in dirs:
    if os.path.isdir(sub_dir):
        continue

    if not os.path.isfile(sub_dir[:-1]):
        os.makedirs(sub_dir)
        print('Created directory', sub_dir)

for data_dir in data:
    if os.path.isdir(data_dir):
        continue

    else:
        print('Directoy {} does not exists. Please either put the training/test data in the appropriate directories or '
              'change the path.'.format(data_dir))


for file in files:
    if os.path.isfile(file):
        continue

    else:
        print('File {} does not exists. Please either put the file in the appropriate directories or '
              'change the path.'.format(file))
