import json
import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE

### Load data & preprocessing
observ_daterange = '1910_1912'
label_daterange = '2001'
try_date = '200825'
version = 'vv10'
desc = '?locationInfo=location_token&scaler=standardscaler'

##### (1) Sequence data in json
with open('make_sequence__observ_{}__labeled_{}_{}_{}_{}'.format(observ_daterange, label_daterange, try_date, version, desc)) as f:
    all_seq = json.load(f)
print(f'json data loaded. {type(all_seq)}')
##### (2) Actor_label_table
actor_label = pd.read_csv('actorLabelTable__observ_{}__labeled_{}_{}_{}_{}.csv'.format(observ_daterange, label_daterange, try_date, version, desc))
uid = actor_label.actor_id.values
##### (3) Original df after feature generation
df_train = pd.read_csv('featureGeneration__observ_{}__labeled_{}_{}_{}_{}.csv'.format(observ_daterange, label_daterange, try_date, version, desc))
print(f'Data loaded: {df_train.shape}')
##### (4) Raw data
df_origin = pd.read_csv('observ_{}__labeled_{}.csv'.format(observ_daterange, label_daterange), dtype={
    'atm_location': 'string',
    'source_bank_code': 'string',
    'target_bank_code': 'string'
})
print(f'Data loaded: {df_origin.shape}')


### check MAX_TIMESTEP from label=1 or 2
steps_each_actor = df_train.groupby(['actor_id', 'label']).size().reset_index().rename(columns={0: 'steps'}).sort_values('steps', ascending=False)
MAX_TIMESTEP = steps_each_actor[steps_each_actor['label'] != 0.0]['steps'].max()
print(f'The max timestep from label=1 or 2 is {MAX_TIMESTEP}.')


### Split data & check label num
train_X, test_X, train_Y, test_Y = train_test_split(all_seq, actor_label, test_size=0.3, random_state=99, stratify=actor_label.label)
train_x, val_x, train_y, val_y = train_test_split(train_X, train_Y, test_size=0.25, random_state=43, stratify=train_Y.label)
print(train_y.groupby(['label']).size())
print(val_y.groupby(['label']).size())
print(test_Y.groupby(['label']).size())

### Scaling data
#### training set
sd = StandardScaler()
trainset_for_scaling = df_train[df_train['actor_id'].isin(train_y.actor_id.unique())]
sd.fit(trainset_for_scaling.iloc[:, np.r_[2, 4:30]])
trainset_for_scaling.iloc[:, np.r_[2, 4:30]] = sd.transform(trainset_for_scaling.iloc[:, np.r_[2, 4:30]])
#### validation set
valset_for_scaling = df_train[df_train['actor_id'].isin(val_y.actor_id.unique())]
valset_for_scaling.iloc[:, np.r_[2, 4:30]] = sd.transform(valset_for_scaling.iloc[:, np.r_[2, 4:30]])
#### testing set
testset_for_scaling = df_train[df_train['actor_id'].isin(test_Y.actor_id.unique())]
testset_for_scaling.iloc[:, np.r_[2, 4:30]] = sd.transform(testset_for_scaling.iloc[:, np.r_[2, 4:30]])

### Update scaled data to sequence format & Create pre-padding
#### training set
train_x_seq = {}
for u in tqdm(trainset_for_scaling.actor_id.unique()):
    train_x_seq[u] = {}
    # update scaled data
    for c in trainset_for_scaling.columns[2:]:
        train_x_seq[u][c] = trainset_for_scaling[trainset_for_scaling['actor_id'] == u][c].values.tolist()

_train_x_seq = []
for u in tqdm(trainset_for_scaling.actor_id.unique()):
    # create pre-padding
    collect = []
    for key, value in train_x_seq[u].items():
        if key in ['label','clust_index']:
            pass
        else:
            tmp = value
            if len(tmp) < MAX_TIMESTEP:
                diff_to_pad = MAX_TIMESTEP - len(tmp)
                tmp = np.insert([0.0]*diff_to_pad, diff_to_pad, tmp).tolist()
            else:
                tmp = tmp[-MAX_TIMESTEP:]
                if key == 'time_diff':
                    tmp[0] = 0
            collect.append(tmp)

    collect = np.array(collect).T
    _train_x_seq.append(collect)

_train_x_seq = np.array(_train_x_seq)
print(_train_x_seq.shape)
### Store to NPY
np.save('train_x_seq__observ_{}__labeled_{}_{}_{}_{}.npy'.format(observ_daterange, label_daterange, try_date, version, desc), _train_x_seq)
np.save('train_y__observ_{}__labeled_{}_{}_{}_{}.npy'.format(observ_daterange, label_daterange, try_date, version, desc), train_y.label.values)
print('training sequence data finish storage.')

#### validation set
val_x_seq = {}
for u in tqdm(valset_for_scaling.actor_id.unique()):
    val_x_seq[u] = {}
    # update scaled data
    for c in valset_for_scaling.columns[2:]:
        val_x_seq[u][c] = valset_for_scaling[valset_for_scaling['actor_id'] == u][c].values.tolist()

_val_x_seq = []
for u in tqdm(valset_for_scaling.actor_id.unique()):
    # create pre-padding
    collect = []
    for key, value in val_x_seq[u].items():
        if key in ['label','clust_index']:
            pass
        else:
            tmp = value
            if len(tmp) < MAX_TIMESTEP:
                diff_to_pad = MAX_TIMESTEP - len(tmp)
                tmp = np.insert([0.0]*diff_to_pad, diff_to_pad, tmp).tolist()
            else:
                tmp = tmp[-MAX_TIMESTEP:]
                if key == 'time_diff':
                    tmp[0] = 0
            collect.append(tmp)

    collect = np.array(collect).T
    _val_x_seq.append(collect)

_val_x_seq = np.array(_val_x_seq)
print(_val_x_seq.shape)
### Store to NPY
np.save('val_x_seq__observ_{}__labeled_{}_{}_{}_{}.npy'.format(observ_daterange, label_daterange, try_date, version, desc), _val_x_seq)
np.save('val_y__observ_{}__labeled_{}_{}_{}_{}.npy'.format(observ_daterange, label_daterange, try_date, version, desc), val_y.label.values)
print('validation sequence data finish storage.')

#### testing set
test_X_seq = {}
for u in tqdm(testset_for_scaling.actor_id.unique()):
    test_X_seq[u] = {}
    # update scaled data
    for c in testset_for_scaling.columns[2:]:
        test_X_seq[u][c] = testset_for_scaling[testset_for_scaling['actor_id'] == u][c].values.tolist()

_test_X_seq = []
for u in tqdm(testset_for_scaling.actor_id.unique()):
    # create pre-padding
    collect = []
    for key, value in test_X_seq[u].items():
        if key in ['label','clust_index']:
            pass
        else:
            tmp = value
            if len(tmp) < MAX_TIMESTEP:
                diff_to_pad = MAX_TIMESTEP - len(tmp)
                tmp = np.insert([0.0]*diff_to_pad, diff_to_pad, tmp).tolist()
            else:
                tmp = tmp[-MAX_TIMESTEP:]
                if key == 'time_diff':
                    tmp[0] = 0
            collect.append(tmp)

    collect = np.array(collect).T
    _test_X_seq.append(collect)

_test_X_seq = np.array(_test_X_seq)
print(_test_X_seq.shape)
### Store to NPY
np.save('test_X_seq__observ_{}__labeled_{}_{}_{}_{}.npy'.format(observ_daterange, label_daterange, try_date, version, desc), _test_X_seq)
np.save('test_Y__observ_{}__labeled_{}_{}_{}_{}.npy'.format(observ_daterange, label_daterange, try_date, version, desc), test_Y.label.values)
print('testing sequence data finish storage.')


### Over-Sampling: SMOTE (only works for 2d data, need to reshape before implement)
train_X_res, train_Y_res = SMOTE(random_state=666).fit_resample(np.reshape(_train_x_seq, (_train_x_seq.shape[0], _train_x_seq.shape[1]*_train_x_seq.shape[2])), train_y.label)
val_X_res, val_Y_res = SMOTE(random_state=666).fit_resample(np.reshape(_val_x_seq, (_val_x_seq.shape[0], _val_x_seq.shape[1]*_val_x_seq.shape[2])), val_y.label)
train_X_res = np.reshape(train_X_res, (train_X_res.shape[0], _train_x_seq.shape[1], _train_x_seq.shape[2]))
val_X_res = np.reshape(val_X_res, (val_X_res.shape[0], _val_x_seq.shape[1], _val_x_seq.shape[2]))
print(train_X_res.shape, train_Y_res.shape)
print(val_X_res.shape, val_Y_res.shape)
### Store to NPY
np.save('OverSamp__train_X_seq__observ_{}__labeled_{}_{}_{}_{}.npy'.format(observ_daterange, label_daterange, try_date, version, desc), train_X_res)
np.save('OverSamp__train_Y__observ_{}__labeled_{}_{}_{}_{}.npy'.format(observ_daterange, label_daterange, try_date, version, desc), train_Y_res)
np.save('OverSamp__val_X_seq__observ_{}__labeled_{}_{}_{}_{}.npy'.format(observ_daterange, label_daterange, try_date, version, desc), val_X_res)
np.save('OverSamp__val_Y__observ_{}__labeled_{}_{}_{}_{}.npy'.format(observ_daterange, label_daterange, try_date, version, desc), val_Y_res)
print('Over-sampling data generated and stored.')


