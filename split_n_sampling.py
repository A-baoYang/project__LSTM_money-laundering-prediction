import json
import argparse
import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE

### Load data & preprocessing
def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--observ_daterange', type=str, required=True,
                        help='Observation date range of the transactions.')
    parser.add_argument('--label_daterange', type=str, required=True,
                        help='Labeled duration of actor_ids.')
    parser.add_argument('--try_date', type=str, required=True,
                        help='Whats the date today?')
    parser.add_argument('--version', type=str, required=False,
                        help='data format versions.')
    parser.add_argument('--desc', type=str, required=False,
                        help='description of the dataset')
    args = parser.parse_args()
    return args

args = parse_arguments()
observ_daterange = args.observ_daterange
label_daterange = args.label_daterange
try_date = args.try_date
version = args.version
desc = args.desc


##### (1) Sequence data in json
with open('make_sequence__observ_{}__labeled_{}_{}_{}_{}.json'.format(observ_daterange, label_daterange, try_date, version, desc), 'r') as f:
    all_seq = json.load(f)
print(f'json data loaded. {type(all_seq)}')
print(all_seq['A10060AE8A7EF2B021'])  # pick one user to check data
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


### make array format
_seq = []
for u in tqdm(uid):
    collect = []
    for key, value in all_seq[u].items():
        if key in ['label','clust_index','time_diff','cash_flow','twd_amt','cat_location','action_time_month','action_time_day','action_time_hour','action_time_dayOfWeek']:
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
    _seq.append(collect)

_seq = np.array(_seq)
print(_seq.shape)


### Split data & check label num
train_X, test_X, train_Y, test_Y = train_test_split(_seq, actor_label, test_size=0.3, random_state=99, stratify=actor_label.label)
train_x, val_x, train_y, val_y = train_test_split(train_X, train_Y, test_size=0.25, random_state=43, stratify=train_Y.label)
print(train_y.groupby(['label']).size())
print(val_y.groupby(['label']).size())
print(test_Y.groupby(['label']).size())

# ### Scaling data
# #### training set
# sd = StandardScaler()
# trainset_for_scaling = df_train[df_train['actor_id'].isin(train_y.actor_id.unique())]
# sd.fit(trainset_for_scaling.iloc[:, np.r_[2, 4:30]])
# trainset_for_scaling.iloc[:, np.r_[2, 4:30]] = sd.transform(trainset_for_scaling.iloc[:, np.r_[2, 4:30]])
# #### validation set
# valset_for_scaling = df_train[df_train['actor_id'].isin(val_y.actor_id.unique())]
# valset_for_scaling.iloc[:, np.r_[2, 4:30]] = sd.transform(valset_for_scaling.iloc[:, np.r_[2, 4:30]])
# #### testing set
# testset_for_scaling = df_train[df_train['actor_id'].isin(test_Y.actor_id.unique())]
# testset_for_scaling.iloc[:, np.r_[2, 4:30]] = sd.transform(testset_for_scaling.iloc[:, np.r_[2, 4:30]])
#
# ### Update scaled data to sequence format & Create pre-padding
# #### training set
# train_x_seq = {}
# for u in tqdm(trainset_for_scaling.actor_id.unique()):
#     train_x_seq[u] = {}
#     # update scaled data
#     for c in trainset_for_scaling.columns[2:]:
#         train_x_seq[u][c] = trainset_for_scaling[trainset_for_scaling['actor_id'] == u][c].values.tolist()
#
# _train_x_seq = []
# for u in tqdm(trainset_for_scaling.actor_id.unique()):
#     # create pre-padding
#     collect = []
#     for key, value in train_x_seq[u].items():
#         if key in ['label','clust_index']:
#             pass
#         else:
#             tmp = value
#             if len(tmp) < MAX_TIMESTEP:
#                 diff_to_pad = MAX_TIMESTEP - len(tmp)
#                 tmp = np.insert([0.0]*diff_to_pad, diff_to_pad, tmp).tolist()
#             else:
#                 tmp = tmp[-MAX_TIMESTEP:]
#                 if key == 'time_diff':
#                     tmp[0] = 0
#             collect.append(tmp)
#
#     collect = np.array(collect).T
#     _train_x_seq.append(collect)
#
# _train_x_seq = np.array(_train_x_seq)
# print(_train_x_seq.shape)
# ### Store to NPY
# np.save('train_x_seq__observ_{}__labeled_{}_{}_{}_{}.npy'.format(observ_daterange, label_daterange, try_date, version, desc), _train_x_seq)
# np.save('train_y__observ_{}__labeled_{}_{}_{}_{}.npy'.format(observ_daterange, label_daterange, try_date, version, desc), train_y.label.values)
# print('training sequence data finish storage.')
#
# #### validation set
# val_x_seq = {}
# for u in tqdm(valset_for_scaling.actor_id.unique()):
#     val_x_seq[u] = {}
#     # update scaled data
#     for c in valset_for_scaling.columns[2:]:
#         val_x_seq[u][c] = valset_for_scaling[valset_for_scaling['actor_id'] == u][c].values.tolist()
#
# _val_x_seq = []
# for u in tqdm(valset_for_scaling.actor_id.unique()):
#     # create pre-padding
#     collect = []
#     for key, value in val_x_seq[u].items():
#         if key in ['label','clust_index']:
#             pass
#         else:
#             tmp = value
#             if len(tmp) < MAX_TIMESTEP:
#                 diff_to_pad = MAX_TIMESTEP - len(tmp)
#                 tmp = np.insert([0.0]*diff_to_pad, diff_to_pad, tmp).tolist()
#             else:
#                 tmp = tmp[-MAX_TIMESTEP:]
#                 if key == 'time_diff':
#                     tmp[0] = 0
#             collect.append(tmp)
#
#     collect = np.array(collect).T
#     _val_x_seq.append(collect)
#
# _val_x_seq = np.array(_val_x_seq)
# print(_val_x_seq.shape)
# ### Store to NPY
# np.save('val_x_seq__observ_{}__labeled_{}_{}_{}_{}.npy'.format(observ_daterange, label_daterange, try_date, version, desc), _val_x_seq)
# np.save('val_y__observ_{}__labeled_{}_{}_{}_{}.npy'.format(observ_daterange, label_daterange, try_date, version, desc), val_y.label.values)
# print('validation sequence data finish storage.')
#
# #### testing set
# test_X_seq = {}
# for u in tqdm(testset_for_scaling.actor_id.unique()):
#     test_X_seq[u] = {}
#     # update scaled data
#     for c in testset_for_scaling.columns[2:]:
#         test_X_seq[u][c] = testset_for_scaling[testset_for_scaling['actor_id'] == u][c].values.tolist()
#
# _test_X_seq = []
# for u in tqdm(testset_for_scaling.actor_id.unique()):
#     # create pre-padding
#     collect = []
#     for key, value in test_X_seq[u].items():
#         if key in ['label','clust_index']:
#             pass
#         else:
#             tmp = value
#             if len(tmp) < MAX_TIMESTEP:
#                 diff_to_pad = MAX_TIMESTEP - len(tmp)
#                 tmp = np.insert([0.0]*diff_to_pad, diff_to_pad, tmp).tolist()
#             else:
#                 tmp = tmp[-MAX_TIMESTEP:]
#                 if key == 'time_diff':
#                     tmp[0] = 0
#             collect.append(tmp)
#
#     collect = np.array(collect).T
#     _test_X_seq.append(collect)
#
# _test_X_seq = np.array(_test_X_seq)
# print(_test_X_seq.shape)
# ### Store to NPY
# np.save('test_X_seq__observ_{}__labeled_{}_{}_{}_{}.npy'.format(observ_daterange, label_daterange, try_date, version, desc), _test_X_seq)
# np.save('test_Y__observ_{}__labeled_{}_{}_{}_{}.npy'.format(observ_daterange, label_daterange, try_date, version, desc), test_Y.label.values)
# print('testing sequence data finish storage.')


### Under-Sampling
print('train_y label 0/1/2: ', train_y.values.tolist().count(0), train_y.values.tolist().count(1), train_y.values.tolist().count(2))
print('val_y label 0/1/2: ', val_y.values.tolist().count(0), val_y.values.tolist().count(1), val_y.values.tolist().count(2))
print('test_Y label 0/1/2: ', test_Y.values.tolist().count(0), test_Y.values.tolist().count(1), test_Y.values.tolist().count(2))
train_y_0_indexes = [index for index, value in enumerate(train_y.values.tolist()) if value == 0]
train_y_1_indexes = [index for index, value in enumerate(train_y.values.tolist()) if value == 1]
train_y_2_indexes = [index for index, value in enumerate(train_y.values.tolist()) if value == 2]
val_y_0_indexes = [index for index, value in enumerate(val_y.values.tolist()) if value == 0]
val_y_1_indexes = [index for index, value in enumerate(val_y.values.tolist()) if value == 1]
val_y_2_indexes = [index for index, value in enumerate(val_y.values.tolist()) if value == 2]
train_y_0_sampled_indexes = np.random.choice(train_y_0_indexes, len(train_y_2_indexes), replace=False)
val_y_0_sampled_indexes = np.random.choice(val_y_0_indexes, len(val_y_2_indexes), replace=False)

train_indexes = []
for ind in train_y_0_sampled_indexes:
    train_indexes.append(ind)
for ind in train_y_1_indexes:
    train_indexes.append(ind)
for ind in train_y_2_indexes:
    train_indexes.append(ind)
train_indexes = np.array(train_indexes)
train_x_undersampled = train_x.iloc[:, 1:].values[train_indexes]
train_y_undersampled = train_y.values[train_indexes]

val_indexes = []
for ind in val_y_0_sampled_indexes:
    val_indexes.append(ind)
for ind in val_y_1_indexes:
    val_indexes.append(ind)
for ind in val_y_2_indexes:
    val_indexes.append(ind)
val_indexes = np.array(val_indexes)
val_x_undersampled = val_x.iloc[:, 1:].values[val_indexes]
val_y_undersampled = val_y.values[val_indexes]
print(train_x_undersampled.shape)
print(val_x_undersampled.shape)


### Over-Sampling: SMOTE (only works for 2d data, need to reshape before implement)
train_X_res, train_Y_res = SMOTE(random_state=666).fit_resample(np.reshape(train_x, (train_x.shape[0], train_x.shape[1]*train_x.shape[2])), train_y.label)
val_X_res, val_Y_res = SMOTE(random_state=666).fit_resample(np.reshape(val_x, (val_x.shape[0], val_x.shape[1]*val_x.shape[2])), val_y.label)
train_X_res = np.reshape(train_X_res, (train_X_res.shape[0], train_x.shape[1], train_x.shape[2]))
val_X_res = np.reshape(val_X_res, (val_X_res.shape[0], val_x.shape[1], val_x.shape[2]))
print(train_X_res.shape, train_Y_res.shape)
print(val_X_res.shape, val_Y_res.shape)
### Store to NPY
np.save('OverSamp__train_x__observ_{}__labeled_{}_{}_{}_{}.npy'.format(observ_daterange, label_daterange, try_date, version, desc), train_X_res)
np.save('OverSamp__train_y__observ_{}__labeled_{}_{}_{}_{}.npy'.format(observ_daterange, label_daterange, try_date, version, desc), train_Y_res)
np.save('OverSamp__val_x__observ_{}__labeled_{}_{}_{}_{}.npy'.format(observ_daterange, label_daterange, try_date, version, desc), val_X_res)
np.save('OverSamp__val_y__observ_{}__labeled_{}_{}_{}_{}.npy'.format(observ_daterange, label_daterange, try_date, version, desc), val_Y_res)
np.save('test_X__observ_{}__labeled_{}_{}_{}_{}.npy'.format(observ_daterange, label_daterange, try_date, version, desc), test_X)
np.save('test_Y__observ_{}__labeled_{}_{}_{}_{}.npy'.format(observ_daterange, label_daterange, try_date, version, desc), test_Y.label)
print('Over-sampling data generated and stored.')


