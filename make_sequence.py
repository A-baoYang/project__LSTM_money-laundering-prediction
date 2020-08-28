import sys
# sys.path.append('./')
import argparse
import json
import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler
sd = StandardScaler()

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


df = pd.read_csv('featureGeneration__observ_{}__labeled_{}_{}_{}_{}.csv'.format(observ_daterange, label_daterange, try_date, version, desc))
print(f'Data loaded: {df.shape}')


### Make sequence data format
all_seq = {}
cut_point = 1  # 1day
uid = df.actor_id.unique()

for u in tqdm(uid):
    all_seq[u] = {}
    time_seq = df[df['actor_id'] == u]['action_time'].values
    time_seq_shift = time_seq
    time_seq_shift = np.insert(time_seq_shift, 0, time_seq[0])
    time_seq_shift = time_seq_shift[:-1]
    time_diff = time_seq - time_seq_shift

    trans_clust_index = []
    flag = 0

    for i in range(0, len(time_diff)):
        if time_diff[i] > cut_point:
            flag += 1
        trans_clust_index.append(flag)

    all_seq[u]['clust_index'] = trans_clust_index
    time_diff = np.round(time_diff, 4)
    all_seq[u]['time_diff'] = time_diff.tolist()

    for c in df.columns[2:]:
        all_seq[u][c] = df[df['actor_id'] == u][c].values.tolist()

### Store to JSON
with open('make_sequence__observ_{}__labeled_{}_{}_{}_{}.json'.format(observ_daterange, label_daterange, try_date, version, desc), 'w') as fp:
    json.dump(all_seq, fp)


### Restore `clust_index`, `time_diff` to CSV
clust_collect = []
diff_collect = []
for u in tqdm(uid):
    for i in range(0, len(all_seq[u]['clust_index'])):
        clust_collect.append(all_seq[u]['clust_index'][i])
        diff_collect.append(all_seq[u]['time_diff'][i])
df = pd.concat([df, pd.DataFrame({'clust_index': clust_collect, 'time_diff': diff_collect})], axis=1)


### time_diff scaling
df['time_diff_scaled'] = sd.fit_transform(df['time_diff'].values.reshape(-1, 1))
for u in tqdm(uid):
    all_seq[u]['time_diff_scaled'] = df[df['actor_id'] == u]['time_diff_scaled'].values.tolist()

with open('make_sequence__observ_{}__labeled_{}_{}_{}_{}.json'.format(observ_daterange, label_daterange, try_date, version, desc), 'w') as fp:
    json.dump(all_seq, fp)

df.to_csv('featureGeneration__observ_{}__labeled_{}_{}_{}_{}.csv'.format(observ_daterange, label_daterange, try_date, version, desc), index=False)
