import sys
# sys.path.append('./')
import json
import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler
sd = StandardScaler()

### Load data & preprocessing
observ_daterange = '1910_1912'
label_daterange = '2001'
try_date = '200825'
version = 'vv10'
desc = '?locationInfo=location_token&scaler=standardscaler'

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
with open('make_sequence__observ_{}__labeled_{}_{}_{}_{}'.format(observ_daterange, label_daterange, try_date, version, desc)) as fp:
    json.dump(all_seq, fp)


### Restore `clust_index`, `time_diff` to CSV
clust_collect = []
diff_collect = []
for u in tqdm(uid):
    for i in range(0, len(all_seq[u]['clust_index'])):
        clust_collect.append(all_seq[u]['clust_index'][i])
        diff_collect.append(all_seq[u]['time_diff'][i])
df = df.concat([df, pd.DataFrame({'clust_index': clust_collect, 'time_diff': diff_collect})], axis=1)


### time_diff scaling
df['time_diff_scaled'] = sd.fit_transform(df['time_diff'].values.reshape(-1, 1))
for u in tqdm(uid):
    all_seq[u]['time_diff_scaled'] = df[df['actor_id'] == u]['time_diff_scaled'].values.tolist()

with open('make_sequence__observ_{}__labeled_{}_{}_{}_{}'.format(observ_daterange, label_daterange, try_date, version, desc)) as fp:
    json.dump(all_seq, fp)

df.to_csv('featureGeneration__observ_{}__labeled_{}_{}_{}_{}.csv'.format(observ_daterange, label_daterange, try_date, version, desc))
