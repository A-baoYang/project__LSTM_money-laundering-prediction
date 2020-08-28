# import sys
# sys.path.append('./')
import argparse
import jieba
jieba.load_userdict('add_location_words.txt')
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
myfont = FontProperties(fname='MSJH.TTF', size=14)
import seaborn as sns
sns.set(font= ['MSJH.TTF'])
import tzlocal
from datetime import datetime
# from wordcloud import WordCloud
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler
sd = StandardScaler()
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import PCA


### Load data & preprocessing
def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--observ_daterange', type=str, required=True,
                        help='Observation date range of the transactions.')
    parser.add_argument('--label_daterange', type=str, required=True,
                        help='Labeled duration of actor_ids.')
    args = parser.parse_args()
    return args

args = parse_arguments()
observ_daterange = args.observ_daterange
label_daterange = args.label_daterange


df = pd.read_csv('CLEANED_data_observ_{}__labeled_{}.csv'.format(observ_daterange, label_daterange))
print(f'Cleaned data loaded: {df.shape}')


# df = df[df['twd_amt']>0]
uid = df.actor_id.unique()
actor_label = pd.DataFrame(uid, columns=['actor_id'])
actor_label = pd.merge(actor_label, df[['actor_id','label']].drop_duplicates(), on='actor_id', how='left')

df['true_amt'] = np.where(df['cash_flow']==-1, 0-df['twd_amt'], df['twd_amt'])
df['weird_amt'] = np.where(df['twd_amt']%1000!=0, 1, 0)
df['less_than_1000'] = np.where(df['twd_amt']<1000, 1, 0)


### User Aggregate Features
user_aggr = []
for u in tqdm(uid):
    tmp_df = df[df['actor_id'] == u]
    twdAmt_avg = tmp_df['twd_amt'].values.mean()
    twdAmt_cov = tmp_df['twd_amt'].values.std() / twdAmt_avg
    sponsor_multiply = tmp_df['action_sponsor'].values.tolist().count('self')*tmp_df['action_sponsor'].values.tolist().count('others')
    currency_uniqueNum = len(tmp_df['txn_currency_code'].unique())
    atmLocation_uniqueNum = len(tmp_df['atm_location'].unique())
    targetAcct_uniqueNum = len(tmp_df['target_acct_nbr'].unique())
    acct_uniqueNum = len(tmp_df['acct_nbr'].unique())
    isCrossBank_01multiply = tmp_df['isCrossBank'].values.tolist().count(0)*tmp_df['isCrossBank'].values.tolist().count(1)
    isSelfTrans_sum = tmp_df['isSelfTrans'].values.sum()
    user_aggr.append((u, twdAmt_avg, twdAmt_cov, sponsor_multiply, currency_uniqueNum, atmLocation_uniqueNum, targetAcct_uniqueNum,
                      acct_uniqueNum, isCrossBank_01multiply, isSelfTrans_sum))

df_userAggr = pd.DataFrame(user_aggr, columns=['actor_id','twdAmt_avg','twdAmt_cov','sponsor_multiply','currency_uniqueNum',
                                               'atmLocation_uniqueNum','targetAcct_uniqueNum','acct_uniqueNum','isCrossBank_01multiply',
                                               'isSelfTrans_sum'])
df_userAggr = pd.merge(df_userAggr, actor_label, on='actor_id', how='left')
df_userAggr = df_userAggr.fillna(0)
print(df_userAggr.head())

### KDE plot example
df_userAggr_2 = df.groupby(['actor_id', 'label'])['true_amt'].sum().reset_index()
df_userAggr_2.groupby('label').true_amt.plot(kind='kde', legend=True)
plt.show()

### catplot example
eda_tmp = df.groupby(['label','action_time_isWeekend'])['actor_id'].count().reset_index()
eda_tmp['isWeekend_percent'] = np.where(eda_tmp['label']==0, eda_tmp['actor_id']/(eda_tmp[eda_tmp['label']==0]['actor_id'].sum()), 0)
eda_tmp['isWeekend_percent'] = np.where(eda_tmp['label']==1, eda_tmp['actor_id']/(eda_tmp[eda_tmp['label']==1]['actor_id'].sum()), eda_tmp['isWeekend_percent'])
eda_tmp['isWeekend_percent'] = np.where(eda_tmp['label']==2, eda_tmp['actor_id']/(eda_tmp[eda_tmp['label']==2]['actor_id'].sum()), eda_tmp['isWeekend_percent'])
sns.catplot(x='action_time_isWeekend', y='isWeekend_percent', hue='label', kind='bar', data=eda_tmp[eda_tmp['isWeekend_percent']>0.001], height=5, aspect=3)
plt.xticks(fontproperties=myfont)


eda_tmp = df.groupby(['label', 'actor_id'])['event'].nunique().reset_index()
df_userAggr_2['kindsOfEvent'] = eda_tmp['event']
eda_tmp = df.groupby(['label', 'actor_id'])['time_diff'].mean().reset_index()
df_userAggr_2['avg_timeDiff_allTrans'] = eda_tmp['time_diff']
eda_tmp = (df.groupby(['label', 'actor_id'])['time_diff'].std() / df.groupby(['actor_id','label'])['time_diff'].mean()).reset_index()
df_userAggr_2['cov_timeDiff_allTrans'] = eda_tmp['time_diff']
eda_tmp = df[df['twd_amt']==0].groupby(['label', 'actor_id'])['time_diff'].mean().reset_index()
df_userAggr_2['avg_timeDiff_allTrans'] = eda_tmp['time_diff']
eda_tmp = (df[df['twd_amt']==0].groupby(['label', 'actor_id'])['time_diff'].std() / df[df['twd_amt']==0].groupby(['actor_id','label'])['time_diff'].mean()).reset_index()
df_userAggr_2['cov_timeDiff_allTrans'] = eda_tmp['time_diff']
eda_tmp = df[df['twd_amt']!=0].groupby(['label', 'actor_id'])['time_diff'].mean().reset_index()
df_userAggr_2['avg_timeDiff_allTrans'] = eda_tmp['time_diff']
eda_tmp = (df[df['twd_amt']!=0].groupby(['label', 'actor_id'])['time_diff'].std() / df[df['twd_amt']==0].groupby(['actor_id','label'])['time_diff'].mean()).reset_index()
df_userAggr_2['cov_timeDiff_allTrans'] = eda_tmp['time_diff']

eda_tmp = df.groupby(['label', 'actor_id'])['twd_amt'].mean().reset_index()
df_userAggr_2['avg_twdAmt_allTrans'] = eda_tmp['twd_amt']
eda_tmp = (df.groupby(['label', 'actor_id'])['twd_amt'].std() / df.groupby(['actor_id','label'])['twd_amt'].mean()).reset_index()
df_userAggr_2['cov_twdAmt_allTrans'] = eda_tmp['twd_amt']
eda_tmp = df[df['twd_amt']!=0].groupby(['label', 'actor_id'])['twd_amt'].mean().reset_index()
df_userAggr_2['avg_twdAmt_allCashTrans'] = eda_tmp['twd_amt']
eda_tmp = (df[df['twd_amt']!=0].groupby(['label', 'actor_id'])['twd_amt'].std() / df[df['twd_amt']==0].groupby(['actor_id','label'])['twd_amt'].mean()).reset_index()
df_userAggr_2['cov_twdAmt_allCashTrans'] = eda_tmp['twd_amt']

eda_tmp = df[df['twd_amt']>0].groupby(['label', 'actor_id'])['less_than_1000'].mean().reset_index()
df_userAggr_2['ratio_lessThan1000_allCashTrans'] = eda_tmp['less_than_1000']
eda_tmp = df[df['twd_amt']>0].groupby(['label', 'actor_id'])['weird_amt'].mean().reset_index()
df_userAggr_2['ratio_weirdAmt_allCashTrans'] = eda_tmp['weird_amt']


