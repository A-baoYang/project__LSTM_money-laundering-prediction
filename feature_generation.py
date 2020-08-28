# import sys
# sys.path.append('./')
import argparse
import jieba
jieba.load_userdict('add_location_words.txt')
import pandas as pd
import numpy as np
# import matplotlib.pyplot as plt
# from matplotlib.font_manager import FontProperties
# myfont = FontProperties(fname='MSJH.TTF', size=14)
# import seaborn as sns
# sns.set(font= ['MSJH.TTF'])
import tzlocal
from datetime import datetime
# from wordcloud import WordCloud
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


df = pd.read_csv('observ_{}__labeled_{}.csv'.format(observ_daterange, label_daterange), dtype={
    'atm_location': 'string',
    'source_bank_code': 'string',
    'target_bank_code': 'string'
})
print(f'Data loaded: {df.shape}')

#### Filtering
df = df[df['event'] != 'cathaybk_browse']
df = df[df['action_sponsor'] == 'self']
df = df[df['customer_class_code'] == 'I']
df = df[df['label'] != 3].reset_index().drop(['index'], axis=1)
print(f'Data filtered: {df.shape}')

#### Missing value
missing_value_cols = ['action_type','txn_currency_code','txn_type_desc','target_acct_nbr','target_bank_code','target_customer_id']
for c in missing_value_cols:
    df[c] = df[c].fillna('unknown')

##### Store a backup as CSV
df.to_csv('CLEANED_data_observ_{}__labeled_{}.csv'.format(observ_daterange, label_daterange), index=False)

#### Current Feature preprocessing
df['atm_location'] = np.where((df['atm_location'].isnull()) & (df['event'] != 'atm_transaction'), '線上交易', df['atm_location'])
df['atm_location'] = df['atm_location'].fillna('unknown')
print(f'Data missing value removed: \n')
print(df.isnull().sum())

#### Add features: `isCrossBank`, `isSelfTrans`
df['isCrossBank'] = np.where(df['target_bank_code'] != '013', 1, 0)
df['isSelfTrans'] = np.where((df['actor_id'] == df['target_customer_id']) & (df['acct_nbr'] != df['target_acct_nbr']), 1, 0)
df = df.drop(['customer_id','object_id','target_customer_id','behavior_yyyymm'], axis=1)

#### (1) action_time - transform & onehot
def action_time_transform(x, transform_type):
    local_timezone = tzlocal.get_localzone()
    local_time = datetime.fromtimestamp(int(x), local_timezone)
    if transform_type == 'month':
        value = local_time.month
    elif transform_type == 'day':
        value = local_time.day
    elif transform_type == 'hour':
        value = local_time.hour
    elif transform_type == 'dayOfWeek':
        value = local_time.weekday()+1
    elif transform_type == 'periodOfDay':
        value = local_time.hour
        if (value >= 1) and (value < 7):
            value = '0'
        elif (value >= 7) and (value < 13):
            value = '1'
        elif (value >= 13) and (value < 19):
            value = '2'
        else:
            value = '3'
    else:
        print('"transform_type" should be chosen from ["month", "day", "hour", "dayOfWeek", "periodOfDay"]')
        value = 'value error'
    return value

time_cols = ['month', 'day', 'hour', 'dayOfWeek', 'periodOfDay']
for col in time_cols:
    df[f'action_time_{col}'] = df['action_time'].apply(lambda x: action_time_transform(x, transform_type=col))
df[f'action_time_isWeekend'] = np.where(df['action_time_dayOfWeek'].isin([6,7]), '1', '0')

dummy_actionTime = pd.get_dummies(df[['action_time_isWeekend', 'action_time_periodOfDay']])
dummy_actionTime = dummy_actionTime.drop(['action_time_isWeekend_0', 'action_time_periodOfDay_0'], axis=1)
df = pd.concat([df.iloc[:, :-2], dummy_actionTime], axis=1)
df['action_time'] = df['action_time'] / (60*60*24)
# df = df.drop(['action_time'], axis=1)
print(f'data ETL finished - action_time: {df.shape}')

#### (2) event - onehot
dummy_event = pd.get_dummies(df[['event']]).drop(['event_mybank_transaction'], axis=1)
df = pd.concat([df, dummy_event], axis=1)
print(f'data ETL finished - event: {df.shape}')

#### (3) isExchangeDeal
df['isExchangeDeal'] = np.where(df['txn_currency_code'].isin(['TWD','unknown']), 0, 1)
print(f'data ETL finished - isExchangeDeal: {df.shape}')

#### (4-1) atm_location - conditional categorization (old method)
df['atm_location'] = df['atm_location'].apply(lambda x: x.replace(u'—', ' '))  # 捷運站 超商分店會有的連結符號
cond_list = [
    df['atm_location'].str.contains('分行'),
    df['atm_location'].str.contains('全家'),
    df['atm_location'].str.contains('萊爾富'),
    df['atm_location'].str.contains('7-ELEVEN'),
    df['atm_location'].str.contains('全聯'),
    df['atm_location'].str.contains('COSTCO'),
    df['atm_location'].str.contains('IKEA'),
    df['atm_location'].str.contains('大潤發'),
    df['atm_location'].str.contains('大買家'),
    df['atm_location'].str.contains('微風|新光三越|TIGER CITY|美麗華百樂園|環球購物中心|大葉高島屋|禮客時尚館|大魯閣草衙道購物中心–商場三樓|大魯閣草衙道購物中心–商場一樓|桃園ATT筷時尚|ATT 4 LIFE|巨城購物中心|維春—日耀百貨|大江國際購物中心1樓'),
    df['atm_location'].str.contains('捷運'),
    df['atm_location'].str.contains('寒舍艾麗酒店|喜來登|大億麗緻酒店'),
    df['atm_location'].str.contains('分局'),
    df['atm_location'].str.contains('醫院'),
    df['atm_location'].str.contains('科大|學院|大學|商學館'),
    df['atm_location'].str.contains('運動中心'),
    df['atm_location'].str.contains('國壽|A3置地大樓|營業部'),
    df['atm_location'].str.contains('誠品'),
    df['atm_location'].str.contains('線上交易'),
    df['atm_location'].str.contains('證券'),
    df['atm_location'].str.contains('unknown')
]
choice_list = [
    '分行','全家','萊爾富','7-ELEVEN','全聯','COSTCO','IKEA','大潤發','大買家','百貨','捷運','餐飲旅宿','分局','醫院','大學','運動中心',
    '國壽相關據點','誠品','線上交易','證券公司','unknown'
]
df['cat_location'] = np.select(condlist=cond_list, choicelist=choice_list, default='其他未歸類')

#### (4-2) atm_location - jieba
stopwords = ['/','-',' ','--','(',')','unknown','店','站','金牌','漾店','分院','中心','服務','鄰店','三文','大樓',
             '三厝','新大樓','通訊','第二','D','棟','1F','22F','國際大樓','愛心店','限','使用','讚','二店','南店','二校區',
             '新','有店','婦幼','櫃','航廈','和','三','二','東店','三店','金','交流','愛萊店','綠']
df['atm_location_edit'] = df['atm_location'].apply(lambda x: [x for x in jieba.cut(x, cut_all=False) if x not in stopwords])
df['atm_location_edit'] = df['atm_location_edit'].apply(lambda x: [item.replace('店', '') for item in x])

#### (4-3) atm_location - tokenize
location_corpus = []
for loca_list in df['atm_location_edit'].values:
    location_corpus.append(' '.join(loca_list))

vectorizer = CountVectorizer(min_df=0.001, max_df=0.99)
location_token = vectorizer.fit_transform(location_corpus).toarray()
print(vectorizer.get_feature_names())
print(location_token.shape)

##### Decomposition
pca = PCA(n_components=10)
location_token_PCA = pca.fit_transform(location_token)
# sns.barplot(x=['PC' + str(x) for x in range(1, 11)], y=pca.explained_variance_ratio_, color='c')

location_token_PCA = pd.DataFrame(location_token_PCA[:, :5])
location_token_PCA.columns = ['location_token_' + str(x) for x in location_token_PCA.columns]

df = pd.concat([df, location_token_PCA], axis=1)
df = df.drop(['atm_location_edit'], axis=1)
print(f'data ETL finished - location_token: {df.shape}')

#### (5) action_type & txn_type_desc - conditional categorization
df['all_action_types'] = df['action_type'] + ' ' + df['txn_type_desc']
cond_list = [
    (df['all_action_types'].str.contains('停車費|國民年金|水費|中華電信|台電|繳稅|遠傳電信|國壽保險費|信用卡費|繳費')) & (df['twd_amt']>0),
    ((df['all_action_types'].str.contains('保單貸款|保單借款')) & (df['all_action_types'].str.contains('轉帳'))) | (df['v'].str.contains('國壽保單借款')) & (df['twd_amt']>0),
    (df['all_action_types'].str.contains('外匯')) & (df['all_action_types'].str.contains('轉帳')) & (df['twd_amt'] > 0),
    (df['all_action_types'].str.contains('理財型房貸')) & (df['all_action_types'].str.contains('轉帳')) & (df['twd_amt'] > 0),
    (df['all_action_types'].str.contains('電子式存單')) & (df['all_action_types'].str.contains('轉帳')) & (df['twd_amt'] > 0),
    (df['all_action_types'].str.contains('網銀約轉')) & (df['all_action_types'].str.contains('轉帳')) & (df['twd_amt'] > 0),
    (df['all_action_types'].str.contains('網銀非約轉')) & (df['all_action_types'].str.contains('轉帳')) & (df['twd_amt'] > 0),
    (df['all_action_types'].str.contains('信用卡')) & (df['all_action_types'].str.contains('轉帳')) & (df['twd_amt'] > 0),
    (df['all_action_types'].str.contains('存單開戶存摺扣款')) & (df['all_action_types'].str.contains('轉帳')) & (df['twd_amt'] > 0),
    (df['all_action_types'].str.contains('轉帳_ unknown|跨行轉帳|轉帳交易|轉出交易')) & (df['twd_amt'] > 0),
    df['all_action_types'].str.contains('申購'),
    df['all_action_types'].str.contains('定存'),
    df['all_action_types'].str.contains('外匯匯款'),
    (df['all_action_types'].str.contains('外幣提款')) & (df['twd_amt'] > 0),
    (df['all_action_types'].str.contains('自行提款')) & (df['twd_amt'] > 0),
    (df['all_action_types'].str.contains('存款')) & (df['all_action_types'].str.contains('現金貸方')),
    (df['all_action_types'].str.contains('保單|理財')) & (df['all_action_types'].str.contains('現金借方')),
    df['all_action_types'].str.contains('現鈔繳信用卡款'),
    df['all_action_types'].str.contains('網路消費|露天付款快手|拍賣'),
    df['all_action_types'].str.contains('愛心捐款'),
    (df['all_action_types'].str.contains('約定帳號查詢')) & (df['all_action_types'].str.contains('查詢')),
    (df['all_action_types'].str.contains('存款')) & (df['all_action_types'].str.contains('查詢')),
    (df['all_action_types'].str.contains('卡片')) & (df['all_action_types'].str.contains('查詢')),
    (~df['all_action_types'].str.contains('約定帳號查詢|存款|卡片')) & (df['all_action_types'].str.contains('查詢')),
    ((df['all_action_types'].str.contains('一般_轉換')) | df['all_action_types'].str.contains('unknown _')) & (df['twd_amt'] > 0),
    (df['all_action_types'] == 'unknown _') & (df['twd_amt'] > 0),
    df['all_action_types'] == 'NCWINQ _',
    df['all_action_types'] == 'RPTSND _',
    df['all_action_types'] == 'ACNIQ6 _',
    df['all_action_types'] == 'CIFIQ1 _',
    df['all_action_types'] == 'PACSET _',
    df['all_action_types'] == 'CIFCHG _',
    df['all_action_types'] == 'NCWSET _',
    df['all_action_types'] == 'NCWPCG _'
]
choice_list = [
    '轉帳_繳費','轉帳_保單','轉帳_外匯','轉帳_理財型房貸','轉帳_電子式存單','轉帳_網銀約轉','轉帳_網銀非約轉',
    '轉帳_信用卡繳費','轉帳_存單開戶存摺扣款','轉帳_個人轉帳','申購','定存','匯款_外匯','提款_外幣','提款_非外幣',
    '現金貸方_存款','現金借方_其他','現鈔繳信用卡款','網路消費','愛心捐款','查詢_約定','查詢_存款','查詢_卡片','查詢_其他',
    '其他_有金額_unknown','其他_無金額_unknown','其他_無金額_NCWINQ','其他_無金額_RPTSND','其他_無金額_ACNIQ6','其他_無金額_CIFIQ1',
    '其他_無金額_PACSET','其他_無金額_CIFCHG','其他_無金額_NCWSET','其他_無金額_NCWPCG'
]

df['cat_action'] = np.select(condlist=cond_list, choicelist=choice_list, default='其他_未歸類')
dummy_cat_action = pd.get_dummies(df['cat_action'])

##### Decomposition
pca = PCA(n_components=10)
cat_action_PCA = pca.fit_transform(dummy_cat_action)
# sns.barplot(x=['PC_' + str(x) for x in range(1, 11)], y=pca.explained_variance_ratio_, color='c')  #當時取4維
cat_action_PCA = pd.DataFrame(cat_action_PCA[:, :4])
cat_action_PCA.columns = ['cat_action_' + str(x) for x in cat_action_PCA.columns]
df = pd.concat([df, cat_action_PCA], axis=1)
print(f'data ETL finished - cat_action: {df.shape}')


#### (5) action_type & txn_type_desc - WordCloud
# wordlist_after_jieba = jieba.cut(' '.join(df.all_action_types.unique()), cut_all=True)
# wl_space_split = ' '.join(wordlist_after_jieba)
# wc = WordCloud(
#     height=800,
#     width=800,
#     background='white',
#     max_words=200,
#     mask=None,
#     max_font_size=None,
#     font_path='MSJH.TTF',
#     random_state=None,
#     prefer_horizontal=0.9
# ).generate(wl_space_split)
# plt.figure(figsize=(10,10))
# plt.imshow(wc)
# plt.axis('off')
# plt.show()

#### (6) twd_amt - 金流流向判斷 `cash_flow`
flowout_filters = ['停車費|國民年金|水費|中華電信|台電|繳稅|遠傳電信|國壽保險費|信用卡費|繳信用卡|繳他人信用卡款|繳費|貸款|外匯|約轉|扣款|轉帳_ unknown|跨行轉帳|自行轉帳|轉帳交易|轉出交易|申購|提款|現金貸方|轉帳貸方|還款|現鈔繳信用卡款|網路消費|一般_轉換|拍賣|露天付款|unknown _|領現|愛心捐款']
df['cash_flow'] = np.where((df['action_sponsor'] == 'self') & (df['twd_amt']>0) & (df['all_action_types'].str.contains(flowout_filters[0])) & (~df['all_action_types'].str.contains('存款|解約')), -1, 1)
df['cash_flow'] = np.where((df['action_sponsor'] == 'self') & (df['twd_amt']==0) & (df['cat_action'].str.contains('查詢|其他_無金額_')), 0, df['cash_flow'])
dummy_cashFlow = pd.get_dummies(df['cash_flow'])
dummy_cashFlow.columns = ['cash_flow_' + str(x) for x in [-1, 0, 1]]
df = pd.concat([df, dummy_cashFlow], axis=1)
df = df.drop(['cash_flow','cash_flow_0'], axis=1)
print(f'data ETL finished - cash_flow: {df.shape}')

### Clean columns
df = df.reset_index()
df['twd_amt_scaled'] = sd.fit_transform(df['twd_amt'].values.reshape(-1, 1))
df = df.drop(['index','event','action_sponsor','action_type','cat_action','txn_currency_code','txn_type_desc','all_action_types',
              'atm_location','cat_location','target_bank_code','target_acct_nbr','acct_nbr','customer_class_code','yyyymm','start_ym'], axis=1)
print(f'data ETL finished - All: {df.shape}')


### Store to CSV
uid = df.actor_id.unique()
actor_label = pd.DataFrame(uid, columns=['actor_id'])
actor_label = pd.merge(actor_label, df[['actor_id','label']].drop_duplicates(), on='actor_id', how='left')
df.to_csv('featureGeneration__observ_{}__labeled_{}_{}_{}_{}.csv'.format(observ_daterange, label_daterange, try_date, version, desc), index=False)
actor_label.to_csv('actorLabelTable__observ_{}__labeled_{}_{}_{}_{}.csv'.format(observ_daterange, label_daterange, try_date, version, desc), index=False)

