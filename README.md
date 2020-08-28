# project__LSTM_money-laundering-prediction
Anti-money laundering project using sequential data about cash transactions on LSTM.

## Query
`query.py`
Hive SQL，用於交易資料撈取


## Feature Generation
`feature_generation.py`
1. 載入資料
2. 目標資料集：條件篩選
    - 排除數量過少的 label=3 (同時為警示戶又為疑似洗錢戶者)
    - 個人戶 (customer_class_code == 'I')
    - 主動交易 (action_sponsor == 'self')
3. 遺失值處理
    - 具遺失值的欄位為類別型變數，各補值 `unknown` 為一種獨立類別 
```['action_type','txn_currency_code','txn_type_desc','target_acct_nbr','target_bank_code','target_customer_id']```

4. 新特徵生成
    - **單筆交易特性**
        - `isCrossBank`：是否為跨行轉帳
        - `isSelfTrans`：是否為自有不同帳戶互轉
        - `isExchangeDeal`：是否為外幣交易
        - `cash_flow`：判斷金流流向
    - **交易時間拆分**
        - `dayOfMonth`：日期
        - `hourOfDay`：當天小時
        - `dayOfWeek`：星期幾
        - `isWeekend`：是否為週末
        - `periodOfDay`：當天時段，共分成 4 個時段 (1am-7am / 7am-1pm / 1pm-7pm / 7pm-1am)
5. 特徵轉換
    - 類別型進行 One-hot 處理
    - **交易地點編碼**
        - 條件判斷
        - 人工新增詞庫 + jieba 斷詞 + 向量轉換 + 降維
    - **交易動作編碼**
        - 條件判斷 + One-hot + 降維
    - 數值欄位進行縮放處理


## Make Sequence
`make_sequence.py`
- loop by `actor_id`
- count `time_diff` for each：計算該用戶交易紀錄中，各筆交易紀錄距離上一次交易的天數 (取到小數點後第4位)
- count `clust_index` for user：
- format: `json`


## Split & Sampling
`split_n_sampling.py`
1. Split：將資料集以 (train:validation:test = 5:2:3) 比例分割
    <!-- 2. Scaling：再使用 trainset scale 對三個資料集進行縮放 -->
2. Sampling：進行 Over-sampling (本專案採用 SMOTE)


## Train, Val & Test
`train_test.py`
- LSTM model structure
- Custom Metrics (F1 Score / Recall / Precision)
- Confusion Matrix Heatmap


### Data version constants
`observ_daterange`：從 hadoop 撈取交易行為資料的時間區間 (ex: 201910-201912)

`label_daterange`：篩選交易帳戶被標記的時間區段 (ex: 202001)

`try_date`：資料流程運行日期

`version`：資料流程版本號

`desc`：對資料處理的描述




