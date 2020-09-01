import spark
# from pyspark.sql import functions as F
# from pyspark.sql import HiveContext
# from pyspark.sql.window import Window
# from pyspark import SparkConf, SparkContext
import pandas as pd


## constants
observ_daterange = '1910_1912'
label_daterange = '2001'


## HIVE SQL
cj_1910_1912 = spark.sql("""
    SELECT distinct actor_id, event, yyyymm AS behavior_yyyymm, action_sponsor, action_time, action_type, object_id, 
    twd_amt, txn_currency_code, txn_type_desc, atm_location, target_acct_nbr, target_bank_code
    FROM vp_bank.customer_journey_event a 
    lateral view json_tuple(a.attrs, "action") attr_a as attrs_action
    lateral view json_tuple(a.attrs, "channel") attr_c as attrs_channel
    lateral view json_tuple(a.attrs, "object") attr_o as attrs_object
    lateral view json_tuple(attr_a.attrs_action, "twd_amt") n3 as twd_amt
    lateral view json_tuple(attr_a.attrs_action, "txn_currency_code") n4 as txn_currency_code
    lateral view json_tuple(attr_a.attrs_action, "txn_type_desc") n5 as txn_type_desc
    lateral view json_tuple(attr_c.attrs_channel, "atm_location") n4 as atm_location
    lateral view json_tuple(attr_o.attrs_object, "target_acct_nbr") n4 as target_acct_nbr
    lateral view json_tuple(attr_o.attrs_object, "target_bank_code") n4 as target_bank_code
    where event in ("atm_transaction", "myatm_transaction", "mybank_transaction") 
        and yyyymm between "201910" and "201912"
    order by actor_id, action_time asc    
""")

accounts_2001 = spark.sql = ("""
    SELECT distinct customer_id, acct_nbr, customer_class_code, yyyymm, start_ym, label
    FROM btmp_cmd.nt83716_cip_acct_base_ds
    WHERE yyyymm = "202001"
""")


## Merging
t_cj = cj_1910_1912.alias('t_cj')
t_acc = accounts_2001.alias('t_acc')

cj_of_accounts = t_cj.join(t_acc, t_cj.object_id == t_acc.acct_nbr, how='inner')
t_acc_rename = t_acc.select('customer_id','acct_nbr').selectExpr("customer_id as target_customer_id", "acct_nbr as target_acct_nbr_marge")
cj_of_accounts = cj_of_accounts.join(t_acc_rename, t_cj.target_acct_nbr == t_acc_rename.target_acct_nbr_marge, how='left')
cj_of_accounts = cj_of_accounts.select('actor_id', 'event', 'behavior_yyyymm', 'action_sponsor', 'action_time', 'action_type',
                                       'object_id', 'twd_amt', 'txn_currency_code', 'txn_type_desc', 'atm_location', 'target_acct_nbr',
                                       'target_bank_code', 'customer_id', 'acct_nbr', 'customer_class_code', 'yyyymm', 'start_ym',
                                       'label', 'target_customer_id')
print(cj_of_accounts.columns)


## Output & store to CSV
result = cj_of_accounts.toPandas()
result.to_csv('observ_{}__labeled_{}.csv'.format(observ_daterange, label_daterange), index=False, encoding='utf-8')
print(result.shape)
print(result.columns)

