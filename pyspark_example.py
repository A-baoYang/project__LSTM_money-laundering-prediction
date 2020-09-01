import pyspark as spark
from pyspark.sql import functions as F
# from pyspark.sql import HiveContext
# from pyspark.sql.window import Window
# from pyspark import SparkConf, SparkContext
import pandas as pd
import numpy as np

focus_customer_id = spark.sql(
    """
    SELECT customer_id, count(1) as consumpt_cnt
    FROM btmp_cmd.nt85610_loc_poc
    WHERE week >= 1 AND week <= 8 AND loc_type_town IS NOT NULL
    GROUP BY customer_id
    HAVING consumpt_cnt >= 10
    """
).select('customer_id')

df_raw = spark.sql("""SELECT * FROM btmp_cmd.nt85610_loc_poc WHERE merchant_flag = 'Y' AND loc_type_town IS NOT NULL""")
train_intent_df = df_raw.filter(F.col('week').between(0, 8)). \
    groupby(['customer_id', 'dayofweek', 'consumption_category_desc', 'loc_type_town']). \
    agg(F.count(F.lit(1)).alias('train_consum_cnt'),
        F.sum('txn_amt').alias('train_consum_tot_amt')). \
    withColumn('train_consum_cnt_rk',
               F.row_number().over(Window.partitionBy(['customer_id', 'dayofweek']). \
                                   orderBy(F.desc('train_consum_cnt'), F.desc('train_consum_tot_amt')))). \
    filter(F.col('train_consum_cnt_rk') == 1). \
    select(['customer_id', 'dayofweek', 'consumption_category_desc', 'loc_type_town'])


train_merchant_df = df_raw.filter(F.col('week').between(0, 8)). \
    groupby(['customer_id', 'consumption_category_desc', 'loc_type_town', 'merchant_name']). \
    agg(F.count(F.lit(1)).alias('train_consum_cnt'),
        F.sum('txn_amt').alias('train_consum_tot_amt')). \
    withColumn('train_consum_cnt_rk',
               F.row_number().over(Window.partitionBy(['customer_id', 'consumption_category_desc']). \
                                   orderBy(F.desc('train_consum_cnt'), F.desc('train_consum_tot_amt')))). \
    select(['customer_id', 'consumption_category_desc', 'loc_type_town', 'merchant_name', 'train_consum_cnt_rk'])


output_df = train_intent_df.join(train_merchant_df,
                                 on = ['customer_id', 'consumption_category_desc', 'loc_type_town'],
                                 how = 'left'). \
                withColumn('train_output_rk',
                           F.row_number().over(Window.partitionBy(['customer_id', 'consumption_category_desc',
                                                                   'loc_type_town', 'dayofweek']). \
                                               orderBy('train_consum_cnt_rk'))). \
                join(focus_customer_id, on=['customer_id'], how='inner'). \
                filter(F.col('train_output_rk') == F.lit(1)). \
                select(['customer_id', 'consumption_category_desc', 'loc_type_town', 'dayofweek', 'merchant_name', 'train_output_rk'])

