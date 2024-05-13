# --- import packages
import os
import sys

sys.path.insert(0, 'dependency.zip')
os.mkdir('downloads')
os.mkdir('uploads')

import json
import boto3
import pyspark
import requests
import warnings
import numpy as np
import pandas as pd
import vowpalwabbit as pyvw
import pyspark.sql.functions as F

from pyspark.sql import SparkSession, Window
from pyspark import (
    SparkConf, SparkContext
)
from pyspark.sql import (
    SparkSession, SQLContext
)

from pyspark.sql.functions import (
    monotonically_increasing_id, row_number
    )
from pyspark.sql import Window


from pyspark.sql.types import *
from pyspark.sql.functions import *
from dependency import (
    feats_eng, modelling, 
    utils, constants
)

utils.download_s3_model_artifacts(
    'floor-ai',
    'ebayk-floor-ai/warm_start_v2/floor_ai.model',
    'downloads/floor_ai.model'
)

utils.download_s3_model_artifacts(
    'floor-ai',
    'ebayk-floor-ai/warm_start_v2/label_encoder.pkl',
    'downloads/label_encoder.pkl'
)

utils.download_s3_model_artifacts(
    'floor-ai',
    'ebayk-floor-ai/warm_start_v2/ordinal_encoder.pkl',
    'downloads/ordinal_encoder.pkl'
)

# --- loading model and transformer objects
model = pyvw.Workspace(
    "--cb 110 -i downloads/floor_ai.model epsilon 0.6", 
    quiet=True
)

label_encoder = utils.import_object('downloads/label_encoder.pkl')
ordinal_encoder = utils.import_object('downloads/ordinal_encoder.pkl')


# --- creating spark session
conf = SparkConf()\
.set('spark.jars.packages','org.apache.hadoop:hadoop-aws:3.2.2,com.amazonaws:aws-java-sdk-bundle:1.11.901')\
.set('spark.executor.memory','512g')\
.set('spark.driver.memory', '512g')\
.set("fs.s3a.connection.maximum", 100)\
.set('spark.sql.shuffle.partitions',300)

sc = SparkContext(conf=conf)

spark = SparkSession(sc).builder.master('local[*]').appName('test').getOrCreate()
spark._jsc.hadoopConfiguration().set("fs.s3a.impl", "org.apache.hadoop.fs.s3a.S3AFileSystem")
spark._jsc.hadoopConfiguration().set("fs.s3a.endpoint", "s3.amazonaws.com")
spark._jsc.hadoopConfiguration().set("fs.s3a.aws.credentials.provider", "com.amazonaws.auth.DefaultAWSCredentialsProviderChain")
spark.sparkContext.setSystemProperty('com.amazonaws.services.s3.enableV4', 'true')

args = {
    'source':'s3a://floor-ai/ebayk-floor-ai/floor-price-data/ebayK_query_for_training_data_2023_02_27.csv',
    'destination':'s3://floor-ai/ebayk-floor-ai/inference/outgoing_predictions/'
}

# --- reading in holiday_df, aids with feature extraction
holiday_df = feats_eng.get_holiday_df()


# --- User Defined Functions for use on PySpark DataFrame
get_holiday_var_UDF = udf(
    lambda x: feats_eng.get_holiday_var(
        x, holiday_df
    )
)
get_weekend_var_UDF = udf(
    lambda x: feats_eng.get_weekend_var(
        x
    )
)
get_adunit_values_UDF = udf(
    lambda x: feats_eng.get_adunit_values(
        x
    ),
    ArrayType(
        StringType()
    )
)

@F.pandas_udf(returnType = ArrayType(IntegerType()))
def ordinally_encode_values(*cols):
    """
    Pandas UDF function to ordinal encode pyspark columns with scikit learn ordinal encoder
    INPUT
    -----
        column names entered as *args
        Expected columns include:
            -  page_type
            - ad_position
            - is_holiday
            - is_weekend
        
    OUTPUT
    ------
        returns columns with ordinally encoded values
    """
    X = pd.concat(
        cols, 
        axis = 1
    )
    X = X.rename(columns = constants.columns_dict)
    X_transformed = ordinal_encoder.transform(X).tolist()
    X_transformed = pd.Series(X_transformed)
    return X_transformed

@F.pandas_udf(returnType = FloatType())
def format_upr_predictions(predicted_uprs):
    """
    Pandas UDF function to convert list of predicted UPRs to column in pyspark dataframe
    
    INPUT
    -----
        predicted_uprs: list of predicted UPRs
    OUTPUT
    ------
        returns pyspark column of predicted UPRs
    """
    predictions = pd.concat(
        pd.Series(
            predicted_uprs
        ), axis = 1
    )
    return predictions
    
def write_predictions_to_json(spark_dataframe, start_date, end_date):
    
    spark_dataframe = spark_dataframe.dropDuplicates(['adunit', 'day', 'hour'])
    spark_dataframe = spark_dataframe.withColumn(
        'upr', col('upr').cast(
            DoubleType()
        )
    )
    spark_dataframe = spark_dataframe.sort(
        [
            'adunit',
            col('day').cast(IntegerType()),
            col('hour').cast(IntegerType())
        ]
    )
    aggregate_on_hour = spark_dataframe.groupBy(
        col('adunit'), col('day')
        ).agg(
        collect_list(
            col('upr')
        ).alias('collated_UPRs')).sort(
        'adunit', 'day'
    )
    aggregate_on_day = aggregate_on_hour.groupBy(
        col('adunit')
        ).agg(
        collect_list(
            col('collated_UPRs')
        ).alias('collated_UPRs')).toPandas()
        
    result = dict(aggregate_on_day.values)
    
    with open(f"uploads/ebayk_floor_price_rec_{start_date}_{end_date}.json", "w") as outfile:
        json.dump(result, outfile)
    
def read_data():
    """
    Function to read inference data from specified input s3 path (args['source'])
    
    INPUT
    -----
        This function takes no input arguements
        
    OUTPUT
    ------
        returns Pyspark DataFrame
        
    """
    data = spark.read.format('csv').option(
        'header','true'
    ).load(
        args['source']
    )
    data.createOrReplaceTempView(
        'inference_data'
    )
    df = spark.sql(
        """select distinct date, adunit, day, hour from inference_data"""
    )
    df = df.cache()
    return df

def get_date_values():
    """
    Function to get beginning and end dates from data
    
    INPUT
    -----
        This function takes no input arguements
        
    OUTPUT
    ------
        Returns start_date and end_date as string values
        These values feed into the slack notification
    """
    df = read_data()
    start_date = df.select(
        F.min(
            'date'
        ).alias(
            'first_date'
        )
    ).collect()[0]['first_date']
    end_date = df.select(
        F.max(
            'date'
        ).alias(
            'last_date'
        )
    ).collect()[0]['last_date']
    return start_date, end_date
    
def send_slack_message(notification):
    """
    Function to dump notification messages in slack channel
    
    INPUT
    -----
        notification: Notification dictionary containing key value pairs (text:'message') to be dumped in slack channel
        
    OUTPUT
    ------
        posts notification
    """
    webhook = 'https://hooks.slack.com/services/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'
    return requests.post(webhook, json.dumps(notification))



# --- data cleaning and processing
def clean_and_process():
    """
    Function to process inference data
    
    INPUT
    -----
        This function takes no input arguements
        
    OUTPUT
    ------
        processed_df: pyspark dataframe with columns in this order:
                     - day
                     - hour
                     - is_holiday
                     - is_weekend
                     - page_type
                     - ad_position
                     
        adunits: Pyspark Column containing adunits
    """
    df = read_data()
    df = df.na.drop()
    processed_df = df.withColumn(
        'is_holiday', 
        get_holiday_var_UDF(
            col('date')
        )
    )\
                     .withColumn(
        'is_weekend',
        get_weekend_var_UDF(
            col('day')
        )
    )\
                      .withColumn(
        'output',
        get_adunit_values_UDF(
            col('adunit')
        )
    )\
                     .withColumn(
        'page_type', 
        col('output')[0]
    )\
                     .withColumn(
        'ad_position', 
        col('output')[1]
    )

    processed_df = processed_df.select(
        'adunit', 
        'hour',
        'day', 
        ordinally_encode_values(
            *constants.ordinal_cols
        ).alias('encoded_output')
    )
    processed_df = processed_df.withColumn(
        'page_type', 
        col('encoded_output')[0]
    )\
                               .withColumn(
        'ad_position',
        col('encoded_output')[1]
    )\
                               .withColumn(
        'is_holiday', 
        col('encoded_output')[2]
    )\
                               .withColumn(
        'is_weekend', 
        col('encoded_output')[3]
    ).drop('encoded_output')
    return processed_df


start_date, end_date = get_date_values()
processed_df = clean_and_process()

rdd_object = processed_df.toLocalIterator()

predictions = []
for row in rdd_object:
    day = str(row['day'])
    hour = str(row['hour'])
    is_holiday = str(row['is_holiday'])
    is_weekend = str(row['is_weekend'])
    page_type = str(row['page_type'])
    ad_position = str(row['ad_position'])
    
    observation = '| ' + day + ' ' + hour + ' ' + is_holiday + ' ' +  is_weekend + ' ' + page_type + ' ' + ad_position
    predictions.append(label_encoder.inverse_transform([model.predict(observation)])[0])
    
output_df = processed_df.select(
    'adunit', 
    'day', 
    'hour'
).toPandas()

print(np.unique(predictions))
output_df['upr'] = predictions
output_df = spark.createDataFrame(output_df)
output_df.show()
try:
    write_predictions_to_json(output_df, start_date, end_date)
    utils.upload_model_artifacts_to_s3(
        bucket = 'floor-ai',
        key = f"ebayk-floor-ai/inference/outgoing_predictions/ebayk_floor_price_rec_{start_date}_{end_date}.json",
        filename = f"uploads/ebayk_floor_price_rec_{start_date}_{end_date}.json",
    )
    print(
        {
            'text':f"Hello <@U04B85QJCRJ>, Floor Prices for Adunits between {start_date} and {end_date} have been saved in a json file (ebayk_floor_price_rec_{start_date}_to_{end_date}.json) and pushed to {args['destination']} for deployment to Google Ad Manager."
        }
    )
except Exception as e:
    print(
        {
            'text':f"Hello <@U04B85QJCRJ>, I failed at predicting Floor Prices for Adunits between {start_date} and {end_date}. I encountered this error: {e}"    
        }
    )
    
