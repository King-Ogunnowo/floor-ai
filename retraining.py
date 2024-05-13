# --- import packages
import os
import sys

sys.path.insert(0, 'dependency.zip')
os.mkdir('uploads')
os.mkdir('downloads')

# --- importing more packages
import json
import boto3
import requests
import pyspark
import warnings
import numpy as np
import pandas as pd
import vowpalwabbit as pyvw
import pyspark.sql.functions as F

from sklearn.preprocessing import (
    OrdinalEncoder, LabelEncoder
)

from pyspark.sql import SparkSession, Window
from pyspark import (
    SparkConf, SparkContext
)
from pyspark.sql import (
    SparkSession, SQLContext
)
from pyspark.sql.types import *
from pyspark.sql.functions import *
from dependency import (
    feats_eng, modelling, 
    utils, feats_eng_pyspark,
    constants
)


# --- loading model and transformer objects

utils.download_s3_model_artifacts(
    'floor-ai',
    'ebayk-floor-ai/warm_start/model_artifacts/ordinal_encoder.pkl',
    'downloads/ordinal_encoder.pkl'
)

label_encoder = LabelEncoder()
ordinal_encoder = utils.import_object('downloads/ordinal_encoder.pkl')

# --- order columns should be arranged after data has been preprocessed
columns = [
    'day', 'hour', 'is_holiday',
    'is_weekend', 'page_type', 'ad_position'
]

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

# --- input inference data s3 path and outgoing predictions s3 path
args = {
    'prev_training_set':'s3a://floor-ai/ebayk-floor-ai/floor-price-data/ebay_k_floor_price_v1.csv',
    'new_training_set':'s3a://floor-ai/ebayk-floor-ai/floor-price-data/ebay_k_floor_price_v1.csv'
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

# --- reading data
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
        args['prev_training_set']
    )
    data = data.cache()
    data.createOrReplaceTempView(
        'feedback_data'
    )
    df = spark.sql(
        f"""select distinct *
                   from feedback_data
         """
    )
    return df
    
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

    
# --- data cleaning and processing
def clean_and_process():
    """
    Function to process inference data
    
    INPUT
    -----
        This function takes no input arguements
        
    OUTPUT
    ------
        processed_df: pyspark dataframe with features
            
    """
    df = read_data()
    df = df.na.drop()
    df = df.withColumn(
        'yield', 
        col('fill_rate') * col('yield')
    ).drop('fill_rate')
    
    processed_df = df.withColumn(
        'day', col(
            'day'
        ).cast(
            IntegerType()
        )
    )\
                    .withColumn(
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
                     
    processed_df = processed_df.toPandas()
    processed_df['action_probability'] = feats_eng.get_probability(
        processed_df
    )
    processed_df = spark.createDataFrame(
        processed_df
    )
    
    processed_df = processed_df.withColumn(
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
        'upr',
        'action_probability',
        'yield',
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
    ).drop('encoded_output', 'output')
    
    unique_uprs = np.unique(processed_df.select('upr').rdd.flatMap(lambda x: x).collect())
    number_of_unique_uprs = len(unique_uprs)
    label_encoder.fit(
        unique_uprs
    )
    return processed_df, label_encoder, number_of_unique_uprs


def retrain_model(processed_df, agent, label_encoder):
    """
    Function to update model with feedback data
    UPRs are encoded with integers with the help of scikit learn label encoder
    
    INPUT
    -----
        Processed_df: 
        
    OUTPUT
    ------
        Trained vowpal wabbit model
    """
    rdd_object = processed_df.toLocalIterator()
    
    predictions = []
    for row in rdd_object:
        
        action = label_encoder.transform([row['upr']])[0]
        cost = -np.round(
            float(
                row['yield']
            ),4
        )
        probability = row['action_probability']
        
        observation_string = ' '.join(
            [
                str(row['day']),
                str(row['hour']),
                str(row['is_holiday']),
                str(row['is_weekend']),
                str(row['page_type']),
                str(row['ad_position'])
            ]
        )
        
        learning_example = (
            str(action) + ':' + str(cost) + ':' + str(probability) + ' | ' + observation_string
        )

        agent.learn(learning_example)
    
    return agent


start_date, end_date = get_date_values()
output_data, label_encoder, number_of_unique_uprs = clean_and_process()
output_data = output_data.cache()

agent = pyvw.Workspace(
    f"--cb {number_of_unique_uprs}", 
    quiet=True
)

try:
    model = retrain_model(
        output_data,
        agent,
        label_encoder
    )
    
    model.save('uploads/floor_ai.model')
    utils.export_object(label_encoder, 'uploads/label_encoder.pkl')
    
    utils.upload_model_artifacts_to_s3(
        bucket = 'floor-ai',
        key = 'ebayk-floor-ai/warm_start/model_art_dud_folder/floor_ai.model',
        filename = 'uploads/floor_ai.model'
    )
    
    utils.upload_model_artifacts_to_s3(
        bucket = 'floor-ai',
        key = 'ebayk-floor-ai/warm_start/model_art_dud_folder/label_encoder.pkl',
        filename = 'uploads/label_encoder.pkl'
    )
    
    print(
        {
            'text':f"Hello <@U04B85QJCRJ>, The floor AI model has been retrained with data between {start_date} and {end_date}. The updated model and its associated artifacts (label encoder and ordinal encoder) have been saved in this destination: s3://floor-ai/ebayk-floor-ai/warm_start/model_art_dud_folder/. You can disregard this message as this is just a test of the retraining functionality."
        }    
    )
    
except Exception as e:
    print(
        {
            'text':f"Hello <@U04B85QJCRJ>, I failed at retraining the floor AI Model. In the process of doing so I faced this error: {e}"
        }
    )
