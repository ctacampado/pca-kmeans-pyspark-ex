
# coding: utf-8

# In[11]:


from pyspark.sql.types import *

mySchema = StructType([ StructField("ID", IntegerType(), True)                       ,StructField("Clothing", FloatType(), True)                       ,StructField("Entertainment", FloatType(), True)                       ,StructField("FoodAndBeverages", FloatType(), True)                       ,StructField("FundTransfer", FloatType(), True)                       ,StructField("Hotel", FloatType(), True)                       ,StructField("Insurance", FloatType(), True)                       ,StructField("Savings", FloatType(), True)                       ,StructField("Transportation", FloatType(), True)                       ,StructField("Utilities", FloatType(), True)                       ,StructField("prediction", IntegerType())
])


#start of imported code from notebook
#use 'import credentials' and 'import dataset' to generate code below 
import ibmos2spark

# @hidden_cell
credentials = {
    'endpoint': '<insert IBM object Storage endpoint url here>',
    'api_key': '<insert IBM object Storage api key here>',
    'service_id': '<insert IBM object Storage service ID here>',
    'iam_service_endpoint': '<insert IBM object Storage iam service endpoint here>'}

configuration_name = '<name of config -- must be unique>'
cos = ibmos2spark.CloudObjectStorage(sc, credentials, configuration_name, 'bluemix_cos')

df_data_1 = spark.read  .format('org.apache.spark.sql.execution.datasources.csv.CSVFileFormat')  .option('header', 'false')  .schema(mySchema)  .load(cos.url('<insert training set file name here>', '<project name and some project instance id??>'))
df_data_1.take(5)
#end of imported code

# In[12]:


df_data_1.printSchema()


# In[13]:


df_data_1.show()


# In[14]:


print ("Number of records: " + str(df_data_1.count()))


# In[15]:


splitted_data = df_data_1.randomSplit([.6,.4],24)
train_data = splitted_data[0]
test_data = splitted_data[1]

print ("Number of training records: " + str(train_data.count()))
print ("Number of testing records : " + str(test_data.count()))


# In[16]:


from pyspark.ml.feature import VectorAssembler
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml import Pipeline, Model


# In[17]:


vectorAssembler_features = VectorAssembler(inputCols=["Clothing", "Entertainment", "FoodAndBeverages", "FundTransfer","Hotel","Insurance","Savings","Transportation","Utilities"], outputCol="features")


# In[37]:


rf = RandomForestClassifier(labelCol="label", featuresCol="features")


# In[38]:


pipeline_rf = Pipeline(stages=[vectorAssembler_features, rf])


# In[39]:


traindf = train_data.withColumnRenamed("prediction", "label")
traindf.printSchema()


# In[40]:


model_rf = pipeline_rf.fit(traindf)


# In[41]:


testdf = test_data.withColumnRenamed("prediction", "label")
predictions = model_rf.transform(testdf)
evaluatorRF = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="accuracy")
accuracy = evaluatorRF.evaluate(predictions)

print("Accuracy = %g" % accuracy)
print("Test Error = %g" % (1.0 - accuracy))
print ("Number of records: " + str(testdf.count()))


# In[42]:


#testdf = test_data.withColumnRenamed("prediction", "label")
#again, use insert dataset code here
testdf_2 = spark.read  .format('org.apache.spark.sql.execution.datasources.csv.CSVFileFormat')  .option('header', 'false')  .schema(mySchema)  .load(cos.url('testset_1b.csv', 'fsinnovationb103791316cf404cac6dab7db1175565'))

testdf = testdf_2.withColumnRenamed("prediction", "label")
predictions = model_rf.transform(testdf)
evaluatorRF = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="accuracy")
accuracy = evaluatorRF.evaluate(predictions)

print("Accuracy = %g" % accuracy)
print("Test Error = %g" % (1.0 - accuracy))
print ("Number of records: " + str(testdf.count()))


# In[24]:


import watson_machine_learning_client as wml_lib
import json


# In[25]:

#watson machine learning service credentials
wml_credentials={
  #<insert credentials here>
}


# In[26]:


client = wml_lib.WatsonMachineLearningAPIClient(wml_credentials)


# In[27]:


meta_props = {
    client.repository.ModelMetaNames.NAME: "epc_v1",
    client.repository.ModelMetaNames.EVALUATION_METHOD: "multiclass",
    client.repository.ModelMetaNames.EVALUATION_METRICS: [{
        "name": "accuracy",
        "value": accuracy,
        "threshold": 0.9
    }]
}


# In[28]:


saved_model_details = client.repository.store_model(model_rf, meta_props, pipeline=pipeline_rf, training_data=traindf)
model_uid = client.repository.get_model_uid(saved_model_details)


# In[29]:


client.repository.ModelMetaNames.show()


# In[30]:


print(saved_model_details)

