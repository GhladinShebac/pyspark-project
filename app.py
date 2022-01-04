import pandas as pd
import streamlit as st 
from pyspark.sql import SparkSession
from pyspark.ml.feature import OneHotEncoder, StringIndexer, VectorAssembler
from pyspark.ml import Pipeline
from pyspark.ml.classification import RandomForestClassifier

spark = SparkSession.builder.appName('ml-bank').getOrCreate()
df = spark.read.csv('bank.csv', header = True, inferSchema = True)
cols = df.columns
categoricalColumns = ['job', 'marital', 'education', 'default', 'housing', 'loan', 'contact', 'poutcome']
stages = []
for categoricalCol in categoricalColumns:
    stringIndexer = StringIndexer(inputCol = categoricalCol, outputCol = categoricalCol + 'Index')
    encoder = OneHotEncoder(inputCols=[stringIndexer.getOutputCol()], outputCols=[categoricalCol + "classVec"])
    stages += [stringIndexer, encoder]
label_stringIdx = StringIndexer(inputCol = 'deposit', outputCol = 'label')
stages += [label_stringIdx]
numericCols = ['age', 'balance', 'duration', 'campaign', 'pdays', 'previous']
assemblerInputs = [c + "classVec" for c in categoricalColumns] + numericCols
assembler = VectorAssembler(inputCols=assemblerInputs, outputCol="features")
stages += [assembler]
pipeline = Pipeline(stages = stages)
pipelineModel = pipeline.fit(df)
df = pipelineModel.transform(df)
selectedCols = ['label', 'features'] + cols
df = df.select(selectedCols)
train, test = df.randomSplit([0.7, 0.3], seed = 2018)
rf = RandomForestClassifier(featuresCol = 'features', labelCol = 'label')
rfModel = rf.fit(df)
predictions = rfModel.transform(df)

st.write("""
# Bank
describ.
""")

def user_input_var():
  age = st.slider('age', 0, 100)
  job = st.selectbox('job', ('admin.', 'technician', 'services', 'management', 'retired', 'blue-collar', 'unemployed', 'entrepreneur', 'housemaid', 'unknown', 'self-employed', 'student'))
  marital = st.selectbox('marital', ('married', 'single', 'divorced'))
  education = st.selectbox('education', ('secondary', 'tertiary', 'primary', 'unknown'))
  default = st.radio("default", ('no', 'yes'))
  balance = st.slider('balance', -10000, 100000)
  housing = st.radio("housing", ('no', 'yes'))
  loan = st.radio("loan", ('no', 'yes'))
  contact = st.selectbox('contact', ('unknown', 'cellular', 'telephone'))
  day = st.slider('day', 1, 31)
  month = st.selectbox('month', ('may', 'jun', 'jul', 'aug', 'oct', 'nov', 'dec', 'jan', 'feb', 'mar', 'apr', 'sep'))
  duration = st.slider('duration', 0, 4000)
  campaign = st.slider('campaign', 1, 31)
  pdays = st.slider('pdays', -1, 1000)
  previous = st.slider('previous', 0, 60)
  poutcome = st.selectbox('poutcome', ('unknown', 'other', 'failure', 'success'))
  deposit = st.radio("deposit", ('no', 'yes'))

  data = {'age': age, 'job': job, 'marital': marital, 'education': education,
        'default': default, 'balance': balance, 'housing': housing,
        'loan': loan, 'contact': contact, 'day': day, 'month': month,
        'duration': duration, 'campaign': campaign, 'pdays': pdays,
        'previous': previous, 'poutcome': poutcome, 'deposit': deposit}
  features = pd.DataFrame(data, index=[0])
  return features

df = user_input_var()
df = spark.createDataFrame(df)
cols = df.columns
categoricalColumns = ['job', 'marital', 'education', 'default', 'housing', 'loan', 'contact', 'poutcome']
stages = []
for categoricalCol in categoricalColumns:
    stringIndexer = StringIndexer(inputCol = categoricalCol, outputCol = categoricalCol + 'Index')
    encoder = OneHotEncoder(inputCols=[stringIndexer.getOutputCol()], outputCols=[categoricalCol + "classVec"])
    stages += [stringIndexer, encoder]
label_stringIdx = StringIndexer(inputCol = 'deposit', outputCol = 'label')
stages += [label_stringIdx]
numericCols = ['age', 'balance', 'duration', 'campaign', 'pdays', 'previous']
assemblerInputs = [c + "classVec" for c in categoricalColumns] + numericCols
assembler = VectorAssembler(inputCols=assemblerInputs, outputCol="features")
stages += [assembler]
pipeline = Pipeline(stages = stages)
pipelineModel = pipeline.fit(df)
df = pipelineModel.transform(df)
selectedCols = ['label', 'features'] + cols
df = df.select(selectedCols)


if st.button('Predict'):
    predictions = rfModel.transform(df)
    st.write(predictions)
else: pass
