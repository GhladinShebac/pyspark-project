import streamlit as st 
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression

df = pd.read_csv("bank.csv")
df = pd.get_dummies(df, columns = ['job', 'marital', 'education', 'default', 'housing', 'loan', 'contact', 'month', 'poutcome'])
df['deposit'] = LabelEncoder().fit_transform(df['deposit'])
X, y  = df.drop('deposit', axis=1), df['deposit']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
clf = LogisticRegression(random_state=0).fit(X_train, y_train) 

st.write("""
# PySpark Project.
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
  
  data = {'age': age, 'job': job, 'marital': marital, 'education': education,
        'default': default, 'balance': balance, 'housing': housing,
        'loan': loan, 'contact': contact, 'day': day, 'month': month,
        'duration': duration, 'campaign': campaign, 'pdays': pdays,
        'previous': previous, 'poutcome': poutcome}
  features = pd.DataFrame(data, index=[0])
  return features

df = user_input_var()
df = pd.get_dummies(df, columns = ['job', 'marital', 'education', 'default', 'housing', 'loan', 'contact', 'month', 'poutcome'])

if st.button('Predict'):
    pred = clf.predict(df.asarray())
    st.write(pred)
else: pass
