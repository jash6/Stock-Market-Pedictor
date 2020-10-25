# IMPORTING NECESSARY MODULES
from flask import Flask,request, url_for, redirect, render_template
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
import sqlalchemy
import pymysql
import cufflinks as cf
import chart_studio.plotly as ply
import plotly.express as px
import holidays
import datetime
import tensorflow
import urllib.request
from pprint import pprint
from html_table_parser import HTMLTableParser
from sklearn.preprocessing import MinMaxScaler
from keras.models import load_model
from plotly.offline import download_plotlyjs,init_notebook_mode,plot,iplot
import plotly.graph_objects as go
init_notebook_mode(connected=True)
cf.go_offline()
warnings.filterwarnings('ignore')
window=60  


app = Flask(__name__)


# FUNCTIONS TO READ AND WRITE DATA FROM/TO DATABASE
def writetosql(stock,dataset,con):
  dataset.to_sql(stock,con,index=False,if_exists='replace')

def readsql(stock,con):
  query='SELECT Date,Close FROM '+stock
  dataset=pd.read_sql(query,con)
  return dataset



# FUNCTION TO CHECK IF A PARTICULAR DAY IS WEEKEND OR NOT SINCE STOCK MARKET CLOSED ON WEEKENDS
def isWeekend(date):
    weekno=date.weekday()
    if weekno<5:
        return False
    return True



# FUNCTION TO CHECK IF PARTICULAR DAY IS A HOLIDAY
def isHoliday(date):
    india_holidays=holidays.India(years=datetime.datetime.now().year)
    return (date in india_holidays)



# CONNECT TO DATABASE
def connect_to_db():
  engine=sqlalchemy.create_engine('mysql+pymysql://stock:stock@localhost/stockdb', pool_recycle=3600)
  conn=engine.connect()
  return conn

# GETTING LIVE DATA
def get_contents(url):
    req=urllib.request.Request(url)
    sock=urllib.request.urlopen(req)
    return sock.read()

def get_live_data(url):
  html=get_contents(url).decode('utf-8')
  parser=HTMLTableParser()
  parser.feed(html)
  livedata=pd.DataFrame(parser.tables[0],columns=parser.tables[0][0])[['Date','Close*']].rename(columns={'Close*':'Close'})
  livedata.drop(livedata.index[[0,-1]],inplace=True)
  livedata['Date']=pd.to_datetime(livedata['Date']).dt.date
  livedata=livedata.reindex(index=livedata.index[::-1])
  return livedata


# MERGING EXISTING DATA WITH LIVE DATA
def merge(dataset,livedata):
  dataset=pd.concat([dataset,livedata]).drop_duplicates('Date',keep='last').reset_index(drop=True)
  dataset['Close']=dataset['Close'].astype(str).apply(lambda x: float(x.split()[0].replace(',','')))
  return dataset





@app.route('/')
def hello_world():
    return render_template("stock.html")


@app.route('/predict',methods=['POST','GET'])
def predict():
    stock=next(request.form.values())
    conn=connect_to_db();
    dataset=readsql(stock.lower(),conn)
    url='https://in.finance.yahoo.com/quote/' + stock + '.NS/history/'
    livedata=get_live_data(url)
    dataset=merge(dataset,livedata)
    writetosql(stock.lower(),dataset,conn)
    length=len(dataset.index)
    scaler=MinMaxScaler(feature_range=(0,1))
    training=pd.DataFrame(dataset['Close'])
    testing=training[-window:]
    testing['Scaled']=scaler.fit_transform(testing)
    testing.index=pd.to_datetime(dataset['Date'][-window:])
    model=load_model('Models/'+stock+'.h5')
    for i in range (30):
      x_test=[]
      x_test.append(testing['Scaled'][-window:])
      x_test=np.array(x_test)
      x_test=np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))
      scaled_pred=model.predict(x_test)
      pred=scaler.inverse_transform(scaled_pred)
      next_date=testing.index[-1]+datetime.timedelta(days=1)
      while isWeekend(next_date) or isHoliday(next_date):
          next_date+=datetime.timedelta(days=1)
      testing=testing.append(pd.Series([pred[0][0],scaled_pred[0][0]],name=next_date,index=testing.columns),ignore_index=False)
    prediction=testing[window:]
    prediction.drop(['Scaled'],axis=1,inplace=True)
    return render_template('stock.html',pred='Predicted values are: \n {}'.format(prediction))


if __name__ == '__main__':
    app.run(debug=True)
