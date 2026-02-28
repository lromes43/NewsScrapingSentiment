#SVM

from sklearn.svm import LinearSVC
import pandas as pd 
import polars as pl
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from plotnine import *

long = pl.read_parquet("/Users/lukeromes/Desktop/NewsScrapingSentiment/FinalPolarityData.parquet")
long = long.to_pandas()
long.columns.to_list()

long['value_type'].unique()
long['sentiment_type'].unique()


stock_data = pd.read_feather("/Users/lukeromes/Desktop/NewsScrapingSentiment/Data/FINALSP500Data.feather")
stock_data.columns.to_list()
stock_data = stock_data[['Date', 'Close','Movement']]

vader_polarity = long[long['sentiment_type'] =='vader_polarity']
fin_polarity = long[long['sentiment_type'] =='Fin_Sent_Score']
news_polarity = long[long['sentiment_type'] =='News_Headlines_Score']


vader_polarity['Date'] = pd.to_datetime(vader_polarity['Date'])
vader_stock = pd.merge(vader_polarity, stock_data, how = 'inner', on = 'Date')

fin_polarity['Date'] = pd.to_datetime(fin_polarity['Date'])
fin_stock = pd.merge(stock_data, fin_polarity, how='inner', on = 'Date')

news_polarity['Date'] = pd.to_datetime(news_polarity['Date'])
news_stock = pd.merge(stock_data, news_polarity, how = 'inner', on = 'Date')




#flair was not transferred over in final data will look into 

#no sentiment

X = stock_data[['Close']]
y = stock_data['Movement']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

clf = LinearSVC(max_iter=10000)
clf.fit(X_train_scaled, y_train)

y_pred = clf.decision_function(X_test_scaled)
y_pred_acc = clf.predict(X_test_scaled)
no_sent_acc = accuracy_score(y_test, y_pred_acc)
no_sent_auc= roc_auc_score(y_test, y_pred)
no_sent_auc  #0.837



#vader sentiment 

vader_stock.columns.to_list()
vader_stock['value']

X = vader_stock[['Close', 'value']]
y = vader_stock['Movement']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

clf = LinearSVC(max_iter=10000)
clf.fit(X_train_scaled, y_train)

y_pred = clf.decision_function(X_test_scaled)
y_pred_acc = clf.predict(X_test_scaled)

vader_acc = accuracy_score(y_test, y_pred_acc)
vader_auc= roc_auc_score(y_test, y_pred)
vader_auc #0.7887309520617722


#fin sentiment


fin_stock.columns.to_list()


X = fin_stock[['Close', 'value']]
y = fin_stock['Movement']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

clf = LinearSVC(max_iter=10000)
clf.fit(X_train_scaled, y_train)

y_pred = clf.decision_function(X_test_scaled)
y_pred_acc = clf.predict(X_test_scaled)

fin_acc = accuracy_score(y_test, y_pred_acc)
fin_auc = roc_auc_score(y_test, y_pred)
fin_auc #0.7888416858673032




#news


news_stock.columns.to_list()


X = news_stock[['Close', 'value']]
y = news_stock['Movement']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

clf = LinearSVC(max_iter=10000)
clf.fit(X_train_scaled, y_train)

y_pred = clf.decision_function(X_test_scaled)
y_pred_acc = clf.predict(X_test_scaled)

news_acc = accuracy_score(y_test, y_pred_acc)
News_auc = roc_auc_score(y_test, y_pred)
News_auc #0.788842510955221





#comparing



models = ['No_Sentiment', 'Vader', 'Financial Sentiment', 'News Sentiment']
accuracies = [no_sent_acc, vader_acc, fin_acc, news_acc]
aucs= [no_sent_auc, vader_auc, fin_auc, News_auc]



df = pd.DataFrame({
    'Model': models,
    'Accuracy': accuracies,
    'AUC': aucs
})

print(df)


#now going to just plot 

#acc

plot = ggplot(df, aes(x= 'models', y = 'Accuracy', fill = 'models')) + geom_col() + labs(title = 'Model Accuracy By Sentiment')
plot.show()


#auc 


plot = ggplot(df, aes(x= 'models', y = 'AUC', fill = 'models')) + geom_col() + labs(title = 'Model AUC By Sentiment')
plot.show()