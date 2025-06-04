import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model  import Ridge
from sklearn.linear_model  import Lasso
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
df=pd.read_csv(r"C:\Users\Asus\OneDrive - IIT Kanpur\Desktop\project_data.csv")
print(df.head().to_string() )
print(df.columns )
#print(df.info())
print(df.describe().to_string() )
#sns.heatmap (df,annot=True)
#plt.show()
df["windspeed"]=df["windspeed"].astype(int)
df["ttimestamplocal"]=pd.to_datetime(df["ttimestamplocal"])
print(df.info())

del df["generation"]
del df["ttimestamplocal"]
turbine=   pd.get_dummies(df["unitlocation"])
df_final=pd.concat ([df,turbine],axis=1)
#print(df_final.info())
df_final.drop(["unitlocation"],inplace=True,axis=1)
print(df_final.info())
plt.figure(figsize=(12,12))
sns.heatmap (df_final.corr(),annot=True)
plt.show()
for i in df_final .index :
    if df_final.loc[i,"power"]<=0:
        df_final.drop(i,inplace=True)

df_final.reset_index(inplace=True)
#print(df_final.to_string() )

from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
X=df_final[["windspeed","wind direction Angle","rtr_rpm","pitch Angle","wheel hub temperature","ambient Temperature","Tower bottom ambient temperature","failure time"]]
y=df_final[["power"]]
print(X.head())
print(y.head())
X_train,X_test,y_train,y_test=train_test_split(X,y,random_state=50,train_size= .95)
print(X_train .head())
print(X_test .head())
print(y_train .head())
print(y_test.head())
regr=linear_model .LinearRegression ()
regr.fit(X_train,y_train)
y_predict=regr.predict(X_test)

print(y_test.head())
y_pred=pd.DataFrame (y_predict )
print(y_pred .head())
"""y_compare=pd.concat([y_test,y_pred],axis=1)
print(y_compare.to_string() )"""

print(regr.score(X_test ,y_test ))
#print(r2_score(y_test,y_predict ) )
mse=cross_val_score(regr,y_test,y_pred,scoring="neg_mean_squared_error",cv=5)
mean_mse_lin=np.mean(mse)
print("mean squared error for linear regressorr=",mean_mse_lin)