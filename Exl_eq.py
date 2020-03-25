import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


df_train = pd.read_csv("EXL_EQ_2020_Train_datasets.csv")
y=df_train.iloc[:,39]
X=df_train.iloc[:,0:39]
X_cat=X.iloc[:,[3,6,11,14,16,21,27,28,29,30,31,32,33,34]]
X_cont=X.iloc[:,[0,1,2,4,5,7,8,9,10,12,13,15,17,18,19,20,22,23,24,25,26]]
X_cont=X_cont.values
X_cont[pd.isnull(X_cont)]='NaN'
X_cat=X_cat.values

n=pd.isnull(X_cont)
X_cat[pd.isnull(X_cat)]='NaN'

from sklearn.preprocessing import LabelEncoder,OneHotEncoder,Imputer,StandardScaler
label_x=LabelEncoder()
label_y=LabelEncoder()
X_cat[:,7]=label_x.fit_transform(X_cat[:,7])
X_cat[:,8]=label_x.fit_transform(X_cat[:,8])
X_cat[:,9]=label_x.fit_transform(X_cat[:,9])
X_cat[:,10]=label_x.fit_transform(X_cat[:,10])
X_cat[:,11]=label_x.fit_transform(X_cat[:,11])
X_cat[:,12]=label_x.fit_transform(X_cat[:,12])
X_cat[:,13]=label_x.fit_transform(X_cat[:,13])

y=label_y.fit_transform(y)

'''hot_x=OneHotEncoder(categorical_features=[0,1,2,3,4,5,6,7,8,9,10,11,12,13])
X_cat=hot_x.fit_transform(X_cat).toarray()'''

imp=Imputer(missing_values='NaN',axis=0)
X_cont=imp.fit_transform(X_cont)

sc_x=StandardScaler()
X_cont=sc_x.fit_transform(X_cont)

X_left=X.iloc[:,35:39].values

x=np.zeros((300000,39))

x[:,0:14]=X_cat
x[:,14:35]=X_cont
x[:,35:]=X_left

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(x,y,test_size=0.1,random_state=42)

'''from sklearn.utils import class_weight
class_weights = class_weight.compute_class_weight('balanced',
                                                  np.unique(y_train),
                                                  y_train)
class_weight = {0: 0.1,
                1: 20.349,
                2: 20.560,
                3: 50.164}
'''
from sklearn.ensemble import RandomForestClassifier
clf=RandomForestClassifier(n_estimators = 300, random_state = 42,class_weight=class_weight)
clf.fit(X_train,y_train)
#from sklearn.svm import SVC
#svc= SVC(C=1.0,degree=2,kernel='sigmoid')
#svc.fit(X_train,y_train)

y_pred=clf.predict(X_test)
from sklearn.metrics import accuracy_score
accuracy=accuracy_score(y_pred,y_test)

from sklearn.metrics import confusion_matrix
conf=confusion_matrix(y_pred,y_test)














