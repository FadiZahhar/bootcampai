from sklearn.model_selection import train_test_split
from sklearn import preprocessing
import numpy as np
from pandas import read_csv
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from keras.utils import np_utils
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
import pandas as pd
from sklearn.metrics import roc_curve
from sklearn import metrics
df = pd.read_csv('cardio_train.csv',delimiter=';')
df.rename({'cardio':'Heart_Disease'}, axis=1, inplace=True)
df.head()
from sklearn import preprocessing
# Create the Scaler object
float_cols = ['age','height','weight','ap_hi','ap_lo']
cat_cols = ['gluc','cholesterol','gender']
scaler = preprocessing.StandardScaler()
# Fit your data on the scaler object
scaled_df = scaler.fit_transform(df[float_cols].values)
scaled_df = pd.DataFrame(scaled_df, columns=float_cols, index = df.index)
df[float_cols] = scaled_df
# Transform categorical columns in dummy variables
df_final = pd.get_dummies(df, prefix_sep="__", columns=cat_cols)
predictor_columns = [x for x in df_final.columns if x not in ["Heart_Disease", "id"]]
# Set randomness
np.random.seed(42)
# Split the data into train and test pieces for both X and Y
X_train, X_test, y_train, y_test = train_test_split(df_final[predictor_columns], df_final.Heart_Disease, train_size=0.7,test_size=0.3)

def baseline_model():
   model = Sequential()
   model.add(Dense(100,input_dim=(16),activation='sigmoid'))
   model.add(Dense(200,activation='relu'))
   model.add(Dense(1,activation='sigmoid'))
   model.compile(optimizer='adam',loss='binary_crossentropy',metrics = ['accuracy'])
   return model
estimator = KerasClassifier(build_fn=baseline_model, epochs=5, batch_size=10, verbose=2)
estimator.fit(X_train,y_train)
y_hat = estimator.predict(X_test)
from sklearn.metrics import jaccard_similarity_score
jaccard_similarity_score(y_test, y_hat)
conf_matrix = metrics.confusion_matrix(y_test, y_hat,labels=[1, 0]).T
confusion_matrix_one = pd.DataFrame(conf_matrix, columns=['p', 'n'], index=['Y', 'N'])
confusion_matrix_one