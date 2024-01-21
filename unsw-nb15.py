import pandas as pd
import numpy as np
from keras.layers import LSTM, ReLU, BatchNormalization, Convolution1D, MaxPooling1D, Reshape, GlobalAveragePooling1D, \
    Dropout, Bidirectional, concatenate
from sklearn import metrics
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.model_selection import StratifiedKFold
from keras.layers import Dense, Embedding, Flatten, Input, Conv1D, Concatenate
from keras.models import Model
import tensorflow as tf

from att.atten import Attention
from keras import backend as K
from sklearn.metrics import classification_report
from functools import partial
import os
import pickle
gpus = tf.config.list_physical_devices(device_type='GPU')
tf.config.set_visible_devices(devices=gpus[0], device_type='GPU')
df = pd.read_csv(r'D:\codework\CANET-main (1)\CANET-main\data\UNSW NID\UNSW_NB15_test-set.csv')
qp = pd.read_csv(r'D:\codework\CANET-main (1)\CANET-main\data\UNSW NID\UNSW_NB15_train-set.csv')
# print(qp.head())
# Dropping the last columns of training set
df = df.drop('id', 1)
df = df.drop('label', 1)
qp = qp.drop('id', 1)
qp = qp.drop('label', 1)
cols = ['proto','state','service']


# One-hot encoding
def one_hot(df, cols):
    """
    @param df pandas DataFrame
    @param cols a list of columns to encode
    @return a DataFrame with one-hot encoding
    """
    for each in cols:
        dummies = pd.get_dummies(df[each], prefix=each, drop_first=False)
        df = pd.concat([df, dummies], axis=1)
        df = df.drop(each, 1)
    return df


combined_data = pd.concat([df,qp])
tmp = combined_data.pop('attack_cat')
combined_data = one_hot(combined_data,cols)


def normalize(df, cols):
    """
    @param df pandas DataFrame
    @param cols a list of columns to encode
    @return a DataFrame with normalized specified features
    """
    result = df.copy() # do not touch the original df
    for feature_name in cols:
        max_value = df[feature_name].max()
        min_value = df[feature_name].min()
        if max_value > min_value:
            result[feature_name] = (df[feature_name] - min_value) / (max_value - min_value)
    return result


new_train_df = normalize(combined_data,combined_data.columns)
# new_train_df = combined_data
new_train_df["Class"] = tmp
y_train = new_train_df["Class"]
combined_data_X = new_train_df.drop('Class', 1)
from sklearn.ensemble import RandomForestClassifier
# def feature(x,y):
#     rf_model = RandomForestClassifier()
#     rf_model.fit(x,y)
#     feat_importances = pd.Series(rf_model.feature_importances_, index=x.columns)
#     df_imp_feat= feat_importances.nlargest(20)
#     df_imp_feat.plot(kind='barh')
#     plt.show()
#     print(df_imp_feat)
#     return df_imp_feat
# determine 20 most important features


acc = []
dr = []
fpr = []
F1 = []
cfn = []
selector = SelectKBest(chi2, k=32).fit(combined_data_X, y_train)
# 将原始特征矩阵转换为只包含选中特征的子集
selected_features = selector.transform(combined_data_X)
selected_feature_names = [new_train_df.columns[i] for i in selector.get_support(indices=True)]

# 创建DataFrame，将特征矩阵转换为DataFrame，并设置列名为选中特征名称
combined_data_X= pd.DataFrame(selected_features, columns=selected_feature_names)

# combined_data_X = pd.DataFrame(combined_data_X)
combined_data_X


kfold = StratifiedKFold(n_splits=6, shuffle=True, random_state=42)
kfold.get_n_splits(combined_data_X, y_train)
num_classes=10
# def create_model(optimizer='adam'):
#     main_input=Input(shape=(combined_data_X.shape[1],1),dtype='float64')
#     embed=main_input
#     # re = tf.squeeze(embed,axis=-1)
#     at=Attention(units=256)(embed)
#     cnn=Convolution1D(64,kernel_size=8,padding='same',activation='relu')(embed)
#     cnn=MaxPooling1D(pool_size=2)(cnn)
#     cnn=BatchNormalization()(cnn)
#     bilstm=Bidirectional(LSTM(64,return_sequences=True))(cnn)
#     # bilstm=Reshape((128, 1), input_shape = (128, ))(bilstm)
#     bilstm=MaxPooling1D(pool_size=(2))(bilstm)
#     bilstm=BatchNormalization()(bilstm)
#     bilstm2=Bidirectional(LSTM(128, return_sequences=True))(bilstm)
#     bilstm3=Attention(units=256)(bilstm2)
#     att=Convolution1D(64,kernel_size=8,padding='same',activation='relu')(embed)
#     att=MaxPooling1D(pool_size=2)(att)
#     att=Attention(units=256)(att)
#     fla=concatenate([bilstm2,att,at,bilstm3],axis=-1)
#     drop=Dropout(0.5)(fla)
#     main_output=Dense(num_classes,activation='softmax')(drop)
#     model=Model(inputs=main_input,outputs=main_output)
#     model.compile(loss='categorical_crossentropy', optimizer='Adam',  metrics=['accuracy'])
#     return model
def create_model(optimizer='adam'):
    main_input=Input(shape=(combined_data_X.shape[1],1),dtype='float64')
    embed=main_input
    cnn=Convolution1D(64,kernel_size=8,padding='same',activation='relu')(embed)
    cnn=MaxPooling1D(pool_size=5)(cnn)
    cnn=BatchNormalization()(cnn)
    bilstm=Bidirectional(LSTM(64,return_sequences=False))(cnn)
    bilstm=Reshape((128, 1), input_shape = (128, ))(bilstm)
    bilstm=MaxPooling1D(pool_size=(5))(bilstm)
    bilstm=BatchNormalization()(bilstm)
    bilstm=Bidirectional(LSTM(128, return_sequences=False))(bilstm)
    att=Convolution1D(64,kernel_size=8,padding='same',activation='relu')(embed)
    att=MaxPooling1D(pool_size=5)(att)
    att=Attention(units=256)(att)
    fla=concatenate([bilstm,att],axis=-1)
    drop=Dropout(0.5)(fla)
    main_output=Dense(num_classes,activation='softmax')(drop)
    model=Model(inputs=main_input,outputs=main_output)
    model.compile(loss='categorical_crossentropy', optimizer='Adam',  metrics=['accuracy'])
    return model

model = create_model()
model.summary()

from zs.log import Log

for train_index, test_index in kfold.split(combined_data_X,y_train):
    train_X, test_X = combined_data_X.iloc[train_index], combined_data_X.iloc[test_index]
    train_y, test_y = y_train.iloc[train_index], y_train.iloc[test_index]
    # train_X_over, train_y_over = train_X, train_y

    # x_columns_train = new_train_df.columns.drop('Class')
    x_train_array = train_X[selected_feature_names].values
    x_train_1=np.reshape(x_train_array, (x_train_array.shape[0], x_train_array.shape[1], 1))

    dummies = pd.get_dummies(train_y) # Classification
    outcomes = dummies.columns
    num_classes = len(outcomes)
    y_train_1 = dummies.values

    # x_columns_test = new_train_df.columns.drop('Class')
    x_test_array = test_X[selected_feature_names].values
    x_test_2=np.reshape(x_test_array, (x_test_array.shape[0], x_test_array.shape[1], 1))

    dummies_test = pd.get_dummies(test_y) # Classification
    outcomes_test = dummies_test.columns
    num_classes = len(outcomes_test)
    y_test_2 = dummies_test.values

    history = model.fit(x_train_1, y_train_1, validation_split=0.1, epochs=100, batch_size=256)

    pred = model.predict(x_test_2)
    pred = np.argmax(pred,axis=1)
    y_eval = np.argmax(y_test_2,axis=1)
    target_names = ['Analysis', 'Backdoor', 'DoS', 'Exploits', 'Fuzzers', 'Generic', 'Normal', 'Reconnaissance', 'Shellcode', 'Worms']
    accuracy = metrics.accuracy_score(y_eval, pred)
    recall = metrics.recall_score(y_eval, pred, average='macro')
    precision = metrics.precision_score(y_eval, pred, average='macro')
    f1score = metrics.f1_score(y_eval, pred, average='macro')
    classification = metrics.classification_report(y_eval, pred, target_names=target_names)
    cm = metrics.confusion_matrix(y_eval, pred)
    mcc=metrics.matthews_corrcoef(y_eval, pred)
    kappa=metrics.cohen_kappa_score(y_eval, pred)
    tp = cm[0][0]+cm[1][1]+cm[2][2]+cm[3][3]+cm[4][4]+cm[5][5]+cm[7][7]+cm[8][8]+cm[9][9]
    fn = cm[0][6]+cm[1][6]+cm[2][6]+cm[3][6]+cm[4][6]+cm[5][6]+cm[7][6]+cm[8][6]+cm[9][6]
    fp = cm[6][0]+cm[6][1]+cm[6][2]+cm[6][3]+cm[6][4]+cm[6][5]+cm[6][7]+cm[6][8]+cm[6][9]
    tn = cm[6][6]
    DR = tp/(tp+fn)
    FPR = fp/(fp+tn)
    dr.append(DR)
    fpr.append(FPR)
    Log.console("===========unsw-chi2val===========")
    Log.console("traindata:\n{}".format(train_X.shape))
    Log.console("testdata:\n{}".format(test_X.shape))
    Log.console("matrics_mat:\n{}".format(cm))
    Log.console("evaluation matrix of data:\n{}".format(classification))
    Log.console("evaluation accuracy of data:{}".format(accuracy))
    Log.console("evaluation precision of data:{}".format(precision))
    Log.console("evaluation recall of data:{}".format(recall))
    Log.console("evaluation f1score of data:{}".format(f1score))
    Log.console("mcc:{}".format(mcc))
    Log.console("kappa:{}".format(kappa))
    print("DR=recall:", DR)
    print("FPR:", FPR)


print(dr)
print(fpr)
#





