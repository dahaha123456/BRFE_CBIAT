import pandas as pd
import numpy as np
from keras.layers import BatchNormalization, MaxPooling1D, Convolution1D, Bidirectional, LSTM, Reshape, Dropout
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from keras.layers import Dense, Flatten, Input, Conv1D
from keras.models import Model
import tensorflow as tf
from zs.log import Log
from zs.IG import information_gain
from sklearn.feature_selection import SelectKBest, chi2, RFE, SelectFromModel
from sklearn.metrics import classification_report
import os
from keras.layers import concatenate
from att.atten import Attention
from functools import partial
# os.environ['CUDA_VISIBLE_DEVICES'] = '1'
gpus = tf.config.list_physical_devices(device_type='GPU')
tf.config.set_visible_devices(devices=gpus[0], device_type='GPU')
# physical_devices = tf.config.list_physical_devices('GPU')
# tf.config.experimental.set_memory_growth(physical_devices[0], True)
# tf.compat.v1.disable_eager_execution()
df = pd.read_csv(r'D:\codework\CANET-main (1)\CANET-main\data\NSL-KDD\KDDTrain+.txt', header=None)
qp = pd.read_csv(r'D:\codework\CANET-main (1)\CANET-main\data\NSL-KDD\KDDTest+.txt', header=None)
df.columns = ['duration', 'protocol_type', 'service', 'flag', 'src_bytes',
'dst_bytes', 'land', 'wrong_fragment', 'urgent', 'hot',
'num_failed_logins', 'logged_in', 'num_compromised', 'root_shell',
'su_attempted', 'num_root', 'num_file_creations', 'num_shells',
'num_access_files', 'num_outbound_cmds', 'is_host_login',
'is_guest_login', 'count', 'srv_count', 'serror_rate',
'srv_serror_rate', 'rerror_rate', 'srv_rerror_rate', 'same_srv_rate',
'diff_srv_rate', 'srv_diff_host_rate', 'dst_host_count',
'dst_host_srv_count', 'dst_host_same_srv_rate','dst_host_diff_srv_rate', 'dst_host_same_src_port_rate',
'dst_host_srv_diff_host_rate', 'dst_host_serror_rate',
'dst_host_srv_serror_rate', 'dst_host_rerror_rate',
'dst_host_srv_rerror_rate', 'subclass', 'difficulty_level']
qp.columns = ['duration', 'protocol_type', 'service', 'flag', 'src_bytes',
'dst_bytes', 'land', 'wrong_fragment', 'urgent', 'hot',
'num_failed_logins', 'logged_in', 'num_compromised', 'root_shell',
'su_attempted', 'num_root', 'num_file_creations', 'num_shells',
'num_access_files', 'num_outbound_cmds', 'is_host_login',
'is_guest_login', 'count', 'srv_count', 'serror_rate',
'srv_serror_rate', 'rerror_rate', 'srv_rerror_rate', 'same_srv_rate',
'diff_srv_rate', 'srv_diff_host_rate', 'dst_host_count',
'dst_host_srv_count', 'dst_host_same_srv_rate','dst_host_diff_srv_rate', 'dst_host_same_src_port_rate',
'dst_host_srv_diff_host_rate', 'dst_host_serror_rate',
'dst_host_srv_serror_rate', 'dst_host_rerror_rate',
'dst_host_srv_rerror_rate', 'subclass', 'difficulty_level']
df = df.drop('difficulty_level', 1)  # we don't need it in this project
qp = qp.drop('difficulty_level', 1)
df.isnull().values.any()
qp.isnull().values.any()
cols = ['protocol_type', 'service', 'flag']


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


# Merging train and test data
combined_data = pd.concat([df, qp])
# Applying one hot encoding to combined data
combined_data = one_hot(combined_data, cols)


# Function to min-max normalize
def normalize(df, cols):
    """
    @param df pandas DataFrame
    @param cols a list of columns to encode
    @return a DataFrame with normalized specified features
    """
    result = df.copy()  # do not touch the original df
    for feature_name in cols:
        max_value = df[feature_name].max()
        min_value = df[feature_name].min()
        if max_value > min_value:
            result[feature_name] = (df[feature_name] - min_value) / (max_value - min_value)
    return result


# Dropping subclass column for training set
tmp = combined_data.pop('subclass')
new_train_df = normalize(combined_data, combined_data.columns)
# Fixing labels for training set
classlist = []
check1 = ("apache2","back","land","neptune","mailbomb","pod","processtable","smurf","teardrop","udpstorm","worm")
check2 = ("ipsweep","mscan","nmap","portsweep","saint","satan")
check3 = ("buffer_overflow","loadmodule","perl","ps","rootkit","sqlattack","xterm")
check4 = ("ftp_write","guess_passwd","httptunnel","imap","multihop","named","phf","sendmail","Snmpgetattack","spy","snmpguess","warezclient","warezmaster","xlock","xsnoop")

DoSCount = 0
ProbeCount = 0
U2RCount = 0
R2LCount = 0
NormalCount = 0

for item in tmp:
    if item in check1:
        classlist.append("DoS")
        DoSCount = DoSCount+1
    elif item in check2:
        classlist.append("Probe")
        ProbeCount = ProbeCount+1
    elif item in check3:
        classlist.append("U2R")
        U2RCount = U2RCount+1
    elif item in check4:
        classlist.append("R2L")
        R2LCount = R2LCount+1
    else:
        classlist.append("Normal")
        NormalCount = NormalCount + 1

new_train_df["Class"] = classlist

y_train = new_train_df["Class"]
# print(y_train.shape)
combined_data_X = new_train_df.drop('Class', 1)
# print(combined_data_X.shape)
oos_pred = []
dr = []
F1 = []
fpr = []
# selector = SelectKBest(chi2, k=32).fit(combined_data_X, y_train)
# 将原始特征矩阵转换为只包含选中特征的子集
# selected_features = selector.transform(combined_data_X)
####
# 使用RFE进行特征选择
import sys
# sys.setrecursionlimit(2000)
rf = RandomForestClassifier(n_estimators=100)
rfe = RFE(estimator=rf, n_features_to_select=32)
selector = rf.fit(combined_data_X, y_train)
# features = selector.feature_importances_
selected_features = SelectFromModel(selector, threshold=0.01)
# selected_features.fit(combined_data_X, y_train)
# selected_features = selected_features.transform(combined_data_X)
# ######
# 获取选中的特征名称
selected_feature_names = [new_train_df.columns[i] for i in selected_features]

# 创建DataFrame，将特征矩阵转换为DataFrame，并设置列名为选中特征名称
combined_data_X= pd.DataFrame(selected_features, columns=selected_feature_names)

# combined_data_X = pd.DataFrame(combined_data_X)
combined_data_X
from imblearn.over_sampling import SMOTE, ADASYN
from sklearn.preprocessing import LabelEncoder

combined_data_X, y_train = ADASYN(sampling_strategy='minority').fit_resample(combined_data_X, y_train)
# combined_test_X, yy_test = ADASYN(sampling_strategy='minority').fit_resample(combined_data_test, y_test)
# combined_data_X, y_train = oversample.fit_resample(combined_data_X, y_train)


kfold = StratifiedKFold(n_splits=6, shuffle=True, random_state=42)
kfold.get_n_splits(combined_data_X, y_train)
num_classes=5
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

for train_index, test_index in kfold.split(combined_data_X,y_train):
    train_X, test_X = combined_data_X.iloc[train_index], combined_data_X.iloc[test_index]
    train_y, test_y = y_train.iloc[train_index], y_train.iloc[test_index]
    print(train_X)
    # x_columns_train = selected_feature_names.columns

    x_train_array = train_X[selected_feature_names].values
    x_train_1=np.reshape(x_train_array, (x_train_array.shape[0], x_train_array.shape[1], 1))
    # print(x_train_1.shape)
    dummies = pd.get_dummies(train_y) # Classification
    outcomes = dummies.columns
    num_classes = len(outcomes)
    y_train_1 = dummies.values

    # x_columns_test = selected_feature_names.columns
    x_test_array = test_X[selected_feature_names].values
    x_test_2=np.reshape(x_test_array, (x_test_array.shape[0], x_test_array.shape[1], 1))

    dummies_test = pd.get_dummies(test_y) # Classification
    outcomes_test = dummies_test.columns
    num_classes = len(outcomes_test)
    y_test_2 = dummies_test.values


    model.fit(x_train_1, y_train_1,validation_split=0.1, epochs=100, batch_size=256)
    target_names = ['Dos', 'Normal', 'Probe', 'R2L', 'U2R']
    pred = model.predict(x_test_2)
    pred = np.argmax(pred, axis=1)
    y_eval = np.argmax(y_test_2,axis=1)
    accuracy = metrics.accuracy_score(y_eval, pred)
    recall = metrics.recall_score(y_eval, pred, average='macro')
    precision = metrics.precision_score(y_eval, pred, average='macro')
    f1score = metrics.f1_score(y_eval, pred, average='macro')
    classification = metrics.classification_report(y_eval, pred, target_names=target_names)
    cm = metrics.confusion_matrix(y_eval, pred)
    mcc=metrics.matthews_corrcoef(y_eval, pred)
    kappa=metrics.cohen_kappa_score(y_eval, pred)
    tp = cm[0][0]+cm[2][2]+cm[3][3]+cm[4][4]
    fn = cm[0][1]+cm[2][1]+cm[3][1]+cm[4][1]
    fp = cm[1][0]+cm[1][2]+cm[1][3]+cm[1][4]
    tn = cm[1][1]
    DR = tp/(tp+fn)
    FPR = fp/(fp+tn)
    Log.console("===========NSL-Rfeval===========")
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


print(oos_pred)
print(dr)
print(fpr)
print(F1)
