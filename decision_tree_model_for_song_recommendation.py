import pandas as pd
import numpy as np
import gc
import lightgbm as lgb
from sklearn.model_selection import (train_test_split, GridSearchCV)
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm

# read the first 99 rows for demo
source = "https://s3.amazonaws.com/hackerday.datascience/102/"
train = pd.read_csv(source+'train.csv', nrows=99)
test = pd.read_csv(source+'test.csv', nrows=99)
songs = pd.read_csv(source+'songs.csv')
members = pd.read_csv(source+'members.csv')

#merge datasets with song attribute
song_col = ['song_id', 'artist_name', 'genre_ids' ,'song_length', 'language']
train = train.merge(songs[song_col], on='song_id', how = 'left')
test = test.merge(songs[song_col], on='song_id', how = 'left')

# merge datasets with member feature
members['registration_year'] = members['registration_init_time'].apply(lambda x: int(str(x)[0:4]))
members['registration_month'] = members['registration_init_time'].apply(lambda x: int(str(x)[4:6]))
members['registration_date'] = members['registration_init_time'].apply(lambda x: int(str(x)[6:8]))

members['expiration_year'] = members['expiration_date'].apply(lambda x: int(str(x)[0:4]))
members['expiration_month'] = members['expiration_date'].apply(lambda x: int(str(x)[4:6]))
members['expiration_date'] = members['expiration_date'].apply(lambda x: int(str(x)[6:8]))
members = members.drop(['registration_init_time'], axis = 1)

member_col = members.columns
train = train.merge(members[member_col], on='msno', how = 'left')
test = test.merge(members[member_col], on='msno', how = 'left')

train = train.fillna(-1)
test = test.fillna(-1)

del members, songs; gc.collect(); # deleting members,songs and gc.collect which is collected as numerical values

cols = list(train.columns)
cols.remove('target')

for col in tqdm(cols):
    if train[col].dtype == 'object':    # converting columns to string 
        train[col] = train[col].apply(str)
        test[col] = test[col].apply(str)
        
        le = LabelEncoder()
        train_vals = list(train[col].unique())
        test_vals = list(test[col].unique())
        le.fit(train_vals + test_vals)
        # tranforming data into 0s and 1s
        train[col] = le.transform(train[col])
        test[col] = le.transform(test[col])
        
x = np.array(train.drop(['target'], axis=1)) # creating array of numerical values using Gbdt 
y = train['target'].values
 
# choose reasonable parameters for improving the accuracy of the model
params= {
        'application' : 'binary', # for binary classification
        'num_class' : 1,
        'boosting' : 'gbdt', # gradient boosting decision tree(gbdt)
        'num_iteration' : 100,
        'learning_rate' : 0.05,
        'num_leaves' : 62,
        'device' : 'cpu',
        'max_depth' : -1, # <0 means no limit
        'max_bin' : 510, # small no of bins may reduce training accuracy but can deal with over-fitting
        'lambda_l1' : 5, # l1 regularization
        'lambda_l2' : 10, # l2 regularization
        'metric' : 'binary_error',
        'subsample_for_bin' : 200, # number of samples for constructing bins
        'subsample' : 1, # subsample ratio for training instance
        'colsample_bytree' : 0.8, # subsample ratio of columns when constructing the tree
        'min_split_gain' : 0.5, # min loss reduction required for further partition on a leaf node of the tree
        'min_child_weight' : 1, # min sun of instanceweight(hessian) needed in a leaf
        'min_child_samples' : 5 # min num of data needed in a leaf
            }
# initiate classifier to use
mdl = lgb.LGBMClassifier(boosting_type = 'gbdt',
                         objective = 'binary',
                         n_jobs = 5,
                         silent = True,
                         max_depth = params['max_depth'],
                         max_bin = params['max_bin'],
                         subsample_for_bin = params['subsample_for_bin'],
                         subsample = params['subsample'],
                         min_split_gain = params['min_split_gain'],
                         min_child_weight = params['min_child_weight'],
                         min_child_samples = params['min_child_samples'])
# to view the default model parameters
mdl.get_params().keys()

# gridsearch parameters
gridParams = {
        'learning_rate' : [0.005,0.01],
        'n_estimators' : [8,16,24],
        'num_leaves' : [6,8,12,16], # larger number helps accuracy but tends to over-fitting
        'boosting_type' : ['gbdt','dart'],
        'objective' : ['binary'],
        'max_bin' : [255,510],
        'random_state' : [500],
        'colsample_bytree' : [0.64,0.65,0.66],
        'subsample' : [0.7,0.75],
        'reg_alpha' : [1,1.2],
        'reg_lambda' : [1,1.2,1.4],
        }
grid = GridSearchCV(mdl,gridParams, verbose = 1,cv = 4, n_jobs =-1)  
# run the grid
grid.fit(x,y)

# print the best parameters found
print(grid.best_params_)
print(grid.best_score_)     

# building the model using best grid data

params['colsample_bytree'] = grid.best_params_['colsample_bytree']
params['learning_rate'] = grid.best_params_['learning_rate']
params['max_bin'] = grid.best_params_['max_bin']
params['num_leaves'] = grid.best_params_['num_leaves']
params['reg_alpha'] = grid.best_params_['reg_alpha']
params['reg_lambda'] = grid.best_params_['reg_lambda']
params['subsample'] = grid.best_params_['subsample']

x_test = np.array(test.drop(['id'],axis = 1))
ids = test['id'].values

x_train,x_valid,y_train,y_valid = train_test_split(x,y,test_size=0.1,random_state =12)

del x, y; gc.collect();
    
d_train = lgb.Dataset(x_train, label = y_train)
d_valid = lgb.Dataset(x_valid, label = y_valid)

watchlist = [d_train, d_train]
model = lgb.train(params, train_set = d_train, num_boost_round = 1000, valid_sets = watchlist, early_stopping_rounds = 50, verbose_eval = 4)
p_test = model.predict(x_test)
