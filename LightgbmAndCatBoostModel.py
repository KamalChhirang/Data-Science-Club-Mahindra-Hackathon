import numpy as np 
import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import lightgbm as lgb
from catboost import CatBoostRegressor

def featureModify(isTrain):
    rowstoread = None
    if isTrain:
        train = pd.read_csv('../input/train.csv',nrows=rowstoread)
        test = pd.read_csv('../input/test.csv',nrows=rowstoread)
        test['amount_spent_per_room_night_scaled'] = 0

        df = pd.concat([train,test], axis=0)
        print(df.shape)
    else:
        df = pd.read_csv('../input/test.csv',nrows=rowstoread)
    df['booking_date_date'] = df['booking_date'].str.slice(start=0, stop=2).astype('int')
    df['booking_date_month'] = df['booking_date'].str.slice(start=3, stop=5).astype('int')
    df['booking_date_year'] = df['booking_date'].str.slice(start=6, stop=8).astype('int')

    
    df['checkin_date_date'] = df['checkin_date'].str.slice(start=0, stop=2).astype('int')
    df['checkin_date_month'] = df['checkin_date'].str.slice(start=3, stop=5).astype('int')
    df['checkin_date_year'] = df['checkin_date'].str.slice(start=6, stop=8).astype('int')
    
    df['checkout_date_date'] = df['checkout_date'].str.slice(start=0, stop=2).astype('int')
    df['checkout_date_month'] = df['checkout_date'].str.slice(start=3, stop=5).astype('int')
    df['checkout_date_year'] = df['checkout_date'].str.slice(start=6, stop=8).astype('int')

    df['booking_date'] = pd.to_datetime("20" + df['booking_date_year'].astype('str') + "-" + df['booking_date_month'].astype('str') + "-" + df['booking_date_date'].astype('str'))
    df['checkin_date'] = pd.to_datetime("20" + df['checkin_date_year'].astype('str') + "-" + df['checkin_date_month'].astype('str') + "-" + df['checkin_date_date'].astype('str'))
    df['checkout_date'] = pd.to_datetime("20" + df['checkout_date_year'].astype('str') + "-" + df['checkout_date_month'].astype('str') + "-" + df['checkout_date_date'].astype('str'))

    df['booking_dayofweek'] = df['booking_date'].dt.dayofweek
    df['checkin_dayofweek'] = df['checkin_date'].dt.dayofweek
    df['checkout_dayofweek'] = df['checkout_date'].dt.dayofweek


    #df['stayOnDiwali17'] = 0
    #df.loc[(df['checkout_date_year']==17) & (df['checkout_date_month']==10 | df['checkin_date_month']==10) & (df['checkout_date_date']<=22) & (df['checkin_date_date']>=16)  ,'stayOnDiwali17'] = 1
    #print(df['stayOnDiwali17'].value_counts())

    df['stayOnXMas'] = 0
    df.loc[ (df['checkout_date_month']==12) & (df['checkout_date_date']<=28) & (df['checkin_date_date']>=22)  ,'stayOnXMas'] = 1

    df['Diwali'] = 0
    df.loc[ (df['checkin_date'] <= "2017-10-19") &  ("2017-10-19" <= df['checkout_date'])  ,'Diwali'] = 1
    df.loc[ (df['checkin_date'] <= "2018-11-07") &  ("2018-11-07" <= df['checkout_date'])  ,'Diwali'] = 1
    df.loc[ (df['checkin_date'] <= "2016-10-30") &  ("2016-10-30" <= df['checkout_date'])  ,'Diwali'] = 1
    df.loc[ (df['checkin_date'] <= "2015-11-11") &  ("2015-11-11" <= df['checkout_date'])  ,'Diwali'] = 1
    df.loc[ (df['checkin_date'] <= "2014-10-23") &  ("2014-10-23" <= df['checkout_date'])  ,'Diwali'] = 1
    print(df['Diwali'].value_counts())



    df['booking_date_checkin_diff'] = (df['checkin_date'] - df['booking_date']).dt.days
    df['checkin_date_checkout_date_diff'] = (df['checkout_date'] - df['checkin_date']).dt.days
    df['booking_date_checkout_date_diff'] = (df['checkout_date'] - df['booking_date']).dt.days
    df['numberOfNightsBooked_diff_ActualStay'] = df['roomnights'] - df['checkin_date_checkout_date_diff']

    df['member_age_buckets'] = df['member_age_buckets'].astype('category')
    df['member_age_buckets'] = df['member_age_buckets'].cat.codes
    
    df['cluster_code'] = df['cluster_code'].astype('category')
    df['cluster_code'] = df['cluster_code'].cat.codes
    
    df['reservationstatusid_code'] = df['reservationstatusid_code'].astype('category')
    df['reservationstatusid_code'] = df['reservationstatusid_code'].cat.codes


    df['resort_id'] = df['resort_id'].astype('category')
    df['resort_id'] = df['resort_id'].cat.codes
    
    
    df['totalMembers_diff_AdultPlusChild'] = (df['numberofadults'] + df['numberofchildren'])-df['total_pax']
    
    numberOfTimeAMemeberVisited = df.groupby(['memberid']).size().reset_index(name='numberOfTimeAMemeberVisited')
    df = pd.merge(df, numberOfTimeAMemeberVisited, how='left', on=['memberid'])

    groupByMemberId = df.groupby(['memberid'])['checkin_date_checkout_date_diff'].agg('mean')
    df = pd.merge(df, groupByMemberId.reset_index(), suffixes=["", "_OfAllTime"], how='left', on=['memberid'])

    groupByMemberId = df.groupby(['memberid'])['roomnights'].agg('mean')
    df = pd.merge(df, groupByMemberId.reset_index(), suffixes=["", "_OfAllTime"], how='left', on=['memberid'])

    groupByMemberId = df.groupby(['memberid'])['total_pax'].agg('mean')
    df = pd.merge(df, groupByMemberId.reset_index(), suffixes=["", "_OfAllTime"], how='left', on=['memberid'])

    groupByMemberId = df.groupby(['memberid'])['numberofadults'].agg('mean')
    df = pd.merge(df, groupByMemberId.reset_index(), suffixes=["", "_OfAllTime"], how='left', on=['memberid'])
    
    groupByMemberId = df.groupby(['memberid'])['roomnights'].agg('sum')
    df = pd.merge(df, groupByMemberId.reset_index(), suffixes=["", "_OfAllTime_Sum"], how='left', on=['memberid'])

    groupByMemberId = df.groupby(['memberid'])['totalMembers_diff_AdultPlusChild'].agg('mean')
    df = pd.merge(df, groupByMemberId.reset_index(), suffixes=["", "_OfAllTime"], how='left', on=['memberid'])
    
    groupByMemberId = df.groupby(['memberid'])['booking_date_checkin_diff'].agg('mean')
    df = pd.merge(df, groupByMemberId.reset_index(), suffixes=["", "_OfAllTime"], how='left', on=['memberid'])


    groupByMemberId = df.groupby(['resort_id'])['numberOfTimeAMemeberVisited'].agg('mean')
    df = pd.merge(df, groupByMemberId.reset_index(), suffixes=["", "_OfAllTime"], how='left', on=['resort_id'])

    
    df = df.drop(['reservation_id','booking_date','checkin_date','checkout_date','memberid'],axis=1)

    return df

#train = featureModify(True)
#test = featureModify(False)
df = featureModify(True)

cat_fe = ['booking_date_date','booking_date_month','booking_date_year','checkin_date_date','checkin_date_month'
    ,'checkin_date_year','checkout_date_date','checkout_date_month','checkout_date_year','channel_code','main_product_code'
    ,'persontravellingid','resort_region_code','resort_type_code','room_type_booked_code','season_holidayed_code'
    ,'state_code_residence','state_code_resort','member_age_buckets','booking_type_code','cluster_code','resort_id'
    ,'booking_dayofweek','checkin_dayofweek','checkout_dayofweek']

df = pd.get_dummies(df, columns=cat_fe)



train = df.iloc[0:341424,:]
test = df.iloc[341424:,:]

y = train['amount_spent_per_room_night_scaled']
train = train.drop(['amount_spent_per_room_night_scaled'],axis=1)
test = test.drop(['amount_spent_per_room_night_scaled'],axis=1)



params = {
            "objective" : "regression", 
            "metric" : "rmse", 
            "num_leaves" : 31, 
            "learning_rate" : 0.02, 
            "bagging_fraction" : 1.0,
            "bagging_seed" : 3, 
            "num_threads" : 4,
            'min_data_in_leaf':20, 
            'min_split_gain':0.0,
            'lambda_l2':0
    }
    

trainforsplit = train
yforsplit = y
y_pred = np.zeros(test.shape[0] )
totalCVError = 0
totalFolds = 10
for i in range(totalFolds):
    train, train_test, y, y_test = train_test_split(trainforsplit, yforsplit, test_size=0.2, shuffle=True,random_state=i)
    
    train_set = lgb.Dataset(train, label=y)#, categorical_feature=cat_fe)
    valid_set = lgb.Dataset(train_test, label=y_test)#, categorical_feature=cat_fe)
    
    model = lgb.train(  params, 
                        train_set = train_set,
                        num_boost_round=10000,
                        early_stopping_rounds=200,
                        verbose_eval=200, 
                        valid_sets=[train_set,valid_set]
                      )
    
    y_pred+=model.predict(test, num_iteration=model.best_iteration)
    cv_pred = model.predict(train_test, num_iteration=model.best_iteration)
    thisCVError = mean_squared_error(y_test, cv_pred)
    totalCVError += thisCVError
    print('Fold', i, ' Error :', thisCVError)
  
totalCVError /= totalFolds
print('LightGBMAverage CV Error: ', totalCVError)

lightGBMPred = y_pred/totalFolds


#================================================
#================================================
#================================================
#================================================
#================================================
#================================================
#================================================
#================================================
#================================================
#CATBoost

for i in range(totalFolds):
    train, train_test, y, y_test = train_test_split(trainforsplit, yforsplit, test_size=0.2, shuffle=True,random_state=i)

    model = CatBoostRegressor(iterations=100000,
                                 learning_rate=0.05,
                                 depth=7,
                                 eval_metric='RMSE',
                                 colsample_bylevel=1,
                                 random_seed = 42,
                                 bagging_temperature = 0.8,
                                 metric_period = None,
                                 early_stopping_rounds=200
                                )
    model.fit(train, y,
                 eval_set=(train_test, y_test),
                 use_best_model=True,
                 verbose=100)
    
    cv_pred = model.predict(train_test)
    y_pred += model.predict(test)

    thisCVError = mean_squared_error(y_test, cv_pred)
    totalCVError += thisCVError
    print('Fold', i, ' Error :', thisCVError)
    
totalCVError /= totalFolds
print('Catboost Average CV Error: ', totalCVError)
y_pred = y_pred/totalFolds
y_pred = (y_pred+lightGBMPred)/2

df_sub = pd.DataFrame()
df_test = pd.read_csv('../input/test.csv',nrows=None)
print(df_test.shape)
df_sub['reservation_id'] = df_test['reservation_id']
df_sub['amount_spent_per_room_night_scaled'] = y_pred

df_sub.to_csv("catboost.csv", index=False)