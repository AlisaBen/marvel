import pandas as pd
from config.config import *
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from model.SingleFeat import SingleFeat
# @staticmethod


def process_data(args,train_path, test_path,is_test = False):
    if is_test:
        train_data_all = pd.read_csv(train_path)
        test_data_all = pd.read_csv(test_path)
    else:
        train_data_all = pd.read_csv(train_path,sep='\t', names=[
        'uid', 'user_city', 'item_id', 'author_id', 'item_city', 'channel', 'finish', 'like', 'music_id', 'did', 'creat_time', 'video_duration'])
        test_data_all = pd.read_csv(test_path,sep='\t', names=[
        'uid', 'user_city', 'item_id', 'author_id', 'item_city', 'channel', 'finish', 'like', 'music_id', 'did', 'creat_time', 'video_duration'])
    # print(train_data_all.info())
    # print(test_data_all.info())
    data = pd.concat([train_data_all,test_data_all],axis=0)
    # data = data.astype('float32')
    # print(data.info())

    train_num = train_data_all.shape[0]

    # sparse_feature_map = [FeatureToIndexMap('uid',0),FeatureToIndexMap('user_city',1),FeatureToIndexMap]
    if args.TASK == 'like':
        sparse_features = ['uid', 'user_city', 'item_id', 'author_id', 'item_city', 'channel',
                       'music_id', 'did', ]
        dense_features = ['video_duration']  # 'creat_time',
    else:
        sparse_features = ['uid', 'user_city', 'item_id', 'author_id', 'item_city', 'channel',
                       'music_id', 'did', ]
        dense_features = ['video_duration']  # 'creat_time',
    # sparse_indexs = [2,3,4,5,6,7,8,9]
    # dense_indexs = [11]

    target = ['finish', 'like']

    like_label = data[target[1]]
    finish_label = data[target[0]]

    data = data[sparse_features + dense_features]

    for feat in sparse_features:
        lbe = LabelEncoder()
        data[feat] = lbe.fit_transform(data[feat]) 
    mms = MinMaxScaler(feature_range=(0, 1))
    data[dense_features] = mms.fit_transform(data[dense_features])

    sparse_index_list = [SingleFeat(feat, data.iloc[:, feat].nunique()) for feat in range(len(sparse_features))]
    dense_index_list = [SingleFeat(feat + len(sparse_features), 0) for feat in range(len(dense_features))]

    feature_dict = {'sparse':sparse_index_list,'dense':dense_index_list}
    # print(data.info())
    data[sparse_features + dense_features] = data[sparse_features + dense_features].astype('float32')
    # print(data.info())
    # print(type(like_label))
    # pd.to_numeric(like_label, errors='coerce')
    # pd.to_numeric(finish_label, errors='coerce')

    like_label = like_label.astype('float32')
    finish_label = finish_label.astype('float32')
    # print(like_label.dtype)

    train_feature = data.iloc[:train_num].values
    test_feature = data.iloc[train_num:].values
    train_like_label = like_label.iloc[:train_num].values
    test_like_label = like_label.iloc[train_num:].values
    train_finish_label = finish_label.iloc[:train_num].values
    test_finish_label = finish_label.iloc[train_num:].values

    # result_tmp = test_data_all[['uid', 'item_id']].copy()

    return train_feature,train_like_label,train_finish_label,\
           test_feature,test_like_label,test_finish_label,test_data_all[['uid', 'item_id']],feature_dict

    # return train_data, test_data, result_tmp



def load_data(process_train_data_path, process_test_data_path, test_path):
    print('load data')
    train_features = pd.read_csv(process_train_data_path)
    test_features = pd.read_csv(process_test_data_path)
    train_data = {}
    test_data = {}
    train_data['label_like'] = train_features['label_like'].values.tolist()
    train_data['label_finish'] = train_features['label_finish'].values.tolist()
    train_data['feature_values'] = train_features['feature_values'].values.tolist()
    # test_data['feature_values']
    # print(train_features.shape)
    # print(train_features.iloc[:, :train_features.shape[1]-2].shape)
    # train_data = pd.DataFrame()
    # train_data['label_like'] = train_features['label_like']
    # train_data['label_finish'] = train_features['label_finish']
    # # print('------------------------------------------------')
    # # print(train_features.iloc[:, train_features.shape[1]-2].values.tolist())
    # # print(type(train_features.iloc[:, :train_features.shape[1]-2].values.tolist()))
    # # print(len(train_features.iloc[:, :train_features.shape[1]-2].values.tolist()[0]))
    # # print(train_features.columns)
    # # print(test_features.columns)
    # # print(len(train_features['feature_values'].values.tolist()[0]))
    # train_data['feature_values'] = train_features.iloc[:, :train_features.shape[1]-2].values.tolist()
    # test_data = pd.DataFrame()
    # test_data['feature_values'] = test_features.iloc[:, :test_features.shape[1]-2].values.tolist()

    result_tmp = pd.read_csv(test_path)
    result_tmp = result_tmp[['uid', 'item_id']]
    # return train_data, test_data, result_tmp
    return train_features,test_features,result_tmp


# if __name__ == '__main__':
#     args = Args
    # train_data, test_data, result_tmp = process_data(args.TRAINING_PATH,args.TEST_PATH,args.SPARSE_SPLIT,
    #                                                  args.PROCESS_DATA_PATH[0],args.PROCESS_DATA_PATH[1])
    #
    # train_data, test_data, result_tmp = load_data(args.PROCESS_DATA_PATH[0], args.PROCESS_DATA_PATH[1], args.TEST_PATH)
    # print(len(train_data['feature_values'][0]))
    # load_data(args.TRAINING_PATH,args.TEST_PATH,args.SPARSE_SPLIT)
