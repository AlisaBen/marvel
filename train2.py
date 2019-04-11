from config.config import *
from model.xDeepFM import xDeepFM
from model.DeepFM import DeepFM
from model.merge_model import MergeModel
from mxnet import init
from util.util import *
import pandas as pd
from model.process_data2 import *
from mxnet.gluon import loss as gloss, nn, data as gdata, Trainer
import time

# pd.DataFrame()
def train(args,feature_dict,feature_values,label,ctx,validation_feature,validation_label):
    if args.MODEL == 'deepfm':
        model = DeepFM(feature_dict,args, ctx, args.TASK)
    elif args.MODEL == 'xdeepfm':
        model = xDeepFM(feature_dict,args, ctx, args.TASK)
    else:
        model = MergeModel(feature_dict,args, ctx, args.TASK)
    if args.TASK == 'finish':
        if args.FINISH_MODEL_PATH is not None:
            model.initialize(init=init.Xavier(), ctx=ctx)
            model.load_params(args.FINISH_MODEL_PATH)
        else:
            model.initialize(init=init.Xavier(),ctx=ctx)
    else:
        if args.LIKE_MODEL_PATH is not None:
            model.initialize(init=init.Xavier(), ctx=ctx)
            model.load_params(args.LIKE_MODEL_PATH)
        else:
            model.initialize(init=init.Xavier(),ctx=ctx)
    # print(model.collect_params())
    # train_iter = gdata.DataLoader(gdata.ArrayDataset(feature_values, label),
    #                               batch_size=args.BATCH_SIZE,shuffle=True)  # pd.read_csv去掉name
    if args.TASK == 'finish':
        lr = args.FINISH_LEARNING_RATE
    else :
        lr = args.LIKE_LEARNING_RATE
    if args.OPTIMIZER == 'adam':
        model_trainer = Trainer(model.collect_params(), args.OPTIMIZER,
                               {'learning_rate': lr, 'wd': args.WEIGHT_DECAY})
    else:
        model_trainer = Trainer(model.collect_params(), args.OPTIMIZER,
                               {'learning_rate': lr})
    if args.TASK == 'finish':
        epochs = args.FINISH_NUM_EPOCHS
        batch = args.FINISH_BATCH_SIZE

    else:
        epochs = args.LIKE_NUM_EPOCHS
        batch = args.LIKE_BATCH_SIZE

    for epoch in range(epochs):
        train_iter = gdata.DataLoader(gdata.ArrayDataset(feature_values, label),
                                      batch_size=batch, shuffle=True)  # pd.read_csv去掉name
        time_start = time.time()
        train_epoch_loss,train_acc = model.train_epoch(epoch, train_iter,model_trainer)

        test_iter = gdata.DataLoader(gdata.ArrayDataset(validation_feature,validation_label),batch_size=batch,shuffle=False)
        epoch_loss,epoch_test_acc = model.eval_model(test_iter)
        # epoch_test_acc = evaluate_accuracy(test_iter,model)
        # epoch_loss = 0.0
        train_num = len(feature_values)
        # print(train_num)
        print('\n[%s] net_name:[%s] ,EPOCH FINISH [%d],,time_cost [%d]s,average_loss:[%f] eval_model_loss:[%f],train_acc:[%f],test_acc:[%f]' %
              (time.strftime("%Y-%m-%d %H:%M:%S"), model.task, epoch+1, np.int(time.time() - time_start),
               train_epoch_loss / train_num,
               epoch_loss/len(validation_feature),
               train_acc,
               epoch_test_acc
               # epoch_loss
               ))
        if epoch % 1 == 0:
            filename = args.SAVE_PARAMS_PATH_PREFIX + '/net_'+args.MODEL + '_' + model.task + '_' + args.CONFIG_NAME + '_' + time.strftime("%Y%m%d_%H%M%S") + '.model'
            model.save_params(filename)
    return model


# def evaluate_accuracy(self,data_iter, net):
#     acc = 0
#     for X, y in data_iter:
#         acc += self.accuracy(net(X), y)
#     return acc / len(data_iter)


def predict(args,model,test_feature,result_tmp):
    print('predict')
    res = model.predict(test_feature)

    # result_tmp['finish_probability'] = finish_res
    # result_tmp['like_probability'] = like_res
    if args.TASK == 'finish':
        result_tmp['finish_probability'] = res
        result_tmp.to_csv(args.FINISH_SAVE_PATH, index=False)
        if os.path.exists(args.LIKE_SAVE_PATH):
            merge(args.LIKE_SAVE_PATH,args.FINISH_SAVE_PATH,args.SUBMISSION_PATH)
    else:
        result_tmp['like_probability'] = res
        result_tmp.to_csv(args.LIKE_SAVE_PATH, index=False)
        if os.path.exists(args.FINISH_SAVE_PATH):
            merge(args.LIKE_SAVE_PATH,args.FINISH_SAVE_PATH,args.SUBMISSION_PATH)

    # if os.path.exists(args.SAVE_PATH1):
    #     result_tmp = pd.read_csv(args.SAVE_PATH1)
    #     if args.TASK == 'finish':
    #         result_tmp['like_probability'] = res
    #     else:
    #         result_tmp['finish_probability'] = res
    #     result_tmp.to_csv(args.SUBMISSION_PATH,index=False)
    # else:
    #     if args.TASK == 'finish':
    #         result_tmp['like_probability'] = res
    #     else:
    #         result_tmp['finish_probability'] = res
    #     result_tmp.to_csv(args.SAVE_PATH,index=False)
    # print('save success')


def accuracy(y_hat, y):
    return (nd.argmax(y_hat, axis=1) == y).asnumpy().mean()

def evaluate_accuracy(data_iter, net):
    acc = 0
    for X, y in data_iter:
        acc += accuracy(net(X), y)
    return acc / len(data_iter)

def run(args,test=False):
    # args = Args
    print('reading args over')
    time_start = time.time()

    train_feature, train_like_label, train_finish_label, test_feature, test_like_label, test_finish_label, result_tmp,\
    feature_dict = process_data(args,args.TRAINING_PATH,args.TEST_PATH,test)
    # print(train_feature)
    # print(train_feature)
    # print('-----------------------------------')
    # print(train_like_label)
    # print(train_finish_label)
    train_size = len(train_feature)
    train_size = int(train_size * (1 - args.VALIDATION_RATE))
    validation_feature = train_feature[train_size:]
    train_feature = train_feature[:train_size]

    validation_finish_label = train_finish_label[train_size:]
    train_finish_label = train_finish_label[:train_size]

    validation_like_label = train_like_label[train_size:]
    train_like_label = train_like_label[:train_size]

    # print(train_data)
    ctx = try_gpu(args.GPU_INDEX)
    print('process data over,time_cost:[%d]s' % np.int(time.time() - time_start))
    # args.FEATURE_SIZE = len(train_data['feature_values'][0])
    args.FIELD_NUM = len(train_feature[0])
    # print(train_data.shape)
    # print(type(train_data['feature_values']))
    # print(len(train_data['feature_values']))
    print('feature num:')
    print(args.FIELD_NUM)
    # test_feature = test_data['feature_values']
    print('test_num:')
    print(len(test_feature))
    # feature_values = nd.array(train_data['feature_values'].tolist(), ctx=ctx)
    # feature_values = train_data['feature_values']
    time_start = time.time()

    if args.TASK == 'like':
        model = train(args,feature_dict,train_feature,train_like_label,ctx,validation_feature,validation_like_label)
    else:
        model = train(args,feature_dict,train_feature, train_finish_label, ctx,validation_feature,validation_finish_label)
    print('train_time_cost:[%d]s' % np.int(time.time() - time_start))
    time_start = time.time()

    predict(args,model,test_feature,result_tmp)
    print('predict_time_cost:[%d]s' % np.int(time.time() - time_start))


if __name__ == '__main__':
    MXNET_CUDNN_AUTOTUNE_DEFAULT = 1
    np.set_printoptions(threshold=np.inf)
    args = Args
    run(args,False)
    # merge(args.LIKE_SAVE_PATH,args.FINISH_MODEL_PATH,args.SUBMISSION_PATH)



