import mxnet as mx
from mxnet import nd, autograd,init,initializer,metric
from mxnet.gluon import loss as gloss, nn, data as gdata, Trainer
import time
import numpy as np
from util import util
import logging
from model.SingleFeat import SingleFeat
from collections import OrderedDict
import os
from util.util import *

class MergeModel(nn.Block):

    def __init__(self, feature_dict, args, ctx, task,**kwargs):
        """{"sparse":[SingleFeat],"dense":[SingleFeat]}"""
        super(MergeModel, self).__init__(**kwargs)  # ??
        util.mkdir_if_not_exist(args.SAVE_PARAMS_PATH_PREFIX)
        # self.feature_sizes = args.FEATURE_SIZE
        self.field_size = args.FIELD_NUM
        self.feature_dict = feature_dict
        print('field_size:')
        print(self.field_size)
        if args.TASK == 'finish':
            self.embedding_size = args.FINISH_EMBEDDING_SIZE
            self.batch_size = args.FINISH_BATCH_SIZE
        else:
            self.embedding_size = args.LIKE_EMBEDDING_SIZE
            self.batch_size = args.LIKE_BATCH_SIZE
        self.config_name = args.CONFIG_NAME
        # self.dropout_prob = args.DROPOUT_PROB
        self.task = task

        # self.loss = gloss.SigmoidBinaryCrossEntropyLoss()
        if args.LOSS == 'l2loss':
            self.loss = gloss.L2Loss()
        else:
            self.loss = gloss.SigmoidBinaryCrossEntropyLoss()
        self.ctx = ctx
        self.embedding_dict = OrderedDict()
        self.dense_dict = OrderedDict()
        with self.name_scope():
            if self.task == 'finish':
                self.layer_list = [np.int(x) for x in args.FINISH_LAYER]
                self.dropout = args.FINISH_DROPOUT_PROB
            else:
                self.layer_list = [np.int(x) for x in args.LIKE_LAYER]
                self.dropout = args.LIKE_DROPOUT_PROB
            # self.params.get('v',shape=(self.field_size,self.embedding_size))
            self.dnn_out = nn.Dense(1,use_bias=False)

            self.register_child(self.dnn_out)

            for feat in feature_dict['sparse']:
                self.embedding_dict[feat.feat_name] = nn.Embedding(feat.feat_num, self.embedding_size)

            for feat in feature_dict['dense']:
                self.dense_dict[feat.feat_name] = nn.Dense(self.embedding_size)

            for emb_k, emb_v in self.embedding_dict.items():
                self.register_child(emb_v)
            for den_k, den_v in self.dense_dict.items():
                self.register_child(den_v)
            self.linear_logit_dense = nn.Dense(1,use_bias=False)
            self.register_child(self.linear_logit_dense)
            self.linear_logit_embedding_bn = nn.BatchNorm()
            self.register_child(self.linear_logit_embedding_bn)
            self.bn_embedding = nn.BatchNorm()
            self.register_child(self.bn_embedding)
            self.dense_list = []
            self.dropout_list = []
            self.bn_list = []
            self.activation_list = []
            for i in range(len(self.layer_list)):
                self.dense_list.append(nn.Dense(self.layer_list[i]))
                self.dropout_list.append(nn.Dropout(self.dropout))
                self.bn_list.append(nn.BatchNorm())
                self.activation_list.append(nn.Activation('relu'))
                self.register_child(self.dense_list[i])
                self.register_child(self.dropout_list[i])
                self.register_child(self.bn_list[i])
                self.register_child(self.activation_list[i])
            # if True:
            print('true')
            self.layer_size = [np.int(x) for x in args.CONV1D_LAYER]
            # self.cin_net = CIN(self.embedding_size,self.field_size, (128, 64), self.ctx)
            # print('oo')
            # self.cin_net.initialize()
            # print('uu')
            # self.register_child(self.cin_net)

            self.cin_dense = nn.Dense(1)
            self.register_child(self.cin_dense)
            self.cin_bn = nn.BatchNorm()
            self.register_child(self.cin_bn)

            self.field_nums = [self.field_size]
            self.conv_list = []
            for idx, size in enumerate(self.layer_size):
                self.conv_list.append(nn.Conv1D(channels=size, kernel_size=1, strides=1, padding=0, activation='relu',
                                                in_channels=self.field_nums[0] * self.field_nums[-1],weight_initializer=init.Uniform()))
                self.field_nums.append(size)
                self.register_child(self.conv_list[idx])

    def get_embedding_array(self,input_sample):
        y = nd.zeros(shape=(input_sample.shape[0], 1, self.embedding_size),ctx=self.ctx)
        for single_feat in self.feature_dict['sparse']:
            x = input_sample[:, single_feat.feat_name].reshape((-1,1))
            y1 = self.embedding_dict[single_feat.feat_name](x) #b,1,e
            y = nd.concat(y, y1, dim=1) # b,n+1,e
        return y[:, 1:]#b,n,e

    def get_dense_array(self,input_sample):
        y = nd.zeros(shape=(input_sample.shape[0],1,self.embedding_size),ctx=self.ctx)
        for single_feat in self.feature_dict['dense']:
            x = input_sample[:, single_feat.feat_name].reshape((-1,1))
            y1 = self.dense_dict[single_feat.feat_name](x).reshape((-1,1,self.embedding_size))
            y = nd.concat(y,y1,dim=1) # b,n+1,e
        return y[:, 1:]

    def matmul(self, x, y, transpose_a=False,transpose_b=False):
        batch = x.shape[0]#batch
        m = x.shape[1]#field
        h_k = y.shape[1]
        x = nd.split(x, self.embedding_size, 2)
        y = nd.split(y, self.embedding_size, 2)
        res = nd.zeros(shape=(1,batch,m,h_k),ctx=self.ctx)
        for idx in range(self.embedding_size):
            array = nd.batch_dot(x[idx], y[idx], transpose_a,transpose_b=transpose_b).reshape((1,-1,m,h_k))
            res = nd.concat(res,array,dim=0) # embedding+1,batch,field,field
        return res[1:,:,:,:]

    def get_linear_logit(self,embedding_part_sparse,dense_input):
        embedding_part_sparse = embedding_part_sparse.sum(axis=1).sum(axis=1).reshape((-1,1))
        net = nn.Sequential()
        net.add(self.linear_logit_dense)
        dense_linear_output = net(dense_input)
        net_embedding = nn.Sequential()
        net_embedding.add(self.linear_logit_embedding_bn)
        embedding_part_sparse = net_embedding(embedding_part_sparse)
        return embedding_part_sparse + dense_linear_output.reshape(embedding_part_sparse.shape)

    def cin(self,X):
        batch = X.shape[0]
        hidden_nn_layers = [X]
        split_tensor0 = hidden_nn_layers[0]
        final_result = nd.arange(batch *self.embedding_size,ctx=self.ctx).reshape((batch,1,self.embedding_size))
        for idx,layer_size in enumerate(self.layer_size):
            # time_start = time.time()
            split_tensor = hidden_nn_layers[-1]
            dot_result_m = self.matmul(split_tensor0, split_tensor, transpose_b=True)
            # print('dot_rsult_m')
            # print(dot_result_m.shape)
            # print('batch_matmul_time:[%d]s' % np.int(time.time() - time_start))

            # time_start = time.time()
            dot_result_o = dot_result_m.reshape((self.embedding_size, -1, self.field_nums[0] * self.field_nums[idx]))
            dot_result = nd.transpose(dot_result_o, axes=(1, 0, 2))
            dot_result = nd.transpose(dot_result,axes=(0, 2, 1))
            # print(dot_result.shape)
            net = nn.Sequential()
            net.add(self.conv_list[idx])
            curr_out = net(dot_result)
            # print('batch_cov_time_cost:[%d]s' % np.int(time.time() - time_start))
            # print(curr_out.shape)
            # if final_result is None:
            #     final_result = curr_out
            # else:
            final_result = nd.concat(final_result,curr_out,dim=1)
            hidden_nn_layers.append(curr_out)
        final_result = final_result[:,1:,:]
        result = final_result.sum(axis=2)
        net = nn.Sequential()
        net.add(self.cin_dense)
        net.add(self.cin_bn)
        result = net(result)
        return result

    def get_linear_dense_input(self, input_sample):
        y = nd.zeros(shape=(input_sample.shape[0], 1), ctx=self.ctx)
        for single_feat in self.feature_dict['dense']:
            x = input_sample[:, single_feat.feat_name].reshape((-1,1))
            y = nd.concat(y,x,dim=1)
            # y1 = self.dense_dict[single_feat.feat_name](x).reshape((-1,1,self.embedding_size))
            # y = nd.concat(y,y1,dim=1) # b,n+1,e
        return y[:, 1:]

    def forward(self, input_sample):
        embedding_part_sparse = self.get_embedding_array(input_sample) #(?,n1,e)
        # time_start = time.time()

        linear_dense_input = self.get_linear_dense_input(input_sample)
        linear_logit = self.get_linear_logit(embedding_part_sparse,linear_dense_input)
        # print('linear_logit_time_cost:[%d]s' % np.int(time.time() - time_start))
        # time_start = time.time()
                # print("linear_logit:")
        # print(linear_logit.mean().asscalar())

        dense_part_dense = self.get_dense_array(input_sample) # (?,n2,e)
        merge_sparse_dense = nd.concat(embedding_part_sparse,dense_part_dense,dim=1) # ?,f,e

        xv = nd.broadcast_mul(merge_sparse_dense,self.params.get('v').data())
        fm_embedding_part = nd.square(xv.sum(axis=1)) - nd.square(xv).sum(axis=1)
        fm_embedding_part = fm_embedding_part.sum(axis=1).reshape((-1,1)) / 2 # (?,1)
        # print(fm_embedding_part)
        net_embedding = nn.Sequential()
        net_embedding.add(self.bn_embedding)
        fm_embedding_part = net_embedding(fm_embedding_part)

        deep_input = merge_sparse_dense.flatten() # ?,f*e

        cin_output = self.cin(merge_sparse_dense)
        # print('cin_time_cost:[%d]s' % np.int(time.time() - time_start))
        # time_start = time.time()

        # print("cin_output:")
        # print(cin_output.mean().asscalar())

        net = nn.Sequential()
        for i in range(len(self.dense_list)):
            net.add(self.dense_list[i])
            net.add(self.bn_list[i])
            net.add(self.activation_list[i])
            net.add(self.dropout_list[i])
        net.add(self.dnn_out)
        deep_output = net(deep_input)
        # print('dnn_time_cost:[%d]s' % np.int(time.time() - time_start))

        # print('deep_output:')
        # print(deep_output.mean().asscalar())
        # print('----')
        # print(deep_output)
        deep_fm = nd.sigmoid(linear_logit + deep_output + cin_output)
        # print('==========')
        # print('deep_fm:')
        # print(deep_fm.mean().asscalar())
        # print('---------------------')
        return deep_fm

    def accuracy(self,y_hat, y):
        return (nd.argmax(y_hat, axis=1) == y).asnumpy().mean()

    def evaluate_accuracy(self,data_iter):
        acc = 0
        for X, y in data_iter:
            acc += self.accuracy(self.forward(X), y)
        return acc / len(data_iter)

    def train_epoch(self, epoch, train_iterator,trainer):
        train_epoch_loss = 0.0
        train_batch_loss = 0.0
        n_batch = 0
        train_acc = 0.0
        for batch_i, (X, y) in enumerate(train_iterator):
            X = X.as_in_context(self.ctx)
            y = nd.array(y).reshape((-1, 1))
            y = y.as_in_context(self.ctx)
            # time_start = time.time()
            with autograd.record():
                target = self.forward(X)
                ls = self.loss(target, y.reshape(target.shape))
            # print('batch_time_cost:[%d]s' % np.int(time.time() - time_start))
            # time_start = time.time()
            train_acc += self.accuracy(target,y.reshape(target.shape))

            ls.backward()
            trainer.step(self.batch_size)
            # print('update_time_cost:[%d]s' % np.int(time.time() - time_start))

            loss = ls.mean().asscalar() * y.shape[0]
            n_batch += y.shape[0]
            train_batch_loss += loss
            train_epoch_loss += loss
            if batch_i % 50 == 0:
                print('\n[%s] net_name:[%s] ,EPOCH [%d],BATCH [%d],average_loss:[%f]' %
                      (time.strftime("%Y-%m-%d %H:%M:%S"), self.task, epoch + 1, batch_i, train_batch_loss / n_batch))
                train_batch_loss = 0.0
                n_batch = 0
        return train_epoch_loss,train_acc / len(train_iterator)

    def predict(self,feature_values):
        train_iter = gdata.DataLoader(gdata.ArrayDataset(feature_values), batch_size=self.batch_size,
                                      shuffle=False)  # pd.read_csv去掉name
        result = []
        for X in train_iter:
            X = X.as_in_context(self.ctx)
            target = self.forward(X)
            result.append(target.reshape((-1,)).asnumpy().tolist())
        result = [num for l in result for num in l]
        return result

    def eval_model(self,test_iter):
        eval_loss = 0.0
        test_acc = self.evaluate_accuracy(test_iter)
        # right_num = 0
        for X,y in test_iter:
            X = X.as_in_context(self.ctx)
            y = nd.array(y).reshape((-1, 1))
            y = y.as_in_context(self.ctx)
            y_hat = self.forward(X)
            loss = self.loss(y_hat, y.reshape(y_hat.shape))
            eval_loss += loss.sum().asscalar()

        return eval_loss,test_acc

    def predict_and_save(self,args,test_feature,result_tmp):
        print('predict')
        res = self.predict(test_feature)
        if args.TASK == 'finish':
            result_tmp['finish_probability'] = res
            result_tmp.to_csv(args.FINISH_SAVE_PATH, index=False)
            if os.path.exists(args.LIKE_SAVE_PATH):
                merge(args.LIKE_SAVE_PATH, args.FINISH_SAVE_PATH, args.SUBMISSION_PATH)
        else:
            result_tmp['like_probability'] = res
            result_tmp.to_csv(args.LIKE_SAVE_PATH, index=False)
            if os.path.exists(args.FINISH_SAVE_PATH):
                merge(args.LIKE_SAVE_PATH, args.FINISH_SAVE_PATH, args.SUBMISSION_PATH)
