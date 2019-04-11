import mxnet as mx
from mxnet import nd, autograd,init,initializer,metric
from mxnet.gluon import loss as gloss, nn, data as gdata, Trainer
import time


def train_epoch(net, epoch, train_iterator, trainer):
    train_epoch_loss = 0.0
    train_batch_loss = 0.0
    n_batch = 0
    for batch_i, (X, y) in enumerate(train_iterator):
        X = X.as_in_context(net.ctx)
        y = nd.array(y).reshape((-1, 1))
        y = y.as_in_context(net.ctx)
        with autograd.record():
            target = net.forward(X)
            ls = net.loss(target, y.reshape(target.shape))
        ls.backward()
        trainer.step(net.batch_size)
        loss = ls.mean().asscalar() * y.shape[0]
        n_batch += y.shape[0]
        train_batch_loss += loss
        train_epoch_loss += loss
        if batch_i % 50 == 0:
            print('\n[%s] net_name:[%s] ,EPOCH [%d],BATCH [%d],average_loss:[%f]' %
                  (time.strftime("%Y-%m-%d %H:%M:%S"), net.task, epoch + 1, batch_i, train_batch_loss / n_batch))
            train_batch_loss = 0.0
            n_batch = 0
    return train_epoch_loss


def predict(net,feature_values):
    train_iter = gdata.DataLoader(gdata.ArrayDataset(feature_values), batch_size=net.batch_size,
                                  shuffle=False)  # pd.read_csv去掉name
    result = []
    for X in train_iter:
        X = X.as_in_context(net.ctx)
        target = net.forward(X)
        result.append(target.reshape((-1,)).asnumpy().tolist())
    result = [num for l in result for num in l]
    return result


def eval_model(net,test_iter):
    eval_loss = 0.0
    # right_num = 0
    for X, y in test_iter:
        X = X.as_in_context(net.ctx)
        y = nd.array(y).reshape((-1, 1))
        y = y.as_in_context(net.ctx)
        y_hat = net.forward(X)
        loss = net.loss(y_hat, y.reshape(y_hat.shape))
        eval_loss += loss.sum().asscalar()

    return eval_loss