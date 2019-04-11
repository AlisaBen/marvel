import mxnet as mx
from mxnet import nd, autograd,init,initializer,metric
from mxnet.gluon import loss as gloss, nn, data as gdata, Trainer



class CIN(nn.Block):
    def __init__(self,embedding_size,field_num,layer_size,ctx,**kwargs):
        super(CIN, self).__init__(**kwargs)
        self.embedding_size = embedding_size
        self.field_num = field_num
        self.ctx = ctx
        self.layer_size = layer_size
        self.field_nums = [self.field_num]
        self.conv_list = []
        for idx, size in enumerate(self.layer_size):
            self.conv_list.append(nn.Conv1D(channels=size, kernel_size=1, strides=1, padding=0,activation='relu',in_channels=self.field_nums[0] * self.field_nums[-1]))
            self.field_nums.append(size)
            self.register_child(self.conv_list[idx])

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

    def forward(self, X):
        hidden_nn_layers = [X]
        split_tensor0 = hidden_nn_layers[0]
        batch = X.shape[0]
        # final_result = None
        final_result = nd.arange(batch *self.embedding_size,ctx=self.ctx).reshape((batch,1,self.embedding_size))

        for idx,layer_size in enumerate(self.layer_size):
            split_tensor = hidden_nn_layers[-1]
            dot_result_m = self.matmul(split_tensor0, split_tensor, transpose_b=True)
            print('dot_rsult_m')
            print(dot_result_m.shape)
            dot_result_o = dot_result_m.reshape((self.embedding_size, -1, self.field_nums[0] * self.field_nums[idx]))
            dot_result = nd.transpose(dot_result_o, axes=(1, 0, 2))
            dot_result = nd.transpose(dot_result,axes=(0, 2, 1))
            print(dot_result.shape)
            net = nn.Sequential()
            net.add(self.conv_list[idx])
            curr_out = net(dot_result)
            print('curr_out shape:')
            print(curr_out.shape)
            # if final_result is None:
            #     final_result = curr_out
            # else:
            print('final result:')
            print(final_result.shape)

            final_result = nd.concat(final_result,curr_out,dim=1)
            hidden_nn_layers.append(curr_out)
        final_result = final_result[:, 1:, :]
        result = final_result.sum(axis=2)
        return result


if __name__ == '__main__':
    X = nd.arange(1, 5 * 16 * 25 + 1).reshape((5, 16, 25))
    # x1= nd.split(X, 25, 2)
    # print(nd.array(x1))
    # print(X.shape)
    # print(x1)
    # print(x1[0].shape)
    net = CIN(25, 16, [128,64],mx.cpu())
    net.initialize()
    y = net(X)
    print(y.shape)
    print('kkk')
    # X = X.reshape((25,5,16,1))
    # y = nd.dot(X,X,transpose_b=True)
    # print(y.shape) # 25,5,16,25,5,16