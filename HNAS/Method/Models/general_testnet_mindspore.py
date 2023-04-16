# 用于非Ramos方法，cell不重复，不添加任何操作
import copy
import mindspore.nn as nn
import mindspore.ops as ops

class GeneralMindsporeNet(nn.Cell):
    def __init__(self,in_channel,final_module,channels):
        super(GeneralMindsporeNet, self).__init__()
        self.in_channel = in_channel
        self.final_module = final_module
        self.channels = channels
    def construct(self,x):
        # 各节点的张量。
        tensors = []
        # 判断某张量是否有初始值。
        tensors_isnull = [True] * len(self.channels)
        tensors.append(x)
        tensors_isnull[0] = False
        for i in range(len(self.channels) - 1):
            # 随意赋一个同类型的初始值
            tensors.append(True)
        final_point = 0
        for eachOperation in self.final_module:
            fromIndex = eachOperation.fromIndex
            final_point = eachOperation.toIndex
            input = tensors[fromIndex]
            toIndex = eachOperation.toIndex
            # 本NAS规定所有操作出通道数与入通道数相同。
            operator_in_channel = self.channels[fromIndex]*self.in_channel
            operator = eachOperation.operator

            # if operator != -1 and operator != 0:
            #     operator = -1
            # indentity
            if operator == -1:
                # print("mindspore执行了了操作-1 ", eachOperation.fromIndex, " ", eachOperation.toIndex)
                if tensors_isnull[toIndex] == True:
                    tensors_isnull[toIndex] = False
                    tensors[toIndex] = copy.deepcopy(tensors[fromIndex])
                    # #log
                    # print("tensor"+str(toIndex)+":")
                    # print(tensors[toIndex])
                else:
                    # concat用于数组之间的操作，因此需要先进行类型转换。
                    temp = ops.Concat(1)((tensors[toIndex], input))
                    tensors[toIndex] = copy.deepcopy(temp)
                    # #log
                    # print("tensor"+str(toIndex)+":")
                    # print(tensors[toIndex])
            # 1 × 1 convolution of C channels
            elif operator == 1:
                # print("mindspore执行了操作1 ", eachOperation.fromIndex, " ", eachOperation.toIndex)
                if tensors_isnull[toIndex] == True:
                    tensors_isnull[toIndex] = False
                    # 这是1*1卷积的代码
                    # filter参数顺序:OutChannel、InChannel、H、W
                    thisresult = nn.Conv2d(in_channels = operator_in_channel
                                           ,out_channels=self.in_channel
                                           ,kernel_size=1,
                                           pad_mode='valid',
                                           stride=1,padding=0,weight_init = 'ones')(input)
                    tensors[toIndex] = copy.deepcopy(thisresult)
                    # #log
                    # print("tensor"+str(toIndex)+":")
                    # print(tensors[toIndex])
                else:
                    # 这是1*1卷积的代码。
                    thisresult = nn.Conv2d(in_channels = operator_in_channel
                                           ,out_channels=self.in_channel
                                           ,kernel_size=1,
                                           pad_mode='valid',
                                           stride=1,padding=0,weight_init = 'ones')(input)
                    tensors[toIndex] = copy.deepcopy(ops.Concat(1)((tensors[toIndex], thisresult)))
                    # #log
                    # print("tensor"+str(toIndex)+":")
                    # print(tensors[toIndex])
            # 3 × 3 depthwise convolution
            elif operator == 2:
                # print("mindspore执行了操作2", eachOperation.fromIndex, " ", eachOperation.toIndex)
                if tensors_isnull[toIndex] == True:
                    tensors_isnull[toIndex] = False
                    #注：mindspore是先右后左，均匀补0。tensorflow。而tensorflow步长为奇数时两端补0，步长为偶数时右下补0.
                    #因此步长为奇数时是一致的。
                    thisresult = nn.Conv2d(in_channels = operator_in_channel
                                           ,out_channels= operator_in_channel
                                           ,group=operator_in_channel
                                           ,kernel_size=3,
                                           stride=1,pad_mode = 'same',weight_init = 'ones')(input)

                    tensors[toIndex] = copy.deepcopy(thisresult)
                    # #log
                    # print("tensor"+str(toIndex)+":")
                    # print(tensors[toIndex])
                else:
                    thisresult = nn.Conv2d(in_channels = operator_in_channel
                                           ,out_channels= operator_in_channel
                                           ,group=operator_in_channel
                                           ,kernel_size=3,
                                           stride=1,pad_mode = 'same',weight_init = 'ones')(input)

                    tensors[toIndex] = copy.deepcopy(ops.Concat(1)((tensors[toIndex], thisresult)))
                    # #log
                    # print("tensor"+str(toIndex)+":")
                    # print(tensors[toIndex])
            elif operator == 3:
                # print("mindspore执行了操作3", eachOperation.fromIndex, " ", eachOperation.toIndex)
                if tensors_isnull[toIndex] == True:
                    tensors_isnull[toIndex] = False
                    depthwise_temp = nn.Conv2d(in_channels = operator_in_channel
                                           ,out_channels= operator_in_channel
                                           ,group=operator_in_channel
                                           ,kernel_size=3,
                                           stride=1,pad_mode = 'same',weight_init = 'ones')(input)
                    pointwise_temp = nn.Conv2d(in_channels=operator_in_channel
                                    , out_channels=self.in_channel
                                    , kernel_size=1
                                    , stride=1
                                    , padding=0
                                    , pad_mode='valid'
                                    , weight_init='ones')(depthwise_temp)
                    tensors[toIndex] = copy.deepcopy(pointwise_temp)
                    # #log
                    # print("tensor"+str(toIndex)+":")
                    # print(tensors[toIndex])
                else:
                    input = nn.Pad(paddings=((0,0),(0,0),(1,1),(1,1)),mode='CONSTANT')(input)
                    depthwise_temp = nn.Conv2d(in_channels=operator_in_channel
                                    , out_channels=operator_in_channel
                                    , group=operator_in_channel
                                    , kernel_size=3, stride=1, padding=0, pad_mode='valid',
                                    weight_init='ones')(input)
                    pointwise_temp = nn.Conv2d(in_channels=operator_in_channel
                                    , out_channels=self.in_channel
                                    , kernel_size=1
                                    , stride=1
                                    , padding=0
                                    , pad_mode='valid'
                                    , weight_init='ones')(depthwise_temp)
                    tensors[toIndex] = copy.deepcopy(ops.Concat(1)((tensors[toIndex], pointwise_temp)))
                    # #log
                    # print("tensor"+str(toIndex)+":")
                    # print(tensors[toIndex])
            elif operator == 4:
                # print("mindspore执行了了操作4 ", eachOperation.fromIndex, " ", eachOperation.toIndex)
                if tensors_isnull[toIndex] == True:
                    tensors_isnull[toIndex] = False
                    result = nn.MaxPool2d(kernel_size=3,stride=1,pad_mode='same')(input)
                    tensors[toIndex] = copy.deepcopy(result)
                    # #log
                    # print("tensor"+str(toIndex)+":")
                    # print(tensors[toIndex])
                else:
                    result = nn.MaxPool2d(kernel_size=3,stride=1,pad_mode='same')(input)
                    tensors[toIndex] = copy.deepcopy(ops.Concat(1)((tensors[toIndex], result)))
                    # #log
                    # print("tensor"+str(toIndex)+":")
                    # print(tensors[toIndex])
            elif operator == 5:
                # print("mindspore执行了了操作5 ", eachOperation.fromIndex, " ", eachOperation.toIndex)
                if tensors_isnull[toIndex] == True:
                    tensors_isnull[toIndex] = False
                    # 注意：一定要有count_include_pad=False,不计算补的0，和tensorflow保持一致。
                    result = nn.AvgPool2d(kernel_size=3,stride=1,pad_mode='same')(input)
                    tensors[toIndex] = copy.deepcopy(result)
                    # #log
                    # print("tensor"+str(toIndex)+":")
                    # print(tensors[toIndex])
                else:
                    result = nn.AvgPool2d(kernel_size=3,stride=1,pad_mode='same')(input)
                    tensors[toIndex] = copy.deepcopy(ops.Concat(1)((tensors[toIndex], result)))
                    # #log
                    # print("tensor"+str(toIndex)+":")
                    # print(tensors[toIndex])
            # 3 × 3 convolution of C channels
            elif operator == 6:
                # print("mindspore执行了操作6 ", eachOperation.fromIndex, " ", eachOperation.toIndex)
                if tensors_isnull[toIndex] == True:
                    tensors_isnull[toIndex] = False
                    # 这是1*1卷积的代码
                    # filter参数顺序:OutChannel、InChannel、H、W
                    thisresult = nn.Conv2d(in_channels=operator_in_channel
                                           , out_channels=self.in_channel
                                           , kernel_size=3,
                                           stride=1, pad_mode='same',
                                           weight_init = 'ones')(input)
                    tensors[toIndex] = copy.deepcopy(thisresult)
                    # #log
                    # print("tensor"+str(toIndex)+":")
                    # print(tensors[toIndex])
                else:
                    # 这是3*3卷积的代码。
                    thisresult = nn.Conv2d(in_channels=operator_in_channel
                                           , out_channels=self.in_channel
                                           , kernel_size=3,
                                           stride=1, pad_mode='same',
                                           weight_init = 'ones')(input)
                    tensors[toIndex] = copy.deepcopy(ops.Concat(1)((tensors[toIndex], thisresult)))
                    # #log
                    # print("tensor"+str(toIndex)+":")
                    # print(tensors[toIndex])
            #3*3 conv2D_transpose of C channels
            #注：参数代表的是正向卷积的过程，但我要做的是反卷积。
            elif operator == 7:
                # print("mindspore执行了操作7 ", eachOperation.fromIndex, " ", eachOperation.toIndex)
                if tensors_isnull[toIndex] == True:
                    tensors_isnull[toIndex] = False
                    # 这是3*3卷积的代码
                    # filter参数顺序:OutChannel、InChannel、H、W
                    thisresult = nn.Conv2dTranspose(in_channels=operator_in_channel
                                           , out_channels=self.in_channel
                                           , kernel_size=3,
                                           stride=1, pad_mode='same',
                                           weight_init = 'ones')(input)
                    tensors[toIndex] = copy.deepcopy(thisresult)
                    # #log
                    # print("tensor"+str(toIndex)+":")
                    # print(tensors[toIndex])
                else:
                    # 这是3*3卷积的代码。
                    thisresult = nn.Conv2dTranspose(in_channels=operator_in_channel
                                           , out_channels=self.in_channel
                                           , kernel_size=3,
                                           stride=1, pad_mode='same',
                                           weight_init = 'ones')(input)
                    tensors[toIndex] = copy.deepcopy(ops.Concat(1)((tensors[toIndex], thisresult)))
                    # #log
                    # print("tensor"+str(toIndex)+":")
                    # print(tensors[toIndex])
            #relu
            elif operator == 8:
                # print("mindspore执行了了操作8 ", eachOperation.fromIndex, " ", eachOperation.toIndex)
                if tensors_isnull[toIndex] == True:
                    tensors_isnull[toIndex] = False
                    result = nn.layer.ReLU()(input)
                    tensors[toIndex] = copy.deepcopy(result)
                    # #log
                    # print("tensor"+str(toIndex)+":")
                    # print(tensors[toIndex])
                else:
                    result = nn.layer.ReLU()(input)
                    tensors[toIndex] = copy.deepcopy(ops.Concat(1)((tensors[toIndex], result)))
                    # #log
                    # print("tensor"+str(toIndex)+":")
                    # print(tensors[toIndex])
            #sigmoid
            elif operator == 9:
                # print("mindspore执行了了操作9 ", eachOperation.fromIndex, " ", eachOperation.toIndex)
                if tensors_isnull[toIndex] == True:
                    tensors_isnull[toIndex] = False
                    result = nn.layer.Sigmoid()(input)
                    tensors[toIndex] = copy.deepcopy(result)
                    # #log
                    # print("tensor"+str(toIndex)+":")
                    # print(tensors[toIndex])
                else:
                    result = nn.layer.Sigmoid()(input)
                    tensors[toIndex] = copy.deepcopy(ops.Concat(1)((tensors[toIndex], result)))
                    # #log
                    # print("tensor"+str(toIndex)+":")
                    # print(tensors[toIndex])
            #tanh
            elif operator == 10:
                # print("mindspore执行了了操作10 ", eachOperation.fromIndex, " ", eachOperation.toIndex)
                if tensors_isnull[toIndex] == True:
                    tensors_isnull[toIndex] = False
                    result = nn.layer.Tanh()(input)
                    tensors[toIndex] = copy.deepcopy(result)
                    # #log
                    # print("tensor"+str(toIndex)+":")
                    # print(tensors[toIndex])
                else:
                    result = nn.layer.Tanh()(input)
                    tensors[toIndex] = copy.deepcopy(ops.Concat(1)((tensors[toIndex], result)))
                    # #log
                    # print("tensor"+str(toIndex)+":")
                    # print(tensors[toIndex])
            #leakyrelu
            elif operator == 11:
                # print("mindspore执行了了操作11 ", eachOperation.fromIndex, " ", eachOperation.toIndex)
                if tensors_isnull[toIndex] == True:
                    tensors_isnull[toIndex] = False
                    result = nn.LeakyReLU()(input)
                    tensors[toIndex] = copy.deepcopy(result)
                    # #log
                    # print("tensor"+str(toIndex)+":")
                    # print(tensors[toIndex])
                else:
                    result = nn.LeakyReLU()(input)
                    tensors[toIndex] = copy.deepcopy(ops.Concat(1)((tensors[toIndex], result)))
                    # #log
                    # print("tensor"+str(toIndex)+":")
                    # print(tensors[toIndex])
            #prelu
            elif operator == 12:
                # print("mindspore执行了了操作12 ", eachOperation.fromIndex, " ", eachOperation.toIndex)
                if tensors_isnull[toIndex] == True:
                    tensors_isnull[toIndex] = False
                    result = input
                    tensors[toIndex] = copy.deepcopy(result)
                    # #log
                    # print("tensor"+str(toIndex)+":")
                    # print(tensors[toIndex])
                else:
                    result = input
                    tensors[toIndex] = copy.deepcopy(ops.Concat(1)((tensors[toIndex], result)))
                    # #log
                    # print("tensor"+str(toIndex)+":")
                    # print(tensors[toIndex])
            #elu
            elif operator == 13:
                # print("mindspore执行了了操作13 ", eachOperation.fromIndex, " ", eachOperation.toIndex)
                if tensors_isnull[toIndex] == True:
                    tensors_isnull[toIndex] = False
                    result = nn.ELU()(input)
                    tensors[toIndex] = copy.deepcopy(result)
                    # #log
                    # print("tensor"+str(toIndex)+":")
                    # print(tensors[toIndex])
                else:
                    result = nn.ELU()(input)
                    tensors[toIndex] = copy.deepcopy(ops.Concat(1)((tensors[toIndex], result)))
                    # #log
                    # print("tensor"+str(toIndex)+":")
                    # print(tensors[toIndex])
            #batchnorm
            elif operator == 14:
                # print("mindspore执行了了操作14 ", eachOperation.fromIndex, " ", eachOperation.toIndex)
                if tensors_isnull[toIndex] == True:
                    tensors_isnull[toIndex] = False
                    result = nn.BatchNorm2d(num_features=operator_in_channel, eps=1e-05, momentum=0.9,
                                                affine=True,
                                                gamma_init="ones", beta_init="zeros",
                                                moving_mean_init="zeros", moving_var_init="ones",
                                                use_batch_statistics=True, data_format="NCHW")(input)
                    tensors[toIndex] = copy.deepcopy(result)
                    # #log
                    # print("tensor"+str(toIndex)+":")
                    # print(tensors[toIndex])
                else:
                    result = nn.BatchNorm2d(num_features=operator_in_channel, eps=1e-05, momentum=0.9,
                                                affine=True,
                                                gamma_init="ones", beta_init="zeros",
                                                moving_mean_init="zeros", moving_var_init="ones",
                                                use_batch_statistics=True, data_format="NCHW")(input)
                    tensors[toIndex] = copy.deepcopy(ops.Concat(1)((tensors[toIndex], result)))
                    # #log
                    # print("tensor"+str(toIndex)+":")
                    # print(tensors[toIndex])
        return copy.deepcopy(tensors[final_point])