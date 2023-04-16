# coding=utf-8

import numpy as np
from DataStruct.population import Population
from DataStruct.genetypeQueue import GenetypeQueue


class GlobalConfig:
    # alternative Method: in [predoo,ramos,cradle,muffin,lemon]
    method = "predoo"
    # data generator(except Ramos)
    data_generator = None
    # structure generator(except Ramos)
    structure_generator = None
    # Record of Model Construction Failure
    fail_time = 0
    # Present Size of Corpus
    N = 0
    # Total Layers of Hierarchical Structure
    L = 3
    # Number of Motifs Contained by Each Layer
    operatorNum = np.array([3, 3, 1])
    # Number of Vertexes of the Motifs in Each Layer
    pointNum = [5, 5, 5]
    # DAG Representation
    flatOperatorMaps = []
    # Total Mutation Time for Cell Search
    maxMutateTime = 200
    # Record of how many rounds has been carried out
    alreadyMutatetime = 0
    # Corpus
    P = Population()
    # Selected Mutation Materials
    Q = GenetypeQueue()
    # Error Modes: max, avg are available, max is recommended
    error_cal_mode = "max"
    # Total Mutation Times Required in Initialization
    initMutateTime = 200
    # Operator Sequence of Final Module
    final_module = []
    # Channels of Each Primitive Operator in Final Module
    channels = []
    # Applied Datasets: random, cifar10, mnist, fashion_mnist, imagenet, sinewave, price and predoo
    dataset = 'predoo'
    # Batch of Random
    batch = 1
    # Initial Channel for Random
    c0 = 3
    # Height of Random 8 de beishu
    h = 224
    # Width of Random 8 de beishu
    w = 224
    # K of Tournament Algorithm
    k = 1
    # Feedback mode: 0. Only primitive operators 1. Only composite operators 2. Both
    mode = 2
    # Result File
    outFile = None
    # Trigger of CSV Writer
    writer = None
    # Primitive Operators
    basicOps = ['identity', 'None', '1*1', 'depthwise_conv2D', 'separable_conv2D', 'max_pooling2D',
                'average_pooling2D', 'conv2D', 'conv2D_transpose', 'ReLU', 'sigmoid', 'tanh', 'leakyReLU',
                'PReLU', 'ELU', 'batch_normalization']
    # The weight of Primitive Operators(Containing None and Identity)
    basicWeights = [1] * len(basicOps)
    # Probability(tendency) of Selecting Primitive Operator
    basicProp = 0.8
    # address and port of your http-server
    HttpServer = "http://127.0.0.1:8080/"
    # To Save TFJS model or not
    save_TFJS_model = "FALSE" # TRUE or FALSE
    # Absolute Path of this project
    absolutePath = "E:\\graduate\\HNAS"

    #TODO 配置成自己电脑上的命令行
    Command_chrome_start = "\"C:\\Program Files\\Google\\Chrome\\Application\\chrome.exe\" E:\\graduate\\HNAS\\TFJS\\TFJS_Executor.html"
    Command_edge_start = "\"C:\\Program Files (x86)\\Microsoft\\Edge\\Application\\msedge.exe\" E:\\graduate\\HNAS\\TFJS\\TFJS_Executor.html"
    Command_chrome_shut = "taskkill /f /t /im chrome.exe"
    Command_edge_shut = "taskkill /f /t /im msedge.exe"


