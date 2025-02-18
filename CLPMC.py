
import numpy as np
import datetime
import os
import csv
import pandas as pd
import torch
# from torch import cdist
from scipy.spatial.distance import cdist
from model import *
from torch.utils.data import DataLoader
from loss_functions import *
from train import *
from transformers import BertModel, BertTokenizer, AutoTokenizer

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
filenames = ['AAP', 'ABP', 'ACP', 'ACVP', 'ADP', 'AEP', 'AFP', 'AHIVP', 'AHP', 'AIP', 'AMRSAP', 'APP', 'ATP',
             'AVP',
             'BBP', 'BIP',
             'CPP', 'DPPIP',
             'QSP', 'SBP', 'THP']


def PadEncode(data, label, max_len):
    amino_acids = 'XACDEFGHIKLMNPQRSTVWY'
    data_e, label_e, seq_length, temp = [], [], [], []
    sequence = []
    sign, b = 0, 0
    for i in range(len(data)):
        length = len(data[i])
        elemt, st = [], data[i].strip()
        for j in st:
            if j not in amino_acids:
                sign = 1
                break
            index = amino_acids.index(j)
            elemt.append(index)
            sign = 0

        if length <= max_len and sign == 0:

            temp.append(elemt)
            seq_length.append(len(temp[b]))
            b += 1
            elemt += [0] * (max_len - length)
            data_e.append(elemt)
            label_e.append(label[i])

            sequence.append(data[i])
    return np.array(data_e), np.array(label_e), np.array(seq_length),sequence


def getSequenceData(first_dir, file_name):
    # getting sequence data and label
    data, label = [], []
    path = "{}/{}.txt".format(first_dir, file_name)

    with open(path) as f:
        for each in f:
            each = each.strip()
            if each[0] == '>':
                label.append(np.array(list(each[1:]), dtype=int))  # Converting string labels to numeric vectors
            else:
                data.append(each)

    return data, label

# 最大绝对值归一化
def max_absolute_normalize(data):
    return data / np.max(np.abs(data))
def staticTrainAndTest(y_train, y_test):
    data_size_tr = np.zeros(len(filenames))
    data_size_te = np.zeros(len(filenames))

    for i in range(len(y_train)):
        for j in range(len(y_train[i])):
            if y_train[i][j] > 0:
                data_size_tr[j] += 1

    for i in range(len(y_test)):
        for j in range(len(y_test[i])):
            if y_test[i][j] > 0:
                data_size_te[j] += 1

    return data_size_tr


def main(num, data):
    first_dir = 'dataset'

    max_length = 50  # the longest length of the peptide sequence

    # getting train data and test data
    train_sequence_data, train_sequence_label = getSequenceData(first_dir, 'train')
    test_sequence_data, test_sequence_label = getSequenceData(first_dir, 'test')

    # Converting the list collection to an array
    y_train = np.array(train_sequence_label)
    y_test = np.array(test_sequence_label)




    # The peptide sequence is encoded and the sequences that do not conform to the peptide sequence are removed
    x_train, y_train, train_length,train_sequence = PadEncode(train_sequence_data, y_train, max_length)
    x_test, y_test, test_length,test_sequence = PadEncode(test_sequence_data, y_test, max_length)

    model_name = "chenmBert"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = BertModel.from_pretrained(model_name)

    # 加载模型的状态字典
    model.load_state_dict(torch.load('duibi_model.pth'),False)

    train_inputs = tokenizer(train_sequence, max_length=50, return_tensors="pt", padding=True, truncation=True)['input_ids']
    test_inputs = tokenizer(test_sequence, max_length=50, return_tensors="pt", padding=True, truncation=True)

    dataset_bert = DataLoader(train_inputs, batch_size=64)

    embeddings = []
    with torch.no_grad():

        for data in dataset_bert:
            train_outputs = model(data)
            embeddings.append(train_outputs.last_hidden_state.mean(dim=1))  # 取平均
        test_outputs = model(**test_inputs)

    train_embeddings = torch.cat(embeddings, dim=0)  # 形状为(16*n, 768)

    test_embeddings = test_outputs.last_hidden_state.mean(dim=1)

    import umap as umap

    reducer = umap.UMAP(n_components=100)
    train_reduced_embeddings = reducer.fit_transform(train_embeddings)
    test_reduced_embeddings = reducer.fit_transform(test_embeddings)

    from sklearn.mixture import GaussianMixture

    gmm = GaussianMixture(n_components=21)
    gmm.fit(train_reduced_embeddings)
    cluster_centers = gmm.means_
    x_distances = cdist(train_reduced_embeddings, cluster_centers, 'euclidean')
    test_distances = cdist(test_reduced_embeddings, cluster_centers, 'euclidean')
    train_prob = max_absolute_normalize(x_distances)
    ttest_prob = max_absolute_normalize(test_distances)
    t1 = gmm.predict_proba(train_reduced_embeddings)
    t2 = gmm.predict_proba(test_reduced_embeddings)
    x_train = torch.LongTensor(x_train)  # torch.Size([7872, 50])
    x_test = torch.LongTensor(x_test)  # torch.Size([1969, 50])
    train_prob = train_prob* t1
    ttest_prob = ttest_prob* t2

    train_prob = torch.Tensor(train_prob)
    ttest_prob = torch.Tensor(ttest_prob)

    x_train = torch.cat((x_train, train_prob), dim=1)
    x_test = torch.cat((x_test, ttest_prob), dim=1)


    y_test = torch.Tensor(y_test)
    y_train = torch.Tensor(y_train)


    """Create a dataset and split it"""
    dataset_train = list(zip(x_train, y_train))
    dataset_test = list(zip(x_test, y_test))
    dataset_train = DataLoader(dataset_train, batch_size=256)
    dataset_test = DataLoader(dataset_test, batch_size=256)

    # PATH = os.getcwd()
    # each_model = os.path.join(PATH, 'result', 'Model', 'data', 'tea_data' + str(num) + '.h5')
    # torch.save(dataset_test, each_model)
    # 设置训练参数
    vocab_size = 50
    output_size = 21



    # # 初始化参数训练模型相关参数
    # model = ETFC(vocab_size, data['embedding_size'], output_size, data['dropout'], data['fan_epochs'],
    #              data['num_heads'])
    # rate_learning = data['learning_rate']
    # 初始化参数训练模型相关参数
    model = TSTC(vocab_size, 192, output_size, 0.6, 1,
                 8)
    rate_learning = 0.0018
    optimizer = torch.optim.Adam(model.parameters(), lr=rate_learning)
    lr_scheduler = CosineScheduler(10000, base_lr=rate_learning, warmup_steps=500)


    # criterion = FocalDiceLoss(clip_pos=data['clip_pos'], clip_neg=data['clip_neg'], pos_weight=data['pos_weight'])
    criterion = FocalDiceLoss(clip_pos=0.7, clip_neg=0.3, pos_weight=0.3)
    # 创建初始化训练类
    Train = DataTrain(model, optimizer, criterion, lr_scheduler, device=DEVICE)

    a = time.time()
    # Train.train_step(dataset_train,epochs=data['epochs'], plot_picture=False)
    Train.train_step(dataset_train, epochs=200, plot_picture=False)
    b = time.time()
    test_score = evaluate(model,dataset_test, device=DEVICE)
    runtime = b - a

    "-------------------------------------------保存模型参数-----------------------------------------------"
    PATH = os.getcwd()
    each_model = os.path.join(PATH, 'result', 'Model', 'CLPMC', 'CL_model' + str(num) + '.h5')
    torch.save(model.state_dict(), each_model, _use_new_zipfile_serialization=False)
    "---------------------------------------------------------------------------------------------------"

    "-------------------------------------------输出模型结果-----------------------------------------------"
    print(f"runtime:{runtime:.3f}s")
    print("测试集：")
    print(f'aiming: {test_score["aiming"]:.3f}')
    print(f'coverage: {test_score["coverage"]:.3f}')
    print(f'accuracy: {test_score["accuracy"]:.3f}')
    print(f'absolute_true: {test_score["absolute_true"]:.3f}')
    print(f'absolute_false: {test_score["absolute_false"]:.3f}')
    "---------------------------------------------------------------------------------------------------"


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    clip_pos = 0.7
    clip_neg = 0.5
    pos_weight = 0.3

    batch_size = 256
    epochs = 200
    learning_rate = 0.0018

    embedding_size = 192
    # embedding_size = 256
    dropout = 0.6
    fan_epochs = 1
    num_heads = 8
    para = {'clip_pos': clip_pos,
            'clip_neg': clip_neg,
            'pos_weight': pos_weight,
            'batch_size': batch_size,
            'epochs': epochs,
            'learning_rate': learning_rate,
            'embedding_size': embedding_size,
            'dropout': dropout,
            'fan_epochs': fan_epochs,
            'num_heads': num_heads}
    for i in range(10):
        main(i, para)
