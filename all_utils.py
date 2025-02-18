import torch

import torch.nn.functional as F
class ContrastiveLoss(torch.nn.Module):
    def __init__(self, margin=2.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, output1, output2, label):
        # print('label.shape', label.shape)
        # 计算每个标签的欧式距离，注意要对每个标签的维度进行平方和
        # 计算逐标签的欧式距离
        euclidean_distance = torch.sqrt((output1 - output2) ** 2)


        loss_contrastive = torch.mean((1 - label) * torch.pow(euclidean_distance, 2) +  # calmp夹断用法
                                      (label) * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2))

        return loss_contrastive


def collate(batch):
    device = torch.device("cpu")
    seq1_ls = []
    seq2_ls = []
    label1_ls = []
    label2_ls = []
    label_ls = []

    batch_size = len(batch)
    for i in range(int(batch_size / 2)):
        seq1, label1= batch[i][0], batch[i][1]
        seq2, label2 = batch[i + int(batch_size / 2)][0], batch[i + int(batch_size / 2)][1]
        label1_ls.append(label1.unsqueeze(0).float())
        label2_ls.append(label2.unsqueeze(0).float())
        label = (label1 ^ label2)
        seq1_ls.append(seq1.unsqueeze(0))
        seq2_ls.append(seq2.unsqueeze(0))
        label_ls.append(label.unsqueeze(0))
    seq1 = torch.cat(seq1_ls).to(device)
    seq2 = torch.cat(seq2_ls).to(device)
    label = torch.cat(label_ls).to(device)
    label1 = torch.cat(label1_ls).to(device)
    label2 = torch.cat(label2_ls).to(device)
    return seq1, seq2, label, label1, label2