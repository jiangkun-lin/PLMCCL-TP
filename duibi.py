import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset
from transformers import BertTokenizer, BertModel, AutoTokenizer

from m1.KD_main import getSequenceData, PadEncode
from m1.all_utils import collate,ContrastiveLoss

first_dir = 'dataset'

max_length = 50  # the longest length of the peptide sequence

# getting train data and test data
train_sequence_data, train_sequence_label = getSequenceData(first_dir, 'train')
test_sequence_data, test_sequence_label = getSequenceData(first_dir, 'test')

# Converting the list collection to an array
y_train = np.array(train_sequence_label)
y_test = np.array(test_sequence_label)


# 自定义Dataset类
class CustomDataset(Dataset):
    def __init__(self, features, labels):
        self.features = features  # 存储特征
        self.labels = labels      # 存储标签

    def __len__(self):
        return len(self.features)  # 返回样本的数量

    def __getitem__(self, idx):
        feature = self.features[idx]  # 根据索引获取对应的特征
        label = self.labels[idx]      # 根据索引获取对应的标签
        return feature, label         # 返回特征和标签
# The peptide sequence is encoded and the sequences that do not conform to the peptide sequence are removed
# x_train, y_train, train_length,train_sequence = PadEncode(train_sequence_data, y_train, max_length)
# x_test, y_test, test_length,test_sequence = PadEncode(test_sequence_data, y_test, max_length)


model_name = r"D:\bisai\PreProtac\chenmBert"
tokenizer = AutoTokenizer.from_pretrained(model_name)
# model = BertModel.from_pretrained(model_name)
train_inputs = tokenizer(train_sequence_data, max_length=50, return_tensors="pt", padding=True, truncation=True)
tt = train_inputs['input_ids']
# trian_data = torch.cat((tt,train_sequence_label),dim=1)

test_inputs = tokenizer(test_sequence_data, max_length=50, return_tensors="pt", padding=True, truncation=True)

# 实例化自定义Dataset
train_sequence_label = torch.tensor(train_sequence_label)
dataset = CustomDataset(tt, train_sequence_label)



class me(nn.Module):
    def __init__(self,bert_pretrained_model_dir):
        super(me, self).__init__()
        self.bert = BertModel.from_pretrained(bert_pretrained_model_dir, output_attentions=True)

        self.num_labels = 21
        self.linear_2 = nn.Linear(768, 21)
    def forward(self,input_ids):
        output = self.bert(input_ids=input_ids).pooler_output
        # attention_matrix = output[-1]  # [12, 1, 12, n, n]
        output = self.linear_2(output)
        return output

# 使用DataLoader封装Dataset
dataloader = DataLoader(tt, batch_size=64, shuffle=True)

fusion_out_train = DataLoader(dataset, batch_size=64, shuffle=True, collate_fn=collate)
criterion = nn.BCEWithLogitsLoss()

model = me(model_name)
optimizer = optim.Adam(model.parameters(), lr=0.001)
contrastive_loss_fn = ContrastiveLoss()

for epoch in range(100):
    model.train()
    optimizer.zero_grad()
    for sequence_1, sequence_2, binary_label, label_1, label_2 in fusion_out_train:  # More descriptive names

        # train_embeddings = []
        # train_embeddings =torch.tensor(train_embeddings)
        # with torch.no_grad():
        #     for batch in dataloader:
        #         train_outputs = model(batch)
        #         train_embeddings = train_embeddings+train_outputs[-1]

        # train_embeddings

        # gmm = GaussianMixture(n_components=21)
        # gmm.fit(train_embeddings)
        # cluster_centers = gmm.means_

        class_output_1 = model(sequence_1)
        class_output_2 = model(sequence_2)

        contrastive_loss = contrastive_loss_fn(class_output_1, class_output_2, binary_label)

        class_loss_1 = criterion(class_output_1, label_1)
        class_loss_2 = criterion(class_output_2, label_2)
        total_loss = contrastive_loss + class_loss_1 + class_loss_2
        total_loss.backward()
        optimizer.step()
        print("1")
torch.save(model.state_dict(), 'duibi_model.pth')