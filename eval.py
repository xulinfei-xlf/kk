
import pandas as pd
import torch
from dgl import from_networkx
import torch.nn as nn
import torch as th
import torch.nn.functional as F
import networkx as nx

import random
import matplotlib.pyplot as plt
import numpy as np
import tqdm
from sklearn.preprocessing import LabelEncoder
from model import Model, compute_accuracy

from sklearn.utils import class_weight

from sklearn.metrics import confusion_matrix
import warnings
warnings.filterwarnings("ignore")
# from train import sample_num


class Test():
    def __init__(self, model_path, sample_num):
        self.sample_num = sample_num
        self.model_path = model_path
        self.file_path = "data/save_data.csv"  # 传入数据的文件
        self.cankao_file_name = "data/index_maxmin_data.csv"  # 进行归一化操作的文件
        self.dict = {'Application':0, 'Benign':1, 'Botnet':2, 'DrDoS':3,  'Network':4,  'SlowDDoS':5}

    def label_encode(self, df):
            # 检查是否存在'Lable'列
            if 'Label' in df.columns:
                # 读取'Lable'这一列的数据
                lable_data = df['Label']
                label_num = [self.dict[item] for item in lable_data]
                print(f"all data length: {len(label_num)}")
                return torch.tensor(label_num, dtype=torch.long)
            else:
                print("CSV文件中不存在'Label'这一列。")
                return []

    # 进行数据预处理
    def process(self):

        data = pd.read_csv(self.file_path)
        # 采样顺序保持一致
        max_start_index = len(data) - self.sample_num
        start_index =  np.random.randint(0, max_start_index + 1)
        sampled_df = data.iloc[start_index:start_index + self.sample_num]
        # sampled_df = data.sample(n=self.sample_num)

        label_num = self.label_encode(sampled_df)

        sampled_df.Label.value_counts()
        print(f"all class: {len(sampled_df.Label.value_counts())}")
        class_len = len(sampled_df.Label.value_counts())
        le = LabelEncoder()
        le.fit_transform(sampled_df.Label.values)
        print(f"class name: {le.inverse_transform(range(class_len))}\n")

        sampled_df['Label'] = le.transform(sampled_df['Label'])

        cols_to_norm = sorted(list(set(list(sampled_df.iloc[:, 2:].columns)) - set(list(['Label']))))
        features = sampled_df[cols_to_norm]
        sampled_df[cols_to_norm] = self.normalization(features)
        sampled_df['h'] = sampled_df[cols_to_norm].values.tolist()

        sampled_df['edge_id'] = range(len(sampled_df))  # 生成图之前对每条边加唯一标签，防止生成图之后顺序打乱

        G = nx.from_pandas_edgelist(sampled_df, "saddr", "daddr", ['h', 'edge_id'], create_using=nx.MultiDiGraph())

        G = from_networkx(G, edge_attrs=['h', 'edge_id'])
        # 将节点特征初始化为具有相同形状的值为 1 的张量。
        G.ndata['h'] = th.ones(G.number_of_nodes(), G.edata['h'].shape[1])
        G.edata['train_mask'] = th.ones(len(G.edata['h']), dtype=th.bool)
        G.ndata['h'] = th.reshape(G.ndata['h'], (G.ndata['h'].shape[0], 1, G.ndata['h'].shape[1]))
        G.edata['h'] = th.reshape(G.edata['h'], (G.edata['h'].shape[0], 1, G.edata['h'].shape[1]))

        node_features = G.ndata['h']
        edge_features = G.edata['h']
        # edge_label = G.edata['Label']
        train_mask = G.edata['train_mask']
        edge_id = G.edata['edge_id']

        return G, node_features, edge_features, train_mask,  label_num, edge_id

    def normalization(self, features):
        cankao_file_name = self.file_path
        cankao_data = pd.read_csv(cankao_file_name)
        min_data = cankao_data.min()
        max_data = cankao_data.max()
        for index in features.columns:
            features[index][features[index] > max_data[index]] = max_data[index]
            features[index][features[index] < min_data[index]] = min_data[index]
            features[index] = features[index].astype('float64')
            if max_data[index] - min_data[index] < 0.1:
                features[index] = features[index] - min_data[index]
            else:
                features[index] = (features[index] - min_data[index]) / (max_data[index] - min_data[index])

        return features

    def compute_accuracy(self, pred, labels):
        return (pred == labels).float().mean().item()

    def test(self):
        G, node_features, edge_features, train_mask, label_num, edge_id = self.process()

        model = Model(G.ndata['h'].shape[2], 128, G.ndata['h'].shape[2], F.relu, 0.2)
        model.load_state_dict(torch.load(self.model_path))
        # model = torch.load(self.model_path)
        model.eval()

        pred = torch.argmax(model(G, node_features, edge_features), dim=1)

        sorted_indices = torch.argsort(edge_id)
        edge_id_ori = edge_id[sorted_indices]
        pred = pred[sorted_indices]

        if len(label_num) > 0:
            criterion = nn.CrossEntropyLoss()
            loss = criterion(input=pred.to(dtype=torch.float32), target=label_num.to(dtype=torch.float32))
            acc = self.compute_accuracy(pred, label_num)

            print("数字标签 :")
            print(f"pred   : \n{pred}")
            print(f"traget : \n{label_num}\n")

            dict = {0: 'Application', 1: 'Benign', 2: 'Botnet', 3: 'DrDoS', 4: 'Network', 5: 'SlowDDoS'}
            label = [dict[i.item()] for i in label_num]
            pred = [dict[i.item()] for i in pred]
            print("字符标签 :")
            print(f"pred   : \n{pred}")
            print(f"traget : \n{label}\n")

            print(f"eval_loss: {loss.item()*1e-3}   accuracy: {acc}\n")

            cm = confusion_matrix(label, pred, labels=['Application', 'Benign', 'Botnet', 'DrDoS',  'Network', 'SlowDDoS'])
            print(f"混淆矩阵：\n{cm}")

            import seaborn as sns
            # 使用seaborn绘制混淆矩阵的热图
            plt.figure(figsize=(8, 6))  # 设置图形的大小
            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False)
            plt.xlabel('Predicted label')
            plt.ylabel('True label')
            plt.title('Confusion Matrix')
            plt.savefig("Confusion Matrix.png")
            plt.show()

        else:
            print("数字标签 :")
            print(f"pred   : \n{pred}")
            dict = {0: 'Application', 1: 'Benign', 2: 'Botnet', 3: 'DrDoS', 4: 'Network', 5: 'SlowDDoS'}
            pred = [dict[i.item()] for i in pred]
            print("字符标签 :")
            print(f"pred   :\n{pred}")


# 函数入口
if __name__ == '__main__':
    model_path = r'model.pkl'
    test = Test(model_path=model_path, sample_num=300)
    test.test()
