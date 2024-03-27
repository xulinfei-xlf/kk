
import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
import warnings
import torch
from dgl import from_networkx
import torch as th
import networkx as nx
import random
import numpy as np

from dgl.data import DGLDataset
from tqdm import tqdm

warnings.filterwarnings("ignore")

class MyDataset(DGLDataset):
    def __init__(self, file_path, data_num, sample_path, sample_num):
        super().__init__(name='custom')
        self.data_num = data_num
        self.sample_num = sample_num
        self.file_path = file_path
        self.sample_path = sample_path
        self.diction = {'Application': 0, 'Benign': 1, 'Botnet': 2, 'DrDoS': 3, 'Network': 4, 'SlowDDoS': 5}
        self.cankao_file_name = "data/index_maxmin_data.csv"
        self.G_map, self.Node_features, self.Edge_features, self.Train_mask, self.Edge_label = self.get_data()


    def label_encode(self, df):
            if 'Label' in df.columns:
                lable_data = df['Label']
                label_num = [self.diction[item] for item in lable_data]
                return torch.tensor(label_num, dtype=torch.long)

            else:
                print("CSV文件中不存在'Label'这一列。")
                return None

    def get_data(self):
        data = pd.read_csv(self.file_path)
        G_map =[]
        Node_features = []
        Edge_features = []
        Train_mask = []
        Class_weights = []
        Edge_label = []

        for i in range(self.data_num):
            print(f"load data... : {i+1}|{self.data_num}")

            # 采样顺序保持一致
            max_start_index = len(data) - self.sample_num
            start_index = np.random.randint(0, max_start_index + 1)
            sampled_df = data.iloc[start_index: start_index+self.sample_num]

            label_num = self.label_encode(sampled_df)

            sampled_df.Label.value_counts()
            le = LabelEncoder()
            le.fit_transform(sampled_df.Label.values) # 唯一性：LabelEncoder识别出所有的唯一标签值，并为这些唯一值排序。默认情况下，排序是按照字母顺序进行的。

            sampled_df['Label'] = le.transform(sampled_df['Label']) # label列转为数字编码

            cols_to_norm = sorted(list(set(list(sampled_df.iloc[:, 2:].columns)) - set(list(['Label']))))  # 获取列名，除label列

            features = sampled_df[cols_to_norm]
            sampled_df[cols_to_norm] = self.normalization(features)

            sampled_df['h'] = sampled_df[cols_to_norm].values.tolist() # 创建新列h

            sampled_df['edge_id'] = range(len(sampled_df))  # 生成图之前对每条边加唯一标签，防止生成图之后顺序打乱

            # "saddr"和"daddr"：分别指定DataFrame中表示边起点和终点的列名
            # ['h', 'Label']：指定作为边属性的其他列名列表。这意味着每条边将携带有关'h'和'Label'的信息
            G = nx.from_pandas_edgelist(sampled_df, "saddr", "daddr", ['h', 'Label', 'edge_id'], create_using=nx.MultiDiGraph())

            # G = G.to_directed() #转为有向图

            G = from_networkx(G, edge_attrs=['h','Label','edge_id'])  # edge_attrs参数指定了需要从NetworkX图转移到新图中的边属性

            # 将节点特征初始化为具有相同形状的值为 1 的张量。
            G.ndata['h'] = th.ones(G.number_of_nodes(), G.edata['h'].shape[1])
            G.edata['train_mask'] = th.ones(len(G.edata['h']), dtype=th.bool)

            G.ndata['h'] = th.reshape(G.ndata['h'], (G.ndata['h'].shape[0], 1, G.ndata['h'].shape[1]))
            G.edata['h'] = th.reshape(G.edata['h'], (G.edata['h'].shape[0], 1, G.edata['h'].shape[1]))


            node_features = G.ndata['h']
            edge_features = G.edata['h']

            edge_label = G.edata['Label']

            # print("edge_label: \n", edge_label)
            # print("label__num: \n", label_num)
            # print("ori____ id: \n", G.edata['edge_id'])
            #
            # sorted_indices = torch.argsort(G.edata['edge_id'])
            # edge_id_ori = G.edata['edge_id'][sorted_indices]
            # edge_label = edge_label[sorted_indices]

            # print("edge_label: \n", edge_id_ori)
            # print("label__num: \n", label_num)
            # print("ori____ id: \n", edge_label)

            # edge= num = {}
            # for i, j in zip(edge_label, label_num):
            #     if i.item() in edge:
            #         edge[i.item()] += 1
            #     else:
            #         edge[i.item()] = 1
            #     if j.item() in num:
            #         num[j.item()] += 1
            #     else:
            #         num[j.item()] = 1
            # print("G   Label: ",edge)
            # print("our Label: ",num)

            #
            # num = 0
            # for i in range(len(label_num)):
            #     if label_num[i] == edge_label[i].item():
            #         num = num + 1
            # print(f"顺序匹配度：{num}/{len(label_num)}")

            train_mask = G.edata['train_mask']

            G_map.append(G)
            Node_features.append(node_features)
            Edge_features.append(edge_features)
            Train_mask.append(train_mask)
            Edge_label.append(edge_label)
            # Class_weights.append(class_weights)

        return G_map, Node_features, Edge_features, Train_mask, Edge_label

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
        # data = data.apply(MinMax, axis = 1)
        return features

    def __getitem__(self, item):
        return self.G_map[item], self.Node_features[item], self.Edge_features[item], \
               self.Train_mask[item], self.Edge_label[item]

    def __len__(self):
        return len(self.G_map)



if __name__ == '__main__':
    file_path = r"data/save_data.csv"
    sample_path = r"data/save_data_sample.csv"
    mydataset = MyDataset(file_path=file_path, data_num=1, sample_path=sample_path, sample_num=50)
    # print(torch.__version__)
