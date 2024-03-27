import pandas as pd
import torch
from dgl import from_networkx
import torch.nn as nn
import torch as th
import torch.nn.functional as F
import networkx as nx
import random
import matplotlib.pyplot as plt
import tqdm
from sklearn.preprocessing import LabelEncoder
import utils
from model import Model,compute_accuracy
from dgl.dataloading import GraphDataLoader
import torch.optim as optim
import os


sample_num = 10000  # 从表格里面采样行数


class ModelRun():
    def __init__(self):
        self.epochs = 30000
        self.sava_frequency = 1
        self.sample_num = sample_num
        self.data_num = 100    #训练集数据量大小
        self.val_data_num = 1
        self.batch_size = 1

        self.lr = 1e-2
        self.final_lr = 1e-5
        self.warmup_step = 1000
        self.total_step = 30000

        self.dict = {'Application': 0, 'Benign': 1, 'Botnet': 2, 'DrDoS': 3, 'Network': 4, 'SlowDDoS': 5}

        self.file_path = "data/save_data.csv"
        self.sample_path = "data/save_data_sample.csv"
        self.cankao_file_name = "data/index_maxmin_data.csv"

        print("-" * 20 + "加载数据中..." + "-" * 20)
        # 训练集
        self.mydataset = utils.MyDataset(file_path=self.file_path, data_num=self.data_num,
                                         sample_path=self.sample_path,sample_num=self.sample_num)

        self.MyDataloader = GraphDataLoader(self.mydataset,
                                            batch_size=self.batch_size,
                                            shuffle=True,
                                            # num_workers= 2,
                                            collate_fn=self.my_collate_fn)
        # 验证集
        self.val_dataset = utils.MyDataset(file_path=self.file_path, data_num=self.val_data_num,
                                           sample_path=self.sample_path,sample_num=self.sample_num)

        self.val_Dataloader = GraphDataLoader(self.val_dataset,
                                            batch_size=self.batch_size,
                                            shuffle=True,
                                            # num_workers= 2,
                                            collate_fn=self.my_collate_fn)

        print("-" * 20 + "数据加载完毕" + "-" * 20 + '\n')

        self.G = self.Data()
        self.model = Model(self.G.ndata['h'].shape[2], 128, self.G.ndata['h'].shape[2], F.relu, 0.2)

        # 检测是否继续训练
        model_path = r'model.pkl'
        if os.path.exists(model_path):
            # 文件存在，可以继续训练
            print(f"Model file '{model_path}' exists. You can continue training.")
            self.model.load_state_dict(torch.load(model_path))

        else:
            # 文件不存在，重新训练模型
            print(f"Model file '{model_path}' does not exist. You might need to train a new model.")

        self.opt = th.optim.Adam(self.model.parameters(), lr=self.lr)

        self.tra_loss = []
        self.tra_acc = []
        self.Val_loss = []
        self.Val_acc = []
        self.all_batch = []

        self.warmup_scheduler = optim.lr_scheduler.LambdaLR(self.opt, lr_lambda=self.lr_lambda)



    def lr_lambda(self, step):
        if step < self.warmup_step:
            return step / self.warmup_step
        else: # 线性衰减至final_lr
            decay_rate = (self.lr - self.final_lr) / (self.total_step - self.warmup_step)
            return max((self.lr - decay_rate * (step - self.warmup_step)) / self.lr, self.final_lr / self.lr)

    def save_batch_data(self):
        for batch in self.MyDataloader:
            self.all_batch.append(batch)
        torch.save(self.all_batch, 'batch_data.pth')  # 保存batch数据

    def normalization(self,features):
        cankao_data = pd.read_csv(self.cankao_file_name)
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

    def Data(self):
        data = pd.read_csv(self.file_path)
        sampled = data.sample(n=self.sample_num)
        # 如果你想将这5000行数据导出到一个新的CSV文件
        sampled.to_csv(self.sample_path, index=False)
        sampled_df = pd.read_csv(self.sample_path)

        sampled_df.Label.value_counts()
        le = LabelEncoder()
        le.fit_transform(sampled_df.Label.values)
        sampled_df['Label'] = le.transform(sampled_df['Label'])

        label_ground_truth = sampled_df[["saddr", "daddr", "Label"]]
        cols_to_norm = list(set(list(sampled_df.iloc[:, 2:].columns)) - set(list(['Label'])))
        features = sampled_df[cols_to_norm]
        sampled_df[cols_to_norm] = self.normalization(features)

        sampled_df['h'] = sampled_df[cols_to_norm].values.tolist()
        G = nx.from_pandas_edgelist(sampled_df, "saddr", "daddr", ['h', 'Label'], create_using=nx.MultiDiGraph())
        # G = G.to_directed()
        G = from_networkx(G, edge_attrs=['h', 'Label'])
        # 将节点特征初始化为具有相同形状的值为 1 的张量。
        G.ndata['h'] = th.ones(G.number_of_nodes(), G.edata['h'].shape[1])
        G.edata['train_mask'] = th.ones(len(G.edata['h']), dtype=th.bool)
        G.ndata['h'] = th.reshape(G.ndata['h'], (G.ndata['h'].shape[0], 1, G.ndata['h'].shape[1]))
        G.edata['h'] = th.reshape(G.edata['h'], (G.edata['h'].shape[0], 1, G.edata['h'].shape[1]))

        return G

    def my_collate_fn(self,data):
        # TODO: Implement your function
        # But I guess in your case it should be:
        return tuple(data)

    # 打印模型
    def print_model_details(self):
        total_params = 0
        for name, layer in self.model.named_children():
            layer_params = sum(p.numel() for p in layer.parameters())
            total_params += layer_params
            print(f"Layer: {name} | Structure: {str(layer)} | Params: {layer_params}")
        print(f"Total parameters in the model: {total_params}")

    def compute_accuracy(self, pred, labels):
        return (pred.argmax(1) == labels).float().mean().item()

    def train(self):
        print("-" * 20 + "开始训练..." + "-" * 20)
        step = 0
        for epoch in range(self.epochs):
            for item in self.MyDataloader:
                self.model.train()
                (items,) = item
                # print("items: ",item)
                g_map, node_features, edge_features, train_mask, edge_label = items

                pred = self.model(g_map, node_features, edge_features)

                criterion = nn.CrossEntropyLoss()
                train_loss = criterion(input=pred[train_mask].to(dtype=torch.float32) ,target=edge_label[train_mask].to(dtype=torch.long))

                self.opt.zero_grad()
                train_loss.backward()
                self.opt.step()
                self.warmup_scheduler.step()

                acc = self.compute_accuracy(pred=pred, labels=edge_label)
                step += 1

                self.model.eval()
                for item in self.val_Dataloader:
                    (items,) = item
                    g_map, node_features, edge_features, train_mask, edge_label = items
                    pred = self.model(g_map, node_features, edge_features)
                    criterion = nn.CrossEntropyLoss()
                    val_loss = criterion(input=pred.to(dtype=torch.float32) ,target=edge_label.to(dtype=torch.long))
                    val_acc = self.compute_accuracy(pred=pred, labels=edge_label)

                print(f"step: {step}, train_loss: {train_loss}, acc: {acc}, val_loss: {val_loss}, val_acc: {val_acc},  lr: {self.opt.state_dict()['param_groups'][0]['lr']}")

            if (epoch+1) % self.sava_frequency == 0:
                torch.save(self.model.state_dict(), 'model.pkl')

            self.tra_loss.append(train_loss.item())
            self.tra_acc.append(acc)
            self.Val_loss.append(val_loss.item())
            self.Val_acc.append(val_acc)

        self.mk_graph()

    def mk_graph(self):
        plt.figure()
        plt.plot(self.tra_loss)
        plt.title('Train_loss',fontsize=10)
        plt.ylabel('tra_loss',fontsize=10)
        plt.xlabel('epochs',fontsize=10)
        plt.savefig("train_loss.png")

        plt.figure()
        plt.plot(self.tra_acc)
        plt.title('Train_acc',fontsize=10)
        plt.ylabel('tra_acc',fontsize=10)
        plt.xlabel('epochs',fontsize=10)
        plt.savefig("train_acc.png")

        plt.figure()
        plt.plot(self.Val_loss)
        plt.title('Val_loss',fontsize=10)
        plt.ylabel('val_loss',fontsize=10)
        plt.xlabel('epochs',fontsize=10)
        plt.savefig("val_loss.png")

        plt.figure()
        plt.plot(self.Val_acc)
        plt.title('Val_acc',fontsize=10)
        plt.ylabel('val_acc',fontsize=10)
        plt.xlabel('epochs',fontsize=10)
        plt.savefig("val_acc.png")
        plt.show()

if __name__ == '__main__':
    modelrun = ModelRun()
    modelrun.train()