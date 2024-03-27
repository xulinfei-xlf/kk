
import torch.nn as nn
import torch as th
import torch.nn.functional as F
import dgl.function as fn

class MLPPredictor(nn.Module):
    def __init__(self, in_features, out_classes):
        super().__init__()
        self.W = nn.Linear(in_features * 2, out_classes)

    def apply_edges(self, edges):
        h_u = edges.src['h']
        h_v = edges.dst['h']
        # print(f"h_u:{h_u},\nh_v:{h_v}\n")
        score = self.W(th.cat([h_u, h_v], 1))  # 在边预测中，每条边的特征通过将其两个端点的特征拼接得到
        return {'score': score}

    def forward(self, graph, h):
        with graph.local_scope():
            graph.ndata['h'] = h
            graph.apply_edges(self.apply_edges)
            return graph.edata['score']

def compute_accuracy(pred, labels):
    return (pred.argmax(1) == labels).float().mean().item()


class SAGELayer(nn.Module):
    def __init__(self, ndim_in, edims, ndim_out, activation):
        super(SAGELayer, self).__init__()
        ### force to outut fix dimensions
        self.W_msg = nn .Linear(ndim_in + edims, ndim_out)
        ### apply weight
        self.W_apply = nn.Linear(ndim_in + ndim_out, ndim_out)
        self.activation = activation

    def message_func(self, edges):
        return {'m': self.W_msg(th.cat([edges.src['h'], edges.data['h']], 2))}

    def forward(self, g_dgl, nfeats, efeats):
        with g_dgl.local_scope():  # 确保图的更新不会影响到外部的图结构和特征。
            g = g_dgl
            g.ndata['h'] = nfeats
            # print(f"g.ndata['h']: {g.ndata['h']}\n")
            g.edata['h'] = efeats
            # print(f"g.edata['h']: {g.edata['h']}\n")
            # Eq4
            g.update_all(self.message_func, fn.mean('m', 'h_neigh'))  # 通过g.update_all调用message_func来传递消息，并使用fn.mean('m', 'h_neigh')来聚合邻居节点的消息，结果存储在h_neigh中
            # Eq5
            g.ndata['h'] = F.relu(self.W_apply(th.cat([g.ndata['h'], g.ndata['h_neigh']], 2)))  # 节点更新：使用W_apply线性层和ReLU激活函数来更新每个节点的特征，方法是将原始节点特征和聚合后的邻居特征拼接起来，然后通过线性层和激活函数

            return g.ndata['h']


class SAGE(nn.Module):
    def __init__(self, ndim_in, ndim_out, edim, activation, dropout):
        super(SAGE, self).__init__()
        self.layers = nn.ModuleList()
        self.layers.append(SAGELayer(ndim_in, edim, 128, activation))
        # self.layers.append(SAGELayer(128, edim, 256, activation))
        # self.layers.append(SAGELayer(256, edim, 512, activation))
        # self.layers.append(SAGELayer(512, edim, 1024, activation))
        # self.layers.append(SAGELayer(1024, edim, 512, activation))
        # self.layers.append(SAGELayer(512, edim, 256, activation))
        # self.layers.append(SAGELayer(256, edim, 128, activation))
        self.layers.append(SAGELayer(128, edim, ndim_out, activation))
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, g, nfeats, efeats):
        for i, layer in enumerate(self.layers):
            if i != 0:
                nfeats = self.dropout(nfeats)
            nfeats = layer(g, nfeats, efeats)
        return nfeats.sum(1)

class Model(nn.Module):
    def __init__(self, ndim_in, ndim_out, edim, activation, dropout):
        super().__init__()
        self.gnn = SAGE(ndim_in, ndim_out, edim, activation, dropout)
        self.pred = MLPPredictor(ndim_out, 6)

    def forward(self, g, nfeats, efeats):
        h = self.gnn(g, nfeats, efeats)
        return self.pred(g, h)