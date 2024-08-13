# -*- coding: utf-8 -*-
from utilsdtiseed import *
import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
from dgl.nn.pytorch import GraphConv
from GCNLayer import *
from dgllife.model.gnn import GCN
# from GCN import GCN
from dgl.nn.pytorch import GATConv
from loss import multihead_contrastive_loss

device = "cuda:0" if torch.cuda.is_available() else "cpu"


class SemanticAttention(nn.Module):
    def __init__(self, in_size, hidden_size=32):
        super(SemanticAttention, self).__init__()
        self.project = nn.Sequential(
            nn.Linear(in_size, hidden_size).apply(init),
            nn.Tanh(),
            nn.Linear(hidden_size, 1, bias=False).apply(init)
        )

    def forward(self, z):
        w = self.project(z).mean(0)
        beta = torch.softmax(w, dim=0)
        beta = beta.expand((z.shape[0],) + beta.shape)
        return (beta * z).sum(1)


class HANLayer(nn.Module):

    def __init__(self, meta_paths, in_size, out_size, layer_num_heads, dropout):

        super(HANLayer, self).__init__()
        self.gat_layers = nn.ModuleList()
        self.gat_layers.append(GraphConv(in_size, out_size, activation=F.relu, allow_zero_in_degree=True).apply(init))
        self.semantic_attention = SemanticAttention(in_size=out_size * layer_num_heads)#128 * 1
        self.meta_paths = list(tuple(meta_path) for meta_path in meta_paths)
        self._cached_graph = None
        self._cached_coalesced_graph = {}

    def forward(self, g, h):
        g = g.to(device)
        h = h.to(device)
        semantic_embeddings = []
        if self._cached_graph is None or self._cached_graph is not g:
            self._cached_graph = g
            self._cached_coalesced_graph.clear()
            for meta_path in self.meta_paths:
                self._cached_coalesced_graph[meta_path] = dgl.metapath_reachable_graph(
                    g, meta_path)
        for i, meta_path in enumerate(self.meta_paths):
            new_g = self._cached_coalesced_graph[meta_path]
            semantic_embeddings.append(self.gat_layers[0](new_g, h).flatten(1))

        semantic_embeddings = torch.stack(semantic_embeddings, dim=1)  # (N, M, D * K)

        return self.semantic_attention(semantic_embeddings)  # (N, D * K)


class HAN(nn.Module):
    def __init__(self, meta_paths, in_size, hidden_size, out_size, dropout, num_heads=1):
        super(HAN, self).__init__()
        self.layers = nn.ModuleList()
        self.predict = nn.Linear(hidden_size * num_heads, out_size, bias=False).apply(init)
        self.layers.append(
            HANLayer(meta_paths, in_size, hidden_size, num_heads, dropout)
        )

    def forward(self, g, h):
        for gnn in self.layers:
            h = gnn(g, h)
        return self.predict(h)


class HAN_DTI(nn.Module):
    def __init__(self, all_meta_paths, in_size, hidden_size, out_size, dropout):
        super(HAN_DTI, self).__init__()
        self.sum_layers = nn.ModuleList()

        for i in range(0, len(all_meta_paths)):
            self.sum_layers.append(
                HAN(all_meta_paths[i], in_size[i], hidden_size[i], out_size[i], dropout))

    def forward(self, s_g, s_h_1, s_h_2):
        h1 = self.sum_layers[0](s_g[0], s_h_1)
        h2 = self.sum_layers[1](s_g[1], s_h_2)
        return h1, h2




class GCN1(nn.Module):
    def __init__(self, nfeat, dropout):
        super(GCN1, self).__init__()
        self.gc1 = GraphConvolution(nfeat, 512)
        self.gc2 = GraphConvolution(512, 256)
        self.gc3 = GraphConvolution(256, 128)
        self.dropout = dropout

    def forward(self, x, adj):
        x = x.to(device)
        adj = adj.to(device)#adj 3846,3846
        x1 = F.relu(self.gc1(x, adj), inplace=True)
        x1 = F.dropout(x1, self.dropout)
        x2 = self.gc2(x1, adj)
        x3 = self.gc3(x2, adj)
        res = x3
        return res


class MolecularGCN(nn.Module):
    def __init__(self, in_feats, dim_embedding=128, padding=True, hidden_feats=None, activation=None):
        super(MolecularGCN, self).__init__()
        self.init_transform = nn.Linear(in_feats, dim_embedding, bias=False)
        if padding:
            with torch.no_grad():
                self.init_transform.weight[-1].fill_(0)
        self.gnn = GCN(in_feats=dim_embedding, hidden_feats=hidden_feats, activation=activation)
        self.output_feats = hidden_feats[-1]

    def forward(self, batch_graph):
        node_feats = batch_graph.ndata.pop('h')
        node_feats = self.init_transform(node_feats)
        node_feats = self.gnn(batch_graph, node_feats)

        batch_size = batch_graph.batch_size
        node_feats = node_feats.view(batch_size, -1, self.output_feats)

        return node_feats

class ProteinCNN(nn.Module):
    def __init__(self, embedding_dim=128, num_filters=[128,128,128], kernel_size=[3,6,9], padding=True):
        super(ProteinCNN, self).__init__()
        if padding:
            self.embedding = nn.Embedding(26, embedding_dim, padding_idx=0)
        else:
            self.embedding = nn.Embedding(26, embedding_dim)
        in_ch = [embedding_dim] + num_filters
        self.in_ch = in_ch[-1]
        kernels = kernel_size
        self.conv1 = nn.Conv1d(in_channels=in_ch[0], out_channels=in_ch[1], kernel_size=kernels[0])
        self.bn1 = nn.BatchNorm1d(in_ch[1])
        self.conv2 = nn.Conv1d(in_channels=in_ch[1], out_channels=in_ch[2], kernel_size=kernels[1])
        self.bn2 = nn.BatchNorm1d(in_ch[2])
        self.conv3 = nn.Conv1d(in_channels=in_ch[2], out_channels=in_ch[3], kernel_size=kernels[2])
        self.bn3 = nn.BatchNorm1d(in_ch[3])

    def forward(self, v):
        v = self.embedding(v.long())
        v = v.transpose(2, 1)
        v = self.bn1(F.relu(self.conv1(v)))
        v = self.bn2(F.relu(self.conv2(v)))
        v = self.bn3(F.relu(self.conv3(v)))
        v = v.view(v.size(0), v.size(2), -1)
        return v

class GCN2(nn.Module):
    def __init__(self, nfeat, dropout):
        super(GCN2, self).__init__()
        self.gc1 = GraphConvolution(nfeat, 256)
        self.gc2 = GraphConvolution(256, 128)
        self.dropout = dropout

    def forward(self, x, adj):
        x = x.to(device)
        adj = adj.to(device)
        x1 = F.relu(self.gc1(x, adj), inplace=True)
        x1 = F.dropout(x1, self.dropout)
        x2 = self.gc2(x1, adj)
        res = x2
        return res

class CL_GCN(nn.Module):
    def __init__(self, nfeat, dropout,alpha = 0.8):
        super(CL_GCN, self).__init__()
        self.gcn1 = GCN2(nfeat, dropout)
        self.gcn2 = GCN2(nfeat, dropout)
        self.tau = 0.5
        self.alpha = alpha

    def forward(self, x1, adj1, x2, adj2, clm):
        z1 = self.gcn1(x1, adj1)
        z2 = self.gcn2(x2, adj2)

        loss = self.alpha * self.sim(z1, z2, clm) + (1 - self.alpha) * self.sim(z2, z1, clm)
        return z1, z2, loss

    def sim(self, z1, z2, clm):
        z1_norm = torch.norm(z1, dim=-1, keepdim=True)
        z2_norm = torch.norm(z2, dim=-1, keepdim=True)
        dot_numerator = torch.mm(z1, z2.t())
        dot_denominator = torch.mm(z1_norm, z2_norm.t())
        sim_matrix = torch.exp(dot_numerator / dot_denominator / self.tau)

        sim_matrix = sim_matrix / (torch.sum(sim_matrix, dim=1).view(-1, 1) + 1e-8)
        sim_matrix = sim_matrix.to(device)

        loss = -torch.log(sim_matrix.mul(clm).sum(dim=-1)).mean()
        return loss

    def mix2(self, z1, z2):
        loss = ((z1 - z2) ** 2).sum() / z1.shape[0]
        return loss

class MLP(nn.Module):
    def __init__(self, nfeat):
        super(MLP, self).__init__()
        self.MLP = nn.Sequential(
            nn.Linear(nfeat, 32, bias=False).apply(init),
            nn.ELU(),
            nn.Linear(32, 2, bias=False),
            nn.LogSoftmax(dim=1))
            # nn.Sigmoid())
    def forward(self, x):
        output = self.MLP(x)
        return output

class MLPDecoder(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, binary=2):
        super(MLPDecoder, self).__init__()
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, out_dim)
        self.bn3 = nn.BatchNorm1d(out_dim)
        self.fc4 = nn.Linear(out_dim, binary)


    def forward(self, x):
        x = self.bn1(F.relu(self.fc1(x)))
        x = self.bn2(F.relu(self.fc2(x)))
        x = self.bn3(F.relu(self.fc3(x)))
        x = self.fc4(x)
        return x

class GAT(nn.Module):
    def __init__(self,
                 # g,
                 num_layers,
                 in_dim,
                 num_hidden,
                 heads,
                 activation,
                 feat_drop,
                 attn_drop,
                 negative_slope):
        super(GAT, self).__init__()
        self.num_layers = num_layers
        self.num_hidden = num_hidden
        self.gat_layers = nn.ModuleList()
        self.activation = activation

        self.gat_layers.append(GATConv(
            in_dim, num_hidden, heads[0],
            feat_drop, attn_drop, negative_slope, False, self.activation))
        # hidden layers
        for l in range(1, num_layers):
            # due to multi-head, the in_dim = num_hidden * num_heads
            self.gat_layers.append(GATConv(
                num_hidden * heads[l - 1], num_hidden, heads[l],
                feat_drop, attn_drop, negative_slope, False, self.activation))

    def forward(self, inputs, g):
        heads = []
        h = inputs
        # get hidden_representation
        for l in range(self.num_layers):
            temp = h.flatten(1)
            h =self.gat_layers[l](g, temp)
        # get heads
        for i in range(h.shape[1]):
            heads.append(h[:, i])
        return heads


class HMTCL(nn.Module):
    def __init__(self, all_meta_paths, in_size, hidden_size, out_size, dropout):
        super(HMTCL, self).__init__()
        drug_in_feats = 75
        drug_embedding = 128
        drug_hidden_feats = [128,128,128]
        protein_emb_dim = 128
        num_filters = [128,128,128]
        kernel_size = [3,6,9]
        mlp_in_dim = 128
        mlp_hidden_dim = 512
        mlp_out_dim = 128
        drug_padding = True
        protein_padding = True
        out_binary = 2
        ban_heads = 2
        num_layers = 1
        in_dim = 512
        num_hidden = 32
        heads = [4]
        activation = F.elu
        feat_drop = 0.2
        attn_drop = 0.2
        negative_slope = 0.2
        self.HAN_DTI = HAN_DTI(all_meta_paths, in_size, hidden_size, out_size, dropout)# , 512, 256, 128
        self.semantic_attention = SemanticAttention(in_size=128)  # 128 * 1
        self.drug_extractor = MolecularGCN(in_feats=drug_in_feats, dim_embedding=drug_embedding,
                                           padding=drug_padding,
                                           hidden_feats=drug_hidden_feats)
        self.protein_extractor = ProteinCNN(protein_emb_dim, num_filters, kernel_size, protein_padding)
        self.GAT = GAT(
                 num_layers=num_layers,
                 in_dim=in_dim,
                 num_hidden=num_hidden,
                 heads=heads,
                 activation=activation,
                 feat_drop=feat_drop,
                 attn_drop=attn_drop,
                 negative_slope=negative_slope)
        self.MLP = MLP(128)


    def forward(self, graph, h, cl, dateset_index, data, v_d, v_p, iftrain=True, d=None, p=None):
        if iftrain:
            d, p= self.HAN_DTI(graph, h[0], h[1])

            v_d1 = []
            for i, v_d in enumerate(v_d):
                v_d = v_d.to(device)
                v_d = self.drug_extractor(v_d)
                v_d1.append(v_d)
            v_d1 = torch.cat(v_d1, dim=0)
            v_d1 = self.semantic_attention(v_d1)

            v_p1 = []
            for i, v_p in enumerate(v_p):
                v_p = v_p.to(device)
                v_p = self.protein_extractor(v_p)
                v_p1.append(v_p)
            v_p1 = torch.cat(v_p1, dim=0)
            v_p1 = self.semantic_attention(v_p1)

            d = torch.cat([d, v_d1], dim=1)
            p = torch.cat([p, v_p1], dim=1)
        f_edge, f_feature = constructure_knngraph(data, d, p)#语义图
        f_feature[f_feature > 0] = 1

        g = dgl.from_scipy(f_edge)
        adj = torch.tensor(f_edge.todense())
        g = g.to(device)
        feature = f_feature.to(device)
        adj = adj.to(device)
        n_edges = g.number_of_edges()
        g = dgl.remove_self_loop(g)
        g = dgl.add_self_loop(g)
        # in_dim = num_feats
        heads = self.GAT(feature, g)
        loss = multihead_contrastive_loss(heads, adj, tau=0.1)
        feature = torch.cat((heads[0],heads[1],heads[2],heads[3]),dim=1)
        pred1 = self.MLP(feature[dateset_index])
        # loss = 0
        if iftrain :
            return pred1, d, p, loss
        return pred1





def init(i):
    if isinstance(i, nn.Linear):
        torch.nn.init.xavier_uniform_(i.weight)
