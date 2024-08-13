


def normalize(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sparse.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)

def load_graph(feature_edges, n):
    fedges = np.array(list(feature_edges), dtype=np.int32).reshape(feature_edges.shape)
    fadj = sparse.coo_matrix((np.ones(fedges.shape[0]), (fedges[:, 0], fedges[:, 1])), shape=(n, n),
                             dtype=np.float32)
    fadj = fadj + fadj.T.multiply(fadj.T > fadj) - fadj.multiply(fadj.T > fadj)
    nfadj = normalize(fadj + sparse.eye(fadj.shape[0]))
    nfadj = sparse_mx_to_torch_sparse_tensor(nfadj)

    return nfadj

def constructure_graph(dateset, h1, h2, task="dti", aug=False):
    feature = torch.cat((h1[dateset[:, :1]], h2[dateset[:, 1:2]]), dim=2)
    feature = feature.squeeze(1)
    edge = np.loadtxt(f"{task}edge.txt", dtype=int)

    # for i in range(dateset.shape[0]):
    #     for j in range(i, dateset.shape[0]):
    #         if dateset[i][0] == dateset[j][0] or dateset[i][1] == dateset[j][1]:
    #             edge.append([i, j])
    # fedge = np.array(generate_knn(feature.cpu().detach().numpy()))

    if aug:
        edge_aug = aug_random_edge(np.array(edge))
        edge_aug = load_graph(np.array(edge_aug), dateset.shape[0])
        edge = load_graph(np.array(edge), dateset.shape[0])

        feature_aug = aug_random_mask(feature)
        return edge, feature, edge_aug, feature_aug
    edge = load_graph(np.array(edge), dateset.shape[0])
    return edge, feature

class GCN(nn.Module):
    def __init__(self, nfeat, dropout):
        super(GCN, self).__init__()
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


class HMTCL(nn.Module):
    def __init__(self, all_meta_paths, in_size, hidden_size, out_size, dropout):
        super(HMTCL, self).__init__()
        self.HAN_DTI = HAN_DTI(all_meta_paths, in_size, hidden_size, out_size, dropout)
        self.GCN = GCN(256, dropout)
        self.MLP = MLP(128)

    def forward(self, graph, h, cl, dateset_index, data, iftrain=True, d=None, p=None):
        if iftrain:
            d, p= self.HAN_DTI(graph, h[0], h[1])
        edge, feature = constructure_graph(data, d, p)
        feature1 = self.GCN(feature, edge)

        pred1 = self.MLP(feature1[dateset_index])

        if iftrain:
            return pred1, d, p
        return pred1