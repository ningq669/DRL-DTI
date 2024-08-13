# import dgl
import pandas as pd
from utilsdtiseed import *
# import utilsdtiseed
from modeltestdtiseed import *
from tqdm import tqdm
import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
import dgl.nn.pytorch as dglnn
from sklearn.metrics import roc_auc_score, f1_score
import warnings
import os
from sklearn.metrics.pairwise import cosine_similarity as cos
from dataloader import DTIDataset_Drug,DTIDataset_Protein
from torch.utils.data import DataLoader
from utils import graph_collate_func1, graph_collate_func2


os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
warnings.filterwarnings("ignore")
seed = 47
args = setup(default_configure,seed)
s = 47
in_size = 512
hidden_size = 256
out_size = 128
dropout = 0.5
lr = 0.001
weight_decay = 1e-7
epochs = 500
cl_loss_co = 1
reg_loss_co = 0.0001
fold = 0
dir = "../modelSave"

args['device'] = "cuda:0" if torch.cuda.is_available() else "cpu"

result_best_acc = []
result_best_roc = []
result_best_pr = []
result_best_f1 = []
result_best_recall = []
result_best_precision = []

params1 = {'batch_size': 64, 'shuffle': False, 'num_workers': 0,
              'drop_last': False, 'collate_fn': graph_collate_func1}
drug = pd.read_csv('./data/zeng/drug.txt')
v_d = DTIDataset_Drug(drug.index.values, drug)
v_d = DataLoader(v_d, **params1)

params2 = {'batch_size': 64, 'shuffle': False, 'num_workers': 0,
              'drop_last': False, 'collate_fn': graph_collate_func2}
pro = pd.read_csv('./data/zeng/protein.txt')
v_p = DTIDataset_Protein(pro.index.values, pro)
v_p = DataLoader(v_p, **params2)

for name in ["zheng"]:
# for name in ["heter","Es","GPCRs","ICs","Ns","zheng"]:
    dtidata, graph, num, all_meta_paths = load_dataset(name)
    # dataName heter Es GPCRs ICs Ns zheng
    dti_label = torch.tensor(dtidata[:, 2:3]).to(args['device'])

    hd = torch.randn((num[0], in_size))
    hp = torch.randn((num[1], in_size))
    features_d = hd.to(args['device'])
    features_p = hp.to(args['device'])

    node_feature = [features_d, features_p]
    dti_cl = get_clGraph(dtidata, "dti").to(args['device'])
    cl = dti_cl
    data = dtidata
    label = dti_label

    def main(tr,te,seed):
        all_acc = []
        all_roc = []
        all_pr = []
        all_f1 = []
        all_recall = []
        all_precision = []
        for i in tqdm(range(len(tr))):
            # f = open(f"{i}foldtrain.txt","w",encoding="utf-8")
            train_index = tr[i]
            # print(train_index.shape)8000
            # for train_index_one in train_index:
            #     f.write(f"{train_index_one}\n")
            test_index = te[i]
            # print(test_index.shape)2000
            # f = open(f"{i}foldtest.txt","w",encoding="utf-8")
            # for train_index_one in test_index:
            #     f.write(f"{train_index_one}\n")

            if not os.path.isdir(f"{dir}"):
                os.makedirs(f"{dir}")

            model = HMTCL(
                all_meta_paths=all_meta_paths,
                in_size=[hd.shape[1], hp.shape[1]],
                hidden_size=[hidden_size, hidden_size],
                out_size=[out_size, out_size],
                dropout=dropout,
            ).to(args['device'])
            # model.load_state_dict(torch.load(f'fold{i}.pkl'))
            optim = torch.optim.Adam(lr=lr, weight_decay=weight_decay, params=model.parameters())
            best_acc = 0
            best_roc = 0
            best_pr = 0
            best_f1 = 0
            best_recall = 0
            best_precision = 0
            for epoch in range(epochs):
                loss, train_acc, task1_roc, acc, task1_roc1, task1_pr, task1_f1, task1_recall, task1_precision = train(model, optim, train_index, test_index, epoch, i)
                if acc > best_acc:
                    best_acc = acc
                if task1_roc1 > best_roc:
                    best_roc = task1_roc1
                if task1_pr > best_pr:
                    best_pr = task1_pr
                if task1_f1 > best_f1:
                    best_f1 = task1_f1
                if task1_recall > best_recall:
                    best_recall = task1_recall
                if task1_precision > best_precision:
                    best_precision = task1_precision
                    # torch.save(obj=model.state_dict(), f=f"{dir}/net.pth")
                # print("Epoch {:04d} | TrainLoss {:.4f} ".format(epoch + 1, loss.item()))
            # torch.save(model.state_dict(), f'fold{i}.pkl')
            all_acc.append(best_acc)
            all_roc.append(best_roc)
            all_pr.append(best_pr)
            all_f1.append(best_f1)
            all_recall.append(best_recall)
            all_precision.append(best_precision)
            result_best_acc.append(best_acc)
            result_best_roc.append(best_roc)
            result_best_pr.append(best_pr)
            result_best_f1.append(best_f1)
            result_best_recall.append(best_recall)
            result_best_precision.append(best_precision)
            print(f"fold{i} acc is {best_acc:.4f} auroc is {best_roc:.4f} aupr is {best_pr:.4f} f1 is {best_f1:.4f} recall is {best_recall:.4f} precision is {best_precision:.4f}")

        print("best_acc:", result_best_acc)
        print("best_roc:", result_best_roc)
        print("best_pr:", result_best_pr)
        print("best_f1:", result_best_f1)
        print("best_recall:", result_best_recall)
        print("best_precision:", result_best_precision)
        print(f"{name}, {sum(all_acc) / len(all_acc):.4f}, {sum(all_roc) / len(all_roc):.4f} ,{sum(all_pr) / len(all_pr):.4f}, {sum(all_f1) / len(all_f1):.4f}, {sum(all_recall) / len(all_recall):.4f}, {sum(all_precision) / len(all_precision):.4f}")

    def train(model, optim, train_index, test_index, epoch, fold):
        model.train()
        out, d, p , loss1 = model(graph, node_feature, cl, train_index, data, v_d, v_p)

        train_acc = (out.argmax(dim=1) == label[train_index].reshape(-1)).sum(dtype=float) / len(train_index)

        task1_roc = get_roc(out, label[train_index])

        reg = get_L2reg(model.parameters())

        loss = F.nll_loss(out, label[train_index].reshape(-1).long()) + loss1 + reg_loss_co * reg

        optim.zero_grad()
        loss.backward()
        optim.step()
        print(f"{epoch} epoch loss  {loss:.4f} train is acc  {train_acc:.4f}, task1 roc is {task1_roc:.4f},")
        te_acc, te_task1_roc1, te_task1_pr, te_task1_f1, te_task1_recall, te_task1_precision = main_test(model, d, p, test_index, epoch, fold)

        return loss.item(), train_acc, task1_roc, te_acc, te_task1_roc1, te_task1_pr, te_task1_f1, te_task1_recall, te_task1_precision



    def main_test(model, d, p, test_index ,epoch,fold):
        model.eval()

        out = model(graph, node_feature, cl, test_index, data, v_d, v_p, iftrain=False, d=d, p=p)
        acc1 = (out.argmax(dim=1) == label[test_index].reshape(-1)).sum(dtype=float) / len(test_index)

        task_roc = get_roc(out, label[test_index])

        task_pr = get_pr(out,label[test_index])

        task_f1 = get_f1score(out,label[test_index])

        task_recall = get_recall(out,label[test_index])

        task_precision = get_precision(out,label[test_index])
        # if epoch == 499:
        #     f = open(f"{fold}out.txt","w",encoding="utf-8")
        #     for o in  (out.argmax(dim=1) == label[test_index].reshape(-1)):
        #         f.write(f"{o}\n")
        #     f.close()

        print(f"{epoch}                     test  is acc  {acc1:.4f}, task  roc is {task_roc:.4f},")

        return acc1, task_roc, task_pr, task_f1, task_recall, task_precision

    # train_indeces,test_indeces = get_cross(dtidata)
    train_indeces1 = [[], [], [], [], []]
    test_indeces1 = [[], [], [], [], []]
    for i in range(5):
        with open(f'{i}foldtest.txt', 'r') as file:
            for line in file:
                line = line.strip()
                test_indeces1[i].append(int(line))
    for i in range(5):
        with open(f'{i}foldtrain.txt', 'r') as file:
            for line in file:
                line = line.strip()
                train_indeces1[i].append(int(line))

    main(train_indeces1,test_indeces1,seed)
