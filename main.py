import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.nn as gnn
import numpy as np
import random
from sklearn.metrics import precision_score, recall_score
import pandas as pd

class GCN(torch.nn.Module):
    def __init__(self, feat_dim, num_class):
        super().__init__()
        # self.conv1 = gnn.GCNConv(feat_dim, 16)
        # self.conv2 = gnn.GCNConv(16, 16)
        # self.conv3 = gnn.GCNConv(16, 4)
        self.conv1 = gnn.GATConv(in_channels=feat_dim, out_channels=8, heads=2)
        self.conv2 = gnn.GATConv(in_channels=16, out_channels=4, heads=2)
        self.conv3 = gnn.GATConv(in_channels=8, out_channels=4, heads=1)
        self.classifier = nn.Linear(4, num_class)
        self.dropout = nn.Dropout(p=0.5)

    def forward(self, x, edge_index):
        h = self.conv1(x, edge_index)
        h = h.relu()
        h = self.conv2(h, edge_index)
        h = h.relu()
        h = self.dropout(h)
        h = self.conv3(h, edge_index)
        out = self.classifier(h)
        return out
    
def train(
    model,
    all_feature,
    all_edge_index,
    train_mask,
    train_label,
    valid_mask,
    valid_label,
    criterion,
    optimizer,
    epoch,
    device
):
    criterion = criterion.to(device)
    all_feature, all_edge_index= all_feature.to(device).float(), all_edge_index.to(device)
    train_label, valid_label = train_label.to(device), valid_label.to(device)

    train_loss, train_precision, train_recall = [], [], []
    valid_loss, valid_precision, valid_recall = [], [], []

    for epoch_idx in range(epoch):
        model.train()
        optimizer.zero_grad()
        out = model(all_feature, all_edge_index)

        train_out = out[train_mask]
        loss = criterion(train_out, train_label)
        loss.backward()
        optimizer.step()
        train_loss.append(loss.item())

        train_pred = torch.argmax(train_out, dim=1)
        train_precision.append(precision_score(y_true=train_label.cpu().numpy(), y_pred=train_pred.cpu().numpy()))
        train_recall.append(recall_score(y_true=train_label.cpu().numpy(), y_pred=train_pred.cpu().numpy()))

        model.eval()
        out = model(all_feature, all_edge_index)
        valid_out = out[valid_mask]
        loss = criterion(valid_out, valid_label)
        valid_loss.append(loss.item())
        valid_pred = torch.argmax(valid_out, dim=1)
        valid_precision.append(precision_score(y_true=valid_label.cpu().numpy(), y_pred=valid_pred.cpu().numpy()))
        valid_recall.append(recall_score(y_true=valid_label.cpu().numpy(), y_pred=valid_pred.cpu().numpy()))

        if epoch_idx % 50 == 0:
            print(f"[{epoch_idx}/{epoch}]: train: {sum(train_loss)/len(train_loss):.4f}/{sum(train_precision)/len(train_precision):.4f}/{sum(train_recall)/len(train_recall):.4f} valid_loss: {sum(valid_loss)/len(valid_loss):.4f}/{sum(valid_precision)/len(valid_precision):.4f}/{sum(valid_recall)/len(valid_recall):.4f}")
        
        if epoch_idx % 1000 == 0 and epoch_idx != 0:
            torch.save(model, f"{epoch_idx}.pt")
    return model

if __name__ == "__main__":
    
    seed = 96
    device = torch.device("cuda:0")
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


    # load training dataset
    train_dataset = torch.load("dataset/train_sub-graph_tensor.pt")
    train_edge_index = train_dataset.edge_index
    train_label = train_dataset.label
    train_mask = np.load("dataset/train_mask.npy")

    # split validation dataset
    train_idx = np.arange(train_mask.shape[0])[train_mask]
    train_tup = list(zip(train_idx, train_label))
    random.shuffle(train_tup)
    valid_tup = sorted(train_tup[0:1574], key=lambda x:x[0])
    train_tup = sorted(train_tup[1574:], key=lambda x:x[0])

    train_idx, train_label = [list(t) for t in zip(*train_tup)]
    valid_idx, valid_label = [list(t) for t in zip(*valid_tup)]
    train_label, valid_label = torch.tensor(train_label), torch.tensor(valid_label)

    train_mask = np.zeros(train_mask.shape[0]).astype(bool)
    valid_mask = np.zeros(train_mask.shape[0]).astype(bool)

    valid_mask[valid_idx] = True
    train_mask[train_idx] = True

    # load testing dataset
    test_dataset = torch.load("dataset/test_sub-graph_tensor_noLabel.pt")
    test_edge_index = test_dataset.edge_index
    test_mask = np.load("dataset/test_mask.npy")

    # dataset summary
    print(f"number of train node: {sum(train_mask)} ({sum(train_label)}/{len(train_label)})")
    print(f"number of valid node: {sum(valid_mask)} ({sum(valid_label)}/{len(valid_label)})")
    print(f"number of valid node: {sum(test_mask)}")

    all_feature = train_dataset.feature
    all_edge_index = torch.concat([train_edge_index, test_edge_index], dim=1)

    # build model
    model = GCN(feat_dim=all_feature.shape[1], num_class=2)
    model = model.to(device)
    print(model)

    # train model
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0003, weight_decay=5e-4)
    model = train(
        model=model,
        all_feature=all_feature,
        all_edge_index=all_edge_index,
        train_mask=train_mask,
        train_label=train_label,
        valid_mask=valid_mask,
        valid_label=valid_label,
        criterion=criterion,
        optimizer=optimizer,
        epoch=10000, # best: 5000
        device=device
    )
    torch.save(model, "model_20000.pt")

    # pred
    model = torch.load("model.pt", map_location="cpu")
    model = model.to(device)
    model.eval()
    out = model(all_feature.to(device).float(), all_edge_index.to(device))
    logit = out[test_mask]
    prob = F.softmax(logit, dim=1)
    prob = prob[:, 1].tolist()
    idx = list(np.where(test_mask)[0])
    pd.DataFrame({'node idx': idx, 'node anomaly score': prob}).to_csv("submission.csv", index=False)
    
