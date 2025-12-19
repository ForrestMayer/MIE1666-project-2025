import os
import time
from sklearn.model_selection import train_test_split

from torchmetrics.classification import F1Score, Precision, Recall, Accuracy, BinaryAUROC

from torch_geometric.data import (InMemoryDataset, Data)
from torch_geometric.loader import DataLoader
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import gc

# import GNN architectures from gnn_models folder
from gnn_models.EdgeConv.mip_bipartite_class import SimpleNet as EdgeConv
from gnn_models.EdgeConv.mip_bipartite_simple_class import SimpleNet as EdgeConvSimple

from gnn_models.GIN.mip_bipartite_class import SimpleNet as GIN
from gnn_models.GIN.mip_bipartite_simple_class import SimpleNet as GINSimple

from gnn_models.Sage.mip_bipartite_class import SimpleNet as Sage
from gnn_models.Sage.mip_bipartite_simple_class import SimpleNet as SageSimple

# sanity check
if torch.cuda.is_available():
    print("cuda")



class MyData(Data):
    def __inc__(self, key, value, store=None):
        if key == 'edge_index_var':
            return torch.tensor([self.num_nodes_var, self.num_nodes_con]).view(2, 1)
        elif key == 'edge_index_con':
            return torch.tensor([self.num_nodes_con, self.num_nodes_var]).view(2, 1)
        elif key == 'index':
            return torch.tensor(self.num_nodes_con)
        elif key == 'index_var':
            return torch.tensor(self.num_nodes_var)
        else:
            return 0

class MyTransform(object):
    def __call__(self, data):
        new_data = MyData()
        for key, item in data:
            new_data[key] = item
        return new_data

class GraphDataset(InMemoryDataset):
    def __init__(self, name, data_range, bias_threshold, ovsersampling_factor,
                 transform=None, pre_transform=None, pre_filter=None):
        self.name = name
        self.data_range = data_range
        self.bias_threshold = bias_threshold
        self.oversampling_factor = oversampling_factor
        super(GraphDataset, self).__init__(transform=transform, pre_transform=pre_transform, pre_filter=pre_filter)

        # load processed data if exists
        if os.path.exists(self.processed_paths[0]):
            self.data, self.slices = torch.load(self.processed_paths[0], weights_only=False)
        else:
            self.process()
            self.data, self.slices = torch.load(self.processed_paths[0], weights_only=False)

    @property
    def raw_file_names(self):
        return [self.name]

    @property
    def processed_file_names(self):
        return [self.name]

    def download(self):
        pass

    def process(self):
        data_list = []
        for i in range(self.data_range[0], self.data_range[1] + 1):
            df = pd.read_csv(f"datagen/node_data/graph_data_{i}.csv", index_col=0)

            num_nodes_var = len(df)
            num_nodes_con = 1

            feat_var = [[row["degree"], 1] for _, row in df.iterrows()]

            budget = int(round(len(df) * 0.1))
            feat_con = [[budget, len(df)]]
            feat_rhs = [[budget]]

            y_real = [row["mean"] for _, row in df.iterrows()]
            y = [0 if val <= self.bias_threshold else 1 for val in y_real]

            minority_indices = [idx for idx, val in enumerate(y) if val == 1]
            factor = self.oversampling_factor

            edge_list_var = [[j, 0] for j in range(num_nodes_var)]
            edge_list_con = [[0, j] for j in range(num_nodes_var)]
            edge_features_var = [[1] for _ in range(num_nodes_var)]
            edge_features_con = [[1] for _ in range(num_nodes_var)]

            new_feat_var = feat_var.copy()
            new_y = y.copy()
            new_y_real = y_real.copy()
            new_edge_list_var = edge_list_var.copy()
            new_edge_list_con = edge_list_con.copy()
            new_edge_features_var = edge_features_var.copy()
            new_edge_features_con = edge_features_con.copy()

            # oversampling
            for _ in range(factor):
                for idx in minority_indices:
                    new_feat_var.append(feat_var[idx])
                    new_y.append(y[idx])
                    new_y_real.append(y_real[idx])
                    new_edge_list_var.append([len(new_feat_var) - 1, 0])
                    new_edge_list_con.append([0, len(new_feat_var) - 1])
                    new_edge_features_var.append(edge_features_var[idx])
                    new_edge_features_con.append(edge_features_con[idx])

            num_nodes_var = len(new_feat_var)

            data = Data(
                var_node_features=torch.tensor(new_feat_var, dtype=torch.float),
                con_node_features=torch.tensor(feat_con, dtype=torch.float),
                rhs=torch.tensor(feat_rhs, dtype=torch.float),
                edge_index_var=torch.tensor(new_edge_list_var).t().contiguous(),
                edge_index_con=torch.tensor(new_edge_list_con).t().contiguous(),
                edge_features_var=torch.tensor(new_edge_features_var, dtype=torch.float),
                edge_features_con=torch.tensor(new_edge_features_con, dtype=torch.float),
                y=torch.tensor(new_y, dtype=torch.long),
                y_real=torch.tensor(new_y_real, dtype=torch.float),
                num_nodes_var=num_nodes_var,
                num_nodes_con=num_nodes_con,
                index=torch.tensor([0], dtype=torch.long)
            )

            data_list.append(data)

        data, slices = InMemoryDataset.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])

model_map = {
    "EC": EdgeConv,
    "ECS": EdgeConvSimple,
    "GIN": GIN,
    "GINS": GINSimple,
    "SG": Sage,
    "SGS": SageSimple
}

for rep in [0]:
    for bias in [0.0]: #,0.001, 0.1]:
        for m in ["SGS"]: #"SG", "ECS", "GINS", "EC", "GIN",
            for oversampling_factor in [0]: # 1, 2, 3, 4, 5, 6, 7, 8
                    log = []
                    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                    model = model_map[m](hidden=64, num_layers=4, aggr="mean", regression=False).to(device)
                    model_name = f"{m}_{bias}_{rep}_{oversampling_factor}"
                    print(model_name)

                    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
                    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',
                                                                           factor=0.8, patience=10,
                                                                           min_lr=1e-7)

                    bias_threshold = bias
                    batch_size = 10
                    num_epochs = 30

                    train_dataset = GraphDataset(f"train_{bias}_{oversampling_factor}.pt",
                                                 [200, 999], bias_threshold,
                                                 oversampling_factor, transform=MyTransform()).shuffle()
                    test_dataset = GraphDataset(f"test_{bias}.pt",
                                                [0, 199], bias_threshold,
                                                oversampling_factor, transform=MyTransform()).shuffle()

                    train_index, val_index = train_test_split(range(len(train_dataset)), test_size=0.2)
                    val_dataset = train_dataset[val_index].shuffle()
                    train_dataset = train_dataset[train_index].shuffle()
                    test_dataset = test_dataset.shuffle()

                    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True)
                    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True, pin_memory=True)
                    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, pin_memory=True)

                    start = time.time()

                    zero = torch.tensor([0], device=device)
                    one = torch.tensor([1], device=device)

                    metrics = {
                        "f1": F1Score(task="binary").cpu(),
                        "precision": Precision(task="binary").cpu(),
                        "recall": Recall(task="binary").cpu(),
                        "acc": Accuracy(task="binary").cpu(),
                        "auroc": BinaryAUROC().cpu()
                    }

                    def train(epoch):
                        model.train()
                        loss_all = 0

                        for data in train_loader:
                            data = data.to(device)
                            y = torch.where(data.y_real <= bias_threshold, zero, one)
                            optimizer.zero_grad()
                            output = model(data)
                            loss = F.nll_loss(output, y)

                            loss.backward()
                            optimizer.step()
                            loss_all += batch_size * loss.item()

                            del data, y, output, loss

                        return loss_all / len(train_dataset)

                    @torch.no_grad()
                    def test(loader):
                        model.eval()
                        all_preds, all_labels = [], []

                        for data in loader:
                            data = data.to(device)
                            logits = model(data)
                            y = torch.where(data.y_real <= bias_threshold, 0, 1)

                            all_preds.append(logits.argmax(dim=1).cpu())
                            all_labels.append(y.cpu())

                            del data, logits, y

                        all_preds = torch.cat(all_preds)
                        all_labels = torch.cat(all_labels)

                        results = [metrics["acc"](all_preds, all_labels),
                                   metrics["f1"](all_preds, all_labels),
                                   metrics["precision"](all_preds, all_labels),
                                   metrics["recall"](all_preds, all_labels),
                                   metrics["auroc"](all_preds, all_labels)]

                        del all_preds, all_labels
                        torch.cuda.empty_cache()
                        return results

                    best_val = 0.0
                    test_acc = test_f1 = test_pr = test_re = test_biauc = 0.0

                    for epoch in range(1, num_epochs + 1):
                        torch.cuda.empty_cache()
                        gc.collect()

                        train_loss = train(epoch)
                        train_acc, train_f1, train_pr, train_re, train_biauc = test(train_loader)
                        val_acc, val_f1, val_pr, val_re, val_biauc = test(val_loader)

                        scheduler.step(val_acc)
                        lr = scheduler.optimizer.param_groups[0]['lr']

                        if val_acc > best_val:
                            best_val = val_acc
                            test_acc, test_f1, test_pr, test_re, test_biauc = test(test_loader)
                            torch.save(model.state_dict(), f"training_results/{model_name}")

                        log.append([
                            epoch, train_loss,
                            train_acc.item(), train_f1.item(), train_pr.item(), train_re.item(), train_biauc.item(),
                            val_acc.item(), val_f1.item(), val_pr.item(), val_re.item(), val_biauc.item(),
                            best_val.item() if isinstance(best_val, torch.Tensor) else best_val,
                            test_acc.item(), test_f1.item(), test_pr.item(), test_re.item(), test_biauc.item()
                        ])

                        if lr < 1e-6 or epoch == num_epochs:
                            print([model_name, test_acc.item(), test_f1.item(),
                                   test_pr.item(), test_re.item(), test_biauc.item(),
                                   (time.time() - start)/60])
                            np.savetxt(f"training_results/{model_name}.log",
                                       np.array(log), delimiter=",", fmt='%1.5f')
                            break

                    del model, optimizer, scheduler, train_loader, val_loader, test_loader
                    torch.cuda.empty_cache()
                    gc.collect()