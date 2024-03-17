import numpy as np
import pandas as pd
import argparse, sys, json, random, time
from tqdm import tqdm
import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GAE, VGAE
from torch.optim.lr_scheduler import ReduceLROnPlateau, StepLR
from sklearn.metrics import roc_auc_score, average_precision_score, accuracy_score, precision_score, recall_score, f1_score, r2_score
from sklearn.linear_model import LogisticRegression
from scipy.stats.stats import pearsonr

from utils import *
from model import *

# define required hyper-parameters here
def init_argparse():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--type', type=str, default="test", help="one of these: feature, comparison, test")
    parser.add_argument('--task', type=str, default="connectivity", help="predict SL connectivity or SL pair")
    parser.add_argument('--data_source', type=str, default="K562", help="which cell line to train and predict")
    parser.add_argument('--threshold', type=float, default=-3, help="threshold of SL determination")

    parser.add_argument('--graph_feats', type=lambda s:[item for item in s.split("%") if item != ""], 
                            default=["pathway","PPI-genetic","PPI-physical"], help="lists of graphs to use.")
    parser.add_argument('--global_tabular_feats', type=lambda s:[item for item in s.split("%") if item != ""], 
                            default=["CCLE_exp","CCLE_ess"], help="list of tabular features")
    parser.add_argument('--cell_specific_feats', type=lambda s:[item for item in s.split("%") if item != ""], 
                            default=["multi_omics"], help="cell specific features")

    parser.add_argument('--cell_specific_flag', type=int, default=1, help="whether to use cell-specific features")
    parser.add_argument('--GT_flag', type=int, default=1, help="whether to include graph transformer")

    parser.add_argument('--LR', type=float, default=0.00005, help="learning rate")
    parser.add_argument('--epochs', type=int, default=1000, help="number of maximum training epochs")
    parser.add_argument('--patience', type=int, default=200, help="patience in early stopping")

    parser.add_argument('--embed_size', type=int, default=32, help="dimension of embeddings: query, key and values")
    parser.add_argument('--hidden_size', type=int, default=64, help="hidden dimension in FFN module")
    parser.add_argument('--heads', type=int, default=4, help="number of heads in transformer")
    parser.add_argument('--n_layers', type=int, default=1, help="number of repeated modules in transformer")

    parser.add_argument('--training_percent', type=float, default=0.7, help="proportion of the SL data (genes) as training set")
    parser.add_argument('--val_percent', type=float, default=0.1, help="proportion of the SL data (genes) as validation set")
    parser.add_argument('--save_results', type=int, default=1, help="whether to save test results into json")
    parser.add_argument('--random_seed', type=int, default=5959, help="control the split of data into train, val and test")
    parser.add_argument('--predict_novel_genes', type=int, default=0, help="whether to predict on novel genes not present in the samples")

    args = parser.parse_args()

    return args


def load_model_data(data, SL_data, task, device):
    '''
    1. move lists of data into device
    2. generate SL connectivity data or SL pair data
    '''

    # moving data to GPU
    temp_edge_index_list = []
    for edge_index in data.edge_index_list:
        temp_edge_index_list.append(edge_index.to(device))

    temp_tabular_feats_list = []
    for feats in data.tabular_feats_list:
        temp_tabular_feats_list.append(feats.to(device))
    
    if len(args.cell_specific_feats)>0 and args.data_source != "synlethdb":
        temp_cell_specific_feats_list = []
        for feats in data.cell_specific_feats_list:
            temp_cell_specific_feats_list.append(feats.to(device))
    else:
        temp_cell_specific_feats_list = None
    
    # generate SL data for training
    if task == "connectivity":
        gene_index = torch.tensor(SL_data["gene"].values, dtype=torch.long, device=device)
    elif task == "pair":
        gene_index = torch.tensor([SL_data['gene1'].values, SL_data['gene2'].values], dtype=torch.long, device=device)
    labels = torch.tensor(SL_data["label"].values, dtype=torch.float, device=device)
    
    return temp_edge_index_list, temp_tabular_feats_list, temp_cell_specific_feats_list, gene_index, labels

def train_model(model, optimizer, data, device, SL_data, task):
    model.train()
    optimizer.zero_grad()

    edge_index_list, tabular_feats_list, cell_specific_feats_list, gene_index, labels = load_model_data(data, SL_data, 
                                                                                                            task, device)
    preds = model(edge_index_list, tabular_feats_list, cell_specific_feats_list, gene_index)

    if task == "connectivity":
        loss = F.mse_loss(preds, labels)
    elif task == "pair":
        loss = F.binary_cross_entropy_with_logits(preds, labels)

    loss.backward()
    optimizer.step()

    return float(loss)

def test_model(model, optimizer, data, device, SL_data, task):
    model.eval()
    results = {}
    pred_dict = {}

    edge_index_list, tabular_feats_list, cell_specific_feats_list, gene_index, labels = load_model_data(data, SL_data, 
                                                                                                            task, device)
    
    with torch.no_grad():
        preds = model(edge_index_list, tabular_feats_list, cell_specific_feats_list, gene_index)
        
        if task == "connectivity":
            loss = F.mse_loss(preds, labels)
        elif task == "pair":
            loss = F.binary_cross_entropy_with_logits(preds, labels)
    
    if task == "connectivity":
        pred_dict = dict(zip(gene_index.cpu().numpy().tolist(), preds.cpu().numpy().tolist()))
        results["test_loss"] = float(loss)
        results["r2"] = r2_score(labels.cpu().numpy(), preds.cpu().numpy())
        results["corr"] = pearsonr(labels.cpu().numpy(), preds.cpu().numpy())
    elif task == "pair":
        probs = preds.sigmoid()
        results = evaluate_performance(labels.cpu().numpy(), probs.cpu().numpy())
        results['true_label'] = labels.cpu().numpy().tolist()
        results['pred_score'] = probs.cpu().numpy().tolist()

    return float(loss), results, pred_dict

@torch.no_grad()
def predict_oos(model, optimizer, data, device, SL_data, task):
    pass

if __name__ == "__main__":
    args = init_argparse()
    print(args)
    
    # load SL data and features
    # SL_potential_ stores the number of SL parterns for each gene (SL connectivity)
    # SL_pair_ stores paired data
    data, SL_potential_train, SL_potential_val, SL_potential_test, SL_pair_train, SL_pair_val, \
                                         SL_pair_test, gene_mapping = load_data(args.data_source, 
                                                                                args.threshold, 
                                                                                args.graph_feats, 
                                                                                args.global_tabular_feats,
                                                                                args.cell_specific_feats, 
                                                                                args.training_percent,
                                                                                args.val_percent,
                                                                                args.random_seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print("#####################################")
    print("Number of total genes: {}".format(len(gene_mapping)))
    print("number of train, val, test gene samples:", SL_potential_train.shape[0], SL_potential_val.shape[0], SL_potential_test.shape[0])
    print("number of train, val, test gene pair samples:", SL_pair_train.shape[0], SL_pair_val.shape[0], SL_pair_test.shape[0])
    print("#####################################\n")

    # load model
    num_CCLE = len(args.global_tabular_feats)
    num_network = len(args.graph_feats)
    model = MLEC_iSL(args.hidden_size, args.embed_size, args.heads, args.n_layers,
                num_CCLE=num_CCLE,
                num_network=num_network, 
                cell_specific_flag=args.cell_specific_flag,
                GT_flag=args.GT_flag,
                task=args.task).to(device)
    
    #print(model)

    print("#####################################")
    print("Number of parameters:", sum(p.numel() for p in model.parameters() if p.requires_grad))
    print("#####################################")
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.LR)

    train_losses = []
    valid_losses = []
    
    # initialize the early_stopping object
    random_key = random.randint(1,100000000)
    checkpoint_path = "checkpoint/{}.pt".format(str(random_key))
    early_stopping = EarlyStopping(patience=args.patience, verbose=True, reverse=True, path=checkpoint_path)

    print("Training SL {} prediction model".format(args.task))

    if args.task == "connectivity":
        # predict SL connectivity as a regression task, then use a logistic regression for SL pair prediction
        for epoch in range(1, args.epochs + 1):
            train_loss = train_model(model, optimizer, data, device, SL_potential_train, args.task)
            train_losses.append(train_loss)
            val_loss, results, val_pred_dict = test_model(model, optimizer, data, device, SL_potential_val, args.task)
            valid_losses.append(val_loss)
            print("Epoch: {:03d}, loss: {:.4f}, val_loss: {:.4f}".format(epoch, train_loss, val_loss))
            
            early_stopping(val_loss, model)
            if early_stopping.early_stop:
                print("Early Stopping!!!")
                break
    
        # load the last checkpoint with the best model
        model.load_state_dict(torch.load(checkpoint_path))

        print("Training LR model for SL pair prediction")

        # get predicted SL potential (SL connectivity) for genes in the test set
        test_loss, test_results, test_pred_dict = test_model(model, optimizer, data, device, SL_potential_test, args.task)
        train_SL_potential_dict = dict(zip(SL_potential_train["gene"].values.tolist(), SL_potential_train["label"].values.tolist()))
        pred_SL_potential_dict = test_pred_dict

        # generate training and test data for logistic regression model
        # input features are SL connectivity of two genes in the pair
        train_X, train_y, test_X, test_y = generate_SL_pair_features(train_SL_potential_dict, pred_SL_potential_dict, 
                                                                                        SL_pair_train, SL_pair_test)
        
        clf = LogisticRegression(class_weight="balanced")
        clf.fit(train_X, train_y)
        # prediction results on test data
        pred_score = clf.predict_proba(test_X)[:,1]

        results = evaluate_performance(test_y, pred_score)
        results["SL_connectivity_test_loss"] = test_results["test_loss"]
        results["SL_connectivity_r2"] = test_results["r2"]
        results["SL_connectivity_corr"] = test_results["corr"]
        print(results)
        results["pred_score"] = pred_score.tolist()
        results["true_label"] = test_y.tolist()
        results["connectivity_label"] = SL_potential_test["label"].values.tolist()
        results["connectivity_pred"] = list(test_pred_dict.values())
    elif args.task == "pair":
        # directly train a model for SL prediction as classification task
        for epoch in range(1, args.epochs + 1):
            train_loss = train_model(model, optimizer, data, device, SL_pair_train, args.task)
            train_losses.append(train_loss)
            val_loss, results, _ = test_model(model, optimizer, data, device, SL_pair_val, args.task)
            valid_losses.append(val_loss)
            print('Epoch: {:03d}, loss: {:.4f}, val_loss: {:.4f}, AUC: {:.4f}, AP: {:.4f}, Ranking: {:.4f}'.format(epoch, 
                                            train_loss, val_loss, results['AUC'], results['AUPR'], results['precision@5']))

            early_stopping(-results["AUPR"], model)
            if early_stopping.early_stop:
                print("Early Stopping!!!")
                break
        
        model.load_state_dict(torch.load(checkpoint_path))
        test_loss, results, _ = test_model(model, optimizer, data, device, SL_pair_test, args.task)
        print("\ntest result:")
        print('test_loss: {:.4f}, AUC: {:.4f}, AP: {:.4f}, Ranking: {:.4f}'.format(val_loss, results['AUC'], 
                                                                    results['AUPR'], results['precision@5']))
        
    if args.save_results:
        save_dict = {**vars(args), **results, "train_loss":train_losses, "val_loss":valid_losses}
        with open("../results/{}_{}_{}.json".format(args.task, args.type, str(random_key)),"w") as f:
            json.dump(save_dict, f)
