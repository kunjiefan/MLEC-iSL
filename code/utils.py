import numpy as np
import pandas as pd
from scipy import stats
import networkx as nx
import os, random, time
import pickle, json,itertools
import torch
from torch_geometric.data import Data
from torch_geometric.utils import train_test_split_edges
from torch_geometric.nn import Node2Vec
from sklearn.preprocessing import scale
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.preprocessing import StandardScaler
from networkx.generators.random_graphs import fast_gnp_random_graph,gnp_random_graph
from collections import Counter,defaultdict


def generate_unique_samples(cell_name):
    # process raw data
    if cell_name == "K562":
        df = pd.read_table("../data/raw_SL_experiments/K562/CRISPRi_K562_replicateAverage_GIscores_genes_inclnegs.txt", index_col=0)
    else:
        df = pd.read_table("../data/raw_SL_experiments/Jurkat/CRISPRi_Jurkat_emap_gene_filt.txt", index_col=0)
    
    # remove negative samples
    num_genes = df.shape[0]
    df = df.iloc[:(num_genes-1),:(num_genes-1)]
    
    num_genes = df.shape[0]
    
    # don't consider self interactions
    GI_matrix = df.values
    GI_indexs = np.triu_indices(num_genes, k=1)
    GI_values = GI_matrix[GI_indexs]
    
    row_indexs = GI_indexs[0]
    col_indexs = GI_indexs[1]
    row_genes = df.index[row_indexs]
    col_genes = df.columns[col_indexs]
    
    all_samples = pd.DataFrame({'gene1':list(row_genes),'gene2':list(col_genes),'GI_scores':GI_values})
    all_samples.to_csv("../data/{}_GI_scores.csv".format(cell_name), index=False)
    
    return all_samples

def load_SL_data(cell_name, threshold=-3):
    if cell_name != "synlethdb":
        data = pd.read_csv("../data/{}_GI_scores.csv".format(cell_name))
        data['label'] = (data['GI_scores'] <= threshold).astype(int)
        all_genes = list(set(np.unique(data['gene1'])) | set(np.unique(data['gene2'])))
    else:
        data = pd.read_csv("../data/SynLethDB_SL.csv")
        data.rename(columns={"gene_a.name":"gene1", "gene_b.name":"gene2"}, inplace=True)
        data = data[["gene1","gene2"]]
        data["label"] = 1
        all_genes = list(set(np.unique(data['gene1'])) | set(np.unique(data['gene2'])))
    
    return data, all_genes

def generate_SL_potential_data(cell_name, SL_data, SL_genes, label_type="regression"):
    '''
    generate gene SL potential (SL connectivity) data based on positive SL pairs
    '''
    if cell_name != "synlethdb":
        SL_data_dup = SL_data.reindex(columns=['gene2','gene1','GI_scores','label'])
        SL_data_dup.columns = ["gene1", "gene2", "GI_scores","label"]
        SL_data = SL_data.append(SL_data_dup)
        SL_potential_score = SL_data.groupby("gene1").agg({"GI_scores":"median","label":"sum"}).reset_index()
        SL_potential_score.columns = ["gene", "median_score", "num_SL_partner"]
        
        # define SL potential to four levels: no potential, little potential, medium potential, high potential
        if label_type == "count":
            # use number of SL partners to categorize
            SL_potential_score["label"] = pd.cut(SL_potential_score.num_SL_partner, bins=np.array([0, 3, 10, 30, 100]), labels=np.array([0,1,2,3]))
        elif label_type == "median_score":
            # use median SL score to categorize
            SL_potential_score["label"] = pd.cut(SL_potential_score.median_score, bins=np.array([-2,-0.5,0,0.5,2]), labels=np.array([3,2,1,0]))
        elif label_type == "regression":
            # use the number of SL partners as the target
            SL_potential_score["label"] = SL_potential_score["num_SL_partner"]
    else:
        SL_data_dup = SL_data.reindex(columns=['gene2', 'gene1', 'label'])
        SL_data_dup.columns = ["gene1", "gene2", "label"]
        SL_data = SL_data.append(SL_data_dup)
        SL_potential_score = SL_data.groupby("gene1").agg({"label":"sum"}).reset_index()
        SL_potential_score.columns = ["gene", "num_SL_partner"]
        SL_potential_score["label"] = SL_potential_score["num_SL_partner"]
    
    SL_potential_score = SL_potential_score[["gene","label"]]

    return SL_potential_score, SL_genes

def load_graph_data(graph_type):
    if graph_type == 'PPI-genetic' or graph_type == 'PPI-physical':
        data = pd.read_csv("../data/BIOGRID-9606.csv", index_col=0)
        all_genes = set(data['Official Symbol Interactor A'].unique()) | set(data['Official Symbol Interactor B'].unique())
        
        if graph_type == 'PPI-physical':
            data = data[data['Experimental System Type'] == 'physical']
        else:
            data = data[data['Experimental System Type'] != 'physical']
        #print("Number of edges of {}: {}".format(graph_type, data.shape[0]))

        data = data[['Official Symbol Interactor A','Official Symbol Interactor B']]
        data.rename(columns={'Official Symbol Interactor A':'gene1', 'Official Symbol Interactor B':'gene2'}, inplace=True)

        # make it undirected graph
        data_dup = data.reindex(columns=['gene2','gene1'])
        data_dup.columns = ['gene1','gene2']
        data = data.append(data_dup)

        # for PPI-physical remove genes whose degree of only 1
        if graph_type == "PPI-physical":
            gene_degree = data["gene1"].value_counts()
            kept_genes = list(gene_degree.index[gene_degree>2])
            data = data[(data["gene1"].isin(kept_genes))&(data["gene2"].isin(kept_genes))]
    elif graph_type == "pathway":
        data = pd.read_csv("../data/Opticon_networks.csv")
        data.rename(columns={"Regulator":"gene1", "Target gene":"gene2"}, inplace=True)
    elif graph_type == "random":
        data = pd.read_csv("../data/Opticon_networks.csv")
        data.rename(columns={"Regulator":"gene1", "Target gene":"gene2"}, inplace=True)
        all_genes = set(data['gene1'].unique()) | set(data['gene2'].unique())
        dict_mapping = dict(zip(range(len(all_genes)), all_genes))
        
        num_nodes = len(all_genes)
        num_edges = data.shape[0]
        p = 2*num_edges/(num_nodes*(num_nodes-1))
        # make it more sparse
        p = p/10.0
        
        G = fast_gnp_random_graph(num_nodes, p)
        data = nx.convert_matrix.to_pandas_edgelist(G,source='gene1',target='gene2')
        data["gene1"] = data["gene1"].apply(lambda x:dict_mapping[x])
        data["gene2"] = data["gene2"].apply(lambda x:dict_mapping[x])
        print("generated number of edges of random graph: {}".format(G.number_of_edges()))

    return data

def load_tabular_data(attr, gene_mapping, cell_name):
    num_nodes = len(gene_mapping)
    if attr == "identity":
        x = np.identity(num_nodes)
    elif attr == 'CCLE_random':
        x = np.random.randn(num_nodes, 32)
    elif attr == 'multi_omics':
        # loading cell-specific multi-omics features
        feat_list = ['exp','mut','cnv','ess']
        dict_list = []
        for feat in feat_list:
            temp_df = pd.read_table("../data/cellline_feats/{}_{}.txt".format(cell_name,feat),
                                        names=['gene','value'], sep=' ')
            # filter genes
            temp_df = temp_df[temp_df['gene'].isin(list(gene_mapping.keys()))]
            temp_dict = dict(zip(temp_df['gene'].values, temp_df['value'].values))
            dict_list.append(temp_dict)
        
        x = np.zeros((num_nodes, len(feat_list)))
        for col_idx, feat_dict in enumerate(dict_list):
            for key, value in feat_dict.items():
                row_idx = gene_mapping[key]
                x[row_idx, col_idx] = value
        # standardize features
        x = scale(x)
    elif attr == "CCLE_exp" or attr == "CCLE_ess" or attr == "CCLE_cnv" or attr == "CCLE_mut":
        df = pd.read_csv("../data/CCLE/{}.csv".format(attr), index_col=0)
        df.fillna(0, inplace=True)
        
        if attr != "CCLE_mut":
            df.columns = list(map(lambda x:x.split(" ")[0], df.columns))
            df = df.T
        all_genes = list(df.index)

        # perform PCA for dimension reduction
        values = df.values
        pca = PCA(n_components=32)
        pca.fit(values)
        embed = pca.transform(values)

        x = np.zeros((num_nodes, 32))
        for i, gene in enumerate(all_genes):
            if gene in gene_mapping:
                x[ gene_mapping[gene] ] = embed[i]
        
        x = scale(x)

    return x

def process_graph_data(graph_input, cell_name, SL_data):
    '''
    use cell-specific expression values to filter out unexpressed genes in the network
    remove genes whose expression raw count == 0
    graph_data_list: a list of DataFrames which stores graph edges
    '''
    if cell_name != "synlethdb":
        exp_df = pd.read_table("../data/cellline_feats/{}_exp.txt".format(cell_name),names=['gene','value'], sep=' ')
        exp_df_filtered = exp_df[exp_df["value"]>0]
        kept_genes = exp_df_filtered["gene"].values

    graph_data_list = []
    for graph_type in graph_input:
        print("{} loaded.".format(graph_type))
        graph_data = load_graph_data(graph_type)
        # remove genes whose expression raw count == 0
        if cell_name != "synlethdb":
            graph_data = graph_data[(graph_data["gene1"].isin(kept_genes))&(graph_data["gene2"].isin(kept_genes))]
        if graph_type == "PPI-genetic":
            SL_pos = SL_data[SL_data['label'] == True]
            SL_pos_list = sorted([tuple(r) for r in SL_pos[['gene1','gene2']].to_numpy()] + [tuple(r) for r in SL_pos[['gene2','gene1']].to_numpy()])
            graph_list = [tuple(r) for r in graph_data[['gene1','gene2']].to_numpy()]
            left = list(set(graph_list) - set(SL_pos_list))
            graph_data = pd.DataFrame(left, columns=['gene1','gene2'])
        graph_data_list.append(graph_data)
    
    return graph_data_list

def merge_and_mapping(SL_potential_score, SL_data, SL_genes, graph_data_list):
    '''
    use the union of SL genes and graph genes as all genes
    map gene names to indexs
    '''
    temp_concat_graph_data = pd.concat(graph_data_list)
    graph_genes = set(temp_concat_graph_data['gene1'].unique()) | set(temp_concat_graph_data['gene2'].unique())
    all_genes = sorted(list(set(SL_genes) | graph_genes))

    # gene mapping dictionary
    gene_mapping = dict(zip(all_genes, range(len(all_genes))))

    # converting gene names to id
    # iterating over all graph types
    for i in range(len(graph_data_list)):
        graph_data_list[i]['gene1'] = graph_data_list[i]['gene1'].apply(lambda x:gene_mapping[x])
        graph_data_list[i]['gene2'] = graph_data_list[i]['gene2'].apply(lambda x:gene_mapping[x])
    
    SL_potential_score['gene'] = SL_potential_score['gene'].apply(lambda x:gene_mapping[x])
    SL_data["gene1"] = SL_data["gene1"].apply(lambda x:gene_mapping[x])
    SL_data["gene2"] = SL_data["gene2"].apply(lambda x:gene_mapping[x])

    return SL_potential_score, SL_data, graph_data_list, gene_mapping

def load_data(cell_name, threshold, graph_input, global_tabular_feats, cell_specific_feats, training_percent, val_percent, random_seed):
    # load SL data and network data
    print("loading SL data...")
    SL_data, SL_genes = load_SL_data(cell_name, threshold)
    SL_potential_score, SL_genes = generate_SL_potential_data(cell_name, SL_data, SL_genes, label_type="regression")
    print("loading graph data...")
    graph_data_list = process_graph_data(graph_input, cell_name, SL_data)
        
    # merge, filter and mapping
    print("merging data...")
    SL_potential_score, SL_data, graph_data_list, gene_mapping = merge_and_mapping(SL_potential_score, SL_data, SL_genes, graph_data_list)
    
    # load tabular features
    print("loading population-based CCLE data...")
    tabular_feats_list = []
    for feats in global_tabular_feats:
        tabular_feats_list.append(torch.tensor(load_tabular_data(feats, gene_mapping, cell_name), dtype=torch.float))
        print("{} loaded.".format(feats))

    # load cell-specific features
    if len(cell_specific_feats)>0 and cell_name != "synlethdb":
        print("loading cell-specific multi-omics data...")
        cell_specific_feats_list = []
        for feats in cell_specific_feats:
            temp_feats = torch.tensor(load_tabular_data(feats, gene_mapping, cell_name), dtype=torch.float)
            cell_specific_feats_list.append(temp_feats)
    else:
        cell_specific_feats_list = None

    # generate torch data for graph data
    data_edge_index_list = []
    for graph_data in graph_data_list:
        temp_edge_index = torch.tensor([graph_data['gene1'].values, graph_data['gene2'].values], dtype=torch.long)
        data_edge_index_list.append(temp_edge_index)

    # get input data that contains all three types of features
    data = Data(edge_index_list=data_edge_index_list,
                tabular_feats_list=tabular_feats_list, 
                cell_specific_feats_list=cell_specific_feats_list)

    # split into train, val and test
    print("spliting data into train, validation and test...")
    # gene-level split: genes in train, val and test are different
    num_samples = SL_potential_score.shape[0]
    all_idx = list(range(num_samples))
    np.random.seed(random_seed)
    np.random.shuffle(all_idx)

    SL_potential_train = SL_potential_score.iloc[all_idx[:int(num_samples*training_percent)]]
    SL_potential_val = SL_potential_score.iloc[all_idx[int(num_samples*training_percent):int(num_samples*(training_percent+val_percent))]]
    SL_potential_test = SL_potential_score.iloc[all_idx[int(num_samples*(training_percent+val_percent)):]]

    # generate SL pair data
    training_genes = SL_potential_train["gene"].values
    val_genes = SL_potential_val["gene"].values
    test_genes = SL_potential_test["gene"].values
    # make training and test data balanced
    SL_pair_train = SL_data[(SL_data["gene1"].isin(training_genes))&(SL_data["gene2"].isin(training_genes))]
    SL_pair_val = SL_data[(SL_data["gene1"].isin(val_genes))&(SL_data["gene2"].isin(val_genes))]
    SL_pair_test = SL_data[(SL_data["gene1"].isin(test_genes))&(SL_data["gene2"].isin(test_genes))]
    if cell_name != "synlethdb":
        SL_pair_train_pos = SL_pair_train[SL_pair_train["label"]==1]
        SL_pair_train_neg = SL_pair_train[SL_pair_train["label"]==0].sample(n=SL_pair_train_pos.shape[0])
        SL_pair_train = pd.concat([SL_pair_train_pos, SL_pair_train_neg])
        SL_pair_val_pos = SL_pair_val[SL_pair_val["label"]==1]
        SL_pair_val_neg = SL_pair_val[SL_pair_val["label"]==0].sample(n=SL_pair_val_pos.shape[0])
        SL_pair_val = pd.concat([SL_pair_val_pos, SL_pair_val_neg])
        SL_pair_test_pos = SL_pair_test[SL_pair_test["label"]==1]
        SL_pair_test_neg = SL_pair_test[SL_pair_test["label"]==0].sample(n=SL_pair_test_pos.shape[0])
        SL_pair_test = pd.concat([SL_pair_test_pos, SL_pair_test_neg])
    else:
        SL_pair_train_neg = generate_random_negative_samples(SL_pair_train)
        SL_pair_val_neg = generate_random_negative_samples(SL_pair_val)
        SL_pair_test_neg = generate_random_negative_samples(SL_pair_test)
        SL_pair_train = pd.concat([SL_pair_train, SL_pair_train_neg])
        SL_pair_val = pd.concat([SL_pair_val, SL_pair_val_neg])
        SL_pair_test = pd.concat([SL_pair_test, SL_pair_test_neg])

    return data, SL_potential_train, SL_potential_val, SL_potential_test, SL_pair_train, SL_pair_val, SL_pair_test, gene_mapping

def generate_SL_pair_features(SL_potential_train_dict, SL_potential_test_dict, SL_pair_train, SL_pair_test):
    # generate features for logistic regression model
    SL_pair_train["gene1_potential"] = SL_pair_train["gene1"].apply(lambda x:SL_potential_train_dict[x])
    SL_pair_train["gene2_potential"] = SL_pair_train["gene2"].apply(lambda x:SL_potential_train_dict[x])
    SL_pair_test["gene1_potential"] = SL_pair_test["gene1"].apply(lambda x:SL_potential_test_dict[x])
    SL_pair_test["gene2_potential"] = SL_pair_test["gene2"].apply(lambda x:SL_potential_test_dict[x])

    train_X = SL_pair_train[["gene1_potential", "gene2_potential"]].values
    train_y = SL_pair_train["label"].values
    test_X = SL_pair_test[["gene1_potential", "gene2_potential"]].values
    test_y = SL_pair_test["label"].values

    return train_X, train_y, test_X, test_y

def ranking_metrics(true_labels, pred_scores, top=0.1):
    sorted_index = np.argsort(-pred_scores)
    top_num = int(top * len(true_labels))
    sorted_true_labels = true_labels[sorted_index[:top_num]]
    acc = float(sorted_true_labels.sum())/float(top_num)
    return acc

def evaluate_performance(label, pred):
    AUC = roc_auc_score(label, pred)
    AUPR = average_precision_score(label, pred)
    rank_score_5 = ranking_metrics(label, pred, top=0.05)
    rank_score_10 = ranking_metrics(label, pred, top=0.10)

    performance_dict = {"AUC":AUC, "AUPR":AUPR, "precision@5":rank_score_5, "precision@10":rank_score_10}

    return performance_dict

def generate_random_negative_samples(pos_samples):
    # randomly generate same amount of negative samples as positive samples
    num = pos_samples.shape[0]
    all_genes = list(set(pos_samples['gene1'].unique()) | set(pos_samples['gene2'].unique()))
    neg_candidates_1 = random.choices(all_genes, k=2*num)
    neg_candidates_2 = random.choices(all_genes, k=2*num)
    
    pos_list = [tuple(r) for r in pos_samples[['gene1','gene2']].to_numpy()] + [tuple(r) for r in pos_samples[['gene2','gene1']].to_numpy()]
    sampled_list = list(zip(neg_candidates_1, neg_candidates_2))
    # remove gene pairs that have positive effects
    remained_list = set(sampled_list) - set(pos_list)
    # remove gene pairs where gene1 = gene2
    remained_list = [x for x in remained_list if x[0] != x[1]]
    remained_list = random.sample(remained_list, num)
    
    neg_df = pd.DataFrame({"gene1":[x[0] for x in remained_list],
                           "gene2":[x[1] for x in remained_list],
                           "label":[0]*num})
    return neg_df
