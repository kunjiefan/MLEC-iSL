import numpy as np
import sys
import torch
import torch.nn as nn
import torch_geometric.transforms as T
import torch.nn.functional as F
from torch.autograd import Variable
from torch_geometric.nn import GCNConv, GAE, VGAE
from torch_geometric.nn import GATConv, GATv2Conv
from torch_geometric.nn import ResGatedGraphConv, TransformerConv

class GCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        x = self.conv2(x, edge_index)
        return x
    
class ScaledDotProductAttention(nn.Module):
    def __init__(self, embed_size):
        super(ScaledDotProductAttention, self).__init__()
        self.embed_size = embed_size

    def forward(self, Q, K, V, attn_mask, attn_bias=None):
        '''
        Q: [batch_size, n_heads, len_q, d_k]
        K: [batch_size, n_heads, len_k, d_k]
        V: [batch_size, n_heads, len_v(=len_k), d_v]
        attn_mask: [batch_size, n_heads, seq_len, seq_len]
        '''
        scores = torch.matmul(Q, K.transpose(-1, -2)) / np.sqrt(self.embed_size) # scores : [batch_size, n_heads, len_q, len_k]

        if attn_bias is not None:
            scores = scores + attn_bias

        scores.masked_fill_(attn_mask, -1e9) # Fills elements of self tensor with value where mask is True.
        
        attn = nn.Softmax(dim=-1)(scores)
        context = torch.matmul(attn, V) # [batch_size, n_heads, len_q, d_v]
        return context, attn

class MultiHeadAttention(nn.Module):
    def __init__(self, embed_size, n_heads, input_embeds_size):
        super(MultiHeadAttention, self).__init__()
        self.embed_size = embed_size
        self.n_heads = n_heads
        self.W_Q = nn.Linear(input_embeds_size, embed_size * self.n_heads, bias=False)
        self.W_K = nn.Linear(input_embeds_size, embed_size * self.n_heads, bias=False)
        self.W_V = nn.Linear(input_embeds_size, embed_size * self.n_heads, bias=False)
        self.fc = nn.Linear(embed_size * self.n_heads, embed_size, bias=False)
        self.layerNorm = nn.LayerNorm(embed_size)

    def forward(self, input_Q, input_K, input_V, attn_mask, attn_bias=None):
        residual, batch_size = input_V, input_V.size(0)
        # (B, S, D) -proj-> (B, S, D_new) -split-> (B, S, H, W) -trans-> (B, H, S, W)
        Q = self.W_Q(input_Q).view(batch_size, -1, self.n_heads, self.embed_size).transpose(1,2)  # Q: [batch_size, n_heads, len_q, embed_size]
        K = self.W_K(input_K).view(batch_size, -1, self.n_heads, self.embed_size).transpose(1,2)  # K: [batch_size, n_heads, len_k, embed_size]
        V = self.W_V(input_V).view(batch_size, -1, self.n_heads, self.embed_size).transpose(1,2)  # V: [batch_size, n_heads, len_v(=len_k), embed_size]

        attn_mask = attn_mask.unsqueeze(1).repeat(1, self.n_heads, 1, 1) # attn_mask : [batch_size, n_heads, seq_len, seq_len]

        # context: [batch_size, n_heads, len_q, d_v], attn: [batch_size, n_heads, len_q, len_k]
        context, attn = ScaledDotProductAttention(self.embed_size)(Q, K, V, attn_mask, attn_bias=attn_bias)
        
        context = context.transpose(1, 2).reshape(batch_size, -1, self.n_heads * self.embed_size) # context: [batch_size, len_q, n_heads * embed_size]
        output = self.fc(context) # [batch_size, len_q, embed_size]
        # add residual
        output = self.layerNorm(output + residual)
        return output, attn

class PoswiseFeedForwardNet(nn.Module):
    def __init__(self, embed_size, hidden_size):
        super(PoswiseFeedForwardNet, self).__init__()

        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.fc = nn.Sequential(
            nn.Linear(embed_size, hidden_size, bias=False),
            nn.ReLU(),
            nn.Linear(hidden_size, embed_size, bias=False)
        )
        self.layerNorm = nn.LayerNorm(embed_size)

    def forward(self, inputs):
        '''
        inputs: [batch_size, seq_len, embed_size]
        '''
        residual = inputs
        output = self.fc(inputs)
        output = self.layerNorm(output + residual) # [batch_size, seq_len, embed_size]
        return output 

class EncoderLayer(nn.Module):
    def __init__(self, embed_size, n_heads, hidden_size, input_embeds_size):
        super(EncoderLayer, self).__init__()
        self.enc_self_attn = MultiHeadAttention(embed_size, n_heads, input_embeds_size)
        self.pos_ffn = PoswiseFeedForwardNet(embed_size, hidden_size)

    def forward(self, enc_inputs, enc_self_attn_mask, attn_bias=None):
        enc_outputs, attn = self.enc_self_attn(enc_inputs, enc_inputs, enc_inputs, enc_self_attn_mask, attn_bias=attn_bias)
        enc_outputs = self.pos_ffn(enc_outputs)
        return enc_outputs, attn

class Transformer(nn.Module):
    def __init__(self, embed_size, hidden_size, n_heads, n_layers, input_embeds_size):
        super(Transformer, self).__init__()
        self.layers = nn.ModuleList([EncoderLayer(embed_size, n_heads, hidden_size, input_embeds_size) for _ in range(n_layers)])

    def forward(self, enc_inputs, enc_self_attn_mask):
        '''
        enc_inputs: [1, num_gene, 128 (or 256)]
        enc_self_attn_mask: [1, num_gene, num_gene]
        '''
        enc_self_attns = []
        enc_outputs = enc_inputs
        for layer in self.layers:
            enc_outputs, enc_self_attn = layer(enc_outputs, enc_self_attn_mask)
            enc_self_attns.append(enc_self_attn)
        return enc_outputs.squeeze(0), enc_self_attns
    
class Graph_trans(nn.Module):
    def __init__(self, embed_size, hidden_size, n_heads, n_layers, num_network=3, GT_flag=True):
        super(Graph_trans, self).__init__()
        self.num_network = num_network
        self.GT_flag = GT_flag
        # a list of GCN models for processing different graph inputs
        self.gcn_list = nn.ModuleList([GCN(embed_size, hidden_size, embed_size) for i in range(self.num_network)])
        
        self.transform = nn.Linear(self.num_network*embed_size, embed_size)
        self.transformer = Transformer(embed_size, hidden_size, n_heads, n_layers, embed_size)

    
    def forward(self, x, edge_index_list):
        num_gene = x.shape[0]

        z_list = []
        for i, edge_index in enumerate(edge_index_list):
            temp_z = self.gcn_list[i](x, edge_index)
            z_list.append(temp_z)
        z = torch.cat(z_list, 1)
        z = self.transform(z)

        if self.GT_flag:
            z = z.unsqueeze(0)
            enc_self_attn_mask = torch.zeros((1, num_gene, num_gene), device=z.device).bool()
            z, enc_attns = self.transformer(z, enc_self_attn_mask)
        
        return z

class MVGT_iSL(nn.Module):
    def __init__(self, hidden_size, embed_size, n_heads, n_layers, num_CCLE=4, num_network=3, cell_specific_flag=True, GT_flag=True, task="connectivity"):
        super(MVGT_iSL, self).__init__()
        self.num_CCLE = num_CCLE
        self.num_network = num_network
        self.cell_specific_flag = cell_specific_flag
        self.GT_flag = GT_flag
        self.task = task

        self.linear_transform_CCLE = nn.Linear(self.num_CCLE*embed_size, embed_size)

        self.graph_transformer = Graph_trans(embed_size, hidden_size, n_heads, n_layers, num_network=self.num_network, GT_flag=self.GT_flag)
        
        if self.cell_specific_flag:
            self.cell_specific_encoder = nn.Sequential(
                                            nn.Linear(4, embed_size), nn.ReLU(),
                                            nn.Dropout(0.5),
                                            nn.Linear(embed_size, embed_size)
                                            )
            self.linear_transform_hetero = nn.Linear(2*embed_size, embed_size)

        if self.task == "pair":
            fc1_input_size = 2*embed_size
        elif self.task == "connectivity":
            fc1_input_size = embed_size
        
        self.predictor = nn.Sequential(
                            nn.Linear(fc1_input_size, int(2*fc1_input_size)), nn.ReLU(),
                            nn.Dropout(0.7),
                            nn.Linear(2*fc1_input_size, int(fc1_input_size/2)), nn.ReLU(),
                            nn.Dropout(0.7),
                            nn.Linear(int(fc1_input_size/2), 1)
        )
    
    def return_weights(self):
        return self.fc1.weight
    
    def return_gene_embeds(self):
        return self.gene_embeds
    
    def forward(self, edge_index_list, tabular_feats_list, cell_specific_feats_list, gene_index):
        num_gene = tabular_feats_list[0].shape[0]

        # concat all population-based CCLE features
        z_global = torch.cat(tabular_feats_list, 1)
        # use MLP to integrate all types of CCLE features
        z_global = self.linear_transform_CCLE(z_global)

        if self.cell_specific_flag:
            cell_specific_input = cell_specific_feats_list[0]
            z_specific = self.cell_specific_encoder(cell_specific_input)
            # use MLP to integrate both population-based features and cell-specific features
            z = torch.cat((z_global, z_specific), 1)
            z = self.linear_transform_hetero(z)
        else:
            z = z_global

        # graph transformer
        z = self.graph_transformer(z, edge_index_list)

        # store the gene embeddings
        self.gene_embeds = z
        
        # prediction module
        if self.task == "connectivity":
            z = z[gene_index]
        elif self.task == "pair":
            z = torch.cat((z[ gene_index[0] ], z[ gene_index[1] ]), dim=1)
        
        z = self.predictor(z)
        
        return z.squeeze()


class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=50, verbose=False, reverse=False, delta=0, path='checkpoint.pt', trace_func=print):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement. 
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            path (str): Path for the checkpoint to be saved to.
                            Default: 'checkpoint.pt'
            trace_func (function): trace print function.
                            Default: print            
        """
        self.patience = patience
        self.verbose = verbose
        self.reverse = reverse
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.path = path
        self.trace_func = trace_func
    def __call__(self, val_loss, model):
        if self.reverse == True:
            # loss
            score = -val_loss
        else:
            # AUC/AUPR
            score = val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            self.trace_func(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss