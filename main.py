import os
import copy
import numpy as np
import torch
import torch.nn.functional as F
import dgl.function as fn
from torch.nn import init
import dgl
import argparse
from gat import GAT
from utils import get_data_loader,evaluate_model, test_model
from get_models import collect_model
from plot_utils import loss_logger, parameters
import time

import matplotlib.pyplot as plt
import collections
import random

torch.set_num_threads(1)

def graph_KLDiv(graph, edgex, edgey, reduce='mean'):
    '''
    compute the KL loss for each edges set, used after edge_softmax
    '''
    with graph.local_scope():
        nnode = graph.number_of_nodes()
        graph.ndata.update({'kldiv': torch.ones(nnode,1).to(edgex.device)})
        diff = edgey*(torch.log(edgey)-torch.log(edgex))
        graph.edata.update({'diff':diff})
        graph.update_all(fn.u_mul_e('kldiv', 'diff', 'm'),
                            fn.sum('m', 'kldiv'))
        if reduce == "mean":
            return torch.mean(torch.flatten(graph.ndata['kldiv']))
    

def train_student(args, auxiliary_model, data, device):
    '''
    strategy:
        alignahead: the proposed model
        OC: one-to-one correspondence
    '''
    best_score = np.zeros(args.model_num)
    best_loss = 1000.0*np.ones(args.model_num)
    test_score = np.zeros(args.model_num)

    train_dataloader, valid_dataloader, test_dataloader, fixed_train_dataloader = data
    
    # multi class loss function
    loss_fcn = torch.nn.BCEWithLogitsLoss()
    loss_mse = torch.nn.MSELoss()
    loss_bce = torch.nn.BCELoss()

    #Online KD
    s_model= auxiliary_model['s_model']['model']
    s_optimizer = auxiliary_model['s_model']['optimizer']

    losslogger = loss_logger()

    for epoch in range(args.s_epochs):
        for model_id in range(args.model_num):
            target_model = s_model[model_id]
            target_model.train()

            loss_list = []
            lsp_loss_list = []
            t0 = time.time()
            for batch, batch_data in enumerate( zip(train_dataloader,fixed_train_dataloader) ):
                shuffle_data, fixed_data = batch_data
                subgraph, feats, labels = shuffle_data
                fixed_subgraph, fixed_feats, fixed_labels = fixed_data

                feats = feats.to(device)
                labels = labels.to(device)
                fixed_feats = fixed_feats.to(device)
                fixed_labels = fixed_labels.to(device)

                target_model.g = subgraph
                for layer in target_model.gat_layers:
                    layer.g = subgraph
                
                logits, middle_feats_s  = target_model(feats.float(), middle=True)


                label_loss = loss_fcn(logits,labels.float())  
                                     
                lsp_loss = torch.tensor(0.0).to(device)
                for others in range(args.model_num):
                    if others != model_id:
                        co_model = s_model[others]                                                       
                        #LSP_loss
                        with torch.no_grad():
                            co_model.g = subgraph
                            for layer in co_model.gat_layers:
                                layer.g = subgraph
                            _, middle_feats_co = co_model(feats.float(), middle=True)
                        if args.strategy =='OC':
                            for i in range(args.s_num_layers):
                                dist_s =auxiliary_model['local_model']['model'](subgraph, middle_feats_s[i])
                                dist_co =auxiliary_model['local_model']['model'](subgraph, middle_feats_co[i])                       
                                lsp_loss += graph_KLDiv(subgraph, dist_s, dist_co,reduce='mean')
                        elif args.strategy =='alignahead':
                            for i in range(args.s_num_layers-1):
                                dist_s =auxiliary_model['local_model']['model'](subgraph, middle_feats_s[i+1])
                                dist_co =auxiliary_model['local_model']['model'](subgraph, middle_feats_co[i])                       
                                lsp_loss += graph_KLDiv(subgraph, dist_s, dist_co,reduce='mean')                       
                            dist_s = auxiliary_model['local_model']['model'](subgraph, middle_feats_s[0])
                            dist_co =auxiliary_model['local_model']['model'](subgraph, middle_feats_co[args.s_num_layers-1])
                            lsp_loss += graph_KLDiv(subgraph, dist_s, dist_co,reduce='mean')                        
                lsp_loss = lsp_loss/(args.model_num-1)
                       
                loss = label_loss+args.a*lsp_loss           
                s_optimizer[model_id].zero_grad()
                loss.backward()
                s_optimizer[model_id].step()
                loss_list.append(loss.item())
                lsp_loss_list.append(lsp_loss.item())
                
            loss_data = np.array(loss_list).mean()            
            lsp_loss_data = np.array(lsp_loss_list).mean()
            with open(log_txt, 'a+') as f:
                f.write(f"Epoch {epoch:05d} | ModelID: {model_id:02d}|Loss: {loss_data:.4f}| Time: {time.time()-t0:.4f}s\n")
            print(f"Epoch {epoch:05d} | ModelID: {model_id:02d}|Loss: {loss_data:.4f}| Time: {time.time()-t0:.4f}s")
            if epoch % 10 == 0:
                score = evaluate_model(valid_dataloader, train_dataloader, device, target_model, loss_fcn)
                if score > best_score[model_id] or loss_data < best_loss[model_id]:
                    best_score[model_id] = score
                    best_loss[model_id] = loss_data
                    test_score[model_id] = test_model(test_dataloader, target_model, device, loss_fcn)
                    with open(log_txt, 'a+') as f:
                        f.write('f1 score on testset:{}\n'.format(test_score))
                    print("f1 score on testset:" ,test_score)

    with open(log_txt, 'a+') as f:
        f.write('final f1 score on testset:{}\n'.format(test_score))
    print("final f1 score on testset:" ,test_score)
   
def main(args):
    device = torch.device("cpu") if args.gpu<0 else torch.device("cuda:" + str(args.gpu))
    data, data_info = get_data_loader(args)
    model_dict = collect_model(args, data_info)
    print(f"number of parameter for student model: {parameters(model_dict['s_model']['model'][0])}")
    print("############ train student models#############")
    train_student(args, model_dict, data, device)



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='GAT')
    parser.add_argument("--gpu", type=int, default=0,
                        help="which GPU to use. Set -1 to use CPU.")
    parser.add_argument("--residual", action="store_true", default=True,
                        help="use residual connection")
    parser.add_argument("--in-drop", type=float, default=0,
                        help="input feature dropout")
    parser.add_argument("--attn-drop", type=float, default=0,
                        help="attention dropout")
    parser.add_argument('--alpha', type=float, default=0.2,
                        help="the negative slop of leaky relu")
    parser.add_argument('--batch-size', type=int, default=2,
                        help="batch size used for training, validation and test")
    parser.add_argument('--model_num',type = int ,default=3)
    parser.add_argument('--model_name',type = str ,default='GAT')

    parser.add_argument('--a',type=float,default=1)
    parser.add_argument("--lr", type=float, default=0.005,
                        help="learning rate")
    parser.add_argument('--weight-decay', type=float, default=0,
                        help="weight decay")

    parser.add_argument("--s-epochs", type=int, default=300,
                        help="number of training epochs")
    parser.add_argument("--s-num-heads", type=int, default=3,
                        help="number of hidden attention heads")
    parser.add_argument("--s-num-out-heads", type=int, default=3,
                        help="number of output attention heads")
    parser.add_argument("--s-num-layers", type=int, default=4,
                        help="number of hidden layers")
    parser.add_argument("--s-num-hidden", type=int, default=64,
                        help="number of hidden units")
   
    parser.add_argument("--strategy", type=str, default='alignahead')
    parser.add_argument("--warmup-epoch", type=int, default=600,
                        help="steps to warmup")
    parser.add_argument('--seed', type=int, default=100,
                        help="seed")


    args = parser.parse_args()
    print(args)

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    
    log_txt = 'result/'+ \
          'model_num' + '_' +  str(args.model_num) + '_'+\
          's_layers' + '_' +  str(args.s_num_layers) + '_'+\
          'model' + '_' +  str(args.model_name) + '_'+\
          'strategy' + '_' +  str(args.strategy) + '_'+\
          'seed'+ str(args.seed) +'.txt'   
    main(args)