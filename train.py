import torch
from google.colab import files
from torch import nn
from models import SAINT
from models import CustomNormalLoss
from reshape_keibadata import addstct, data_prep, separate_catcon
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, mean_squared_error, log_loss

from data_openml import data_prep_openml,task_dset_ids,DataSetCatCon
import argparse
from torch.utils.data import DataLoader
import torch.optim as optim
from utils import count_parameters, classification_scores, mean_sq_error, param_CustomNormalLoss
from augmentations import embed_data_mask
from augmentations import add_noise

import os
import numpy as np
import pandas as pd
import warnings
from datetime import datetime, date
today = date.today()
today = str(today)
current_time = datetime.now().time()
formatted_time = current_time.strftime("%H-%M-%S")
date = today+'_'+formatted_time


warnings.simplefilter('ignore')
parser = argparse.ArgumentParser()

# NOTE: 提案のときは，--propをつける
parser.add_argument('--prop', action='store_true')
parser.add_argument('--project', default='saint+', choices = ['saint+','p_SAINT_saraya'])
parser.add_argument('--alpha', default=1.0, type=float)  # loss_gnll の重み
parser.add_argument('--beta', default=1.0, type=float)  # loss_mse の重み
parser.add_argument('--gamma', default=1.0, type=float)  # loss_sum_var の重み

parser.add_argument('--dset_id', required=True, type=int, help="Keiba = 999")
parser.add_argument('--vision_dset', action = 'store_true')
parser.add_argument('--task', required=True, type=str,choices = ['binary','multiclass','regression'])
parser.add_argument('--cont_embeddings', default='MLP', type=str,choices = ['MLP','Noemb','pos_singleMLP'])
parser.add_argument('--embedding_size', default=32, type=int)
parser.add_argument('--transformer_depth', default=6, type=int) #1層で解釈しよう
parser.add_argument('--attention_heads', default=8, type=int)
parser.add_argument('--attention_dropout', default=0.1, type=float)
parser.add_argument('--ff_dropout', default=0.1, type=float)
parser.add_argument('--attentiontype', default='colrow', type=str,choices = ['col','colrow','row','justmlp','attn','attnmlp'])

parser.add_argument('--optimizer', default='AdamW', type=str,choices = ['AdamW','Adam','SGD'])
parser.add_argument('--scheduler', default='cosine', type=str,choices = ['cosine','linear'])

parser.add_argument('--lr', default=0.001, type=float)
parser.add_argument('--epochs', default=100, type=int)
parser.add_argument('--batchsize', default=256, type=int)
parser.add_argument('--savemodelroot', default='./bestmodels', type=str)
parser.add_argument('--run_name', default='testrun', type=str)
parser.add_argument('--set_seed', default= 1 , type=int)
parser.add_argument('--dset_seed', default= 5 , type=int)
parser.add_argument('--active_log', action = 'store_true')

parser.add_argument('--pretrain', action = 'store_true')
parser.add_argument('--pretrain_epochs', default=50, type=int)
parser.add_argument('--pt_tasks', default=['contrastive','denoising'], type=str,nargs='*',choices = ['contrastive','contrastive_sim','denoising'])
parser.add_argument('--pt_aug', default=[], type=str,nargs='*',choices = ['mixup','cutmix'])
parser.add_argument('--pt_aug_lam', default=0.1, type=float)
parser.add_argument('--mixup_lam', default=0.3, type=float)

parser.add_argument('--train_mask_prob', default=0, type=float)
parser.add_argument('--mask_prob', default=0, type=float)

parser.add_argument('--ssl_avail_y', default= 0, type=int)
parser.add_argument('--pt_projhead_style', default='diff', type=str,choices = ['diff','same','nohead'])
parser.add_argument('--nce_temp', default=0.7, type=float)

parser.add_argument('--lam0', default=0.5, type=float)
parser.add_argument('--lam1', default=10, type=float)
parser.add_argument('--lam2', default=1, type=float)
parser.add_argument('--lam3', default=10, type=float)
parser.add_argument('--final_mlp_style', default='sep', type=str,choices = ['common','sep'])

def objective(trail,x,y):
    alpha = trial.suggest_float('param1', 0.0, 100.0)
    gamma = trial.suggest_float('param2', 0.0, 100.0)
    
    loss_mse = criterion(y_outs, y_gts)
    loss_gnll = criterion_gnll(y_outs[:, 0].unsqueeze(1), y_gts, y_outs[:, 1].unsqueeze(1))
    #loss_gnll = criterion_CNL(y_outs[:, 0].unsqueeze(1), y_gts, y_outs[:, 1].unsqueeze(1))
    loss_sum_var = y_outs[:, 1].mean()
    opt.beta = 0.0
    loss = (opt.alpha * loss_gnll) + (opt.beta * loss_mse) + (opt.gamma * loss_sum_var)

    return loss


opt = parser.parse_args()
modelsave_path = os.path.join(os.getcwd(),opt.savemodelroot,opt.task,str(opt.dset_id),opt.run_name)
if opt.task == 'regression':
    opt.dtask = 'reg'
else:
    opt.dtask = 'clf'


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Device is {device}.")

torch.manual_seed(opt.set_seed)
os.makedirs(modelsave_path, exist_ok=True)

if opt.active_log:
    import wandb
    if opt.pretrain:
        wandb.init(project="saint+", group =opt.run_name, name = f'pretrain_{opt.task}_{str(opt.attentiontype)}_{str(opt.dset_id)}_{str(opt.set_seed)}')
    else:
        if opt.task=='multiclass':
            wandb.init(project="saint+_kamal", group =opt.run_name ,name = f'{opt.task}_{str(opt.attentiontype)}_{str(opt.dset_id)}_{str(opt.set_seed)}')
        else:
            if opt.prop:
                wandb.init(project="saint+", group=f'{opt.task}_att-{str(opt.attentiontype)}_dset-{str(opt.dset_id)}_prop-alpha{str(opt.alpha)}-beta{str(opt.beta)}', name=f'seed-{str(opt.set_seed)}')
            else:
                wandb.init(project="saint+", group=f'{opt.task}_att-{str(opt.attentiontype)}_dset-{str(opt.dset_id)}_conv', name=f'seed-{str(opt.set_seed)}')


if opt.dset_id != 999:
    cat_dims, cat_idxs, con_idxs, X_train, y_train, X_valid, y_valid, X_test, y_test, train_mean, train_std = data_prep_openml(opt.dset_id, opt.dset_seed, opt.task, datasplit=[.65, .15, .2])
else:
    # Excelファイルを読み込む dset_id=999は競馬データ
    df2 = pd.read_excel("/content/drive/MyDrive/SAINT/data/source_data/df5.xlsx", index_col = 0)
    #df2 = df2.iloc[:100,:]
    cat_dims, cat_idxs, con_idxs, X_train, y_train, X_valid, y_valid, X_test, y_test, train_mean, train_std, train_indices, test_indices = data_prep_openml(opt.dset_id, opt.dset_seed, opt.task, df2, datasplit=[.65, .15, .2])

print('Downloading and processing the dataset, it might take some time.')
#cat_dims, cat_idxs, con_idxs, X_train, y_train, X_valid, y_valid, X_test, y_test, train_mean, train_std = data_prep_openml(opt.dset_id, opt.dset_seed,opt.task, datasplit=[.65, .15, .2])

nfeat = len(X_train)
continuous_mean_std = np.array([train_mean,train_std]).astype(np.float32) 

##### Setting some hyperparams based on inputs and dataset
#_,nfeat = X_train['data'].shape
if nfeat > 100:
    opt.embedding_size = min(8,opt.embedding_size)
    opt.batchsize = min(64, opt.batchsize)
if opt.attentiontype != 'col':
    #opt.transformer_depth = 1
    opt.attention_heads = min(4,opt.attention_heads)
    opt.attention_dropout = 0.8
    opt.embedding_size = min(32,opt.embedding_size)
    opt.ff_dropout = 0.8

print(nfeat,opt.batchsize)
print(opt)

if opt.active_log:
    wandb.config.update(opt)
train_ds = DataSetCatCon(X_train, y_train, cat_idxs,opt.dtask,continuous_mean_std)
#!!!---------shuffle = False-----------!!!
trainloader = DataLoader(train_ds, batch_size=opt.batchsize, shuffle=True,num_workers=4)

valid_ds = DataSetCatCon(X_valid, y_valid, cat_idxs,opt.dtask, continuous_mean_std)
validloader = DataLoader(valid_ds, batch_size=opt.batchsize, shuffle=False,num_workers=4)

test_ds = DataSetCatCon(X_test, y_test, cat_idxs,opt.dtask, continuous_mean_std)
testloader = DataLoader(test_ds, batch_size=opt.batchsize, shuffle=False,num_workers=4)
if opt.task == 'regression':
    y_dim = 1
    # NOTE: 提案で分散まで計算する
    if opt.prop:
        y_dim = 2
else:
    y_dim = len(np.unique(y_train['data'][:,0]))

cat_dims = np.append(np.array([1]),np.array(cat_dims)).astype(int) #Appending 1 for CLS token, this is later used to generate embeddings.

model = SAINT(
    categories = tuple(cat_dims), 
    num_continuous = len(con_idxs),                
    dim = opt.embedding_size,                           
    dim_out = 1,                       
    depth = opt.transformer_depth,                       
    heads = opt.attention_heads,                         
    attn_dropout = opt.attention_dropout,             
    ff_dropout = opt.ff_dropout,                  
    mlp_hidden_mults = (4, 2),       
    cont_embeddings = opt.cont_embeddings,
    attentiontype = opt.attentiontype,
    final_mlp_style = opt.final_mlp_style,
    y_dim = y_dim,
    variance_mode = opt.prop,
)
vision_dset = opt.vision_dset

if y_dim == 2 and opt.task == 'binary':
    # opt.task = 'binary'
    criterion = nn.CrossEntropyLoss().to(device)
elif y_dim > 2 and  opt.task == 'multiclass':
    # opt.task = 'multiclass'
    criterion = nn.CrossEntropyLoss().to(device)
elif opt.task == 'regression':
    criterion = nn.MSELoss().to(device)
    # ---------------------
    # NOTE: NLL
    criterion_gnll = nn.GaussianNLLLoss(eps=1e-03).to(device)
    criterion_CNL = CustomNormalLoss()
    # ---------------------
else:
    raise'case not written yet'

model.to(device)


if opt.pretrain:
    from pretraining import SAINT_pretrain
    model = SAINT_pretrain(model, cat_idxs, X_train,y_train, continuous_mean_std, opt,device)

## Choosing the optimizer

if opt.optimizer == 'SGD':
    optimizer = optim.SGD(model.parameters(), lr=opt.lr,
                          momentum=0.9, weight_decay=5e-4)
    from utils import get_scheduler
    scheduler = get_scheduler(opt, optimizer)
elif opt.optimizer == 'Adam':
    optimizer = optim.Adam(model.parameters(),lr=opt.lr)
elif opt.optimizer == 'AdamW':
    optimizer = optim.AdamW(model.parameters(),lr=opt.lr)
best_valid_auroc = 0
best_valid_accuracy = 0
best_test_auroc = 0
best_test_accuracy = 0
best_valid_rmse = 100000
best_valid_NLL = 100000

for epoch in range(opt.epochs):
    epoch += 1
    model.train()
    running_loss = 0.0
    loss_mse = torch.tensor([0.0])
    #print(f'{i}回目--------------\n')
    for i, data in enumerate(trainloader, 0):
        optimizer.zero_grad()
        # x_categ is the the categorical data, x_cont has continuous data, y_gts has ground truth ys. cat_mask is an array of ones same shape as x_categ and an additional column(corresponding to CLS token) set to 0s. con_mask is an array of ones same shape as x_cont. 
        x_categ, x_cont, y_gts, cat_mask, con_mask = data[0].to(device), data[1].to(device),data[2].to(device),data[3].to(device),data[4].to(device)

        # We are converting the data to embeddings in the next step
        _ , x_categ_enc, x_cont_enc = embed_data_mask(x_categ, x_cont, cat_mask, con_mask,model,vision_dset)           
        reps, self_out, inter_out = model.transformer(x_categ_enc, x_cont_enc)
        #reps = model.transformer(x_categ_enc, x_cont_enc)
        #select only the representations corresponding to CLS token and apply mlp on it in the next step to get the predictions.
        y_reps = reps[:,0,:]
        
        y_outs = model.mlpfory(y_reps)
        
        if opt.task == 'regression' and opt.prop:
            loss_mse = criterion(y_outs, y_gts)
            loss_gnll = criterion_gnll(y_outs[:, 0].unsqueeze(1), y_gts, y_outs[:, 1].unsqueeze(1))
            #loss_gnll = criterion_CNL(y_outs[:, 0].unsqueeze(1), y_gts, y_outs[:, 1].unsqueeze(1))
            loss_sum_var = y_outs[:, 1].mean()
            loss = (opt.alpha * loss_gnll) + (opt.beta * loss_mse) + (opt.gamma * loss_sum_var)
        elif opt.task == 'regression':
            loss = criterion(y_outs[:, 0].unsqueeze(1), y_gts)
        else:
            loss = criterion(y_outs,y_gts.squeeze()) 
        loss.backward()
        optimizer.step()
        if opt.optimizer == 'SGD':
            scheduler.step()
        running_loss += loss.item()
    # print(running_loss)
    
    if opt.prop:
        print(f'[Info] (train) epoch: {epoch} | train_epoch_loss: {running_loss} | loss_mse: {loss_mse.item()} | loss_gnll: {loss_gnll.item()} | loss_sum_var: {loss_sum_var.item()}')
    
    else:
        print(f'[Info] (train) epoch: {epoch} | train_epoch_loss: {running_loss} | loss: {loss.item()}')
    
    if opt.active_log:
        if opt.prop:
            wandb.log({'epoch': epoch,'train_epoch_loss': running_loss, 'loss_mse': loss.item(), 'loss_gnll': loss_gnll.item()})
        else:
            wandb.log({'epoch': epoch ,'train_epoch_loss': running_loss, 'loss': loss.item()})
        
    if epoch%5==0:
        model.eval()
        with torch.no_grad():
            if opt.task in ['binary','multiclass']:
                accuracy, auroc = classification_scores(model, validloader, device, opt.task,vision_dset)
                test_accuracy, test_auroc = classification_scores(model, testloader, device, opt.task,vision_dset)

                print('[EPOCH %d] VALID ACCURACY: %.3f, VALID AUROC: %.3f' % (epoch, accuracy,auroc))
                print('[EPOCH %d] TEST ACCURACY: %.3f, TEST AUROC: %.3f' % (epoch, test_accuracy,test_auroc))
                if opt.active_log:
                    wandb.log({'valid_accuracy': accuracy ,'valid_auroc': auroc })
                    wandb.log({'test_accuracy': test_accuracy ,'test_auroc': test_auroc })  
                if opt.task =='multiclass':
                    if accuracy > best_valid_accuracy:
                        best_valid_accuracy = accuracy
                        best_test_auroc = test_auroc
                        best_test_accuracy = test_accuracy
                        torch.save(model.state_dict(),'%s/bestmodel.pth' % (modelsave_path))
                else:
                    if accuracy > best_valid_accuracy:
                        best_valid_accuracy = accuracy
                    # if auroc > best_valid_auroc:
                    #     best_valid_auroc = auroc
                        best_test_auroc = test_auroc
                        best_test_accuracy = test_accuracy               
                        torch.save(model.state_dict(),'%s/bestmodel.pth' % (modelsave_path))

            else:
                valid_rmse, valid_NLL,_, _, _, _ = mean_sq_error(model, validloader, device, vision_dset, epoch, opt.epochs, opt.prop)
                test_rmse, test_NLL, y_pred_mu, y_pred_var, _, _ = mean_sq_error(model, testloader, device, vision_dset, epoch, opt.epochs, opt.prop)
                print('[EPOCH %d] VALID RMSE: %.3f' % (epoch, valid_rmse ))
                print('[EPOCH %d] TEST RMSE: %.3f' % (epoch, test_rmse ))
                print('[EPOCH %d] TEST NLL: %.3f' % (epoch, test_NLL ))
                if opt.active_log:
                    wandb.log({'valid_rmse': valid_rmse,'test_rmse': test_rmse})
                if valid_rmse < best_valid_rmse:
                    best_valid_rmse = valid_rmse
                    best_test_rmse = test_rmse
                    torch.save(model.state_dict(),'%s/bestmodel.pth' % (modelsave_path))
                if valid_NLL < best_valid_NLL:
                    best_valid_NLL = valid_NLL
                    best_test_NLL = test_NLL
                    #torch.save(model.state_dict(),'%s/bestmodel.pth' % (modelsave_path))
                    #---validの中にattentionの出力を入れよう
        
        print("[test] (test) y_pred_mu: ", str(y_pred_mu[0:5].tolist()))
        if opt.prop:
            print("[test] (test) y_pred_var: ", str(y_pred_var[0:5].tolist()))
        model.train()

output = pd.DataFrame(y_pred_mu.numpy(),columns = ["平均"])

if opt.prop:
    output['分散'] = y_pred_var

#print("[test] mu: ", y_pred_mu)
#print("[test] var: ", y_pred_var)

#stacked_data = np.vstack(X_test['data'])
#df_result = pd.DataFrame(stacked_data)
#df_result.set_index(test_indices, inplace=True)
#print(df_result)
#print(test_indices.tolist())

output.set_index([test_indices.tolist()],inplace = True)
print(output)
df_result = df2.loc[test_indices.tolist()]
print(df_result)
df_result.drop(columns = '補正タイム',inplace = True)
attribute_names = ['場所','開催何日目','芝・ダート','距離','馬場状態','血統登録番号','斤量','基準タイム','補正タイム総合値_noleak','レース番号','馬番','レースID','rpr_逃げ','rpr_先行','rpr_中団','rpr_後方']
df_result = df_result.rename(columns=dict(zip(df_result.columns[0:], attribute_names)))

#result = pd.concat([df_result,output],axis = 1)
result = pd.merge(df_result, output, left_index=True, right_index=True)

if opt.prop:
    cp = "prop"
else: cp = "conv"

with pd.ExcelWriter(f"/content/drive/MyDrive/SAINT/data/output_{cp}_{opt.epochs}eps_a{opt.alpha}_b{opt.beta}_c{opt.gamma}_{date}.xlsx") as writer:
    result.to_excel(writer)
    
#with pd.ExcelWriter("/content/drive/MyDrive/SAINT/data/output.xlsx") as writer:
#    output.to_excel(writer)

total_parameters = count_parameters(model)
print('TOTAL NUMBER OF PARAMS: %d' %(total_parameters))
if opt.task =='binary':
    print('AUROC on best model:  %.3f' %(best_test_auroc))
elif opt.task =='multiclass':
    print('Accuracy on best model:  %.3f' %(best_test_accuracy))
else:
    print('RMSE on best model:  %.3f' %(best_test_rmse))
    if opt.prop:
        print('NLL on best model:  %.3f' %(best_test_NLL))

if opt.active_log:
    if opt.task == 'regression':
        wandb.log({'total_parameters': total_parameters, 'test_rmse_bestep':best_test_rmse , 
        'cat_dims':len(cat_idxs) , 'con_dims':len(con_idxs) })        
    else:
        wandb.log({'total_parameters': total_parameters, 'test_auroc_bestep':best_test_auroc , 
        'test_accuracy_bestep':best_test_accuracy,'cat_dims':len(cat_idxs) , 'con_dims':len(con_idxs) })
