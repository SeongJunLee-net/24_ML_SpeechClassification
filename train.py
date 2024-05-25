import os
import random
import time
import yaml
from argparse import ArgumentParser
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import wandb
from tqdm import tqdm
from src.models import load_model_from_config
from src.utils import prepare_train, prepare_val, FMCCdataset, label2gen
from src.models.resnet import Bottleneck,BasicBlock


def parse_arguments():
    parser = ArgumentParser()
    parser.add_argument("--config", type=str, default="config.yaml")
    return parser.parse_args()
    
def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed) 
    torch.cuda.manual_seed_all(seed)
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False
    
def init_train(args):
    run_id = time.strftime("%m%d%H%M%S")
    with open(args.config) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    
    result_dir = f"{config['result_dir']}/{run_id}/"
    os.makedirs(result_dir)
    with open(result_dir + "/config.yaml", 'w') as f:
        yaml.dump(config, f)
    
    if config['use_wandb']:
        wandb.init(name=run_id,
                   config=config, 
                   entity=config['wandb_entity'], 
                   project=config['wandb_project']
                   )
    
    return config, result_dir

if __name__ == "__main__":
    # 0. Initialization
    args = parse_arguments()
    config, result_dir = init_train(args)
    device = torch.device("cuda") if torch.cuda.is_available() else "cpu"
    seed_everything(config['seed'])


    
    # 1. Prepare Dataset
    train_paths, train_targets = prepare_train(config['train_ctl'], config['train_dir'])
    train_dataset = FMCCdataset(train_paths, train_targets, **config)
    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], num_workers=4, shuffle=True)

    ## 변경
    if config['model']['params']['block'] == 'Bottleneck':
        config['model']['params']['block'] = Bottleneck
    else:
        config['model']['params']['block'] = BasicBlock
    

    
    val_paths, val_targets = prepare_val(config['val_ctl'], config['val_dir'])
    val_dataset = FMCCdataset(val_paths, val_targets, **config)
    val_loader = DataLoader(val_dataset, batch_size=config['batch_size'], num_workers=4)

    # 2. Prepare model, optimizer, etc.
    model = load_model_from_config(config)
    model.to(device)
    
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'])

    # 3. Train
    for epoch in range(config['epochs']):
        # Metrics
        train_loss = 0
        val_loss = 0
        val_ys = np.array([])
        val_preds = np.array([])
        best_acc = 0
        
        # Train epoch
        model.train()
        for x, y in tqdm(train_loader, leave=False):
            optimizer.zero_grad()
            x = x.to(device)
            y = y.to(device)

            pred = model(x)
            loss = criterion(pred, y)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        # Validation epoch
        model.eval()
        with torch.no_grad():
            for x, y in tqdm(val_loader, leave=False):
                x = x.to(device)
                y = y.to(device)
                pred = model(x)
                loss = criterion(pred, y)
                val_loss += loss.item()
                
                pred = torch.round(pred)
                val_preds = np.append(val_preds, pred.squeeze().cpu().numpy())
                val_ys = np.append(val_ys, y.squeeze().cpu().numpy())
        
        # Compute metrics
        
        train_loss /= len(train_loader)
        val_loss /= len(val_loader)
        val_acc = np.mean(val_ys == val_preds)
    
        
        # Logging & Update
        print(f"[train_loss: {train_loss}] [val_loss: {val_loss}] [val_acc: {val_acc}]")
        if config['use_wandb']:
            wandb.log({
                "trian_loss": train_loss,
                "val_loss": val_loss,
                "val_acc": val_acc,
                "conf_mat" : wandb.plot.confusion_matrix(
                    y_true=val_ys,
                    preds=val_preds,
                    class_names=[label2gen[0], label2gen[1]] # ['male', 'feml']
                )
            })

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), result_dir + f"/model.pt")
        





        
