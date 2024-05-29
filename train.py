import os
import random
import time
import yaml
from argparse import ArgumentParser
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from ema_pytorch import EMA
import wandb
from tqdm import tqdm
from src.models import load_model_from_config
from src.models.utils import BCELoss
from src.utils import prepare_train, prepare_val, FMCCdataset, label2gen
from src.models.utils import get_optimizer_class, get_scheduler_class


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
    run_id = time.strftime("%m%d%H%M%S") # time_tuple
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


    
    ## 1. Prepare Dataset
    train_paths, train_targets = prepare_train(config['train_ctl'], config['train_dir'])
    train_dataset = FMCCdataset(train_paths, train_targets, **config)
    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], num_workers=4, shuffle=True)
    
    
    val_paths, val_targets = prepare_val(config['val_ctl'], config['val_dir'])
    val_dataset = FMCCdataset(val_paths, val_targets, **config)
    val_loader = DataLoader(val_dataset, batch_size=config['batch_size'], num_workers=4)

    # 2. Prepare model, optimizer, etc.
    model = load_model_from_config(config)
    model.to(device)
    ema = EMA(
        model,
        beta=0.9999,
        update_after_step=100,
        update_every=10
    )
    
    criterion = BCELoss(**config)
    OptimizerCls = get_optimizer_class(config['optimizer']['target'])
    optimizer = OptimizerCls(model.parameters(), config['lr'])

    schedulerCls = get_scheduler_class(config['scheduler']['target'])
    scheduler = schedulerCls(optimizer, **config['scheduler']['params']) if schedulerCls else None

    # 3. Train
    for epoch in range(config['epochs']):
        # Metrics
        train_loss = 0
        val_loss = 0
        val_ys = np.array([])
        val_preds = np.array([])
        best_val_acc = 0
        
        ema_loss = 0
        ema_ys = np.array([])
        ema_preds = np.array([])
        best_ema_acc = 0
        
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
            
            ema.update()
            train_loss += loss.item()
        
        # Validation epoch
        model.eval()
        ema.eval()
        with torch.no_grad():
            for x, y in tqdm(val_loader, leave=False):
                x = x.to(device)
                y = y.to(device)
                val_pred = model(x)
                val_loss = criterion(val_pred, y)
                val_loss += val_loss.item()
                
                val_pred = torch.round(val_pred)
                val_preds = np.append(val_preds, val_pred.squeeze().cpu().numpy())
                val_ys = np.append(val_ys, y.squeeze().cpu().numpy())
                
                ema_pred = ema(x)
                ema_loss = criterion(ema_pred, y)
                ema_loss += ema_loss.item()
                
                ema_pred = torch.round(ema_pred)
                ema_preds = np.append(ema_preds, ema_pred.squeeze().cpu().numpy())
                ema_ys = np.append(ema_ys, y.squeeze().cpu().numpy())
                
        # Compute metrics
        
        train_loss /= len(train_loader)
        val_loss /= len(val_loader)
        ema_loss /= len(val_loader)
        val_acc = np.mean(val_ys == val_preds)
        ema_acc = np.mean(ema_ys == ema_preds)
    
        
        # Logging & Update
        print(f"[Epoch: {epoch+1}/{config['epochs']}] [train_loss: {train_loss}]")
        print(f"[val_loss: {val_loss}] [val_acc: {val_acc}]")
        print(f"[ema_loss]: {ema_loss}] [ema_acc: {ema_acc}]\n")

        if config['use_wandb']:
            wandb.log({
                "train_loss": train_loss,
                "val_loss": val_loss,
                "val_acc": val_acc,
                "ema_loss": ema_loss,
                "ema_acc": ema_acc,
                "val_conf_mat" : wandb.plot.confusion_matrix(
                    y_true=val_ys,
                    preds=val_preds,
                    class_names=[label2gen[0], label2gen[1]] # ['male', 'feml']
                ),
                "ema_conf_mat" : wandb.plot.confusion_matrix(
                    y_true=ema_ys,
                    preds=ema_preds,
                    class_names=[label2gen[0], label2gen[1]] # ['male', 'feml']
                )
            })

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), result_dir + f"/model.pt")
        if ema_acc > best_ema_acc:
            best_ema_acc = ema_acc
            torch.save(ema.ema_model.state_dict(), result_dir + f"/ema.pt")




        
