import os
import random
import yaml
from argparse import ArgumentParser
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from src.models import build_model_from_config
from src.utils import prepare_test, FMCCdataset, label2gen, print_torchsummary


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed) 
    torch.cuda.manual_seed_all(seed)
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False
    
def parse_arguments():
    parser = ArgumentParser()
    parser.add_argument("--input_path", type=str, default="fmcc_test.ctl")
    parser.add_argument("--result_path", type=str, default="Perceptron_test_results.txt")
    parser.add_argument("--data_dir", type=str, default="dataset/raw16k/test/")
    parser.add_argument("--model_checkpoints", type=lambda s: s.split(','), help="Model checkpoints, Comma-Seperated")
    parser.add_argument("--use_ema", action='store_true', help="Whether or not to use EMA model. If False, use original model instead.")
    return parser.parse_args()
    

if __name__ == "__main__":
    # 0. Initialization
    args = parse_arguments()
    device = torch.device("cuda") if torch.cuda.is_available() else "cpu"
    paths = prepare_test(args.input_path, args.data_dir)
 
    merged_preds = []
    for i, checkpoint in enumerate(args.model_checkpoints):
        # Read config from the checkpoint
        with open(checkpoint + "/config.yaml") as f:
            config = yaml.load(f, Loader=yaml.FullLoader)
        seed_everything(config['seed'])
        
        # Load model
        model = build_model_from_config(config)
        if args.use_ema:
            model.load_state_dict(torch.load(checkpoint + '/ema.pt'))
        else:
            model.load_state_dict(torch.load(checkpoint + '/model.pt'))
        model.eval()
       
        # Dataset
        dataset = FMCCdataset(paths, **config)
        loader = (DataLoader(
            dataset,
            batch_size=config['batch_size'], 
            num_workers=1, 
            shuffle=False
        ))
    
        input_size = next(iter(loader)).shape[1:]
        model.to(device)
        print_torchsummary(model, input_size=input_size, batch_size=1)

        model_preds = []
        with torch.no_grad():
            for x in tqdm(loader, desc=f"Model ({i}/{len(args.model_checkpoints)}) Inference.."):
                x = x.to(device)
                pred = model(x).squeeze().cpu().tolist()
                model_preds.extend(pred)
                
            merged_preds.append(model_preds)
            
        with torch.no_grad():
            for x in tqdm(loader, desc=f"Model ({i}/{len(args.model_checkpoints)}) Inference.."):
                x = x.to(device)
                pred = model(x).squeeze().cpu().tolist()
                model_preds.extend(pred)
                
            merged_preds.append(model_preds)
        merged_preds = torch.Tensor(merged_preds).mean(dim=0).squeeze()
        print(merged_preds)
        merged_preds = merged_preds > 0.5

    assert len(merged_preds) == len(paths)
    with open(args.result_path, 'w') as f:
        for path, pred, in zip(paths, merged_preds):
            line = f"{path} {label2gen[int(pred.item())]}\n"
            f.write(line)