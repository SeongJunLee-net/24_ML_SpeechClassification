import os
import yaml
from argparse import ArgumentParser
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from src.models import load_model_from_config
from src.utils import prepare_test, FMCCdataset, label2gen


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
    models = []
    loaders = []
    paths = prepare_test(args.input_path, args.data_dir)
    # 1. Load Data, Models
    for checkpoint in args.model_checkpoints:
        with open(checkpoint + "/config.yaml") as f:
            config = yaml.load(f, Loader=yaml.FullLoader)
            
        # model
        model = load_model_from_config(config)
        if args.use_ema:
            model.load_state_dict(torch.load(checkpoint + '/ema.pt'))
        else:
            model.load_state_dict(torch.load(checkpoint + '/model.pt'))
        models.append(model)
        
        # dataset
        dataset = FMCCdataset(paths, **config)
        loaders.append(DataLoader(dataset, batch_size=config['batch_size'], num_workers=4, shuffle=False))
    

    # 2. Inference with each models
    merged_preds = []
    with torch.no_grad():
        for i, (model, loader) in enumerate(zip(models, loaders)):
            model.to(device)
            model.eval()
            model_preds = []
            
            for x in tqdm(loader, desc=f"Model ({i}/{len(models)}) Inference.."):
                x = x.to(device)
                pred = model(x).squeeze().cpu().numpy().tolist()
                model_preds.extend(pred)
            merged_preds.append(model_preds)
            
    merged_preds = np.round(np.mean(merged_preds, axis=0))
    print(merged_preds.shape)
    assert len(merged_preds) == len(paths)
    with open(args.result_path, 'w') as f:
        for path, pred, in zip(paths, merged_preds):
            line = f"{path} {label2gen[pred]}\n"
            f.write(line)