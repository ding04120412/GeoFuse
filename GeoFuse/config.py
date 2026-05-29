import argparse
import torch
import os

parser = argparse.ArgumentParser(description='GeoFuse-Final-Stable')


parser.add_argument('--dataset', type=str, default='dblp', help='dblp, uslegis, 或者其他新数据集')
parser.add_argument('--data_pt_path', type=str, default='./data/', help='数据存储路径')
parser.add_argument('--device', type=int, default=0, help='GPU ID')
parser.add_argument('--seed', type=int, default=1111)
parser.add_argument('--max_epoch', type=int, default=500)
parser.add_argument('--patience', type=int, default=50)
parser.add_argument('--log_interval', type=int, default=1)


parser.add_argument('--nhid', type=int, default=32)
parser.add_argument('--causal_conv_depth', type=int, default=5, help='ASA window size')
parser.add_argument('--evol_weight', type=float, default=0.5, help='LEL alpha weight')
parser.add_argument('--r', type=float, default=2.0)
parser.add_argument('--t', type=float, default=1.0)
parser.add_argument('--eps', type=float, default=1e-15)
parser.add_argument('--curvature', type=float, default=1.0)

args = parser.parse_args()


DATASET_CONFIG = {
    'dblp':      {'lr': 0.0005, 'num_layers': 3, 'test_length': 3, 'dropout': 0.1,  'weight_decay': 5e-7},
    'ia-enron':  {'lr': 0.001,  'num_layers': 4, 'test_length': 3, 'dropout': 0.1,  'weight_decay': 5e-7},
    'uslegis':   {'lr': 0.005,  'num_layers': 3, 'test_length': 3, 'dropout': 0.3,  'weight_decay': 0.0},
}

dataset_name = args.dataset.lower()
if dataset_name in DATASET_CONFIG:
    conf = DATASET_CONFIG[dataset_name]
    for key, value in conf.items():
        setattr(args, key, value)
    print(f"--- [INFO] Dataset {args.dataset} recognized. Table 8 config loaded. ---")
else:
    conf = {
        'lr': 0.001, 
        'num_layers': 3, 
        'test_length': 3, 
        'dropout': 0.1,  
        'weight_decay': 5e-7
    }
    for key, value in conf.items():
        setattr(args, key, value)
    print(f"--- [INFO] Dataset '{args.dataset}' NOT recognized. Using generic fallback config. ---")

args.output_path = os.path.join('./output/', args.dataset)
if not os.path.isdir(args.output_path): os.makedirs(args.output_path)
args.log_file = os.path.join(args.output_path, 'train.log')
args.device = torch.device(f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu')