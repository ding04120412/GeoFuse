import os
import torch
import pickle
import pandas as pd
from utils.util import get_logger 

def process_generalized_temporal_data(data_file, args):
    get_logger().info(f"--- [INFO] Triggering Generalized Fallback Loader for {args.dataset} from {data_file} ---")
    
    df = pd.read_csv(data_file, sep=r'\s+', comment='%', header=None)
    
    if df.shape[1] >= 4:
        df.columns = ['u', 'v', 'weight', 'ts'] + list(df.columns[4:])
    elif df.shape[1] == 3:
        df.columns = ['u', 'v', 'ts']
    else:
        raise ValueError(f"Data format not recognized. Expected 3 or 4 columns, got {df.shape[1]}")
    
    unique_nodes = pd.unique(df[['u', 'v']].values.ravel())
    node_map = {old_id: new_id for new_id, old_id in enumerate(unique_nodes)}
    df['u'] = df['u'].map(node_map)
    df['v'] = df['v'].map(node_map)
    num_nodes = len(unique_nodes)

    df = df.sort_values(by='ts')
    
    total_snapshots = 36
    df['snapshot'] = pd.qcut(df['ts'], q=total_snapshots, labels=False, duplicates='drop')
    actual_snapshots = df['snapshot'].nunique()
    
    edge_index_list = []
    for t in range(actual_snapshots):
        snap_df = df[df['snapshot'] == t]
        edge_index = torch.tensor(snap_df[['u', 'v']].values.T, dtype=torch.long)
        edge_index_list.append(edge_index)
        
    std_data = {
        'num_nodes': num_nodes,
        'weights': None, 
        'edge_index_list': edge_index_list,
        'pos_edge_index_list': edge_index_list, 
        'neg_edge_index_list': [None] * actual_snapshots, 
        'new_pos_edge_index_list': None,
        'new_neg_edge_index_list': None,
        'time_length': actual_snapshots
    }
    return std_data

def loader(args):
    folder = os.path.join(args.data_pt_path, args.dataset)
    
    possible_files = [
        os.path.join(args.data_pt_path, f'{args.dataset}.data'),
        os.path.join(folder, f'{args.dataset}.data'),
        os.path.join(args.data_pt_path, f'{args.dataset}.pt')
    ]
    filepath = next((f for f in possible_files if os.path.isfile(f)), None)
    
    if filepath is not None:
        get_logger().info(f'loading {args.dataset} from {filepath} ...')
        try:
            raw_data = torch.load(filepath, map_location='cpu')
        except Exception:
            with open(filepath, 'rb') as f:
                raw_data = pickle.load(f, encoding='latin1')

        def get_v(d, keys, keywords):
            for k in keys:
                if k in d: return d[k]
            for k in d.keys():
                if all(w in k.lower() for w in keywords): return d[k]
            return None

        std_data = {
            'num_nodes': raw_data.get('num_nodes', 0),
            'weights': raw_data.get('weights', None),
            'edge_index_list': get_v(raw_data, ['edge_index_list', 'adjs'], ['edge', 'list']),
            'pos_edge_index_list': get_v(raw_data, ['pos_edge_index_list', 'pedges'], ['pos', 'index']),
            'neg_edge_index_list': get_v(raw_data, ['neg_edge_index_list', 'nedges'], ['neg', 'index']),
            'new_pos_edge_index_list': get_v(raw_data, ['new_pos_edge_index_list', 'new_pedges'], ['new', 'pos']),
            'new_neg_edge_index_list': get_v(raw_data, ['new_neg_edge_index_list', 'new_nedges'], ['new', 'neg']),
        }
        
        if std_data['edge_index_list'] is None: raise KeyError("No edges found!")
        if std_data['pos_edge_index_list'] is None: std_data['pos_edge_index_list'] = std_data['edge_index_list']
        std_data['time_length'] = len(std_data['edge_index_list'])
        return std_data

    possible_edges_files = [
        os.path.join(args.data_pt_path, f'{args.dataset}.edges'),
        os.path.join(folder, f'{args.dataset}.edges'),
        os.path.join(args.data_pt_path, f'{args.dataset}.txt'),
        os.path.join(folder, f'{args.dataset}.txt'),
        os.path.join(args.data_pt_path, f'{args.dataset}-employees.edges'),
        os.path.join(args.data_pt_path, f'{args.dataset}-reply.edges')
    ]
    edges_filepath = next((f for f in possible_edges_files if os.path.isfile(f)), None)
    
    if edges_filepath is not None:
        return process_generalized_temporal_data(edges_filepath, args)

    get_logger().error(f"Cannot find any valid data file for dataset: {args.dataset}")
    return None

def prepare_train_test_data(data, t, device):
    def ensure_2_rows(tensor):
        if tensor is not None and tensor.dim() == 2 and tensor.shape[0] != 2:
            return tensor.t().contiguous()
        return tensor

    edge_index = ensure_2_rows(data['edge_index_list'][t].long().to(device))
    pos_idx = ensure_2_rows(data['pos_edge_index_list'][t].long().to(device))
    
    n_l = data.get('neg_edge_index_list')

    neg_idx = ensure_2_rows(n_l[t].long().to(device)) if (n_l and n_l[t] is not None) else None
    
    np_l = data.get('new_pos_edge_index_list')
    np_idx = ensure_2_rows(np_l[t].long().to(device)) if (np_l and t < len(np_l) and np_l[t] is not None) else None
    
    nn_l = data.get('new_neg_edge_index_list')
    nn_idx = ensure_2_rows(nn_l[t].long().to(device)) if (nn_l and t < len(nn_l) and nn_l[t] is not None) else None
    
    return edge_index, pos_idx, neg_idx, np_idx, nn_idx