import torch
import numpy as np
import networkx as nx
from tqdm import tqdm
from torch_geometric.utils import to_networkx
from torch_geometric.data import Data

def compute_node_hyperbolicity(G, node, k_hop=3, num_samples=100): 

    try:
        subg = nx.ego_graph(G, node, radius=k_hop)
    except:
        return 0.0
        
    nodes_list = list(subg.nodes())
    if len(nodes_list) < 4:
        return 0.0 
    
    deltas = []
    iter_count = min(len(nodes_list) * 2, num_samples) 
    
    for _ in range(iter_count):
        try:
            quad = np.random.choice(nodes_list, 4, replace=False)
            u, v, w, z = quad
            
            try:
                d_uv = nx.shortest_path_length(subg, u, v)
                d_wz = nx.shortest_path_length(subg, w, z)
                d_uw = nx.shortest_path_length(subg, u, w)
                d_vz = nx.shortest_path_length(subg, v, z)
                d_uz = nx.shortest_path_length(subg, u, z)
                d_vw = nx.shortest_path_length(subg, v, w)
            except nx.NetworkXNoPath:
                continue
            
            s1 = d_uv + d_wz
            s2 = d_uw + d_vz
            s3 = d_uz + d_vw
            
            l, m, s = sorted([s1, s2, s3], reverse=True)
            deltas.append((l - m) / 2.0)
        except Exception:
            continue
            
    return max(deltas) if deltas else 0.0

def precompute_deltas(data, k_hop=3, sample_ratio=0.5): 
    
    num_nodes = data['num_nodes']
    
    full_compute = num_nodes < 2000
    mode_str = "Full Compute" if full_compute else f"Sampling (ratio={sample_ratio})"
    print(f"Pre-computing Deltas [{mode_str}] for accurate geometry...")

    all_deltas = []
    
    for t, edge_index in enumerate(data['edge_index_list']):
        edge_index_tensor = edge_index.long().cpu()
        data_obj = Data(edge_index=edge_index_tensor, num_nodes=num_nodes)
        G = to_networkx(data_obj, to_undirected=True)
        
        node_deltas = np.zeros(num_nodes)
        
        if full_compute:
            target_nodes = list(G.nodes())
        else:
            degrees = dict(G.degree())
            active_nodes = [n for n, d in degrees.items() if d > 0]
            if not active_nodes:
                all_deltas.append(torch.zeros(num_nodes))
                continue
            num_samples = int(len(active_nodes) * sample_ratio)
            target_nodes = np.random.choice(active_nodes, num_samples, replace=False)


        for node in tqdm(target_nodes, desc=f"Snapshot {t}", leave=False):
            delta = compute_node_hyperbolicity(G, node, k_hop=k_hop)
            node_deltas[node] = delta
            
        if not full_compute:
            mean_delta = np.mean([x for x in node_deltas if x > 0]) if np.sum(node_deltas) > 0 else 0
            node_deltas[node_deltas == 0] = mean_delta
        
        max_d = np.max(node_deltas)
        if max_d > 0:
            node_deltas = node_deltas / max_d
            
        final_deltas = torch.tensor(node_deltas, dtype=torch.float)
        all_deltas.append(final_deltas)
        
    return all_deltas