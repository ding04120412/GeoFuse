# GeoFuse

PyTorch implementation of the paper:

**"GeoFuse: Adaptive Geometry Fusion for Scalable Temporal Link Prediction"**

This repository contains the anonymous implementation used for the double-blind review process.

---

# 1. Quick Start

Example:

```bash
python main.py --dataset=dblp
```

Some important configuration parameters:

- `--dataset`, `--data_pt_path`  
  Dataset name and data directory path.

- `--test_length`  
  Number of snapshots used for testing.

- `--causal_conv_depth`  
  Historical window size used in the ASA module.

- `--evol_weight`  
  Weight coefficient of the Manifold Evolution Loss (MEL).

- `--r`, `--t`  
  Margin and scaling parameters used in the hyperbolic distance decoding of MEL.

Please refer to `./config.py` for the full list of configuration options.

---

# 2. Data Format and Preprocessing

The data loader (`utils/data_utils.py`) supports two input formats.

## Format A: Preprocessed Serialized Files (`.data` / `.pt`)

If a `.data` or `.pt` file is detected, the loader will directly parse the serialized dictionary object.

The loader supports several commonly used key names automatically:

- `num_nodes`  
  Total number of nodes in the dynamic graph.

- `weights`  
  Node feature tensor. Set to `None` if node features are unavailable.

- `edge_index_list` (or `adjs`)  
  List of graph topology tensors for each snapshot.

- `pos_edge_index_list` (or `pedges`)  
  Positive edge samples used for temporal link prediction.

- `neg_edge_index_list` (or `nedges`)  
  Negative edge samples used for training and evaluation.

- `new_pos_edge_index_list` (or `new_pedges`)  
  Positive edges that appear for the first time at snapshot `t+1`.

- `new_neg_edge_index_list` (or `new_nedges`)  
  Negative samples paired with newly appeared edges.

---

## Format B: Raw Temporal Edge Lists (`.edges` / `.txt`)

The framework also supports loading raw temporal edge lists directly from text files.

Supported input formats:

- `[source, target, timestamp]`
- `[source, target, weight, timestamp]`

The preprocessing pipeline performs the following steps automatically:

1. **Node Remapping**  
   Arbitrary node IDs are remapped into a continuous index range from `0` to `N-1`.

2. **Temporal Discretization**  
   Continuous timestamps are partitioned into 36 chronological snapshots using quantile-based discretization (`pd.qcut`).

3. **Snapshot Construction**  
   Temporal edge streams are converted into synchronized dynamic graph snapshots for training and evaluation.