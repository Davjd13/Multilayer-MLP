import os
import warnings
import pandas as pd
import numpy as np
import networkx as nx
from sklearn.preprocessing import MinMaxScaler
from ts2vg import HorizontalVG

warnings.filterwarnings('ignore')
# intra/inter_adj: 256, 512, 1024
# intra/inter_adj_v1: 2048, 4096
# intra/inter_edge: 256, 512, 1024

def MLPNN(audio_files):
    hvg = HorizontalVG(weighted="slope")
    intra_adj_folder = "intra_edge"
    inter_adj_folder = "inter_edge"
    os.makedirs(intra_adj_folder, exist_ok=True)
    os.makedirs(inter_adj_folder, exist_ok=True)

    # Clear the contents of the folders before generating new adjacency lists
    for filename in os.listdir(intra_adj_folder):
        file_path = os.path.join(intra_adj_folder, filename)
        try:
            if os.path.isfile(file_path):
                os.unlink(file_path)
        except Exception as e:
            print(e)

    for filename in os.listdir(inter_adj_folder):
        file_path = os.path.join(inter_adj_folder, filename)
        try:
            if os.path.isfile(file_path):
                os.unlink(file_path)
        except Exception as e:
            print(e)

    for audio_file in audio_files:
        print("New")
        data = pd.read_csv(os.path.join(audio_folder, audio_file))
        data = data.drop(data.columns[0], axis=1)

        for n in [256, 512, 1024]:
            A = {}
            layers = []
            # intra_layers
            for idx, col in enumerate(data.columns):
                layer = data[col][:n]
                hvg_i = hvg.build(layer)
                G_i = hvg_i.as_networkx()
                filename = os.path.join(intra_adj_folder, f'{audio_file}_{n}_layer{idx}.weighted.edgelist')
                nx.write_weighted_edgelist(G_i, filename)
                layers.append(layer)

            # inter_layers
            for i, layer_i in enumerate(layers):
                layer_i_scaled = MinMaxScaler().fit_transform(layer_i.values.reshape(-1, 1)).flatten()
                for j, layer_j in enumerate(layers):
                    if i != j:
                        layer_j_scaled = MinMaxScaler().fit_transform(layer_j.values.reshape(-1, 1)).flatten()
                        max_ij = np.maximum(layer_i_scaled, layer_j_scaled)
                        A_ij = np.zeros((n, n), dtype=float)
                        for x_idx, x in enumerate(layer_i):
                            for y_idx, y in enumerate(layer_j):
                                if x_idx != y_idx:
                                    if abs(x_idx - y_idx) == 1:
                                        A_ij[x_idx, y_idx] = float((y - x) / (y_idx - x_idx))
                                    else:
                                        z_between = max_ij[min(x_idx, y_idx) + 1: max(x_idx, y_idx)]
                                        if all(z < min(layer_i_scaled[x_idx], layer_j_scaled[y_idx]) for z in
                                               z_between):
                                            A_ij[x_idx, y_idx] = float((y - x) / (y_idx - x_idx))
                        A[f'A_{i}_{j}'] = A_ij

            used_keys = set()
            for name in list(A.keys()):
                i, j = map(int, name[2:].split('_'))
                if name not in used_keys:
                    B_ij = np.block([[np.zeros((n, n), dtype=int), A[f'A_{i}_{j}']],
                                     [A[f'A_{j}_{i}'], np.zeros((n, n), dtype=int)]])
                    used_keys.add(name)
                    used_keys.add(f'A_{j}_{i}')
                    G_i_j = nx.from_numpy_array(B_ij, edge_attr='weight')
                    filename = os.path.join(inter_adj_folder, f'{audio_file}_{n}_layer{i}{j}.weighted.edgelist')
                    nx.write_weighted_edgelist(G_i_j, filename)


audio_folder = 'D:\\TS2VG\\Audio'
audio_files = [f for f in os.listdir(audio_folder) if f.endswith('.csv')]

MLPNN(audio_files)


