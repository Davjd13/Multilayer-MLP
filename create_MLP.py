import os
import warnings
import pandas as pd
import numpy as np
import networkx as nx
from sklearn.preprocessing import MinMaxScaler
from ts2vg import HorizontalVG
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')

def MLPNN(audio_files):
    hvg = HorizontalVG(weighted="abs_slope")
    output_folder_gexf = "/home/davjd313/MultilayerNetwork (BME_4)/Result/MLP.gexf"
    output_folder_edgelist = "/home/davjd313/MultilayerNetwork (BME_4)/Result/MLP.edgelist"

    # Clear the output folder
    if os.path.exists(output_folder_gexf):
        for file in os.listdir(output_folder_gexf):
            file_path = os.path.join(output_folder_gexf, file)
            try:
                if os.path.isfile(file_path):
                    os.unlink(file_path)  # Remove file
                elif os.path.isdir(file_path):
                    os.rmdir(file_path)  # Remove empty directory
            except Exception as e:
                print(f"Error deleting {file_path}: {e}")
    else:
        os.makedirs(output_folder_gexf) 

    if os.path.exists(output_folder_edgelist):
        for file in os.listdir(output_folder_edgelist):
            file_path = os.path.join(output_folder_edgelist, file)
            try:
                if os.path.isfile(file_path):
                    os.unlink(file_path)  # Remove file
                elif os.path.isdir(file_path):
                    os.rmdir(file_path)  # Remove empty directory
            except Exception as e:
                print(f"Error deleting {file_path}: {e}")
    else:
        os.makedirs(output_folder_edgelist)

    # Loop through each audio file
    for audio_file in audio_files:
        print("Processing", audio_file)
        data = pd.read_csv(os.path.join(audio_folder, audio_file))
        data = data.drop(data.columns[0], axis=1)

        # Loop over each value of n
        for n in [128, 256, 512]:
            A = {}
            layers = []
            columns_to_use = [1, 3]
            selected_columns = data.columns[columns_to_use]
            # Generate intra-layer adjacency matrices
            for idx, col in enumerate(selected_columns):
                layer = data[col][:n]
                hvg_i = hvg.build(layer)
                G_i = hvg_i.as_networkx()
                layers.append(layer)
                
                # Convert intra-layer graph to adjacency matrix and store
                A[f'A_{idx}_{idx}'] = nx.to_numpy_array(G_i, nodelist=range(n))

            # Generate inter-layer adjacency matrices
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
                                        A_ij[x_idx, y_idx] = abs(float((y - x) / (y_idx - x_idx)))
                                    else:
                                        z_between = max_ij[min(x_idx, y_idx) + 1: max(x_idx, y_idx)]
                                        if all(z < min(layer_i_scaled[x_idx], layer_j_scaled[y_idx]) for z in z_between):
                                            A_ij[x_idx, y_idx] = abs(float((y - x) / (y_idx - x_idx)))
                        A[f'A_{i}_{j}'] = A_ij

            # Create supra-adjacency matrix
            num_layers = len(layers)
            supra_adj_matrix = np.zeros((num_layers * n, num_layers * n), dtype=float)

            for (key, matrix) in A.items():
                i, j = map(int, key[2:].split('_'))
                row_offset = i * n
                col_offset = j * n
                supra_adj_matrix[row_offset:row_offset + n, col_offset:col_offset + n] = matrix

            # Generate graph from supra-adjacency matrix
            G = nx.from_numpy_array(supra_adj_matrix, create_using=nx.Graph)

            # Save graph as .edgelist file
            columns_str = "_".join(selected_columns)
            output_path_edgelist = os.path.join(output_folder_edgelist, f'MLP_{audio_file.split(".")[0]}_{n}_{columns_str}.weighted.edgelist')
            nx.write_weighted_edgelist(G, output_path_edgelist)

            # Assign layer attributes to nodes
            for node in G.nodes():
                layer = "Layer1" if node < n else "Layer2"
                G.nodes[node]["layer"] = layer

            # Assign edge type attributes
            for u, v, d in G.edges(data=True):
                G.edges[u, v]["connection_type"] = "intra-layer" if (u < n and v < n) or (u >= n and v >= n) else "inter-layer"

            # Save graph as .gexf file 
            output_path_gexf = os.path.join(output_folder_gexf, f'MLP_{audio_file.split(".")[0]}_{n}_{columns_str}.gexf')
            nx.write_gexf(G, output_path_gexf)

audio_folder = "/home/davjd313/MultilayerNetwork (BME_4)/Dataset/Audio"
audio_files = [f for f in os.listdir(audio_folder) if f.endswith('.csv')]

MLPNN(audio_files)

# # Save supra-adjacency matrix to file
# output_file = os.path.join(test_folder, f"{audio_file.split('.')[0]}_supra_adj_matrix_{n}.txt")
# np.savetxt(output_file, supra_adj_matrix, fmt="%.2f")
# print(f"Supra-adjacency matrix for n={n} saved to {output_file}")

# # Visualization of the multilayer network
# pos = {}
# for node in G.nodes:
#     layer = node // n  # Determine the layer based on node index
#     idx = node % n     # Determine the position within the layer
#     pos[node] = (layer * 2, idx)

# plt.figure(figsize=(10, 8))
# nx.draw(G, pos, with_labels=True, node_size=500, node_color="skyblue", edge_color="gray", font_size=10, font_weight="bold")

# # Adjust edge label format to 2 decimal places
# edge_labels = {edge: f"{weight:.2f}" for edge, weight in nx.get_edge_attributes(G, 'weight').items()}
# nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=8)

# plt.title(f"Multilayer Network Visualization for n={n}")
# plt.show()
