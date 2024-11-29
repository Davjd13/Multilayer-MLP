import os
import warnings
import pandas as pd
import numpy as np
import networkx as nx
from sklearn.preprocessing import MinMaxScaler
from ts2vg import HorizontalVG
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')

def Multiplex(audio_files):
    hvg = HorizontalVG(weighted="abs_slope")
    output_folder_gexf = "/home/davjd313/MultilayerNetwork (BME_4)/Result/Multiplex.gexf"
    output_folder_edgelist = "/home/davjd313/MultilayerNetwork (BME_4)/Result/Multiplex.edgelist"

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
            for i in range(len(layers)):
                for j in range(len(layers)):
                    if i != j:
                        # Create identity matrix for inter-layer connections
                        A_ij = np.eye(n)
                        A[f'A_{i}_{j}'] = A_ij

            # Create supra-adjacency matrix
            num_layers = len(layers)
            supra_adj_matrix = np.zeros((num_layers * n, num_layers * n), dtype=float)

            # Fill supra-adjacency matrix with intra- and inter-layer adjacency matrices
            for key, matrix in A.items():
                i, j = map(int, key[2:].split('_'))
                row_offset = i * n
                col_offset = j * n
                supra_adj_matrix[row_offset:row_offset + n, col_offset:col_offset + n] = matrix

            # Generate graph from supra-adjacency matrix
            G = nx.from_numpy_array(supra_adj_matrix, create_using=nx.Graph)

            # Save graph as .edgelist file
            columns_str = "_".join(selected_columns)
            output_path_edgelist = os.path.join(output_folder_edgelist, f'Multiplex_{audio_file.split(".")[0]}_{n}_{columns_str}.weighted.edgelist')
            nx.write_weighted_edgelist(G, output_path_edgelist)

            # Assign layer attributes to nodes
            for node in G.nodes():
                layer = "Layer1" if node < n else "Layer2"
                G.nodes[node]["layer"] = layer

            # Assign edge type attributes
            for u, v, d in G.edges(data=True):
                G.edges[u, v]["connection_type"] = "intra-layer" if (u < n and v < n) or (u >= n and v >= n) else "inter-layer"
            
            # Save graph as .gexf file
            output_path_gexf = os.path.join(output_folder_gexf, f'Multiplex_{audio_file.split(".")[0]}_{n}_{columns_str}.gexf')
            nx.write_gexf(G, output_path_gexf)

audio_folder = "/home/davjd313/MultilayerNetwork (BME_4)/Dataset/Audio"
audio_files = [f for f in os.listdir(audio_folder) if f.endswith('.csv')]

Multiplex(audio_files)
