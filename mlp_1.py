from ts2vg import HorizontalVG
import networkx as nx
import pandas as pd
import csv
import numpy as np
import os
import warnings
import MLE_functions
import itertools
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

warnings.filterwarnings('ignore')


def intra_layer_stat_to_csv(audio_folder):
    with open("intra_layer_v3.csv", 'w', newline='', buffering=1) as csvfile:
        fieldnames = ['key1', 'key2', 'key3', 'n', 'm', 'AD', 'ASPL', 'GE', 'ACC', 'T', 'S', 'Q', 'AWD', 'CC', 'Kmin', 'Fit', 'Param1', 'Param2']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        # Get a list of all files in the audio folder
        audio_files = (f for f in os.listdir(audio_folder))
        # audio_files = list(itertools.islice(os.listdir(audio_folder), 3))

        for audio_file in audio_files:
            # Extracting key1, key2, and key3 from the audio file name
            file_parts = os.path.splitext(audio_file)[0].split('_')
            key1 = "_".join(file_parts[:-2])  # Joining all parts except last two
            key2 = file_parts[-2]  # Second last part
            key3 = file_parts[-1]  # Last part

            # Read adjacency list from the audio file
            with open(os.path.join(audio_folder, audio_file), "rb") as data:
                G = nx.read_weighted_edgelist(data)
                n = nx.number_of_nodes(G)
                m = nx.number_of_edges(G)
                AD = round(2 * m / n, 3)  # average degree
                ASPL = round(nx.average_shortest_path_length(G), 3)  # average shortest path length
                GE = round(nx.global_efficiency(G), 3)
                ACC = round(nx.average_clustering(G), 3)
                T = round(nx.transitivity(G), 3)
                community = nx.community.greedy_modularity_communities(G, weight='weight')
                S = len(community)  # number of communities (greedy_modularity)
                Q = nx.community.modularity(G, community)  # modularity
                weighted_degrees = []
                for node in G.nodes():
                    total_weight = sum(G[node][neighbor]['weight'] for neighbor in G[node])
                    weighted_degrees.append(total_weight)
                AWD = sum(weighted_degrees) / len(weighted_degrees)  # average weighted degree
                CC = round(sum(nx.closeness_centrality(G).values()), 3)
                k_min = min(dict(G.degree()).values())  # Get the minimum degree directly
                fit_result = MLE_functions.fit('Graph', G, k_min=k_min, plot_type='ccdf', save=False)

                writer.writerow({
                    'key1': key1, 'key2': key2, 'key3': key3,
                    'n': n, 'm': m, 'AD': AD, 'ASPL': ASPL, 'GE': GE,
                    'ACC': ACC, 'T': T, 'S': S, 'Q': Q, 'CC': CC, 'AWD': AWD,
                    'Kmin': fit_result[0], 'Fit': fit_result[1],
                    'Param1': np.round(fit_result[2][0][0], 2),
                    'Param2': np.round(fit_result[2][0][1], 2) if len(fit_result[2][0]) > 1 else 0
                })


def inter_layer_stat_to_csv(audio_folder):
    with open("inter_layer_v3.csv", 'w', newline='', buffering=1) as csvfile:
        fieldnames = ['key1', 'key2', 'key3', 'n', 'm', 'AD', 'ASPL', 'GE', 'ACC', 'T', 'S', 'Q', 'AWD', 'CC', 'Kmin', 'Fit', 'Param1', 'Param2']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        audio_files = (f for f in os.listdir(audio_folder))

        for audio_file in audio_files:
            # Extracting key1, key2, and key3 from the audio file name
            file_parts = os.path.splitext(audio_file)[0].split('_')
            key1 = "_".join(file_parts[:-2])  # Joining all parts except last two
            key2 = file_parts[-2]  # Second last part
            key3 = file_parts[-1]  # Last part

            # Read adjacency list from the audio file
            with open(os.path.join(audio_folder, audio_file), "rb") as data:
                G = nx.read_weighted_edgelist(data)
                n = nx.number_of_nodes(G)
                m = nx.number_of_edges(G)
                AD = round(2 * m / n, 3)  # average degree
                if not nx.is_connected(G):
                    ASPLs = []
                    for C in (G.subgraph(c).copy() for c in nx.connected_components(G)):
                        ASPLs.append(round(nx.average_shortest_path_length(C), 3))
                    ASPL = sum(ASPLs) / len(ASPLs) if ASPLs else 0
                else:
                    ASPL = round(nx.average_shortest_path_length(G), 3)
                GE = round(nx.global_efficiency(G), 3)
                ACC = round(nx.average_clustering(G), 3)
                T = round(nx.transitivity(G), 3)
                community = nx.community.greedy_modularity_communities(G, weight='weight')
                S = len(community)
                Q = nx.community.modularity(G, community)
                weighted_degrees = []
                for node in G.nodes():
                    total_weight = sum(G[node][neighbor]['weight'] for neighbor in G[node])
                    weighted_degrees.append(total_weight)
                AWD = sum(weighted_degrees) / len(weighted_degrees)  # average weighted degree
                CC = round(sum(nx.closeness_centrality(G).values()), 3)
                k_min = MLE_functions.degree_list(G).min()
                fit_result = MLE_functions.fit('Graph', G, k_min=k_min, plot_type='ccdf', save=False)

                writer.writerow({
                    'key1': key1, 'key2': key2, 'key3': key3,
                    'n': n, 'm': m, 'AD': AD, 'ASPL': ASPL, 'GE': GE,
                    'ACC': ACC, 'T': T, 'S': S, 'Q': Q, 'CC': CC, 'AWD': AWD,
                    'Kmin': fit_result[0], 'Fit': fit_result[1],
                    'Param1': np.round(fit_result[2][0][0], 2),
                    'Param2': np.round(fit_result[2][0][1], 2) if len(fit_result[2][0]) > 1 else 0
                })


audio_folder_path_intra = r"D:\TS2VG\ts2vg_1\intra_edge"  # r"D:\TS2VG\ts2vg_1\intra_adj"
audio_folder_path_inter = r"D:\TS2VG\ts2vg_1\inter_edge"  # r"D:\TS2VG\ts2vg_1\intra_adj"
intra_layer_stat_to_csv(audio_folder_path_intra)
inter_layer_stat_to_csv(audio_folder_path_inter)
