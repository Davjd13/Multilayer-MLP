import os
import csv
import networkx as nx
import MLE_functions_v2
import numpy as np

# Define the folder containing audio files
# audio_folder = "/home/davjd313/MultilayerNetwork (BME_4)/Result/MLP.edgelist"
audio_folder = "/home/davjd313/MultilayerNetwork (BME_4)/Result/Multiplex.edgelist"

# Get the list of audio files in the folder
audio_files = sorted([os.path.join(audio_folder, f) for f in os.listdir(audio_folder) if f.endswith('.weighted.edgelist')])

# Define the CSV file path for storing results
MLP_stat = "/home/davjd313/MultilayerNetwork (BME_4)/Result/stat_results/MLP_stat.csv"
multiplex_stat = "/home/davjd313/MultilayerNetwork (BME_4)/Result/stat_results/Multiplex_stat.csv"
# Define the function for processing graphs
def stat_MLP(audio_files):
    # Define field names for the CSV
    fieldnames = [
        'key1', 'key2', 'key3', 'n', 'm', 'AD', 'ASPL', 'GE', 'ACC', 'T', 'S', 'Q', 'AWD', 'CC', 'Kmin', 'Fit', 'Param1', 'Param2'
    ]
    
    # Open the CSV file for writing
    # with open(MLP_stat, mode='w', newline='') as csvfile:
    with open(multiplex_stat, mode='w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()  # Write the header row

        for audio_file in audio_files:
            print(f"Processing file: {audio_file}")
            # Read the weighted edge list
            G = nx.read_weighted_edgelist(audio_file)
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
            k_min = MLE_functions_v2.degree_list(G).min()
            fit_result = MLE_functions_v2.fit('Graph', G, k_min=k_min, plot_type='ccdf', save=False)
            file_parts = os.path.splitext(os.path.basename(audio_file))[0].split('_')
            key1 = "_".join(file_parts[:-2])  # Joining all parts except last two
            key2 = file_parts[-2]  # Second last part
            key3 = file_parts[-1].split('.')[0]  # Last part
            
            # Write results to CSV
            writer.writerow({
                'key1': key1, 'key2': key2, 'key3': key3,
                'n': n, 'm': m, 'AD': AD, 'ASPL': ASPL, 'GE': GE,
                'ACC': ACC, 'T': T, 'S': S, 'Q': Q, 'AWD': AWD, 'CC': CC,
                'Kmin': fit_result[0], 'Fit': fit_result[1],
                'Param1': np.round(fit_result[2][0][0], 2),
                'Param2': np.round(fit_result[2][0][1], 2) if len(fit_result[2][0]) > 1 else 0
            })

# Call the function
stat_MLP(audio_files)
