import numpy as np

def generate_file_paths(cluster_indexes):
    # From indexes of clusters generate lists containing their paths
    file_paths = []
    for index in cluster_indexes:
        path = "/Users/marijazelic/github-classroom/CS-433/ml-project-2-girl-mse/Clusters A3M/EX_"
        if index < 10:
            path = path + '00' + str(int(index)) + '.a3m'
        elif index >= 10 and index<= 99:
            path = path + '0' + str(int(index)) + '.a3m'
        else:
            path = path + str(int(index)) + '.a3m'
        
        file_paths.append(path)
    return file_paths

def generate_consenzus_sequences_for_each_cluster(cluster_indexes):
    # For every cluster with a path in file_paths generate consenzus sequence and store it in a list
    
    all_clusters_cs = [] # all clusters consenzus sequences
    file_paths = generate_file_paths(cluster_indexes)
    for path in file_paths:
        all_lines = read_a3m_file(path)
        sequence = make_consenzus_sequence(all_lines)
        all_clusters_cs.append(sequence)
        
    return all_clusters_cs
    
def read_a3m_file(path):
    # Open and read a3m file
    all_lines = []
    with open(path, 'r') as file:
        for line in file:
            all_lines.append(line.strip())
    all_lines = all_lines[1::2]
    return all_lines

def if_exists(aa, all_aa):
    # Checks if aa exists in all_aa
    if len(all_aa) == 0:
        return -1
    else:
        for indx, tuple in enumerate(all_aa):
            if tuple == aa:
                return indx
        return -1
    
 
def most_frequent_aa(position, all_sequences):
    # Finds most frequent amino acid between over all sequences in one cluster
    all_aa = {}
    for sequence in all_sequences:
        index = if_exists(sequence[position], all_aa)
        if index == -1:
            upd = {sequence[position]:1}
            all_aa.update(upd)
        else:
            all_aa[sequence[position]] += 1
    
    max_occurence = -1
    max_occuring_aa = '0'
    for aa in all_aa:
        if all_aa[aa] > max_occurence:
            max_occurence = all_aa[aa]
            max_occuring_aa = aa
            
    return max_occuring_aa
 
def make_consenzus_sequence(all_sequences):
    # Making a consenzus sequence for a cluster
    consenzus_sequence = ''
    N = len(all_sequences[0])
    for i in range(N):
        consenzus_sequence = consenzus_sequence + most_frequent_aa(i, all_sequences)
    
    return consenzus_sequence     
