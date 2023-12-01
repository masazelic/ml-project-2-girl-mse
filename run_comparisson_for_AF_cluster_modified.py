# -*- coding: utf-8 -*-
"""
Created on Sun Jul 30 13:00:22 2023
Modified by Sara Zatezalo, Marija Zelic and Elena Mrdja for the purpose of ML4Science project within the LPBS lab.

The code performs evaluation of structures obtained with AlphaFold2 predictions comparing to the ground truth structure
which is in our case the Chain A of OFF conformation of the EL222 protein.

@author: gligorov
"""
from main_tools import *
import os
import re

def extract_number_from_filename(filename):
    match = re.search(r'\d+', filename)  # Matches one or more digits in the filename
    if match:
        return int(match.group())  # Extracts the matched number as an integer
    else:
        return None  # Return None if no number is found in the filename


#Looks like things before 25 are missing from the chain A

#151-200 radi
#30-140 radi

#Unresolved are A1-24 and A142-148 based on: https://www.rcsb.org/3d-view/3P7N
#make sure that your numbes for annotations are matching these
def align_structures(AF_file, original_pdb_file, choose_contigs, mutation = None):

     #contig notation includes both limits intersected with the LOV domain we are interested in
     #For some reason if you include 222th then it's not working although both files contain it. I'll just skip it
    if choose_contigs == "whole_protein":
        contigs = "A25-141/A149-221" # excluded because of unresolved technical limitiations
        contigs_for_predictions = "A25-141/A149-221"
    elif choose_contigs == "with low-confidenceLOV":
        contigs = "A25-141/A149-164"
        contigs_for_predictions = "A12-128/A136-151"
    elif choose_contigs == "without low-confidenceLOV":
        contigs = "A25-138"
        contigs_for_predictions = "A12-125"
    else:
        contigs = ...
        contigs_for_predictions = ...
        
    """
    if(mutation > 25 and mutation < 141):
        contigs = "A25-"+str(mutation-1)+"/A"+str(mutation+1)+"-141/A149-221"
    elif(mutation>149):
        contigs = "A25-141/A149-"+str(mutation-1)+"/A"+str(mutation+1)+"-221"
    elif(mutation<=25):
        print("Mutation is in the first unresolved few residues")
    elif(mutation<=149 and mutation>=141):
        print("Mutation is in the unresolved residues in the middle") #In this case you can just try aligning to the AF prediction if it's good enough
    #print(contigs)
    """

    rmsd, mean_conf_AF, min_conf_AF, max_conf_AF, confidence_scores, matched_ids, sequence_predictions = CompareTwoPDBs(contigs, contigs_for_predictions, original_pdb_file, AF_file)
    print(f"RMSD: {rmsd:.3f}")
    print(f"Mean confindence AF: {mean_conf_AF:.3f}")
    return rmsd, mean_conf_AF, confidence_scores, matched_ids, sequence_predictions


def extract_number_from_folder_name(folder_name):
    """
    Extract the number from the folder name.

    Parameters:
        folder_name (str): The name of the folder containing a number.

    Returns:
        int: The extracted number from the folder name.
    """
    number = ''.join(filter(str.isdigit, folder_name))
    return int(number)


def run_alignments_on_subfolders(parent_folder, original_pdb_file, choose_contigs):
    """
    Load file from subfolders that contain AF structure predictions in order to perform its comparison
    with the ground truth structure.

    Parameters:
        parent_folder (str): The path to the parent folder containing subfolders.

    Returns:
        dict: A dictionary containing subfolder names as keys and lists of file names as values.
              The lists will be empty for subfolders without '_relaxed_' in their names.
    """
    #Collect all of the cofidence scores
    list_scores = []
    list_matched_ids = []
    list_rmsd = []
    list_sequences = []
    #Each structure should be in a subfolder
    subfolder_files = {}
    cluster_ids = []
    # Get a list of subfolders
    subfolders = [name for name in os.listdir(parent_folder) if os.path.isdir(os.path.join(parent_folder, name))]

    for subfolder in subfolders:
        os.chdir(parent_folder)
        try:
			
            num_of_cluster = extract_number_from_folder_name(subfolder)
            subfolder_path = os.path.join(parent_folder, subfolder)
            
            #changing to the subdirectory
            os.chdir(subfolder_path)
            print(os.getcwd())
            files = os.listdir(os.getcwd())

            # Filter files that contain the structure prediction (.pdb)
            pdb_file = None
            for file in files:
                if file.endswith('.pdb'):
                    pdb_file = file

            if pdb_file == None:
                raise ValueError(f"Error: Subfolder {subfolder} does not contain any .pdb file.")
            else:
                # Extracting the cluster number from the pdb file name
                cluster_ids.append(num_of_cluster)
                rmsd, mean_conf_AF, confidence_scores, matched_ids, sequence_predictions = align_structures(pdb_file, original_pdb_file, choose_contigs)
                list_rmsd.append(rmsd)
                list_scores.append(confidence_scores)
                list_matched_ids.append(matched_ids)
                list_sequences.append(sequence_predictions)
        
        except ValueError as error:
            print(error)
    
    return list_scores, list_matched_ids, list_rmsd, list_sequences, cluster_ids
