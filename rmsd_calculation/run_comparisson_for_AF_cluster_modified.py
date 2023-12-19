
from main_tools import *
import os
import re

def extract_number_from_filename(filename):
    match = re.search(r'\d+', filename)  # Matches one or more digits in the filename
    if match:
        return int(match.group())  # Extracts the matched number as an integer
    else:
        return None  # Return None if no number is found in the filename

#Unresolved are A1-24 and A142-148 based on: https://www.rcsb.org/3d-view/3P7N
#make sure that your numbes for annotations are matching these
def align_structures(AF_file, original_pdb_file, choose_contigs, specific=None):

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

    if(specific == None):
        rmsd, mean_conf_AF, min_conf_AF, max_conf_AF, confidence_scores, matched_ids, sequence_predictions = CompareTwoPDBs(contigs, contigs_for_predictions, original_pdb_file, AF_file)
    else:
        rmsd, mean_conf_AF, min_conf_AF, max_conf_AF, confidence_scores, matched_ids, sequence_predictions = CompareTwoPDBs_specific(contigs, contigs_for_predictions, original_pdb_file, AF_file,specific)

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


def run_alignments_on_subfolders(parent_folder, original_pdb_file, choose_contigs, specific=False):
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
                rmsd, mean_conf_AF, confidence_scores, matched_ids, sequence_predictions = align_structures(pdb_file, original_pdb_file, choose_contigs, specific)
                list_rmsd.append(rmsd)
                list_scores.append(confidence_scores)
                list_matched_ids.append(matched_ids)
                list_sequences.append(sequence_predictions)
        
        except ValueError as error:
            print(error)
    
    return list_scores, list_matched_ids, list_rmsd, list_sequences, cluster_ids
