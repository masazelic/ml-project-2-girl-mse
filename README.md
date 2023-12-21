[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-24ddc0f5d75046c5622901739e7c5dd533143b0c8e959d652212380cedb1ea36.svg)](https://classroom.github.com/a/fEFF99tU)
[![Open in Visual Studio Code](https://classroom.github.com/assets/open-in-vscode-718a45dd9cf7e7f842a935f5ebbe5719a5e09af4491e668f4dbf3b35d5cca122.svg)](https://classroom.github.com/online_ide?assignment_repo_id=13101974&assignment_repo_type=AssignmentRepo)

# ML4Science project: Exploration for mutants with enhanced protein fitness

*Team:*

The Girl MSE Team is composed of the following members:

Sara Zatezalo: @sarazatezalo

Marija Zelic: @masazelic

Elena Mrdja: @elena-mrdja

# Abstract

Efficient exploration of the protein fitness landscape can be crucial for finding proteins with desired functions. Our project aimed to find machine learning methods that can help the LPBS laboratory in finding suitable mutants of the optogenetic protein EL222. Since laboratory testing of protein function poses financial and time limitations, it was important to find a way to unravel the EL222 fitness landscape and find proteins with the most suitable optic traits, that would serve as candidates for further testing. Toward this goal, we combined various approaches such as AlphaFold (AF), AF Cluster, XGBoost, and BERT into a reliable method.

# Project pipeline

For better understanding of the steps this project consists of, refer to the flowchart below.
![Project pipeline](https://github.com/CS-433/ml-project-2-girl-mse/blob/main/Project%20pipeline.png)

## Alpha Fold cluster

First, to generate the MSA that is used for the Alpha Fold Cluster run the notebook: [ColabFold notebook](https://colab.research.google.com/github/sokrypton/ColabFold/blob/main/AlphaFold2.ipynb). In order to get the same results for the EL222 protein, the query sequence should be the following: "GADDTRVEVQPPAQWVLDLIEASPIASVVSDPRLADNPLIAINQAFTDLTGYSEEECVGRNCRFLAGSGTEPWLTDKIRQGVREHKPVLVEILNYKKDGTPFRNAVLVAPIYDDDDELLYFLGSQVEVDDDQPNMGMARRERAAEMLKTLS" (the LOV domain of the protein). 
To obtain Alpha Fold structure prediction for each of cluster (from their respective MSAs), run the [AF Cluster notebook](https://colab.research.google.com/github/HWaymentSteele/AF_Cluster/blob/main/AFcluster.ipynb) corresponding to Wayment-Steele, Ovchinnikov, Colwell, Kern (2022) "Prediction of multiple conformational states by combining sequence clustering with AlphaFold2" [bioRxiv](https://www.biorxiv.org/content/10.1101/2022.10.17.512570v1).

## Comparing the outputs of AlphaFold2 to the ground truth structure of EL222 protein  

The code corresponding to this part of the project is located in the *rmsd_calculation* folder.  
To compare the obtained Alpha Fold structures (.pdb files) to the ground-truth EL222 OFF structure ("EL222_chain_A.pdb"), you should place the files in a folder within the same folder as the notebook "comparison.ipynb"  and follow the steps in the notebook, that is run the code cell by cell. Change the variables parent_folder_path and original_pdb_file to correspond to your paths to the folder containing structures'.pdb' files and the original chain structure. "main_tools.py" and "run_comparisson_for_AF_cluster_modified.py" are auxiliary files containing functions used in the notebook.  

## Classification of the protein sequences using XGBoost

You can find the code for this part of the project in *XGBoost* folder. The sequences used for training and testing, that were found by using the procedure explained in the previous section, can be found within the data folder, specifically:
- off_sequences.txt and on_sequences.txt files that contain consensus sequences belonging to OFF cluster (37 in total) and ON cluster (81 in total).

To reproduce the reported results, the folder structure provided should be perserved. The model was implemented in the xgboost_training.ipynb file, and can be reproduced by running the notebook. 

## Classification of the protein sequences using BERT fine-tuned model and fully-connected classifier   

You can find the code for this part of the project in *BERT classifier* folder. Resulting sequences obtained by following the procedure explained in the previous section, can be found within the data folder, specifically:  

- all_train.csv and all_test.csv files containt training and test data for the auxiliary task (masked language model problem) on the protein sequences  
- all_off_sequences_v2.txt and all_on_sequences.txt files contain all sequences belonging to OFF cluster (3950 in total) and all sequences belonging to ON cluster (105 in total)
- off_sequences_v2.txt and on_sequences_v2.txt files that contain consensus sequences belonging to OFF cluster (28 in total) and ON cluster (19 in total).  

Folder also contains subfolder *utils* that stores all the necessary .py files, as well as run.py file that can be used to perform hyperparameter tuning. Additionally, there is .ipynb that can be used for displaying accuracies and learning curves for the classifier obtained by the hyperparameter tuning.  

In order to reproduce our results in terms of hyperparameter tuning, you should preserve the folder structure provided. If you want to change type of the sequences for which you are performing hyperparameter tuning (balanced or consensus), modify variables OFF_SEQUENCES_PATH and ON_SEQUENCES_PATH according to the explanation provided in the form of comments in the run.py file.
For the coarser grid search, refer to the section commented with Grid search parameters in the train_main_loop() function, within run.py file. After all this adjustments according to your needs, you can run the file by command:  

bash
python run.py
  
Regrading the .ipynb Notebook, it is inteded for running on the Google Colab, since it requires GPU's for the faster execution. For this purpose, upload *data* and *utils* folders to your Google Drive and then just run the cells.
