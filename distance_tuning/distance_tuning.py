import sys
import numpy as np
from prettytable import PrettyTable
from sklearn.cluster import DBSCAN
from sklearn.metrics.cluster import homogeneity_score, completeness_score, v_measure_score
import preprocessing_functions
import global_vars
import hdbscan
from clusters_validity import clusters_validity
from sklearn.metrics.cluster import normalized_mutual_info_score, adjusted_mutual_info_score, v_measure_score, adjusted_rand_score
from sklearn.metrics import fbeta_score
import math

# Used only for extraction
min_samples_per_family = 1
#min_samples_per_family = 10
clustering_mode = 'consensus_only'
requested_cats = None

if (len(sys.argv)!=2):
    print("Sorry, you should specify 1 command-line arguments:\n1-number of config(realistic_dataset or synthetic dataset)")
    exit(1)

if int(sys.argv[1]) == 1:
    feat_dir = "/home/lnouredd/Desktop/PhD/git/datasets/mixed_dataset/pe_dates_featfiles_packers_families/training_set_1/"
    global_vars.file_names = \
        [
        "ASPack",
        "ASProtect",
        "ExeStealth",
        "eXPressor",
        "FSG",
        "InstallShield",
        "MEW",
        "MoleBox",
        "NeoLite",
        "NsPacK",
        "Packman",
        "PECompact",
        "PEPACK",
        "Petite",
        "RLPack",
        "Themida",
        "UPX",
        "UPack",
        "WinRAR",
        "WinZip",
        "Wise",
        "ActiveMARK",
        "FishPE",
        "PCGuard",
        "PESpin",
        "Shrinker",
        "NSIS",
        "InnoSetup",
        "AutoIt"
        #"PEArmor"
        ]

elif int(sys.argv[1]) == 2:
    feat_dir = "/home/lnouredd/Desktop/PhD/git/datasets/mixed_dataset/pe_dates_featfiles_packers_families/training_set_2/"
    global_vars.file_names = global_vars.file_names = \
        ["Armadillo",
         "ASPack",
         "YodaCryptor",
         "CustomAmberPacker",
         "CustomOrigamiPacker",
         "CustomPackerSimple1",
         "CustomPEPacker1",
         "CustomPePacker2",
         "CustomPetoyPacker",
         "CustomSilentPacker",
         "CustomTheArkPacker",
         "CustomUchihaPacker",
         "CustomXorPacker",
         "eXPressor",
         "ezip",
         "FSG",
         "MEW",
         "mPress",
         "NeoLite",
         "Packman",
         "PECompact",
         "PELock",
         "PENinja",
         "Petite",
         "RLPack",
         "telock",
         "Themida",
         "UPX",
         "WinRAR",
         "UPack",
         "WinZip"
         ]

# Preprocessing tasks
preprocessing_functions.assign_log_file_name(clustering_mode, sys.argv[1], min_samples_per_family, requested_cats)
global_vars.family_names = global_vars.file_names
preprocessing_functions.remove_unpacked_files()
preprocessing_functions.load_features_dir(clustering_mode, feat_dir)
preprocessing_functions.fam_to_remove(min_samples_per_family)
preprocessing_functions.assign_labels_and_features()
if preprocessing_functions.is_features_empty():
    exit(1)
if preprocessing_functions.len_features_wrong():
    exit(1)

"""
preprocessing_functions.log(
    'Using ' + str(len(global_vars.feature_names)) + ' features: ' + str(global_vars.feature_names))
preprocessing_functions.log(
    'Found ' + str(len(set(global_vars.features_categories))) + ' feature categories: ' + str(
        sorted(set(global_vars.features_categories), key=global_vars.features_categories.index)))
"""
preprocessing_functions.delete_unrequested_feat_cat(requested_cats)
# ----

# Loadthe final averaged similarities matrix
#offline_averaged_distances_matrix = np.load('../distance_matrices/offline_tapered_levenshtein_distances_matrix.npy')
#offline_averaged_distances_matrix = np.load('../distance_matrices/offline_euclidean_distance_matrix.npy')

#matrix_to_load = '../distance_matrices/offline_averaged_distances_matrix_' + sys.argv[1] + '.npy'
#matrix_to_load = '../distance_matrices/offline_averaged_distances_matrix_' + sys.argv[1] + '.npy'
#matrix_to_load = '../distance_matrices/offline_manhattan_distance_matrix_' + sys.argv[1] + '.npy'
#matrix_to_load = '../distance_matrices/offline_OneHotEncode_cosine_distance_matrix_' + sys.argv[1] + '.npy'
matrix_to_load = '../distance_matrices/offline_OneHotEncode_euclidean_distance_matrix_' + sys.argv[1] + '.npy'
#matrix_to_load = '../distance_matrices/offline_OneHotEncode_manhattan_distance_matrix_' + sys.argv[1] + '.npy'
offline_averaged_distances_matrix = np.load(matrix_to_load)
offline_averaged_distances_matrix = np.float64(offline_averaged_distances_matrix)

# Start the clustering with tuning the best parameters
temp = []
results = []

#eps_list = [0.065, 0.0675, 0.07, 0.0725, 0.075, 0.0775, 0.08, 0.0825, 0.085, 0.0875, 0.09]
eps_list = [0.001, 0.00125, 0.00150, 0.00175, 0.002, 0.0025, 0.00250, 0.00275, 0.003, 0.00325, 0.00350, 0.00375, 0.004, 0.00425, 0.0045, 0.00475, 0.005, 0.00525, 0.0055, 0.00575, 0.006, 0.00625, 0.0065, 0.00675, 0.007, 0.00725, 0.0075, 0.00775, 0.008, 0.00825, 0.0085, 0.00875, 0.009, 0.00925, 0.0095,0.00975, 0.01, 0.01, 0.0125, 0.0150, 0.0175, 0.02, 0.025, 0.0250, 0.0275, 0.03, 0.0325, 0.0350, 0.0375, 0.04, 0.0425, 0.045, 0.0475, 0.05, 0.0525, 0.055, 0.0575, 0.06, 0.0625, 0.065, 0.0675, 0.07, 0.0725, 0.075, 0.0775, 0.08, 0.0825, 0.085, 0.0875, 0.09, 0.0925, 0.095,0.0975, 0.1]
#eps_list = [0.01, 0.0125, 0.0150, 0.0175, 0.02, 0.025, 0.0250, 0.0275, 0.03, 0.0325, 0.0350, 0.0375, 0.04, 0.0425, 0.045, 0.0475, 0.05, 0.0525, 0.055, 0.0575, 0.06, 0.0625, 0.065, 0.0675, 0.07, 0.0725, 0.075, 0.0775, 0.08, 0.0825, 0.085, 0.0875, 0.09, 0.0925, 0.095,0.0975, 0.1,0.125,0.150,0.175, 0.2, 0.225, 0.250, 0.275, 0.3, 0.325, 0.35, 0.375, 0.4, 0.425, 0.45, 0.475, 0.5, 0.525, 0.55, 0.575, 0.6, 0.625, 0.65, 0.675]
#eps_list = [0.08]
#min_sample_list = [3, 4, 5, 6, 7, 8, 9, 10]
min_sample_list = [3]

"""
clusterer = hdbscan.HDBSCAN(metric='precomputed').fit(offline_averaged_distances_matrix)
clustering_homogeneity_score = homogeneity_score(global_vars.labels, clusterer.labels_)
noise=0
for x in clusterer.labels_:
    if x==-1:
        noise+=1
print noise
input()
"""

"""
temp_list=[]
for i in range(0,len(global_vars.family_names)):
    temp_list.extend([i]*global_vars.labels.count(global_vars.family_names[i]))
    print(temp_list)

global_vars.labels=temp_list
"""

for eps in  eps_list:
    for min_sample in min_sample_list:
        #ca = hdbscan.HDBSCAN(metric='precomputed',min_cluster_size=min_sample).fit(averaged_similarities)
        ca = DBSCAN(metric = 'precomputed', eps=eps, min_samples=min_sample).fit(offline_averaged_distances_matrix)
        ca_clustering_labels = ca.labels_
        ca_clustering_labels = ca_clustering_labels.tolist()

        ca_clusters = {}
        has_noise_cluster = (-1 in ca_clustering_labels)

        ca_n_clusters = len(np.unique(ca_clustering_labels))

        first_cluster_index = -1 if has_noise_cluster else 0
        last_cluster_index = (ca_n_clusters - 2) if has_noise_cluster else (ca_n_clusters - 1)

        ca_n_clusters = (ca_n_clusters - 1) if has_noise_cluster else ca_n_clusters

        # Printing usefuls informations
        preprocessing_functions.log("eps: " + str(eps))
        preprocessing_functions.log("min_samples: " + str(min_sample))
        preprocessing_functions.log("Does it have a noise cluster ?: " + str(has_noise_cluster))
        preprocessing_functions.log("The number of clusters it has: " + str(ca_n_clusters))
        preprocessing_functions.log("The first index cluster is: " + str(first_cluster_index))
        preprocessing_functions.log("The last index cluster is: " + str(last_cluster_index))

        # initialization of dictionaries
        for ca_cluster_index in range(first_cluster_index, (last_cluster_index + 1)):
            ca_clusters[ca_cluster_index] = {}
            for fam in global_vars.family_names:
                ca_clusters[ca_cluster_index][fam] = 0

        #"""
        # Affection of binaries and their reduced/full labels in their corresponding cluster
        for list_index, ca_cluster_label in enumerate(ca_clustering_labels):
            ca_clusters[ca_cluster_label][global_vars.labels[list_index]] += 1
        #"""

        #clustering_adjusted_rand_score = adjusted_rand_score(global_vars.labels, ca_clustering_labels)
        #clustering_fB_score = fbeta_score(global_vars.labels, ca_clustering_labels, average='weighted', beta=5)
        #clustering_v_measure_score = v_measure_score(global_vars.labels, ca_clustering_labels)
        clustering_adjusted_mutual_info_score = adjusted_mutual_info_score(global_vars.labels, ca_clustering_labels, average_method="geometric")
        #clustering_adjusted_mutual_info_score = normalized_mutual_info_score(global_vars.labels, ca_clustering_labels)

        #clustering_normalized_mutual_info_score = normalized_mutual_info_score(global_vars.labels, ca_clustering_labels)
        clustering_homogeneity_score = homogeneity_score(global_vars.labels, ca_clustering_labels)
        clustering_completeness_score = completeness_score(global_vars.labels, ca_clustering_labels)
        clustering_v_measure_score = v_measure_score(global_vars.labels, ca_clustering_labels)
        #try:
        #    clustering_quality = clusters_validity(offline_averaged_distances_matrix, np.array(ca_clustering_labels))
        #    clustering_quality=clustering_quality[0]
        #except Exception as e:
        #    clustering_quality=0

        preprocessing_functions.log("clustering_homogeneity_score is: " + str(clustering_homogeneity_score))
        #preprocessing_functions.log("clustering quality is: " + str(clustering_quality))

        # Printing results_with_synthetic_centroids:
        ca_cluster_table = PrettyTable()

        # fiiling fiedl_names for the first table
        ca_field_names = ["num cluster"]
        for fam in global_vars.family_names:
            ca_field_names.append(fam)
        ca_cluster_table.field_names = ca_field_names

        # fiiling rows of the first and second tables
        for ca_cluster_index in range(first_cluster_index, (last_cluster_index + 1)):
            row = [ca_cluster_index]
            for fam in global_vars.family_names:
                row.append(ca_clusters[ca_cluster_index][fam])
            ca_cluster_table.add_row(row)

        preprocessing_functions.log(str(ca_cluster_table))

        #preprocessing_functions.log("\n" + str(ca_cluster_table))
        #preprocessing_functions.log("--------------")
        #preprocessing_functions.log("\n\nThe homogeneity_score for this experiment was : " + str(clustering_homogeneity_score) + "\n\n")
        #preprocessing_functions.log("\n\nThe completeness_score for this experiment was : " + str(clustering_completeness_score) + "\n\n")
        #preprocessing_functions.log("\n\nThe v_measure_score for this experiment was : " + str(clustering_v_measure_score) + "\n\n")
        # TODO: add more evaluation metrics
        #preprocessing_functions.log("OK, good Bye ! \n\n")

        #temp.append(clustering_adjusted_rand_score)
        #temp.append(clustering_fB_score)
        #temp.append(clustering_v_measure_score)

        #personal_score = 100*clustering_homogeneity_score/ 0.9*ca_n_clusters


        temp.append(clustering_adjusted_mutual_info_score)
        #temp.append(personal_score)


        #temp.append(clustering_normalized_mutual_info_score)
        temp.append(clustering_homogeneity_score)
        #temp.append(clustering_quality)
        temp.append(ca_n_clusters)
        temp.append(eps)
        temp.append(min_sample)
        results += [temp]
        temp = []

np.save('offline_clusters_labels_' + str(sys.argv[1]) + ".npy", ca_clustering_labels)
np.save('offline_packers_families_labels_' + str(sys.argv[1]) + ".npy", global_vars.labels)


# Rank the best results_with_synthetic_centroids
results.sort(key=lambda x: x[0], reverse=True)

# Printing results_with_synthetic_centroids:
ca_results_table = PrettyTable()

# fiiling fiedl_names for the table
ca_results_table.field_names = ["adjusted_mutual_info_score", "homogeneity_score", "number_of_clusters", "eps", "min_samples"]

# fiiling rows of the first and second tables
for result in  results:
    row = result
    ca_results_table.add_row(row)

preprocessing_functions.log("\n" + str(ca_results_table))
preprocessing_functions.log("--------------")
