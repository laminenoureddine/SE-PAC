import sys
import datetime
import numpy as np
from prettytable import PrettyTable
from sklearn.cluster import DBSCAN, KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics.cluster import homogeneity_score, completeness_score, v_measure_score
from sklearn.metrics import accuracy_score
import preprocessing_functions
import global_vars
from incremental_dbscan import IncrementalDBSCAN, gowr_distance, tapered_levenshtein_distance, manhattan_distance, get_scaler_values
import pandas as pd
from clusters_validity import clusters_validity
from scipy.spatial.distance import euclidean
from sklearn import metrics
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_extraction import FeatureHasher
from scipy.spatial import distance
import matplotlib.pyplot as plt
import hdbscan
from sklearn.metrics.cluster import normalized_mutual_info_score, adjusted_mutual_info_score, v_measure_score, adjusted_rand_score


global min_max_scaler_min
global min_max_scaler_max
global min_max_scaler_scale


def get_scaler_values(dir, scenario):
    global min_max_scaler_min
    global min_max_scaler_max
    global min_max_scaler_scale

    min_max_scaler_min = np.load(dir + "min_max_scaler_min_" + str(scenario) + ".npy")
    min_max_scaler_max = np.load(dir + "min_max_scaler_max_" + str(scenario) + ".npy")
    min_max_scaler_scale = np.load(dir + "min_max_scaler_scale_" + str(scenario) + ".npy")

def get_samples_indexes(dataset_of_mnemonics_sequences):
    """
    :param dataset: the dataset of mnemonics sequences features in pd dataframe or list of pd dataframes of mnemonics sequences features
    """
    mnemonic_features_tuples = list(map(tuple, np.array(dataset_of_mnemonics_sequences)))
    unique_elements_features = list(set(mnemonic_features_tuples))
    unique_elements_features.sort(key=mnemonic_features_tuples.index)

    unique_elements_indexes = []
    for j in range(len(unique_elements_features)):
        l = [i for i in range(len(mnemonic_features_tuples)) if mnemonic_features_tuples[i] == unique_elements_features[j]]
        unique_elements_indexes.append(l)

    return unique_elements_features, unique_elements_indexes

def update_pairewise_disance_matrix(scenario, offline_averaged_distances_matrix, online_dataset, old_unique_elements_indexes,
                                    new_unique_elements_indexes, old_unique_elements_features,
                                    new_unique_elements_features, list_of_samples_to_update):
    """
    This function updates the distance matrix
    :param offline_averaged_matrix: the pairewise averaged distance matrix in numpy format
    :param online_dataset: the dataset of features in pd dataframe
    :param old_unique_elements_indexes: indices of the unique offlines samples
    :param new_unique_elements_indexes: indices of the unique online samples
    :param old_unique_elements_features: unique offline samples
    :param new_unique_elements_features: unique online samples
    :param list_new_samples: list of one or multiples samples in pd dataframe
    :return: the updated pairewise distance matrix
    """

    offline_matrix_size = offline_averaged_distances_matrix.shape[0]
    new_samples_size = len(list_of_samples_to_update)
    online_matrix_size = offline_matrix_size + new_samples_size

    # Initiliaze the online averaged pairewise matrix
    online_averaged_pairewise_matrix = np.zeros([online_matrix_size, online_matrix_size], dtype=float)

    # Initiliaze the online tapered_levenshtein pairewise matrix
    online_tapered_levenshtein_pairewise_matrix = np.zeros([online_matrix_size, online_matrix_size], dtype=float)

    # Initiliaze the online manhattan pairewise matrix
    online_manhattan_pairewise_matrix = np.zeros([online_matrix_size, online_matrix_size], dtype=float)

    # Assign the offline pairewise matrix to the online averaged pairewise matrix
    online_averaged_pairewise_matrix[0: offline_matrix_size, 0: offline_matrix_size] = offline_averaged_distances_matrix

    # Update the tapered_levenshtein distances
    max_len_seq = 50 # not used here
    for i in range(len(new_unique_elements_features)):
        for j in range(len(old_unique_elements_features)):
            s1 = new_unique_elements_features[i]
            s2 = old_unique_elements_features[j]
            s1 = pd.DataFrame([[s1]], columns=['Unpacking_code_sequence'])
            s2 = pd.DataFrame([[s2]], columns=['Unpacking_code_sequence'])
            scaled_distance = tapered_levenshtein_distance(s1.iloc[0], s2.iloc[0])

            online_tapered_levenshtein_pairewise_matrix[
                np.ix_(new_unique_elements_indexes[i], old_unique_elements_indexes[j])] = scaled_distance
            online_tapered_levenshtein_pairewise_matrix[
                np.ix_(old_unique_elements_indexes[j], new_unique_elements_indexes[i])] = scaled_distance

        for j in range(len(new_unique_elements_features)):
            s1 = new_unique_elements_features[i]
            s2 = new_unique_elements_features[j]
            s1 = pd.DataFrame([[s1]], columns=['Unpacking_code_sequence'])
            s2 = pd.DataFrame([[s2]], columns=['Unpacking_code_sequence'])
            scaled_distance = tapered_levenshtein_distance(s1.iloc[0], s2.iloc[0])

            online_tapered_levenshtein_pairewise_matrix[
                np.ix_(new_unique_elements_indexes[i], new_unique_elements_indexes[j])] = scaled_distance
            online_tapered_levenshtein_pairewise_matrix[
                np.ix_(new_unique_elements_indexes[j], new_unique_elements_indexes[i])] = scaled_distance

    # Update the manhattan distances
    # scale first the dbscan dataset
    pe_array_dbscan_dataset = np.array(dbscan.dataset.iloc[:, 0:len(global_vars.features[0])])

    if dbscan.local_server_image == 'server':
        get_scaler_values("scaler_values/", scenario)
    else:
        get_scaler_values("../scaler_values/", scenario)
    pe_array_dbscan_dataset_scaled = np.zeros([pe_array_dbscan_dataset.shape[0], pe_array_dbscan_dataset.shape[1]])
    for i in range(len(pe_array_dbscan_dataset[0])):
        pe_array_dbscan_dataset_scaled[:, i] = min_max_scaler_scale[i] * pe_array_dbscan_dataset[:, i] + 0 - min_max_scaler_min[i] * min_max_scaler_scale[i]

    # then scale new samples to update
    pe_list_new_samples = [list_of_samples_to_update[i].iloc[0] for i in range(len(list_of_samples_to_update))]
    pe_list_new_samples = [[pe_list_new_samples[i]['PE_feature_' + str(j + 1)] for j in range(len(global_vars.features[0]))] for i in range(len(pe_list_new_samples))]
    pe_array_new_samples = np.array(pe_list_new_samples)

    pe_array_new_samples_scaled = np.zeros([pe_array_new_samples.shape[0], pe_array_new_samples.shape[1]])
    for i in range(len(pe_array_new_samples[0])):
        pe_array_new_samples_scaled[:, i] = min_max_scaler_scale[i] * pe_array_new_samples[:, i] + 0 - min_max_scaler_min[i] * min_max_scaler_scale[i]

    #new_sample_pairewise_distances = [[(manhattan_distance(list_of_samples_to_update[i].iloc[0], dbscan.dataset.iloc[j])) for j in range(len(dbscan.dataset))] for i in range(len(list_of_samples_to_update))]
    #new_sample_pairewise_distances = np.array(new_sample_pairewise_distances)
    new_sample_pairewise_distances=distance.cdist(pe_array_new_samples_scaled, pe_array_dbscan_dataset_scaled,'cityblock') / len(global_vars.features[0])

    online_manhattan_pairewise_matrix[offline_matrix_size:online_matrix_size, 0: online_matrix_size] = new_sample_pairewise_distances
    online_manhattan_pairewise_matrix[0: online_matrix_size, offline_matrix_size:online_matrix_size] = new_sample_pairewise_distances.transpose()

    # Code that removes -1 from the lev matrix and replaces it with it corresponding value in the manh distance.
    indexes_to_replaced = np.argwhere(online_tapered_levenshtein_pairewise_matrix == -1)
    for i in range(len(indexes_to_replaced)):
        online_tapered_levenshtein_pairewise_matrix[indexes_to_replaced[i][0], indexes_to_replaced[i][1]] = len(global_vars.features[0]) * online_manhattan_pairewise_matrix[indexes_to_replaced[i][0], indexes_to_replaced[i][1]]
        online_manhattan_pairewise_matrix[indexes_to_replaced[i][0], indexes_to_replaced[i][1]] = len(global_vars.features[0]) * online_manhattan_pairewise_matrix[indexes_to_replaced[i][0], indexes_to_replaced[i][1]]

    # Update the final online averaged pairewise distance matrix
    online_averaged_pairewise_matrix[offline_matrix_size:online_matrix_size,:] = (online_tapered_levenshtein_pairewise_matrix[offline_matrix_size:online_matrix_size,:] + online_manhattan_pairewise_matrix[offline_matrix_size:online_matrix_size,:]) / 2
    online_averaged_pairewise_matrix[:, offline_matrix_size:online_matrix_size] = (online_tapered_levenshtein_pairewise_matrix[:,offline_matrix_size:online_matrix_size] + online_manhattan_pairewise_matrix[:,offline_matrix_size:online_matrix_size]) / 2

    return online_averaged_pairewise_matrix

def generate_INC_clusters_table(final_dataset, family_names, true_labels):
    """
    :param final_dataset: the final dataset in pandas dataframe
    :param family_names: the family names
    :param true labels: the true labels
    :return: a clustering results table
    """

    clustering_labels = list(final_dataset['Label'])
    clusters = {}
    has_noise_cluster = (-1 in clustering_labels)
    num_clusters_found = len(np.unique(clustering_labels))
    first_cluster_index = -1 if has_noise_cluster else 0
    last_cluster_index = (num_clusters_found - 2) if has_noise_cluster else (num_clusters_found - 1)

    # initialization of dictionaries
    for cluster_index in range(first_cluster_index, (last_cluster_index + 1)):
        clusters[cluster_index] = {}
        for fam in family_names:
            clusters[cluster_index][fam] = 0

    # Printing results:
    clusters_table = PrettyTable()

    # fiiling fields_names for the table
    field_names = ["num cluster"]
    for fam in family_names:
        field_names.append(fam)
    clusters_table.field_names = field_names

    # Affection of binaries and their reduced/full labels in their corresponding cluster
    for list_index, cluster_label in enumerate(clustering_labels):
        clusters[cluster_label][true_labels[list_index]] += 1

    # fiiling rows of the first and second tables
    for cluster_index in range(first_cluster_index, (last_cluster_index + 1)):
        row = [cluster_index]
        for fam in family_names:
            row.append(clusters[cluster_index][fam])
        clusters_table.add_row(row)

    return (clusters_table)


def generate_RS_clusters_table(RS_labels, family_names, true_labels):
    """
    :param RS_labels: The labels generated by DBSCAN/HDBSCAN from scratch
    :param family_names: the family names
    :param true labels: the true labels
    :return: a clustering results table
    """
    clusters = {}
    has_noise_cluster = (-1 in RS_labels)
    num_clusters_found = len(np.unique(RS_labels))
    first_cluster_index = -1 if has_noise_cluster else 0
    last_cluster_index = (num_clusters_found - 2) if has_noise_cluster else (num_clusters_found - 1)

    # initialization of dictionaries
    for cluster_index in range(first_cluster_index, (last_cluster_index + 1)):
        clusters[cluster_index] = {}
        for fam in family_names:
            clusters[cluster_index][fam] = 0

    # Printing results:
    clusters_table = PrettyTable()

    # fiiling fields_names for the table
    field_names = ["num cluster"]
    for fam in family_names:
        field_names.append(fam)
    clusters_table.field_names = field_names

    # Affection of binaries and their reduced/full labels in their corresponding cluster
    for list_index, cluster_label in enumerate(RS_labels):
        clusters[cluster_label][true_labels[list_index]] += 1

    # fiiling rows of the first and second tables
    for cluster_index in range(first_cluster_index, (last_cluster_index + 1)):
        row = [cluster_index]
        for fam in family_names:
            row.append(clusters[cluster_index][fam])
        clusters_table.add_row(row)

    return (clusters_table)

#### The main starts here:
if (len(sys.argv)!=9):
    print("Sorry, you should specify 8 command-line arguments:\n1-eps\n2-min_samples\n3-for number of the config\n4-shuffling_index(-1 in the case of the second config)\n5-SRPs factor (i.e. k=0.1, or k=0.5, or k=2, ...)\n6-number months of testing(i.e. number of distinct packers)\n7-Please tapae local or server version of the program)\n7-Integrate gemoetric quality (DBCV) in the anlysis(tape 0 for No, 1 for Yes))\n")
    print("Example:\n")
    print("./run_incremental_packers_clustering.sh 0.08 3 1 1 0.5 1 local 0\n\n")
    exit(1)
# eps + number of samples
eps=float(sys.argv[1])
min_samples=int(sys.argv[2])
DBCV_integration = int(sys.argv[8])

# config number
if int(sys.argv[3]) == 1:
    if sys.argv[7] == "server":
        training_set_feat_dir = "datasets/mixed_dataset/pe_dates_featfiles_packers_families/training_set_1/"
        training_set_mnemonics_feat_dir = "datasets/mixed_dataset/radare2_dates_featfiles_packers_families/training_set_1/"
        # Loadthe final averaged similarities matrix
        offline_averaged_distances_matrix = np.load('distance_matrices/offline_averaged_distances_matrix_1.npy')
        offline_averaged_distances_matrix = np.float64(offline_averaged_distances_matrix)
    else:
        training_set_feat_dir = "/home/lnouredd/Desktop/PhD/git/datasets/mixed_dataset/pe_dates_featfiles_packers_families/training_set_1/"
        training_set_mnemonics_feat_dir = "/home/lnouredd/Desktop/PhD/git/datasets/mixed_dataset/radare2_dates_featfiles_packers_families/training_set_1/"
        # Loadthe final averaged similarities matrix
        offline_averaged_distances_matrix = np.load('../distance_matrices/offline_averaged_distances_matrix_1.npy')
        offline_averaged_distances_matrix = np.float64(offline_averaged_distances_matrix)

    my_xticks = ['training', 'training', 'training']

    #number of core points
    #if int(sys.argv[5]) in range(1,150):
    SRPs_factor = float(sys.argv[5])
    #else:
    #    print("Sorry, the number of core samples can't exceed 150, please respecify correctly this number again")
    #    exit(1)

    # number of shuffling index
    if int(sys.argv[4]) in [1,2,3]:
        shuffle_index=int(sys.argv[4])
        if sys.argv[7] == "server":
            testing_set_feat_dir = "datasets/mixed_dataset/pe_dates_featfiles_packers_families/test_set_1/shuffle_" + str(shuffle_index) + "/2017-"
            testing_set_mnemonics_feat_dir = "datasets/mixed_dataset/radare2_dates_featfiles_packers_families/test_set_1/shuffle_" + str(shuffle_index) + "/2017-"
        
        else:
            testing_set_feat_dir = "/home/lnouredd/Desktop/PhD/git/datasets/mixed_dataset/pe_dates_featfiles_packers_families/test_set_1/shuffle_" + str(shuffle_index) + "/2017-"
            testing_set_mnemonics_feat_dir = "/home/lnouredd/Desktop/PhD/git/datasets/mixed_dataset/radare2_dates_featfiles_packers_families/test_set_1/shuffle_" + str(shuffle_index) + "/2017-"

        prefix_date = "2017-"
        months_of_tests = range(1, int(sys.argv[6]) + 1)
        results_file_name = "results/results_realistic_dataset_as_training_and_synthetic_as_testing_shuffle_" + str(shuffle_index)  + "_and_" + str(SRPs_factor) + "_representative_core_samples.txt"
        figure_title = 'realistic dataset as training set and synthetic dataset as testing set: shuffled test samples ' + str(shuffle_index) + " and " + str(SRPs_factor) + " representative_core_samples"
        figure_name = 'results/results_realistic_dataset_as_training_and_synthetic_as_testing_shuffle_' + str(shuffle_index) + "_and_" + str(SRPs_factor) + "_representative_core_samples.png"
    else:
        print("Sorry, you are in config 1, you have to specify the right number of the shuffle_index: 1 or 2 or 3")
        exit(1)

#config number
elif int(sys.argv[3]) == 2:
    # number of shuffling index
    if int(sys.argv[4]) in [1, 2, 3]:
        shuffle_index = int(sys.argv[4])
        # number of core points
        # if int(sys.argv[5]) in range(1, 51):
        SRPs_factor = float(sys.argv[5])
        # else:
        #    print("Sorry, the number of core samples can't exceed 10, please respecify correctly this number again")

        if sys.argv[7] == "server":
            training_set_feat_dir = "datasets/mixed_dataset/pe_dates_featfiles_packers_families/training_set_2/"
            training_set_mnemonics_feat_dir = "datasets/mixed_dataset/radare2_dates_featfiles_packers_families/training_set_2/"
            testing_set_feat_dir = "datasets/mixed_dataset/pe_dates_featfiles_packers_families/test_set_2/shuffle_" + str(shuffle_index) + "/2017-"
            testing_set_mnemonics_feat_dir = "datasets/mixed_dataset/radare2_dates_featfiles_packers_families/test_set_2/shuffle_" + str(shuffle_index) + "/2017-"
            # Loadthe final averaged similarities matrix
            offline_averaged_distances_matrix = np.load('distance_matrices/offline_averaged_distances_matrix_2.npy')
            offline_averaged_distances_matrix = np.float64(offline_averaged_distances_matrix)
        else:
            training_set_feat_dir = "/home/lnouredd/Desktop/PhD/git/datasets/mixed_dataset/pe_dates_featfiles_packers_families/training_set_2/"
            training_set_mnemonics_feat_dir = "/home/lnouredd/Desktop/PhD/git/datasets/mixed_dataset/radare2_dates_featfiles_packers_families/training_set_2/"
            testing_set_feat_dir = "/home/lnouredd/Desktop/PhD/git/datasets/mixed_dataset/pe_dates_featfiles_packers_families/test_set_2/shuffle_" + str(
                shuffle_index) + "/2017-"
            testing_set_mnemonics_feat_dir = "/home/lnouredd/Desktop/PhD/git/datasets/mixed_dataset/radare2_dates_featfiles_packers_families/test_set_2/shuffle_" + str(
                shuffle_index) + "/2017-"
            # Loadthe final averaged similarities matrix
            offline_averaged_distances_matrix = np.load('../distance_matrices/offline_averaged_distances_matrix_2.npy')
            offline_averaged_distances_matrix = np.float64(offline_averaged_distances_matrix)

    else:
        print("Sorry, you are in config 2, you have to specify the right number of the shuffle_index: 1 or 2 or 3")
        exit(1)

    my_xticks = ['training', 'training', 'training']
    prefix_date = "2017-"
    months_of_tests = range(1, int(sys.argv[6]) + 1)
    results_file_name = "results/results_synthetic_dataset_as_training_and_realistic_as_testing_and_" + str(SRPs_factor) + "_representative_core_samples.txt"
    figure_title = 'synthetic dataset as training set and realistic dataset as testing set and ' + str(SRPs_factor) + ' representative_core_samples'
    figure_name = 'results/results_synthetic_dataset_as_training_and_realistic_as_testing_and_' + str(SRPs_factor) + '_representative_core_samples.png'

else:
    print("Sorry, you have to specify the right number of the config: 1 or 2")
    exit(1)

# variable in which we will save all labels incrementally
all_labels=[]

# Settings
min_samples_per_family = 1
clustering_mode = 'consensus_only'
#requested_cats = ['section', 'entropy', 'imp_func']
requested_cats = ['section']

# Preprocessing tasks:
file_names_backup=global_vars.file_names[:]
preprocessing_functions.assign_log_file_name(clustering_mode, sys.argv[3], min_samples_per_family, requested_cats)
global_vars.family_names = global_vars.file_names
preprocessing_functions.remove_unpacked_files()
preprocessing_functions.load_features_dir(clustering_mode, training_set_feat_dir)
preprocessing_functions.load_mnemonics_features_dir(clustering_mode, training_set_mnemonics_feat_dir)
preprocessing_functions.fam_to_remove(min_samples_per_family)
preprocessing_functions.assign_labels_and_features()
if preprocessing_functions.is_features_empty():
    exit(1)
if preprocessing_functions.len_features_wrong():
    exit(1)
preprocessing_functions.delete_unrequested_feat_cat(requested_cats)

all_labels.extend(global_vars.labels)

# Load the object incremental dbscan
dbscan = IncrementalDBSCAN()
dbscan.set_local_server_image(local_server_image=sys.argv[7])
dbscan.set_distance_matrix(offline_averaged_distances_matrix)
dbscan.set_SRPs_factor(SRPs_factor)
dbscan.set_eps(eps)
dbscan.set_min_samples(min_samples)
dbscan.set_scenario_config(int(sys.argv[3]))
dbscan.set_num_months_test(int(sys.argv[6]))

# load features into dbscan dataset's pd dataframe
for i in range(len(global_vars.features[0])):
    dbscan.dataset['PE_feature_'+str(i+1)] = global_vars.features[:,i]
dbscan.dataset['Unpacking_code_sequence'] = global_vars.features_mnemonics

# Run the offline phase for DBSCAN with training set samples
dbscan.batch_dbscan(distance_matrix=offline_averaged_distances_matrix)

# Run the offline phase for HDBSCAN with training set samples
HDB = hdbscan.HDBSCAN(metric='precomputed', min_cluster_size=dbscan.min_samples, min_samples=dbscan.min_samples).fit(offline_averaged_distances_matrix)

# Fit A random forest
h = FeatureHasher(n_features=50, input_type='string')
features_mnemonics_hashed = [h.transform([" ".join(s)]).toarray() if len(s)!=1 else h.transform([""]).toarray() for s in global_vars.features_mnemonics]
features_mnemonics_hashed = np.array(features_mnemonics_hashed)
features_mnemonics_hashed = np.squeeze(features_mnemonics_hashed, axis=1)
Rf_features=np.concatenate((np.array(global_vars.features), features_mnemonics_hashed), axis=1)
Rf = RandomForestClassifier(n_estimators=100, random_state=42)
Rf.fit(Rf_features,all_labels)

# Offline scores of measures FOR dbscan
offline_DB_ami_score= adjusted_mutual_info_score(global_vars.labels, np.array(dbscan.labels).reshape(1,len(dbscan.labels))[0], average_method="geometric")
offline_DB_homogeneity_score=homogeneity_score(global_vars.labels, np.array(dbscan.labels).reshape(1,len(dbscan.labels))[0])
if DBCV_integration==1:
    offline_DB_clustering_validity=clusters_validity(offline_averaged_distances_matrix, np.array(dbscan.labels).reshape(1,len(dbscan.labels))[0], d=len(global_vars.features[0]))
#offline_DB_silhouette_coefficient=metrics.silhouette_score(offline_averaged_distances_matrix,np.array(dbscan.labels).reshape(1,len(dbscan.labels))[0])
offline_Rf_accuracy_score = accuracy_score(all_labels,Rf.predict(Rf_features))

# Offline scores of measures for HDBSCAN
offline_HDB_ami_score= adjusted_mutual_info_score(global_vars.labels, np.array(HDB.labels_).reshape(1,len(HDB.labels_))[0], average_method="geometric")
offline_HDB_homogeneity_score=homogeneity_score(global_vars.labels, np.array(HDB.labels_).reshape(1,len(HDB.labels_))[0])
if DBCV_integration==1:
    offline_HDB_clustering_validity=clusters_validity(offline_averaged_distances_matrix, np.array(HDB.labels_).reshape(1,len(HDB.labels_))[0], d=len(global_vars.features[0]))
#offline_HDB_silhouette_coefficient=metrics.silhouette_score(offline_averaged_distances_matrix,np.array(HDB.labels_).reshape(1,len(HDB.labels_))[0])

print(offline_DB_homogeneity_score)
if DBCV_integration==1:
    print(offline_DB_clustering_validity)
#print(offline_DB_silhouette_coefficient)
print(offline_HDB_homogeneity_score)
if DBCV_integration==1:
    print(offline_HDB_clustering_validity)
#print(offline_HDB_silhouette_coefficient)
print(offline_Rf_accuracy_score)
# Till here everyhing is OK

###### The online phase starts here

# For printing results later:
update_results_table = PrettyTable()

# fiiling fiedl_names for the first table
#update_results_fields_names = ["date of update", "number_of_samples", "INC_number_of_clusters_found", "RS_DB_number_of_clusters_found", "RS_HDB_number_of_clusters_found", "INC_adjusted_nmi","RS_DB_adjusted_nmi", "RS_HDB_adjusted_nmi", "INC_homogeneity_score", "RS_DB_homogeneity_score", "RS_HDB_homogeneity_score", "RandomForest_accuracy_score", "INC_clusters_validity", "RS_DB_clusters_validity", "RS_HDB_clusters_validity", "INC_Silhouette Coefficient",  "RS_DB_Silhouette Coefficient", "RS_HDB_Silhouette Coefficient"]
if DBCV_integration==1:
    update_results_fields_names = ["date of update", "number_of_samples", "INC_number_of_clusters_found", "RS_DB_number_of_clusters_found", "RS_HDB_number_of_clusters_found", "INC_adjusted_nmi","RS_DB_adjusted_nmi", "RS_HDB_adjusted_nmi",  "INC_homogeneity_score", "RS_DB_homogeneity_score", "RS_HDB_homogeneity_score", "INC_clusters_validity", "RS_DB_clusters_validity", "RS_HDB_clusters_validity", "RandomForest_accuracy_score"]
else:
    update_results_fields_names = ["date of update", "number_of_samples", "INC_number_of_clusters_found",
                                   "RS_DB_number_of_clusters_found", "RS_HDB_number_of_clusters_found",
                                   "INC_adjusted_nmi", "RS_DB_adjusted_nmi", "RS_HDB_adjusted_nmi",
                                   "INC_homogeneity_score", "RS_DB_homogeneity_score", "RS_HDB_homogeneity_score",
                                   "RandomForest_accuracy_score"]

update_results_table.field_names = update_results_fields_names

# a list of clusters tables that we print at the end
INC_clusters_tables=[]
RS_DB_clusters_tables=[]
RS_HDB_clusters_tables=[]

INC_adjusted_nmi=offline_DB_ami_score
RS_DB_adjusted_nmi=offline_DB_ami_score
RS_HDB_adjusted_nmi=offline_HDB_ami_score

INC_homogeneity=offline_DB_homogeneity_score
RS_DB_homogeneity=offline_DB_homogeneity_score
RS_HDB_homogeneity=offline_HDB_homogeneity_score

if DBCV_integration==1:
    INC_clustering_validity=offline_DB_clustering_validity
    RS_DB_clustering_validity=offline_DB_clustering_validity
    RS_HDB_clustering_validity=offline_HDB_clustering_validity

# Used after for plotting results

RS_DB_offline_labels=list(np.array(dbscan.labels).reshape(1,len(dbscan.labels))[0])

x=[]
x.extend([1,2,3])

number_samples=[]
number_samples.extend([0,0,len(RS_DB_offline_labels)])

INC_ami=[]
INC_ami.extend([0,0,offline_DB_ami_score*100])
INC_homo=[]
INC_homo.extend([0,0,offline_DB_homogeneity_score*100])

if DBCV_integration==1:
    INC_DBCV=[]
    INC_DBCV.extend([0,0,offline_DB_clustering_validity[0]*100])
#INC_Silh=[]
#INC_Silh.extend([0,0,offline_DB_silhouette_coefficient*100])
INC_clus=[]
INC_clus.extend([0,0,(len(set(RS_DB_offline_labels)) - (1 if -1 in RS_DB_offline_labels else 0))])

DB_ami=[]
DB_ami.extend([0,0,offline_DB_ami_score*100])
DB_homo=[]
DB_homo.extend([0,0,offline_DB_homogeneity_score*100])

if DBCV_integration==1:
    DB_DBCV=[]
    DB_DBCV.extend([0,0,offline_DB_clustering_validity[0]*100])

#DB_Silh=[]
#DB_Silh.extend([0,0,offline_DB_silhouette_coefficient*100])
DB_clus=[]
DB_clus.extend([0,0,(len(set(RS_DB_offline_labels)) - (1 if -1 in RS_DB_offline_labels else 0))])

HDB_ami=[]
HDB_ami.extend([0,0,offline_HDB_ami_score*100])
HDB_homo=[]
HDB_homo.extend([0,0,offline_HDB_homogeneity_score*100])

if DBCV_integration==1:
    HDB_DBCV=[]
    HDB_DBCV.extend([0,0,offline_HDB_clustering_validity[0]*100])

#HDB_Silh=[]
#HDB_Silh.extend([0,0,offline_HDB_silhouette_coefficient*100])
HDB_clus=[]
HDB_clus.extend([0,0,(len(set(HDB.labels_)) - (1 if -1 in HDB.labels_ else 0))])

rf=[]
rf.extend([0,0,offline_Rf_accuracy_score*100])

#variable used only for printing
number_months=0
old_avg_elapsed = -1
total_number_samples_processed = 0

for i in months_of_tests:
    # Load features of testing set
    # Preprocessing tasks:

    number_months+=1

    date = prefix_date + (str(i).zfill(2))
    global_vars.dumped_files_suffix = ''
    global_vars.logfile = ''
    global_vars.features = []
    global_vars.features_mnemonics = []
    global_vars.feature_names = []
    global_vars.feature_names_short = []
    global_vars.features_categories = []
    global_vars.feature_types = []
    global_vars.feature_types_range = []
    global_vars.features_desc_and_comments = []
    global_vars.features_dict = {}
    global_vars.features_mnemonics_dict = {}
    global_vars.binaries_names = []
    global_vars.family_names = []
    global_vars.labels = []
    global_vars.scores = []
    global_vars.file_names=file_names_backup[:]

    preprocessing_functions.assign_log_file_name(clustering_mode, sys.argv[3], min_samples_per_family, requested_cats)
    global_vars.family_names = global_vars.file_names
    preprocessing_functions.remove_unpacked_files()
    preprocessing_functions.load_features_dir(clustering_mode, testing_set_feat_dir + str(i).zfill(2) + "/")
    preprocessing_functions.load_mnemonics_features_dir(clustering_mode, testing_set_mnemonics_feat_dir + str(i).zfill(2) + "/")
    preprocessing_functions.fam_to_remove(min_samples_per_family)
    preprocessing_functions.assign_labels_and_features()
    if preprocessing_functions.is_features_empty():
        exit(1)
    if preprocessing_functions.len_features_wrong():
        exit(1)
    preprocessing_functions.delete_unrequested_feat_cat(requested_cats)

    all_labels.extend(global_vars.labels)

    new_sample_columns = ['PE_feature_'+str(i+1) for i in range(len(global_vars.features[0]))] + ['Unpacking_code_sequence']

    print(dbscan.final_dataset)

    # Get indexes of the old dataset:
    # Warning: Put this before updating dbscan.dataset
    old_unique_elements_features,old_unique_elements_indexes = get_samples_indexes(dbscan.dataset['Unpacking_code_sequence'])

    list_of_samples_to_update = []
    list_2_of_samples_to_update = []

    # Do incremental learning
    number_test_samples=len(global_vars.features)
    for j in range(number_test_samples):
        feature_values = list(global_vars.features[j])
        feature_values.append(global_vars.features_mnemonics[j])
        new_sample = pd.DataFrame([feature_values], columns=new_sample_columns)
        dbscan.set_data(new_sample)
        list_2_of_samples_to_update.append(new_sample)
        list_of_samples_to_update.append(new_sample)

        # Prepare the dataset of mneminics sequences features
        list_of_mnemonic_sequences_to_update = [list_of_samples_to_update[k]['Unpacking_code_sequence'][0] for k in range(len(list_of_samples_to_update))]

        # Get indexes of the new samples we want to update
        new_unique_elements_features, new_unique_elements_indexes = get_samples_indexes(list_of_mnemonic_sequences_to_update)

        # Adjust the starting index
        new_unique_elements_indexes = [[x + offline_averaged_distances_matrix.shape[0] for x in new_unique_elements_indexes[k]] for k in range(len(new_unique_elements_indexes))]
        # Update the pairewise distance matrix
        online_averaged_pairewise_matrix = update_pairewise_disance_matrix(sys.argv[3], offline_averaged_distances_matrix,dbscan.dataset, old_unique_elements_indexes,new_unique_elements_indexes,old_unique_elements_features,new_unique_elements_features,list_of_samples_to_update)
        #Update the distance matrix for dbscan object
        dbscan.set_distance_matrix(distance_matrix=online_averaged_pairewise_matrix)
        # Finally, launch the incremental dbscan

        start_time = datetime.datetime.now()
        dbscan.incremental_dbscan_()
        elapsed = datetime.datetime.now() - start_time
        if old_avg_elapsed == -1:
            old_avg_elapsed = elapsed.total_seconds()
        avg_elapsed = (elapsed.total_seconds() + old_avg_elapsed)/2
        old_avg_elapsed = avg_elapsed

        # Update the offline distance matrix before next iteration
        offline_averaged_distances_matrix = online_averaged_pairewise_matrix

        # Update the old unique elements
        old_unique_elements_features, old_unique_elements_indexes = get_samples_indexes(dbscan.dataset['Unpacking_code_sequence'])

        # Make empty the list, since, we are updating the matrix at each iteration
        list_of_samples_to_update=[]

        total_number_samples_processed += 1

        print(dbscan.final_dataset)
        print("You are in a ", dbscan.local_server_image, " version of this program")
        print("Core points factor: k=", dbscan.SRPs_factor)
        print("Number of clusters found right now is: ", dbscan.number_clusters)
        print("Current month of test is: ", i, " from a total of ", dbscan.num_months_test, "months")
        print("Number of samples processed so far is: ", total_number_samples_processed)
        print("INC_ami: ", INC_adjusted_nmi)
        print("INC_homogeneity: ", INC_homogeneity)
        print("RS_DB_homogeneity: ", RS_DB_homogeneity)
        print("RS_HDB_homogeneity: ", RS_HDB_homogeneity)
        if DBCV_integration == 1:
            print("INC_quality: ", INC_clustering_validity[0])
            print("DB_quality: ", RS_DB_clustering_validity[0])
            print("HDB_quality: ", RS_HDB_clustering_validity[0])
            print("number of months spent: ", number_months)
        print("avg time to update one simple is: ", avg_elapsed, " seconds")

    #Predict with Random forest
    h = FeatureHasher(n_features=50, input_type='string')
    list_of_mnemonic_sequences_to_update = [list_2_of_samples_to_update[i]['Unpacking_code_sequence'][0] for i in range(len(list_2_of_samples_to_update))]
    new_samples_features_mnemonics_hashed = [h.transform([" ".join(s)]).toarray() if len(s) != 1 else h.transform([""]).toarray() for s in list_of_mnemonic_sequences_to_update]
    new_samples_features_mnemonics_hashed = np.array(new_samples_features_mnemonics_hashed)
    new_samples_features_mnemonics_hashed = np.squeeze(new_samples_features_mnemonics_hashed, axis=1)
    pe_list_new_samples = [list_2_of_samples_to_update[i].iloc[0] for i in range(len(list_2_of_samples_to_update))]
    pe_list_new_samples = [[pe_list_new_samples[i]['PE_feature_' + str(j + 1)] for j in range(len(global_vars.features[0]))] for i in range(len(pe_list_new_samples))]
    pe_array_new_samples = np.array(pe_list_new_samples)
    new_samples_Rf_features = np.concatenate((pe_array_new_samples, new_samples_features_mnemonics_hashed), axis=1)
    online_Rf_accuracy_score = accuracy_score(global_vars.labels, Rf.predict(new_samples_Rf_features))

    #Retrain from scratch with DBSCAN: number of clusters found + homogeinity score + measures
    RS_DB = DBSCAN(metric='precomputed', eps=dbscan.eps, min_samples=dbscan.min_samples).fit(online_averaged_pairewise_matrix)
    RS_DB_labels = RS_DB.labels_
    RS_DB_n_clusters=len(set(RS_DB_labels)) - (1 if -1 in RS_DB_labels else 0)
    #RS_DB_Silhouette_Coefficient = metrics.silhouette_score(online_averaged_pairewise_matrix, RS_DB_labels)
    if DBCV_integration == 1:
        RS_DB_clustering_validity = clusters_validity(online_averaged_pairewise_matrix,np.array(RS_DB_labels).reshape(1, len(RS_DB_labels))[0], d=len(global_vars.features[0]))
    RS_DB_homogeneity = homogeneity_score(all_labels, np.array(RS_DB_labels).reshape(1, len(RS_DB_labels))[0])
    RS_DB_adjusted_nmi = adjusted_mutual_info_score(all_labels, np.array(RS_DB_labels).reshape(1, len(RS_DB_labels))[0], average_method="geometric")

    # Retrain from scratch with HDBSCAN: number of clusters found + homogeinity score + measures
    RS_HDB = hdbscan.HDBSCAN(metric='precomputed', min_cluster_size=dbscan.min_samples, min_samples=dbscan.min_samples).fit(online_averaged_pairewise_matrix)
    RS_HDB_labels = RS_HDB.labels_
    RS_HDB_n_clusters = len(set(RS_HDB_labels)) - (1 if -1 in RS_HDB_labels else 0)
    #RS_HDB_Silhouette_Coefficient = metrics.silhouette_score(online_averaged_pairewise_matrix, RS_HDB_labels)
    if DBCV_integration == 1:
        RS_HDB_clustering_validity = clusters_validity(online_averaged_pairewise_matrix,np.array(RS_HDB_labels).reshape(1, len(RS_HDB_labels))[0], d=len(global_vars.features[0]))
    RS_HDB_homogeneity = homogeneity_score(all_labels, np.array(RS_HDB_labels).reshape(1, len(RS_HDB_labels))[0])
    RS_HDB_adjusted_nmi = adjusted_mutual_info_score(all_labels, np.array(RS_HDB_labels).reshape(1, len(RS_HDB_labels))[0], average_method="geometric")

    # Incremental dbscan: number of clusters found + homogeinity score + measures
    INC_clustering_labels = list(dbscan.final_dataset['Label'])
    INC_n_clusters = len(set(INC_clustering_labels)) - (1 if -1 in INC_clustering_labels else 0)
    #INC_Silhouette_Coefficient=metrics.silhouette_score(online_averaged_pairewise_matrix,INC_clustering_labels)
    if DBCV_integration == 1:
        INC_clustering_validity=clusters_validity(online_averaged_pairewise_matrix, np.array(INC_clustering_labels).reshape(1,len(INC_clustering_labels))[0], d=len(global_vars.features[0]))
    INC_homogeneity=homogeneity_score(all_labels, np.array(INC_clustering_labels).reshape(1,len(INC_clustering_labels))[0])
    INC_adjusted_nmi = adjusted_mutual_info_score(all_labels, np.array(INC_clustering_labels).reshape(1,len(INC_clustering_labels))[0], average_method="geometric")

    #Update the table of results
    #row = [date,  len(dbscan.dataset), INC_n_clusters, RS_DB_n_clusters, RS_HDB_n_clusters, INC_adjusted_nmi, RS_DB_adjusted_nmi, RS_HDB_adjusted_nmi, INC_homogeneity, RS_DB_homogeneity, RS_HDB_homogeneity, online_Rf_accuracy_score, INC_clustering_validity, RS_DB_clustering_validity, RS_HDB_clustering_validity, INC_Silhouette_Coefficient, RS_DB_Silhouette_Coefficient, RS_HDB_Silhouette_Coefficient]
    if DBCV_integration == 1:
        row = [date,  len(dbscan.dataset), INC_n_clusters, RS_DB_n_clusters, RS_HDB_n_clusters, INC_adjusted_nmi, RS_DB_adjusted_nmi, RS_HDB_adjusted_nmi, INC_homogeneity, RS_DB_homogeneity, RS_HDB_homogeneity, INC_clustering_validity, RS_DB_clustering_validity, RS_HDB_clustering_validity, online_Rf_accuracy_score]
    else:
        row = [date, len(dbscan.dataset), INC_n_clusters, RS_DB_n_clusters, RS_HDB_n_clusters, INC_adjusted_nmi,
               RS_DB_adjusted_nmi, RS_HDB_adjusted_nmi, INC_homogeneity, RS_DB_homogeneity, RS_HDB_homogeneity,
               online_Rf_accuracy_score]

    update_results_table.add_row(row)

    # Update the table clusters found for each month for incremental learning and for RS_DB and RS_HDB
    INC_clusters_tables.append((date, generate_INC_clusters_table(dbscan.final_dataset, np.unique(np.array(all_labels)), all_labels)))
    RS_DB_clusters_tables.append((date, generate_RS_clusters_table(RS_DB_labels, np.unique(np.array(all_labels)), all_labels)))
    RS_HDB_clusters_tables.append((date, generate_RS_clusters_table(RS_HDB_labels, np.unique(np.array(all_labels)), all_labels)))

    # Update the offline distance matrix before next iteration
    #offline_averaged_distances_matrix=online_averaged_pairewise_matrix

    # fill the corresponding x and y axis values, for plotting later
    x.append(x[-1]+1)
    my_xticks.append(date)

    number_samples.append(online_averaged_pairewise_matrix.shape[0])

    INC_ami.append(INC_adjusted_nmi*100)
    INC_homo.append(INC_homogeneity*100)
    if DBCV_integration == 1:
        INC_DBCV.append(INC_clustering_validity[0]*100)
    #INC_Silh.append(INC_Silhouette_Coefficient*100)
    INC_clus.append(INC_n_clusters)

    DB_ami.append(RS_DB_adjusted_nmi* 100)
    DB_homo.append(RS_DB_homogeneity * 100)
    if DBCV_integration == 1:
        DB_DBCV.append(RS_DB_clustering_validity[0] * 100)
    #DB_Silh.append(RS_DB_Silhouette_Coefficient * 100)
    DB_clus.append(RS_DB_n_clusters)

    HDB_ami.append(RS_HDB_adjusted_nmi* 100)
    HDB_homo.append(RS_HDB_homogeneity * 100)
    if DBCV_integration == 1:
        HDB_DBCV.append(RS_HDB_clustering_validity[0] * 100)
    #HDB_Silh.append(RS_HDB_Silhouette_Coefficient * 100)
    HDB_clus.append(RS_HDB_n_clusters)

    rf.append(online_Rf_accuracy_score*100)

    config = "eps_" + str(dbscan.eps) + "_min_samples_" + str(dbscan.min_samples) + "_scenario_" + str(sys.argv[3]) + "_shuffle_" + str(sys.argv[4]) + "_k_" + str(sys.argv[5]) + "_"

    coordinates_path = "results/" + config + "x.npy"
    np.save(coordinates_path, np.array(x))
    coordinates_path = "results/" + config + "my_xticks.npy"
    np.save(coordinates_path, np.array(my_xticks))
    coordinates_path = "results/" + config + "number_samples.npy"
    np.save(coordinates_path, np.array(number_samples))
    coordinates_path = "results/" + config + "INC_adjusted_nmi.npy"
    np.save(coordinates_path, np.array(INC_ami))
    coordinates_path = "results/" + config + "RS_DB_adjusted_nmi.npy"
    np.save(coordinates_path, np.array(DB_ami))
    coordinates_path = "results/" + config + "RS_HDB_adjusted_nmi.npy"
    np.save(coordinates_path, np.array(HDB_ami))
    coordinates_path = "results/" + config + "INC_homogeneity.npy"
    np.save(coordinates_path, np.array(INC_homo))
    coordinates_path = "results/" + config + "RS_DB_homogeneity.npy"
    np.save(coordinates_path, np.array(DB_homo))
    coordinates_path = "results/" + config + "RS_HDB_homogeneity.npy"
    np.save(coordinates_path, np.array(HDB_homo))
    if DBCV_integration == 1:
        coordinates_path = "results/" + config + "INC_clustering_validity.npy"
        np.save(coordinates_path, np.array(INC_DBCV))
        coordinates_path = "results/" + config + "RS_DB_clustering_validity.npy"
        np.save(coordinates_path, np.array(DB_DBCV))
        coordinates_path = "results/" + config + "RS_HDB_clustering_validity.npy"
        np.save(coordinates_path, np.array(HDB_DBCV))
    #coordinates_path = "results/" + config + "INC_Silhouette_Coefficient.npy"
    #np.save(coordinates_path , np.array(INC_Silh))
    #coordinates_path = "results/" + config + "RS_DB_Silhouette_Coefficient.npy"
    #np.save(coordinates_path, np.array(DB_Silh))
    #coordinates_path = "results/" + config + "RS_HDB_Silhouette_Coefficient.npy"
    #np.save(coordinates_path, np.array(HDB_Silh))
    coordinates_path = "results/" + config + "INC_n_clusters.npy"
    np.save(coordinates_path, np.array(INC_clus))
    coordinates_path = "results/" + config + "INC_avg_time_per_sample.npy"
    np.save(coordinates_path, np.array(avg_elapsed))
    coordinates_path = "results/" + config + "DB_n_clusters.npy"
    np.save(coordinates_path, np.array(DB_clus))
    coordinates_path = "results/" + config + "HDB_n_clusters.npy"
    np.save(coordinates_path, np.array(HDB_clus))
    coordinates_path = "results/" + config + "Rf_accuracy_score.npy"
    np.save(coordinates_path, np.array(rf))
    coordinates_path = "results/" + config + "packers_families_labels.npy"
    np.save(coordinates_path, np.array(all_labels))
    coordinates_path = "results/" + config + "final_clusters_labels.npy"
    np.save(coordinates_path, np.array(INC_clustering_labels))
    if float(sys.argv[5]) < 1:
        coordinates_path = "results/" + config + "final_matrix.npy"
        np.save(coordinates_path, online_averaged_pairewise_matrix)
    coordinates_path = "results/" + config + "core_points.npy"
    np.save(coordinates_path, np.array(dbscan.core_samples_indexes))

    # Print the results of the two tables for each month, and write them as a log file
    print(update_results_table)

    with open(results_file_name, "w") as f:
        f.write(str(update_results_table))
        for elt in zip(INC_clusters_tables,RS_DB_clusters_tables,RS_HDB_clusters_tables):
            f.write("\n\n\n" + "results for INC: " + "\n")
            f.write("date of update: " + elt[0][0] + "\n" )
            f.write(str(elt[0][1]))
            f.write("\n\n\n" + "results for RS_DB: " + "\n")
            f.write("date of update: " + elt[1][0] + "\n")
            f.write(str(elt[1][1]))
            f.write("\n\n\n" + "results for RS_HDB: " + "\n")
            f.write("date of update: " + elt[2][0] + "\n")
            f.write(str(elt[2][1]))
    f.close()