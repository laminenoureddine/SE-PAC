import sys
import preprocessing_functions
import global_vars
import re
#from gower_distance import gower_distances
import numpy as np
from scipy.spatial import distance
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
import pickle


# Used only for extraction
min_samples_per_family = 1
#min_samples_per_family = 10
clustering_mode = 'consensus_only'
#requested_cats = ['section', 'imp_func', 'entropy']
requested_cats = ['section']


if (len(sys.argv)!=2):
    print("Sorry, you should specify 1 command-line arguments:\n1-number of config(realistic_dataset or synthetic dataset)")
    exit(1)

if int(sys.argv[1]) == 1:
    feat_dir = "/home/lnouredd/Desktop/PhD/git/datasets/mixed_dataset/pe_dates_featfiles_packers_families/training_set_1/"
    global_vars.file_names = [
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
        "Wise"
    ]

elif int(sys.argv[1]) == 2:
    feat_dir = "/home/lnouredd/Desktop/PhD/git/datasets/mixed_dataset/pe_dates_featfiles_packers_families/training_set_2/"
    global_vars.file_names = ["Armadillo",
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
preprocessing_functions.log(
    'Using ' + str(len(global_vars.feature_names)) + ' features: ' + str(global_vars.feature_names))
preprocessing_functions.log(
    'Found ' + str(len(set(global_vars.features_categories))) + ' feature categories: ' + str(
        sorted(set(global_vars.features_categories), key=global_vars.features_categories.index)))
preprocessing_functions.delete_unrequested_feat_cat(requested_cats)
# ----

# Get the manhattan distances matrix
#global_vars.features = StandardScaler().fit_transform(global_vars.features)

min_max_scaler = MinMaxScaler().fit(global_vars.features)
np.save("../scaler_values/min_max_scaler_min.npy", min_max_scaler.data_min_)
np.save("../scaler_values/min_max_scaler_max.npy", min_max_scaler.data_max_)
np.save("../scaler_values/min_max_scaler_scale.npy", min_max_scaler.scale_)

global_vars.features = min_max_scaler.transform((global_vars.features))


len_features = len(global_vars.features[0])

manhattan_distance_matrix = distance.cdist(global_vars.features, global_vars.features, 'cityblock') / len_features
np.save('../distance_matrices/offline_manhattan_distance_matrix.npy', manhattan_distance_matrix)
