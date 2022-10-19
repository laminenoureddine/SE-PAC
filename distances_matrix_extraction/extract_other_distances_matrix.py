import sys
import preprocessing_functions
import global_vars
import re
import numpy as np
from collections import Counter
from sklearn.feature_extraction import FeatureHasher
from sklearn.preprocessing import OneHotEncoder
from scipy.spatial import distance
from sklearn.metrics.pairwise import euclidean_distances, manhattan_distances, cosine_distances
#import editdistance

# Used only for extraction
min_samples_per_family = 1
#min_samples_per_family = 10
clustering_mode = 'consensus_only'
requested_cats = ['section']

if (len(sys.argv)!=2):
    print("Sorry, you should specify 1 command-line arguments:\n1-number of config(realistic_dataset or synthetic dataset)")
    exit(1)

if int(sys.argv[1]) == 1:
    feat_dir = "/home/lnouredd/Desktop/PhD/git/datasets/mixed_dataset/radare2_dates_featfiles_packers_families/training_set_1/"
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
        "Wise",
        "ActiveMARK",
        "FishPE",
        "PCGuard",
        "PESpin",
        "Shrinker",
        "NSIS",
        "InnoSetup",
        "AutoIt"
        ]

elif int(sys.argv[1]) == 2:
    feat_dir = "/home/lnouredd/Desktop/PhD/git/datasets/mixed_dataset/radare2_dates_featfiles_packers_families/training_set_2/"
    global_vars.file_names = \
        [
             "Armadillo",
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

preprocessing_functions.assign_log_file_name(clustering_mode, min_samples_per_family, requested_cats)
global_vars.family_names = global_vars.file_names
#preprocessing_functions.remove_unpacked_files()
preprocessing_functions.load_mnemonics_features_dir(clustering_mode, feat_dir)
preprocessing_functions.fam_to_remove(min_samples_per_family)
preprocessing_functions.assign_mnemonics_labels_and_features()
# -----------------------------------

features_memeonic_with_equal_size = []

for i, list_memeonic in enumerate(global_vars.features):
    temp = list_memeonic
    if len(temp) < 50:
        for j in range((len(temp)+1), 51):
            temp.append("null")
        features_memeonic_with_equal_size.append(temp)
    else:
        features_memeonic_with_equal_size.append(temp)

global_vars.features= []
# Used only for extraction
min_samples_per_family = 1
#min_samples_per_family = 10
clustering_mode = 'consensus_only'
requested_cats = ['section']

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
        "Wise",
        "ActiveMARK",
        "FishPE",
        "PCGuard",
        "PESpin",
        "Shrinker",
        "NSIS",
        "InnoSetup",
        "AutoIt"
        ]

elif int(sys.argv[1]) == 2:
    feat_dir = "/home/lnouredd/Desktop/PhD/git/datasets/mixed_dataset/pe_dates_featfiles_packers_families/training_set_2/"
    global_vars.file_names = \
        [
        "Armadillo",
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

min_samples_per_family = 1
#min_samples_per_family = 10
clustering_mode = 'consensus_only'
requested_cats = ['section']

preprocessing_functions.assign_log_file_name(clustering_mode, min_samples_per_family, requested_cats)
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

concatenated_features = []

#I fixed the pb, so the bloc of instructions below is no longer uselful
#if int(sys.argv[1]) == 2:
#    print(len(features_memeonic_with_equal_size))
#    del list(global_vars.features)[-1]
#   global_vars.features = list(global_vars.features)
#    global_vars.features.pop()
#    print(len(global_vars.features))

for i, item in enumerate(global_vars.features):
    print(i)
    temp = list(global_vars.features[i])
    temp.extend(features_memeonic_with_equal_size[i])
    concatenated_features.append(temp)
    print(concatenated_features[i])

enc = OneHotEncoder(handle_unknown='ignore')
f = enc.fit_transform(np.array(concatenated_features)).toarray()

len_feat = len(concatenated_features[0])

#OneHotEncode_manhattan_distance_matrix = distance.cdist(f, f, 'cityblock') / len(concatenated_features[0])/len_feat
#matrix_long_name_path = '../distance_matrices/offline_OneHotEncode_manhattan_distance_matrix_'+ sys.argv[1] + '.npy'
#np.save(matrix_long_name_path, OneHotEncode_manhattan_distance_matrix)

OneHotEncode_euclidean_distance_matrix = euclidean_distances(f, f)/len_feat
matrix_long_name_path = '../distance_matrices/offline_OneHotEncode_euclidean_distance_matrix_'+ sys.argv[1] + '.npy'
np.save(matrix_long_name_path, OneHotEncode_euclidean_distance_matrix)

#OneHotEncode_cosine_distance_matrix = cosine_distances(f, f)/len_feat
#matrix_long_name_path = '../distance_matrices/offline_OneHotEncode_cosine_distance_matrix_'+ sys.argv[1] + '.npy'
#np.save(matrix_long_name_path, OneHotEncode_cosine_distance_matrix)