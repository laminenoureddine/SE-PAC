import sys
import preprocessing_functions
import global_vars
import re
import numpy as np
from collections import Counter
#import editdistance


def tapered_levenshtein(s1, s2):
    max_len = float(max(len(s1), len(s2)))
    if len(s1) < len(s2):
        return tapered_levenshtein(s2, s1)

    # len(s1) >= len(s2)
    if len(s2) == 0:
        return len(s1)

    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            taper = 1.0 - min(i, j) / max_len
            insertions = previous_row[j + 1] + taper # j+1 instead of j since previous_row and current_row are one character longer
            deletions = current_row[j] + taper       # than s2
            substitutions = previous_row[j] + (c1 != c2) * taper
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row

    return previous_row[-1]


# Used only for extraction
min_samples_per_family = 1
#min_samples_per_family = 10
clustering_mode = 'consensus_only'
requested_cats = None

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

max_len_seq = 50

tapered_levenshtein_distances_matrix = np.empty([len(global_vars.features), len(global_vars.features)], dtype=float)
features_tuples = map(tuple,np.array(global_vars.features))
unique_elements_features  = sorted(set(features_tuples), key=features_tuples.index)

unique_elements_indexes = []
for j in range(len(unique_elements_features)):
    l=[i for i in range(len(features_tuples)) if features_tuples[i] == unique_elements_features[j]]
    unique_elements_indexes.append(l)


for i in range(len(unique_elements_features)):
    print (i)
    for j in range(len(unique_elements_features)):
        s1 = unique_elements_features[i]
        s2 = unique_elements_features[j]
        if s1 == ('',):
            s1 = ('')
        if s2 == ('',):
            s2 = ('')

        # If the two sequences are empty, it doesn't make sense to put a similarity to 1, so ...
        if (s1 == ('') and s2 == ('')):
            scaled_distance = -1

        else:
            distance = tapered_levenshtein(s1, s2)
            scaled_distance = distance / float(max_len_seq)

        for k in range(len(unique_elements_indexes[i])):
            tapered_levenshtein_distances_matrix[unique_elements_indexes[i][k],unique_elements_indexes[j]] = scaled_distance

        tapered_levenshtein_distances_matrix[np.ix_(unique_elements_indexes[i], unique_elements_indexes[j])]
        tapered_levenshtein_distances_matrix[np.ix_(unique_elements_indexes[j], unique_elements_indexes[i])]

matrix_long_name_path = '../distance_matrices/offline_tapered_levenshtein_distances_matrix_' + sys.argv[1] + '.npy'
np.save(matrix_long_name_path, tapered_levenshtein_distances_matrix)
