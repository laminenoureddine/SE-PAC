import preprocessing_functions
import global_vars
import numpy as np
np.set_printoptions(edgeitems=15)
np.core.arrayprint._line_width = 250
import pandas as pd
pd.set_option('display.width', 250)
pd.set_option('display.max_columns', 15)
import sys
import argparse
import json
import itertools
from time import time, strftime
from math import isinf, isnan, factorial
from os.path import isfile, exists
from os import makedirs, remove
from prettytable import PrettyTable
from ast import literal_eval
from sklearn import preprocessing
from sklearn.externals import joblib
from collections import Counter, OrderedDict
from sklearn.preprocessing import minmax_scale
from sklearn.decomposition import PCA
from sklearn.preprocessing import scale
from sklearn.ensemble import ExtraTreesClassifier


def main_process(feat_dir,
                 consensus_only,
                 # save_clustering_models
                 # save_results,
                 # score_type,
                 # feats_to_print,
                 printed_digits,
                 use_short_feat_names,
                 min_samples_per_family,
                 requested_cats,
                 ):
    format_string = '.' + str(printed_digits) + 'f'  # format string for outputted floats

    clustering_mode = 'consensus_only' if consensus_only else 'no_consensus_only'
    preprocessing_functions.assign_log_file_name(clustering_mode, min_samples_per_family, requested_cats)
    global_vars.family_names = global_vars.file_names
    preprocessing_functions.remove_unpacked_files()
    preprocessing_functions.load_features_dir(clustering_mode, feat_dir)
    preprocessing_functions.fam_to_remove(min_samples_per_family)
    preprocessing_functions.assign_labels_and_features()

    # TODO: add scale and PCA as functions and as command line options for the script

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


    # --------------The code used for calculating statistics starts here----------
    """
    with open('harcoded_data/results_of_pepac.txt') as f:
        binaries_full_labels = [line.rstrip('\n') for line in f]
    f.close()
    # need it as an array
    binaries_full_labels = np.array(binaries_full_labels)
    """

    feature_values_dict = OrderedDict()
    for i in range(0, len(global_vars.features[0])):
        features_column_values = []
        features_column_values = [row[i] for row in global_vars.features]

        for index, feature_value in enumerate(features_column_values):
            try:
                feature_values_dict[global_vars.feature_names_short[i], feature_value].append(index)
            except KeyError:
                feature_values_dict[global_vars.feature_names_short[i], feature_value] = []
                feature_values_dict[global_vars.feature_names_short[i], feature_value].append(index)

    binary_full_labels_by_feature = OrderedDict()
    for i in feature_values_dict:
        binary_full_labels_by_feature[i] = []
        binary_full_labels_by_feature[i] = [global_vars.labels[index] for index in feature_values_dict[i]]
        c = Counter(binary_full_labels_by_feature[i])
        binaries_total_amout_by_feature = sum(c.values())
        binaries_labels_amount_by_feature = c.items()
        preprocessing_functions.log("\n\n\n\nStatisctics about binaries belonging to the key " + str(i) + " :")
        preprocessing_functions.log(
            "The total amount of binaries in this key is: " + str(binaries_total_amout_by_feature))
        for element in binaries_labels_amount_by_feature:
            preprocessing_functions.log(str(element))
        preprocessing_functions.log("\n")
    preprocessing_functions.log("\n\n\n\n\n\n")

    exit(0)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="This script uses ML clustering algorithms for packing clustering")
    parser.add_argument("-p", "--printed-digits", nargs='?', type=int, dest='printed_digits', default=5,
                        help="How many digits to print when printing floats (default 5)")
    parser.add_argument("-m", "--min-samples-per-family", nargs='?', type=int, dest='min_samples_per_family',
                        default=10, help="Discard families with less than this number of samples (default 10)")
    restricted_features = parser.add_mutually_exclusive_group()
    # restricted_features.add_argument("-f", "--feats-to-print", nargs='?', type=int, dest='feats_to_print', default=0,
    #                                 help="how many features to print in the result table (in descending order of "
    #                                      "importance); set to 0 to disable feature selection/testing")
    restricted_features.add_argument("--feat-cat", action="append",
                                     help="Restrict to only these feature categories (can be used more than once to"
                                          " add categories). Allowed categories are: header, section,"
                                          " entropy, ep64, imp_func, other"
                                     )

    restricted_task = parser.add_mutually_exclusive_group()
    parser.add_argument("-o", "--consensus-only", default=False, action="store_true", dest="consensus_only",
                        help="Use only consensus files from the target directory  as sources "
                             "(files in the form consensus_<FAMILY_NAME> )")
    # parser.add_argument("-s", "--save-classifiers", default=False, action="store_true", dest="save_classifiers",
    #                    help="Saves the trained classifiers in trained_classifiers/<NAME>.pkl")
    # parser.add_argument("-r", "--save-results_with_synthetic_centroids", default=False, action="store_true", dest="save_results",
    #                    help="Saves the testing results_with_synthetic_centroids in results_with_synthetic_centroids/results_<SETTINGS>.{txt,json} ")

    parser.add_argument("-n", "--use-short-feat-names", default=False, action="store_true", dest="use_short_feat_names",
                        help="Use short names for the features")
    # parser.add_argument("-t", "--score-type", default="f1_micro", dest='score_type',
    #                    help="Type of score to check for testing")
    parser.add_argument("feat_dir", nargs='?', default='featfiles/',
                        help="Directory containing the feature files to scan.")

    args = parser.parse_args()
    feat_dir = args.feat_dir
    consensus_only = args.consensus_only
    # score_type = args.score-type
    # save_clusering_models = args.save_classifiers
    # save_results = args.save_results
    # feats_to_print = args.feats_to_print
    printed_digits = args.printed_digits
    use_short_feat_names = args.use_short_feat_names
    min_samples_per_family = args.min_samples_per_family
    requested_cats = args.feat_cat

    main_process(feat_dir,
                 consensus_only,
                 # save_clustering models, (like before for classifiers models)
                 # save_results,
                 # feats_to_print,
                 # score_type
                 printed_digits,
                 use_short_feat_names, min_samples_per_family, requested_cats)
