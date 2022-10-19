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

    # --------------The code used for calculating statistics starts here----------

    # get hardcoded informations
    with open('harcoded_data/feature_types.txt') as f:
        feature_types = [line.rstrip('\n') for line in f]
    f.close()

    with open('harcoded_data/feature_types_range.txt') as f:
        feature_types_range = [line.rstrip('\n') for line in f]
    f.close()

    with open('harcoded_data/features_description_with_comments.txt') as f:
        global_vars.features_desc_and_comments = [line.rstrip('\n') for line in f]
    f.close()

    with open('harcoded_data/results_of_pepac.txt') as f:
        binaries_full_labels = [line.rstrip('\n') for line in f]

    f.close()
    # Need it as an array
    binaries_full_labels = np.array(binaries_full_labels)

    # delete all unrequested features, feature_names, feature_type, feature_type_range, feature_desc_comments ...
    preprocessing_functions.delete_unrequested_feat_cat(requested_cats)

    # get features importance
    Et_n_estimators = 18
    Et_criterion = 'gini'
    Et_max_depth = 29
    Et_class_weight = 'balanced'
    clf = ExtraTreesClassifier(n_estimators=Et_n_estimators, criterion=Et_criterion, max_depth=Et_max_depth,
                               class_weight=Et_class_weight)
    clf.fit(global_vars.features, global_vars.labels)
    feature_importances = np.array(clf.feature_importances_)
    feature_importance_desc_rank = np.argsort(-feature_importances).tolist()

    # For getting amount of samples in each family: need it later
    unique_fam_amount = OrderedDict()
    for elt in global_vars.labels:
        try:
            unique_fam_amount[elt] += 1
        except KeyError:
            unique_fam_amount[elt] = 1

    fam_version_features_column = {}
    # This small block is for initializing dictionaries of unique_fam_version
    unique_fam_version, unique_fam_index = np.unique(binaries_full_labels, return_index=True)
    unique_fam_version = binaries_full_labels[np.sort(unique_fam_index)]
    for j in range(0, len(unique_fam_version)):
        fam_version_features_column[unique_fam_version[j]] = []

    # Only one comparative table will be printed
    features_fam_table = PrettyTable()
    table_field_names = []
    table_field_names = ["feature_number", "feature_name", "feature_importance_rank", "type", "type_range",
                         "global_range"]
    table_field_names.extend(unique_fam_version.tolist())
    table_field_names.append("features_comments")
    features_fam_table.field_names = table_field_names

    global_min = np.min(global_vars.features, axis=0)
    global_max = np.max(global_vars.features, axis=0)

    # main loop for doing the stuff of statistics for each family version ...
    for i in range(0, len(global_vars.features[0])):
        binaries_full_labels_index = 0
        list_fam_version_range_values = []
        first_fam_index = 0

        for j in range(0, len(global_vars.features)):
            fam_version_features_column[binaries_full_labels[j]] += [global_vars.features[j][i]]
            fam_version_features_column[binaries_full_labels[j]].sort()

        for j in range(0, len(unique_fam_version)):
            fam_version_min = fam_version_features_column[unique_fam_version[j]][0]
            fam_version_max = fam_version_features_column[unique_fam_version[j]][-1]

            # formatting strings for printing: drop trailing zeros and limit the number of digits after comma
            fam_version_min = format(fam_version_min, format_string)
            fam_version_min = preprocessing_functions.drop_traling_zeros(float(fam_version_min))
            fam_version_max = format(fam_version_max, format_string)
            fam_version_max = preprocessing_functions.drop_traling_zeros(float(fam_version_max))
            fam_version_range_values = '[' + str(fam_version_min) + ',' + str(fam_version_max) + ']'
            list_fam_version_range_values.append(fam_version_range_values)
            fam_version_features_column[unique_fam_version[j]] = []

        global_min_i = global_min[i]
        global_max_i = global_max[i]

        # Also formating these two above values
        global_min_i = format(global_min_i, format_string)
        global_min_i = preprocessing_functions.drop_traling_zeros(float(global_min_i))
        global_max_i = format(global_max_i, format_string)
        global_max_i = preprocessing_functions.drop_traling_zeros(float(global_max_i))
        global_range_i = '[' + str(global_min_i) + ',' + str(global_max_i) + ']'

        feature_names_short_i = global_vars.feature_names_short[i]
        feature_importance_desc_rank_i = feature_importance_desc_rank[i]
        features_type_i = global_vars.feature_types[i]
        features_type_range_i = global_vars.feature_types_range[i]
        features_desc_and_comments_i = global_vars.features_desc_and_comments[i]

        row = [(i + 1), feature_names_short_i, feature_importance_desc_rank_i, features_type_i, features_type_range_i,
               global_range_i]
        row.extend(list_fam_version_range_values)
        row.append(features_desc_and_comments_i)
        features_fam_table.add_row(row)

    preprocessing_functions.log("\n\nComparative table: features range between different family versions")
    preprocessing_functions.log('\n' + str(features_fam_table))
    """
    for item in unique_fam_amout.items():
        fam = item[0]
        last_fam_index = first_fam_index + item[1]
        features_fam_array = np.copy(global_vars.features[first_fam_index:last_fam_index])
        features_fam_column = [row[i] for row in features_fam_array]
        features_fam_column.sort()
        features_fam_min = features_fam_column[0]
        features_fam_max = features_fam_column[-1]

        # formatting strings for printing: drop trailing zeros and limit the number of digits after comma
        features_fam_min = format(features_fam_min, format_string)
        features_fam_min = basic_functions.drop_traling_zeros(float(features_fam_min))
        features_fam_max = format(features_fam_max, format_string)
        features_fam_max = basic_functions.drop_traling_zeros(float(features_fam_max))

        fam_range_values[fam] = '[' + str(features_fam_min) + ',' + str(features_fam_max) + ']'
        list_fam_range_values.append(fam_range_values[fam])
        first_fam_index += item[1]
    """
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
