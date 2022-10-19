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
from sklearn.preprocessing import minmax_scale, StandardScaler, RobustScaler, PowerTransformer, MinMaxScaler,  Normalizer
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

    # getting features importance
    Et_n_estimators = 18
    Et_criterion = 'gini'
    Et_max_depth = 29
    Et_class_weight = 'balanced'
    clf = ExtraTreesClassifier(n_estimators=Et_n_estimators, criterion=Et_criterion, max_depth=Et_max_depth, class_weight=Et_class_weight)
    clf.fit(global_vars.features, global_vars.labels)
    feature_importances = np.array(clf.feature_importances_)
    feature_importance_desc_rank = np.argsort(-feature_importances).tolist()

    # getting amount of samples in each family: need it later
    unique_fam_amout = OrderedDict()
    for elt in global_vars.labels:
        try:
            unique_fam_amout[elt] += 1
        except KeyError:
            unique_fam_amout[elt] = 1

    #global_vars.features = StandardScaler().fit_transform(global_vars.features)
    #global_vars.features = RobustScaler(quantile_range=(25, 75)).fit_transform(global_vars.features)
    #global_vars.features = MinMaxScaler(copy=True, feature_range=(-1, 1)).fit_transform(global_vars.features)
    # global_vars.features =  Normalizer().fit_transform(global_vars.features)
    # print global_vars.features

    # main loop for doing the stuff of statistics for each family...
    first_fam_index = 0
    to_format = {}
    for item in unique_fam_amout.items():
        last_fam_index = first_fam_index + item[1]
        features_fam_array = np.copy(global_vars.features[first_fam_index:last_fam_index])
        features_fam_table = PrettyTable()
        features_fam_table.field_names = ["feature_number", "feature_short_name", "feature_importance_rank", "min",
                                          "minor_outlier", "lowerQ", "median", "average", "upperQ", "major_outlier",
                                          "max"]

        preprocessing_functions.log("\n\n------------------------------")
        preprocessing_functions.log("Statistics about " + item[0] + " family: \n")
        for i in range(0, len(features_fam_array[0])):
            features_fam_column = [row[i] for row in features_fam_array]
            features_fam_column.sort()
            min = features_fam_column[0]
            max = features_fam_column[-1]
            features_actual_range = '[' + str(min) + ',' + str(max) + ']'
            avg = sum(features_fam_column) / float(len(features_fam_column))
            median = np.percentile(np.array(features_fam_column), 50)
            lowerQ = np.percentile(np.array(features_fam_column), 25)
            upperQ = np.percentile(np.array(features_fam_column), 75)
            interQ = upperQ - lowerQ
            major_outlier = upperQ + interQ * (1.5)
            minor_outlier = lowerQ - interQ * (1.5)
            no_major_outlier = False
            no_minor_outlier = False
            if major_outlier > max:
                no_major_outlier = True
            if minor_outlier < min:
                no_minor_outlier = True

            # feature_importances_i = feature_importances[i]
            feature_names_short_i = global_vars.feature_names_short[i]
            feature_importance_desc_rank_i = feature_importance_desc_rank[i]

            # formatting strings for printing: drop trailing zeros and limit the number of digits after comma
            min = format(min, format_string)
            to_format["min"] = preprocessing_functions.drop_traling_zeros(float(min))
            max = format(max, format_string)
            to_format["max"] = preprocessing_functions.drop_traling_zeros(float(max))
            avg = format(avg, format_string)
            to_format["avg"] = preprocessing_functions.drop_traling_zeros(float(avg))
            median = format(median, format_string)
            to_format["median"] = preprocessing_functions.drop_traling_zeros(float(median))
            lowerQ = format(lowerQ, format_string)
            to_format["lowerQ"] = preprocessing_functions.drop_traling_zeros(float(lowerQ))
            upperQ = format(upperQ, format_string)
            to_format["upperQ"] = preprocessing_functions.drop_traling_zeros(float(upperQ))

            if no_minor_outlier == True:
                to_format["minor_outlier"] = "no_minor_outlier"
            else:
                minor_outlier = format(minor_outlier, format_string)
                to_format["minor_outlier"] = preprocessing_functions.drop_traling_zeros(float(minor_outlier))
            if no_major_outlier == True:
                to_format["major_outlier"] = "no_major_outlier"
            else:
                major_outlier = format(major_outlier, format_string)
                to_format["major_outlier"] = preprocessing_functions.drop_traling_zeros(float(major_outlier))

            row = [(i + 1), feature_names_short_i, feature_importance_desc_rank_i, to_format["min"],
                   to_format["minor_outlier"], to_format["lowerQ"], to_format["median"],
                   to_format["avg"], to_format["upperQ"], to_format["major_outlier"], to_format["max"]]

            features_fam_table.add_row(row)
        first_fam_index += item[1]
        preprocessing_functions.log('\n' + str(features_fam_table))

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
