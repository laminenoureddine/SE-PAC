from os.path import isfile, exists
from os import makedirs, remove
from time import time, strftime
from math import isinf, isnan, factorial
import global_vars
import numpy as np

def log(message, print_time=True):
    if print_time:
        message = strftime('%Y-%m-%d %H:%M:%S') + ': ' + message
    print message
    with open(global_vars.logfile, 'a') as logoutfile:
        logoutfile.write(message + '\n')


def drop_traling_zeros(num):
  if num % 1 == 0:
    return int(num)
  else:
    return num


def parse_special_line(line):
    # assumes line is a string and starts with #. if it is a known parameter parse it, otherwise it's a comment and
    # we can ignore it
    if line.startswith('#Short_feature_name'):
        global_vars.feature_names_short = map(str.rstrip, line.split(',')[1:])
    elif line.startswith('#Long_feature_name'):
        global_vars.feature_names = map(str.rstrip, line.split(',')[1:])
    elif line.startswith('#Features_category'):
        global_vars.features_categories = map(str.rstrip, line.split(',')[1:])


"""
def get_feat_names():
    if use_short_feat_names:
        return global_vars.feature_names_short
    else:
        return global_vars.feature_names
"""


def assign_log_file_name(clustering_mode, scenario_mode, min_samples_per_family, requested_cats):
    if clustering_mode == "consensus_only":
        global_vars.dumped_files_suffix = global_vars.dumped_files_suffix + "_consensus"
    else:
        global_vars.dumped_files_suffix = global_vars.dumped_files_suffix + "_noconsensus"
    if min_samples_per_family > 0:
        global_vars.dumped_files_suffix = global_vars.dumped_files_suffix + "_min" + str(min_samples_per_family)
    if requested_cats is not None:
        global_vars.dumped_files_suffix = global_vars.dumped_files_suffix + '_' + '+'.join(sorted(requested_cats))

    logdir = 'logs/'
    if not exists(logdir):
        makedirs(logdir)

    global_vars.logfile = logdir + 'log' + '_' + str(scenario_mode) + global_vars.dumped_files_suffix + '.txt'

    if isfile(global_vars.logfile):
        remove(global_vars.logfile)


def remove_unpacked_files():
    if 'unpacked' in global_vars.family_names:
        global_vars.family_names.remove('unpacked')


def load_feature_file(features_file):
    with open(features_file) as my_file:
        for l in my_file:
            try:
                if l.startswith('#'):
                    parse_special_line(l)
                    continue

                if "Invalid PE File" in l:
                    continue
                data = l.split(',')
                if len(data) != len(global_vars.feature_names) + 1:
                    log('Found feature line with wrong number of features (' + str(
                        len(data) - 1) + ' instead of ' + str(len(global_vars.feature_names)) + '), will be ignored')
                    continue
                if True in map(isinf, map(float, data[1:])):
                    log('WARNING: Infinity in features of file ' + str(data[0]) + ' within ' + str(
                        my_file.name) + ', will be ignored')
                    continue
                if True in map(isnan, map(float, data[1:])):
                    log('WARNING: NaN in features of file ' + str(data[0]) + ' within ' + str(
                        my_file.name) + ', will be ignored')
                    continue
                yield map(float, data[1:])
                (global_vars.binaries_names).append(data[0:1][0])
            except AssertionError:
                log("Assertion Error with line " + str(l.split(',')))
            except ValueError:
                log("Value Error with line " + str(l.split(',')))


def load_features_dir(clustering_mode, feat_dir):
    for file in global_vars.file_names:
        filename = feat_dir + 'consensus_' + file
        if not isfile(filename):
            log(filename + ' not found; will be ignored.')
        else:
            loaded_features_array = np.array(list(load_feature_file(filename)))
            for featline in loaded_features_array:
                if True in map(isinf, featline):
                    log('ERROR: Infinity in features of file ' + filename)
                if True in map(isnan, featline):
                    log('ERROR: NaN in features of file' + filename)
            log('Loaded ' + str(len(loaded_features_array)) + ' features from file ' + filename)

            fam = file
            if fam in global_vars.features_dict:
                global_vars.features_dict[fam] = np.concatenate((global_vars.features_dict[fam], loaded_features_array))
            else:
                global_vars.features_dict[fam] = loaded_features_array

        if clustering_mode == "no_consensus_only":
            non_cons_filename = feat_dir + 'no_consensus_' + file  # + '.features.txt'
            if not isfile(non_cons_filename):
                log(non_cons_filename + ' not found; will be ignored.')
            else:
                loaded_features_array = np.array(list(load_feature_file(non_cons_filename)))
                log('Loaded ' + str(len(loaded_features_array)) + ' features from file ' + non_cons_filename)

                fam = file
                if fam in global_vars.features_dict:
                    global_vars.features_dict[fam] = np.concatenate((global_vars.features_dict[fam], loaded_features_array))
                else:
                    global_vars.features_dict[fam] = loaded_features_array


def load_mnemonics_feature_file(features_file):
    data = []
    with open(features_file) as my_file:
        for l in my_file:
            l = l.replace("'", "")
            l = l.replace("[", "")
            l = l.replace("]", "")
            l = l.strip("\n")
            l = l.split(',')
            l = [x.strip(' ') for x in l]
            data += [l]
    """
        for l in my_file:
            try:
                if l.startswith('#'):
                    parse_special_line(l)
                    continue

                if "Invalid PE File" in l:
                    continue
                data = l.split(',')
                if len(data) != len(global_vars.feature_names) + 1:
                    log('Found feature line with wrong number of features (' + str(
                        len(data) - 1) + ' instead of ' + str(len(global_vars.feature_names)) + '), will be ignored')
                    continue
                if True in map(isinf, map(float, data[1:])):
                    log('WARNING: Infinity in features of file ' + str(data[0]) + ' within ' + str(
                        my_file.name) + ', will be ignored')
                    continue
                if True in map(isnan, map(float, data[1:])):
                    log('WARNING: NaN in features of file ' + str(data[0]) + ' within ' + str(
                        my_file.name) + ', will be ignored')
                    continue
                yield map(float, data[1:])
                (global_vars.binaries_names).append(data[0:1][0])
            except AssertionError:
                log("Assertion Error with line " + str(l.split(',')))
            except ValueError:
                log("Value Error with line " + str(l.split(',')))     
        """
    return data

def load_mnemonics_features_dir(clustering_mode, feat_dir):
    for file in global_vars.file_names:
        filename = feat_dir + 'consensus_' + file
        if not isfile(filename):
            log(filename + ' not found; will be ignored.')
        else:
            fam = file
            if fam in global_vars.features_dict:
                global_vars.features_dict[fam] += load_mnemonics_feature_file(filename)
            else:
                global_vars.features_dict[fam] = load_mnemonics_feature_file(filename)

        if clustering_mode == "no_consensus_only":
            non_cons_filename = feat_dir + 'no_consensus_' + file  # + '.features.txt'
            if not isfile(non_cons_filename):
                log(non_cons_filename + ' not found; will be ignored.')
            else:
                log('Loaded ' + str(len(load_mnemonics_feature_file(non_cons_filename))) + ' features from file ' + non_cons_filename)

                fam = file
                if fam in global_vars.features_dict:
                    global_vars.features_dict[fam] += load_mnemonics_feature_file(non_cons_filename)
                else:
                    global_vars.features_dict[fam] = load_mnemonics_feature_file(non_cons_filename)

def fam_to_remove(min_samples_per_family):
    toremove = []
    log('Loaded source files:')
    for fam in global_vars.family_names:
        if fam in global_vars.features_dict:
            num_files = len(global_vars.features_dict[fam])
            if num_files < min_samples_per_family:
                log(fam + ' : ' + str(len(global_vars.features_dict[fam])) + ' (will not be used: less than ' + str(min_samples_per_family) + ' samples)')
                toremove.append(fam)
            else:
                log(fam + ' : ' + str(len(global_vars.features_dict[fam])))
        else:
            log(fam + ' : 0 (will not be used: no features found)')
            toremove.append(fam)

    for missingfam in toremove:
        global_vars.family_names.remove(missingfam)
        global_vars.features_dict.pop(missingfam, None)

    log('Families that will be used: ' + str(global_vars.family_names))


def assign_labels_and_features():
    first_concatenation = True  # to avoid some problems of np.empty, ...
    for i, fam in enumerate(global_vars.family_names):
        global_vars.labels += [fam] * len(global_vars.features_dict[fam])
        if (first_concatenation):
            global_vars.features = global_vars.features_dict[fam]
            first_concatenation = False
        else:
            global_vars.features = np.concatenate((global_vars.features, global_vars.features_dict[fam]))

def assign_mnemonics_labels_and_features():
    for i, fam in enumerate(global_vars.family_names):
        global_vars.labels += [fam] * len(global_vars.features_dict[fam])
        global_vars.features += global_vars.features_dict[fam]


def is_features_empty():
    if len(global_vars.features) == 0:
        log('ERROR: No features loaded. Aborting.')
        return 1
    else:
        return 0


def len_features_wrong():
    if len(global_vars.features[0]) != len(global_vars.feature_names):
        log('ERROR: number of features and of feature names do not match. Aborting.')
        return 1
    else:
        return 0


def delete_unrequested_feat_cat(requested_cats):
    if requested_cats is None:
        log('All feature categories will be used.')
    else:
        features_categories_uniq = set(global_vars.features_categories)
        for cat in requested_cats:
            # Check whether the category exists, if not exit
            if cat not in features_categories_uniq:
                log("ERROR: feature category " + cat + " does not exist; available feature categories are " + str(features_categories_uniq))
                exit(1)
        cats_to_remove = list(features_categories_uniq - set(requested_cats))
        to_delete = []
        for cat in cats_to_remove:
            to_delete.extend([i for i, j in enumerate(global_vars.features_categories) if j == cat])

        global_vars.features = np.delete(global_vars.features, to_delete, axis=1)
        global_vars.feature_names = np.delete(np.array(global_vars.feature_names), to_delete, axis=0)
        global_vars.feature_names_short = np.delete(np.array(global_vars.feature_names_short), to_delete, axis=0)

        if len(global_vars.feature_types) != 0:
            global_vars.feature_types = np.delete(np.array(global_vars.feature_types), to_delete, axis=0)
        if len(global_vars.feature_types_range) != 0:
            global_vars.feature_types_range = np.delete(np.array(global_vars.feature_types_range), to_delete, axis=0)
        if len(global_vars.features_desc_and_comments) != 0:
            global_vars.features_desc_and_comments = np.delete(np.array(global_vars.features_desc_and_comments), to_delete, axis=0)
        log('Feature categories used: ' + str(requested_cats))

"""
# this returns a ditionary havign as keys the unique elements in features_categories and as values the list of results_with_synthetic_centroids of
# testing the classifier against the given score if the corresponding category is not considered.
def get_score_losses(classifier, crossvalidator, my_features, my_labels, score_type, features_categories, save_classifiers, name):
    response = {}
    unique_categories = set(features_categories)  # set is a trick to make the list unique, it loses the order though
    num_of_categories = len(unique_categories)

    for size in range(0, num_of_categories, 1):
        for cats in combinations(unique_categories, size):
            features_temp = my_features[:]
            labels_temp = my_labels[:]
            features_temp = delete_categories(features_temp, cats, features_categories)
            start_time = time()
            classifier.fit(features_temp, labels_temp)
            end_training_time = time() - start_time
            if save_classifiers:
                used_cats = unique_categories - set(cats)
                class_directory = 'trained_classifiers/'
                if not exists(class_directory):
                    makedirs(class_directory)
                outfile_name = name + '_' + '+'.join(used_cats)
                classifier._categories = used_cats
                joblib.dump(classifier, class_directory + outfile_name + '.pkl')
                log('Classifier saved in ' + class_directory + outfile_name + '.pkl')
            cross_val_values = cross_val_score(classifier, features_temp, labels_temp, cv=crossvalidator, verbose=0,
                                               scoring=score_type).tolist()
            cross_val_time = time() - (end_training_time + start_time)
            response[str(cats)] = dict(values=cross_val_values, training_time=end_training_time,
                                       cross_val_time=cross_val_time
                                       )
    return response
"""
