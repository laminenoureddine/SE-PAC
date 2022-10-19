import pandas as pd
import io
from sklearn.cluster import DBSCAN
import numpy as np
from collections import Counter
import gower_distance
import copy
import global_vars
import math
from scipy.spatial.distance import cdist
from scipy.spatial import distance
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from itertools import count

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

def GreedyFarthestPoint(pairwise_distance, k, cluster_indexes):
    # Initialize the variables
    two_first_farthest_points_indexes = []
    solution_set_indexes = []
    remaining_points_indexes = cluster_indexes

    # Calculate the number of SRPs
    num_SRPs = int(math.ceil(k*(math.sqrt(len(cluster_indexes)))))

    # Check if the size of the matrix is not lower than the value of k
    if pairwise_distance.shape[0] < num_SRPs:
        return cluster_indexes

    if len(cluster_indexes) <= num_SRPs:
        return cluster_indexes
    else:
        # Take the two farthest points
        cluster_partial_pairwise_distance = pairwise_distance[np.ix_(cluster_indexes, cluster_indexes)]
        two_first_farthest_points_indexes_in_partial_matrix = np.unravel_index(
            np.argmax(cluster_partial_pairwise_distance, axis=None), cluster_partial_pairwise_distance.shape)
        two_first_farthest_points_indexes.append(
            cluster_indexes[two_first_farthest_points_indexes_in_partial_matrix[0]])
        two_first_farthest_points_indexes.append(
            cluster_indexes[two_first_farthest_points_indexes_in_partial_matrix[1]])

        solution_set_indexes.append(two_first_farthest_points_indexes[0])
        solution_set_indexes.append(two_first_farthest_points_indexes[1])

        if (num_SRPs == 2):
            return solution_set_indexes

        remaining_points_indexes.remove(solution_set_indexes[0])
        try:
            remaining_points_indexes.remove(solution_set_indexes[1])
        except:
            return [solution_set_indexes[0]]

        # Initialize the vector of sum_distances, and add the third element to solution, with removing it from remining points and sum_distances
        sum_distances = []
        sum_d = 0

        for i in remaining_points_indexes:
            for j in solution_set_indexes:
                sum_d += pairwise_distance[i][j]
            sum_distances.append(sum_d)
            sum_d = 0

        best_index = sum_distances.index(max(sum_distances))
        solution_set_indexes.append(remaining_points_indexes[best_index])
        remaining_points_indexes.pop(best_index)
        sum_distances.pop(best_index)

        # Add to the solution set, the remaining points for which the sum to them is higher/greater than all other remaining points

        for s in range(4, num_SRPs + 1):
            for i, r in enumerate(remaining_points_indexes):
                sum_distances[i] += pairwise_distance[solution_set_indexes[-1]][r]

            best_index = sum_distances.index(max(sum_distances))
            solution_set_indexes.append(remaining_points_indexes[best_index])
            remaining_points_indexes.pop(best_index)
            sum_distances.pop(best_index)

    return solution_set_indexes


def euclidean_distance(self, element_1, element_2):
    """
    This function calculates the euclidean distance
    """
    #:param element_1:  the current element that needs to be checked
    #:param element_2:  the element to check the euclidean_distance from
    #:returns euclidean_distance: the Euclidean euclidean_distance between the element_1 and the element_2(float)

    squares_sum = [(element_1['PE_feature_' + str(i + 1)] - element_2['PE_feature_' + str(i + 1)]) ** 2 for i in
                   range(len(global_vars.features[0]))]
    euclidean_distance = (sum(squares_sum)) ** (1 / 2)

    return euclidean_distance


def gowr_distance(element_1, element_2):
    """
    This function calculates the Gower distance function
    """
    element_1_feature_values = [element_1['PE_feature_' + str(i + 1)] for i in range(len(global_vars.features[0]))]
    element_2_feature_values = [element_2['PE_feature_' + str(i + 1)] for i in range(len(global_vars.features[0]))]

    input_array = np.array([element_1_feature_values, element_2_feature_values])
    gowr_distance = gower_distance.gower_distances(input_array)

    return gowr_distance[1][0]


def manhattan_distance(element_1, element_2):
    """
    This function calculates the manhattan distance function
    """
    element_1_feature_values = [element_1['PE_feature_' + str(i + 1)] for i in range(len(global_vars.features[0]))]
    element_2_feature_values = [element_2['PE_feature_' + str(i + 1)] for i in range(len(global_vars.features[0]))]

    element_1_feature_values_scaled = [min_max_scaler_scale[i] * element_1_feature_values[i] + 0 - min_max_scaler_min[i] * min_max_scaler_scale[i] for i in range(len(element_1_feature_values))]
    element_2_feature_values_scaled = [min_max_scaler_scale[i] * element_2_feature_values[i] + 0 - min_max_scaler_min[i] * min_max_scaler_scale[i] for i in range(len(element_2_feature_values))]

    manhattan_distance = \
        distance.cdist(np.array([element_1_feature_values_scaled]), np.array([element_2_feature_values_scaled]),
                       'cityblock')[0][0]
    # manhattan_distance=distance.cityblock(a,b)
    return manhattan_distance / len(global_vars.features[0])


def tapered_levenshtein_distance(element_1, element_2):
    """
    This function calculates the tappered levenshtein distance function
    """
    if element_1['Unpacking_code_sequence'] == ('',):
        element_1['Unpacking_code_sequence'] = ('')
    if element_2['Unpacking_code_sequence'] == ('',):
        element_2['Unpacking_code_sequence'] = ('')

    if (element_1['Unpacking_code_sequence'] == ('') and element_2['Unpacking_code_sequence'] == ('')):
        return -1

    max_len = float(max(len(element_1['Unpacking_code_sequence']), len(element_2['Unpacking_code_sequence'])))
    if len(element_1['Unpacking_code_sequence']) < len(element_2['Unpacking_code_sequence']):
        return tapered_levenshtein_distance(element_2, element_1)

    # len(element_1['Unpacking_code_sequence']) >= len(element_2['Unpacking_code_sequence'])
    if len(element_2['Unpacking_code_sequence']) == 0:
        return len(element_1['Unpacking_code_sequence']) / 50

    previous_row = range(len(element_2['Unpacking_code_sequence']) + 1)
    for i, c1 in enumerate(element_1['Unpacking_code_sequence']):
        current_row = [i + 1]
        for j, c2 in enumerate(element_2['Unpacking_code_sequence']):
            taper = 1.0 - min(i, j) / max_len
            insertions = previous_row[j + 1] + taper  # j+1 instead of j since previous_row and current_row are one character longer
            deletions = current_row[j] + taper  # than element_2['Unpacking_code_sequence']
            substitutions = previous_row[j] + (c1 != c2) * taper
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row

    return previous_row[-1] / 50


class IncrementalDBSCAN:

    def __init__(self, eps=0.06, min_samples=3):
        """
        Constructor the Incremental_DBSCAN class.
        :param eps:  the  maximum radius that an element should be in order to formulate a cluster
        :param min_samples:  the minimum samples required in order to formulate a cluster
        In order to identify the optimum eps and min_samples we need to  make a KNN
        """
        self.pe_features_columns = ['PE_feature_' + str(i + 1) for i in range(len(global_vars.features[0]))]
        self.unpacking_code_sequences_columns = ['Unpacking_code_sequence']
        self.all_features_columns = self.pe_features_columns + self.unpacking_code_sequences_columns
        self.dataset = pd.DataFrame(columns=self.all_features_columns)
        self.labels = pd.DataFrame(columns=['Label'])
        self.final_dataset = pd.DataFrame(columns=(self.all_features_columns + ['Label']))
        self.pe_features_mean_core_elements = pd.DataFrame(columns=(self.pe_features_columns + ['Label']))
        self.unpacking_code_sequence_mean_core_elements = pd.DataFrame(
            columns=self.unpacking_code_sequences_columns + ['Label'])
        self.unpacking_code_sequence_mean_core_indexes = []
        # self.max_core_elements = 0
        self.max_representative_core_elements = 0
        self.eps = eps
        self.min_samples = min_samples
        self.max_unpacking_code_sequence_lenght = 50
        self.largest_cluster = -1
        self.cluster_limits = 0
        self.largest_cluster_limits = 0
        self.distance_matrix = []
        self.last_modified_cluster_label = -1
        self.representative_core_points = {}
        self.number_clusters = 0
        self.local_server_image = ''
        self.scenario_config = 0
        self.num_months_test = 0
        self.core_samples_indexes = []

    def set_data(self, new_sample):
        """
        This function is used to gather the new packed sample. It appends the newly arrived data to the
        dataset used for clustering.
        :param new_sample:  The new packed sample in panda dataframe
        """
        self.dataset = self.dataset.append(new_sample, ignore_index=True)
        new_sample['Label'] = -1
        self.final_dataset = self.final_dataset.append(new_sample, ignore_index=True)

    def set_scenario_config(self, scenario):
        self.scenario_config = scenario

    def set_local_server_image(self, local_server_image):
        self.local_server_image = local_server_image

    def set_num_months_test(self, num_months_test):
        self.num_months_test = num_months_test

    def set_core_samples_indexes(self, core_sample_indexes):
        self.core_samples_indexes.extend(core_sample_indexes)

    """
    def set_max_core_elements(self, max_core_elements): 
       This function is used to set the max number of core samples
       :param max_core_elements:  The new max number of cores samples
        self.max_core_elements = max_core_elements
    """

    def set_SRPs_factor(self, k):
        """
        This function is used to set the max number of representative core samples
        :param: K: max number of representative core elements
        """
        self.SRPs_factor = k

    def set_eps(self, eps):
        """
       This function is used to set eps
       :param eps:  eps
       """
        self.eps = eps

    def set_min_samples(self, min_samples):
        """
       This function is used to set eps
       :param min_samples:  min_samples
       """
        self.min_samples = min_samples

    def set_distance_matrix(self, distance_matrix):
        """
       This function is used to set the distance matrix
       :param distance_matrix:  The pairwise distance matrix
       """
        self.distance_matrix = distance_matrix

    def set_last_modified_cluster_label(self, label):
        """
        This function is used to set the representative core points of each cluster
       :param label: label of the last modified cluster
       """
        self.last_modified_cluster_label = label

    def set_representative_core_points(self, label, list_of_indexes):
        """
        This function is used to set the representative core points of each cluster
       :param label: label of the cluster
       :param list_of_indexes: list of representative indexes in the pairwise distance matrix
       """
        self.representative_core_points[label] = list_of_indexes

    def set_number_clusters(self, number_clusters):
        self.number_clusters = number_clusters

    def calculate_or_update_representative_core_points(self, last_modified_cluster_label, all, k):
        if all == True:
            unique_labels = set(self.final_dataset['Label'])
            try:
                unique_labels.remove(-1)
            except:
                pass

            for label in unique_labels:
                cluster_indexes = self.final_dataset.index[self.final_dataset['Label'] == label].tolist()
                cluster_representative_indexes = GreedyFarthestPoint(pairwise_distance=self.distance_matrix, k=k, cluster_indexes=cluster_indexes)
                self.set_representative_core_points(label, cluster_representative_indexes)
        else:
            last_modified_cluster_indexes = self.final_dataset.index[self.final_dataset['Label'] == last_modified_cluster_label].tolist()
            last_modified_cluster_representative_indexes = GreedyFarthestPoint(pairwise_distance=self.distance_matrix, k=k, cluster_indexes=last_modified_cluster_indexes)
            self.set_representative_core_points(last_modified_cluster_label, last_modified_cluster_representative_indexes)

    def batch_dbscan(self, distance_matrix):
        """
        The DBSCAN algorithm taken from the sklearn library. It is used to formulate the clusters the first time.
        Based on the outcomes of this algorithm the Incremental_DBSCAN algorithm
        """
        # batch_dbscan = DBSCAN(eps=self.eps, min_samples=self.min_samples).fit(self.dataset)

        batch_dbscan = DBSCAN(metric='precomputed', eps=self.eps, min_samples=self.min_samples).fit(distance_matrix)
        # Get the number of the clusters created

        self.add_labels_to_dataset(batch_dbscan.labels_)
        n_clusters = len(set(batch_dbscan.labels_)) - (1 if -1 in batch_dbscan.labels_ else 0)
        self.set_number_clusters(n_clusters)
        self.set_core_samples_indexes(list(batch_dbscan.core_sample_indices_))

        # Cast everything in the final_dataset as integer.
        # If this line is missing, it throws an error
        # TODO: Be careful, I commented the line below
        # self.final_dataset = self.final_dataset.astype(int)
        # self.incremental_dbscan_()
        # self.sort_dataset_based_on_labels()
        # self.find_pe_features_mean_core_elements()
        # response = self.calculate_min_distance_centroid()
        # # print(response)
        # if response is not None:
        #     self.check_min_samples_in_eps_or_outlier(min_dist_index=response)
        # self.largest_cluster = self.find_largest_cluster()

    def add_labels_to_dataset(self, labels):
        """
        This function adds the labels on the dataset after the batch DBSCAN is done
        :param labels: The labels param should be a list that  describes the cluster of each element.
        If an element is considered as an outlier it should be equal to -1
        """
        self.labels = pd.DataFrame(labels, columns=['Label'])
        self.final_dataset = pd.concat([self.dataset, self.labels], axis=1)

    def update_labels(self, final_dataset):
        """
        This function update the labels attribute after the incremental DBSCAN is done
        :param final_dataset: The final_dataset param on which labels have been set.
        """
        self.labels = final_dataset['Label']

    def sort_dataset_based_on_labels(self):
        """
        This function sorts the dataset based on the Label of each cluster.
        """
        # print(self.final_dataset)
        self.final_dataset = self.final_dataset.sort_values(by=['Label'])
        # Cast everything in the final_dataset as integer.
        # If this line is missing, it throws an error
        self.final_dataset = self.final_dataset.astype(int)

    """
    def find_pe_features_mean_core_elements(self, mean_core_indexes):

        This function calculates the average core elements of each cluster, for the synthetic PE-extracted features
        Note: It does not calculate an average core element for the outliers.

        # Exclude rows labeled as outliers
        self.pe_features_mean_core_elements = self.final_dataset.loc[self.final_dataset['Label'] != -1]
        self.pe_features_mean_core_elements = self.pe_features_mean_core_elements.reset_index(drop=True)
        temp_pe_features_mean_core_elements = pd.DataFrame(columns=(self.pe_features_columns + ['Label']))

        for i in range(len(mean_core_indexes)):
            temp = self.pe_features_mean_core_elements.iloc[mean_core_indexes[i][1], :].groupby('Label')[
                self.pe_features_columns].mean()
            temp = pd.DataFrame(temp.reset_index())
            temp = temp[self.pe_features_columns + ['Label']]
            temp_pe_features_mean_core_elements = temp_pe_features_mean_core_elements.append(temp, ignore_index=True)

        self.pe_features_mean_core_elements = temp_pe_features_mean_core_elements
        print(self.pe_features_mean_core_elements)

        # Find the mean core elements of each cluster
        # self.pe_features_mean_core_elements = self.pe_features_mean_core_elements.groupby('Label')[self.pe_features_columns].mean()
        # response = self.calculate_min_distance_centroid()
        # # print(response)
        # if response is not None:
        #     self.check_min_samples_in_eps_or_outlier(min_dist_index=response)
    """

    # No longer used
    """
    def fill_with_Nones(self, list_input):
        # This function fills temporarely a list with Nones
        list_temp = list_input.copy()
        for i in range(len(list_temp), self.max_unpacking_code_sequence_lenght):
            list_temp.append(None)
        return list_temp

    def remove_Nones(self, list_input):
        # This function removes temporarely Nones from a list
        temp_list = list_input.copy()
        return [x for x in temp_list if x is not None]
    """

    def extract_row(self, pd):
        for index, row_element in pd.iterrows():
            new_element = row_element
        return new_element

    """"
    def find_unpacking_code_sequence_mean_core_elements(self, max_core_elements):

        This function calculates the average core elements of each cluster, for the unpacking code sequences
        Note: It does not calculate an average core element for the outliers.

        # Exclude rows labeled as outliers
        self.unpacking_code_sequence_mean_core_elements = self.final_dataset.loc[self.final_dataset['Label'] != -1]
        self.unpacking_code_sequence_mean_core_elements = self.unpacking_code_sequence_mean_core_elements.reset_index(
            drop=True)

        # Find the mean core elements of each cluster
        self.unpacking_code_sequence_mean_core_elements.groupby('Label')

        temp_unpacking_code_sequence_mean_core_elements = pd.DataFrame(columns=['Unpacking_code_sequence', 'Label'])

        unique_labels = set(self.unpacking_code_sequence_mean_core_elements['Label'])

        # Calculates agressively representative synthetic core points for unpacking code sequences
        for label in unique_labels:
            temp = self.unpacking_code_sequence_mean_core_elements.loc[
                self.unpacking_code_sequence_mean_core_elements['Label'].isin([label])].copy()
            temp['Unpacking_code_sequence'] = temp['Unpacking_code_sequence'].apply(tuple)
            temp2 = temp.groupby('Unpacking_code_sequence').size().sort_values(ascending=False).head(
                max_core_elements).reset_index()['Unpacking_code_sequence']
            [self.unpacking_code_sequence_mean_core_indexes.append(
                (label, list(temp[temp['Unpacking_code_sequence'] == temp2[i]].index))) for i in range(len(temp2))]
            d = pd.DataFrame()
            d['Unpacking_code_sequence'] = [list(x) for x in temp2.values]
            d['Label'] = label
            temp_unpacking_code_sequence_mean_core_elements = temp_unpacking_code_sequence_mean_core_elements.append(d,
                                                                                                                     ignore_index=True)

        self.unpacking_code_senquence_mean_core_elements = temp_unpacking_code_sequence_mean_core_elements

        print(self.unpacking_code_senquence_mean_core_elements)

        # print(self.mean_core_elements)
        # response = self.calculate_min_distance_centroid()
        # # print(response)
        # if response is not None:
        #     self.check_min_samples_in_eps_or_outlier(min_dist_index=response)
    """
    """
    def calculate_min_distance_centroid(self):

        This function identifies the closest mean_core_element to the incoming element
        that has not yet been added to a cluster or considered as outlier.
        An average between Tappered Levenshtein and euclidean_distance is calculated using the both distances as it is described above.

        :returns min_dist_index: if there is a cluster that is closest to the new entry element
        or None if there are no clusters yet.

        min_dist = None
        min_dist_index = None

        new_element = self.extract_row(self.final_dataset.tail(1))

        # Check if there are elements in the core_elements dataframe.
        # In other words if there are clusters created by the DBSCAN algorithm
        min_dist = None
        min_dist_index = None
        tmp_dist = []

        new_element = self.extract_row(self.final_dataset.tail(1))

        # Check if there are elements in the core_elements dataframe.
        # In other words if there are clusters created by the DBSCAN algorithm
        if not self.pe_features_mean_core_elements.empty and not self.unpacking_code_sequence_mean_core_elements.empty:
            # Iterate over the pe_features_mean_core_elements dataframe and find the minimum euclidean_distance
            [tmp_dist.append(((manhattan_distance(element_1=new_element,
                                                  element_2=current_pe_feature_mean_core_element) + tapered_levenshtein_distance(
                element_1=new_element, element_2=current_unpacking_code_sequence_mean_core_element)) / 2,
                              current_pe_feature_mean_core_element['Label'])) for
             (index, current_pe_feature_mean_core_element), (index2, current_unpacking_code_sequence_mean_core_element)
             in zip(self.pe_features_mean_core_elements.iterrows(),
                    self.unpacking_code_sequence_mean_core_elements.iterrows())]
            if min_dist is None:
                min_dist = min(tmp_dist)[0]
                min_dist_index = min(tmp_dist)[1]
            elif tmp_dist < min_dist:
                min_dist = min(tmp_dist)[0]
                min_dist_index = min(tmp_dist)[1]

            print('Minimum distance is: ', min_dist, ' at cluster ', min_dist_index)
            return min_dist_index
        else:
            return None
    """

    def calculate_min_distance_cluster(self):

        min_dist = None
        min_dist_index = None
        tmp_dist = []

        new_element = self.extract_row(self.final_dataset.tail(1))

        # Check if there are elements in the representative core_elements dataframe.
        # In other words if there are clusters created by the DBSCAN algorithm
        if not self.representative_core_points == {}:
            # Iterate over the representative core points
            #for cluster_index in self.representative_core_points.keys():
            #    for representative_index in self.representative_core_points[cluster_index]:
                    # representative_point = self.final_dataset.iloc[representative_index]
                    #tmp_dist = (manhattan_distance(element_1=new_element,element_2=representative_point) + tapered_levenshtein_distance(element_1=new_element, element_2=representative_point)) / 2
                    #tmp_dist = self.distance_matrix[representative_index][-1]
                    #tmp_dist.append((self.distance_matrix[representative_index][-1], cluster_index, representative_index))
                    #if min_dist is None:
                    #    min_dist = tmp_dist
                    #    min_dist_index = cluster_index
                    #elif tmp_dist < min_dist:
                    #    min_dist = tmp_dist
                    #    min_dist_index = cluster_index

            temp = [(tapered_levenshtein_distance(element_1=new_element, element_2=self.final_dataset.iloc[representative_index]), manhattan_distance(element_1=new_element, element_2=self.final_dataset.iloc[representative_index]),cluster_index, representative_index) for cluster_index, representative_indexes in self.representative_core_points.items() for representative_index in representative_indexes]
            tmp_dist = [((x[0] + x[1])/float(2), x[2], x[3]) if x[0] != -1 else (len(global_vars.features[0])*x[1], x[2], x[3]) for x in temp ]
            # tmp_dist = [(self.distance_matrix[representative_index][-1], cluster_index, representative_index) for cluster_index, representative_indexes in self.representative_core_points.items() for representative_index in representative_indexes]
            tmp_dist.sort(key=lambda tmp_dist: tmp_dist[0])
            min_dist = tmp_dist[0][0]
            cluster_min_dist_index = tmp_dist[0][1]
            SRP_min_dist_index = tmp_dist[0][2]
            print('Minimum distance is: ', min_dist, ' at cluster ', cluster_min_dist_index, ' at SRP index ', SRP_min_dist_index)
            return min_dist, cluster_min_dist_index, SRP_min_dist_index
        else:
            return None

    def check_min_samples_in_eps_or_outlier(self, min_dist, cluster_min_dist_index, SRP_min_dist_index):
        """
        This function checks whether there are at least min_samples in the given radius from the new
        entry element.
        If there are at least min_samples this element will be added to the cluster and the
        mean_core_element of the current cluster has to be re-calculated.
        If not, there are two options.
            1. Check if there are at least min_samples  outliers in the given radius in order to create a new
                cluster, or
            2.  Consider it as a new outlier

        :param min_dist_index: This is the parameter that contains information related to the closest
        mean_core_element to the current element.
        """

        # Use only the elements of the closest cluster from the new entry element
        nearest_cluster_elements = self.final_dataset[self.final_dataset['Label'] == cluster_min_dist_index]
        """
        nearest_cluster_indexes_with_new_element = []
        nearest_cluster_indexes_with_new_element = list(nearest_cluster_elements.index)
        nearest_cluster_indexes_with_new_element.append(-1)
        nearest_cluster_with_new_element_pairwise_distance = self.distance_matrix[np.ix_(nearest_cluster_indexes_with_new_element, nearest_cluster_indexes_with_new_element)]
        nearest_cluster_with_new_element_pairwise_distance = nearest_cluster_with_new_element_pairwise_distance[:, -1:]
        nearest_cluster_with_new_element_pairwise_distance = nearest_cluster_with_new_element_pairwise_distance[:-1, ]
        min_samples_count = len(nearest_cluster_with_new_element_pairwise_distance[nearest_cluster_with_new_element_pairwise_distance <= self.eps]) + 1
        new_element = self.extract_row(self.final_dataset.tail(1))
        """

        new_element = self.extract_row(self.final_dataset.tail(1))
        min_samples_count = 1

        """
        for index, cluster_element in nearest_cluster_elements.iterrows():
            if ((manhattan_distance(element_1=new_element, element_2=cluster_element) + tapered_levenshtein_distance(element_1=new_element, element_2=cluster_element)) / 2) <= self.eps:
                min_samples_count += 1
                if min_samples_count >= self.min_samples:
                    break;
        """
        prediction_done = False
        rejected_points_indexes = []
        nearest_cluster_indexes = [index for index,cluster_element in nearest_cluster_elements.iterrows()]
        remaining_points_indexes = nearest_cluster_indexes
        size_nearest_cluster = len(nearest_cluster_indexes)

        if min_dist <= self.eps:
            #Case 1:
            accepted_points = [d for i,d in enumerate(self.distance_matrix[SRP_min_dist_index]) if ((d <= self.eps - min_dist) and (i in nearest_cluster_indexes))]
            if len(accepted_points) >= self.min_samples: #normally it is self.min_samples - 1, but do not forget that the reis a self-distance 0 in tis row
                #Joins the cluster and update the representative core points
                self.final_dataset.loc[self.final_dataset.index[-1], 'Label'] = cluster_min_dist_index
                self.set_last_modified_cluster_label(cluster_min_dist_index)
                #Update if and only if the number of SRPs of the concerned cluster has changed
                if len(self.representative_core_points[self.last_modified_cluster_label]) <  int(math.ceil(self.SRPs_factor*(math.sqrt(size_nearest_cluster+1)))):
                    self.calculate_or_update_representative_core_points(last_modified_cluster_label=self.last_modified_cluster_label, all=False,k=self.SRPs_factor)
                self.set_core_samples_indexes([len(self.distance_matrix) - 1])
                prediction_done = True

            #Case 2:
            else:
                rejected_points_indexes = [i for i,d in enumerate(self.distance_matrix[SRP_min_dist_index]) if ((d > self.eps + min_dist) and (i in nearest_cluster_indexes))]
                remaining_points_indexes =  [r for r in remaining_points_indexes if r not in rejected_points_indexes]

        #Case 3 and 4 together
        else:
            rejected_points_indexes = [i for i, d in enumerate(self.distance_matrix[SRP_min_dist_index]) if (((d < min_dist - self.eps) or (d > min_dist + self.eps)) and (i in nearest_cluster_indexes))]
            remaining_points_indexes = [r for r in remaining_points_indexes if r not in rejected_points_indexes]

        #Do not have choice, forced to continue to compute distances with other
        if prediction_done == False:
            remaining_points_elements = nearest_cluster_elements.loc[remaining_points_indexes]
            for index, cluster_element in remaining_points_elements.iterrows():
                tap_lev_dis = tapered_levenshtein_distance(element_1=new_element, element_2=cluster_element)
                manh_dis = manhattan_distance(element_1=new_element, element_2=cluster_element)
                if tap_lev_dis == -1:
                    if (len(global_vars.features[0]) * manh_dis) <= self.eps:
                        min_samples_count += 1

                else:
                    if ((manh_dis + tap_lev_dis) / 2) <= self.eps:
                        min_samples_count += 1

                if min_samples_count >= self.min_samples:
                    # Joins the cluster and update (if necessary) the representative core points
                    self.final_dataset.loc[self.final_dataset.index[-1], 'Label'] = cluster_min_dist_index
                    self.set_last_modified_cluster_label(cluster_min_dist_index)
                    # Update if and only if the number of SRPs of the concerned cluster has changed
                    if len(self.representative_core_points[self.last_modified_cluster_label]) < int(math.ceil(self.SRPs_factor * (math.sqrt(size_nearest_cluster+1)))):
                        self.calculate_or_update_representative_core_points(last_modified_cluster_label=self.last_modified_cluster_label, all=False, k=self.SRPs_factor)
                    self.set_core_samples_indexes([len(self.distance_matrix) - 1])
                    prediction_done =True
                    break;


        """
        if min_samples_count >= self.min_samples:
            # The new element  has enough cluster labels in the eps range
            #  and is now considered as a new member of the cluster.
            #  The representative core elements of this cluster is re-calculated.
            self.final_dataset.loc[self.final_dataset.index[-1], 'Label'] = min_dist_index
            self.set_last_modified_cluster_label(min_dist_index)
        
        self.calculate_or_update_representative_core_points(last_modified_cluster_label=self.last_modified_cluster_label, all=False, k=self.SRPs_factor)
        # self.find_unpacking_code_sequence_mean_core_elements(self.max_core_elements)
        # self.find_pe_features_mean_core_elements(self.unpacking_code_sequence_mean_core_indexes)
        # self.unpacking_code_sequence_mean_core_indexes = []
        """

        if prediction_done==False:
            # The new element is not added to its closest cluster. Now we have to check
            # whether it is going to be considered an outlier or it will form a new cluster
            # with other outliers.

            outliers = self.final_dataset[self.final_dataset['Label'] == -1]
            """
            outliers_with_new_element = []
            outliers_indexes_with_new_element = list(outliers.index)
            outliers_with_new_element_pairwise_distance = self.distance_matrix[np.ix_(outliers_indexes_with_new_element, outliers_indexes_with_new_element)]
            outliers_with_new_element_pairwise_distance = outliers_with_new_element_pairwise_distance[:,-1:]
            outliers_with_new_element_pairwise_distance = outliers_with_new_element_pairwise_distance[:-1, ]
            min_outliers_count = len(outliers_with_new_element_pairwise_distance[outliers_with_new_element_pairwise_distance <= self.eps]) + 1
            """

            min_outliers_count = 0 # Right ! Because the new point itself is considered as noise at the begining, so when comparing, we are also comparing it against itself
            new_cluster_elements = pd.DataFrame(columns=['Index'])
            for index, outlier in outliers.iterrows():
                tap_lev_dis = tapered_levenshtein_distance(element_1=new_element, element_2=outlier)
                manh_dis = manhattan_distance(element_1=new_element, element_2=outlier)
                if tap_lev_dis == -1:
                    if (len(global_vars.features[0])*manh_dis) <= self.eps:
                        min_outliers_count += 1
                        new_cluster_elements = new_cluster_elements.append({"Index": index}, ignore_index=True)
                else:
                    if ((manh_dis + tap_lev_dis) / 2) <= self.eps:
                        min_outliers_count += 1
                        new_cluster_elements = new_cluster_elements.append({"Index": index}, ignore_index=True)

            if min_outliers_count >= self.min_samples:
                """
                outliers_indexes_to_update = np.argwhere(outliers_with_new_element_pairwise_distance <= self.eps)[:, :1]
                outliers_indexes_to_update = list(np.concatenate(outliers_indexes_to_update, axis=0))
                outliers_indexes_to_update.append(-1)
                outliers_indexes_to_update = [outliers_indexes_with_new_element[x] for x in outliers_indexes_to_update]
                """
                # The new element has enough outliers in its eps radius in order to form a new cluster.
                new_cluster_number = int(self.final_dataset['Label'].max()) + 1

                # Update the number of clusters found
                self.set_number_clusters(new_cluster_number)

                for new_cluster_element in new_cluster_elements.iterrows():
                    self.final_dataset.loc[self.final_dataset.index[int(new_cluster_element[1])], 'Label'] = new_cluster_number
                    self.set_core_samples_indexes([self.final_dataset.index[int(new_cluster_element[1])]])

                # self.final_dataset.loc[outliers_indexes_to_update, 'Label'] = new_cluster_number

                print("A new cluster is now formed out of already existing outliers.")

                # The new cluster's representative core elements are calculated after the cluster's creation.
                self.set_last_modified_cluster_label(new_cluster_number)
                self.calculate_or_update_representative_core_points(last_modified_cluster_label=self.last_modified_cluster_label, all=False,k=self.SRPs_factor)

                # self.find_unpacking_code_sequence_mean_core_elements(self.max_core_elements)
                # self.find_pe_features_mean_core_elements(self.unpacking_code_sequence_mean_core_indexes)
                # self.unpacking_code_sequence_mean_core_indexes = []
                print("New cluster found !")

            else:
                # The new element is an outlier.
                # It is not close enough to its closest in order to be added to it,
                # neither has enough outliers close by to form a new cluster.
                self.final_dataset.loc[self.final_dataset.index[-1], 'Label'] = -1
                print("Outlier !")

        print("The new element in the dataset: \n", self.final_dataset.tail(1))

    def incremental_dbscan_(self):
        # self.find_unpacking_code_sequence_mean_core_elements(self.max_core_elements)
        # self.find_pe_features_mean_core_elements(self.unpacking_code_sequence_mean_core_indexes)
        # self.unpacking_code_sequence_mean_core_indexes = []

        self.calculate_or_update_representative_core_points(last_modified_cluster_label=self.last_modified_cluster_label, all=True,k=self.SRPs_factor)

        if self.local_server_image == 'server':
            get_scaler_values("scaler_values/", self.scenario_config)
        else:
            get_scaler_values("../scaler_values/", self.scenario_config)

        min_dist, cluster_min_dist_index, SRP_min_dist_index = self.calculate_min_distance_cluster()

        if cluster_min_dist_index is not None:
            self.check_min_samples_in_eps_or_outlier(min_dist = min_dist, cluster_min_dist_index=cluster_min_dist_index, SRP_min_dist_index=SRP_min_dist_index)
        self.update_labels(final_dataset=self.final_dataset)

        # TODO: Be careful, I commented the 3 lines below:
        # self.largest_cluster = self.find_largest_cluster()
        # self.find_cluster_limits()
        # self.get_largest_cluster_limits()

    def find_largest_cluster(self):
        """
        This function identifies the largest of the clusters with respect to the number of the core elements.
        The largest cluster is the one with the most core elements in it.

        :returns: the number of the largest cluster. If -1 is returned, then there are no clusters created
        in the first place.
        """
        cluster_size = self.final_dataset.groupby('Label')['Label'].count()
        # cluster_size = cluster_size['PE_feature_1'].value_counts()
        try:
            cluster_size = cluster_size.drop(labels=[-1])
        except ValueError:
            print("The label -1 does not exist")
        largest_cluster = -1
        if not cluster_size.empty:
            largest_cluster = cluster_size.idxmax()
            print('The cluster with the most elements is cluster No: ', cluster_size.idxmax())
            return largest_cluster
        else:
            print('There aren\'t any clusters formed yet')
            return largest_cluster

    def find_cluster_limits(self):
        self.cluster_limits = self.final_dataset \
            .groupby(self.final_dataset['Label']) \
            .agg(['min', 'max'])
        print(self.cluster_limits)
        self.cluster_limits.to_json(r'json_exports/all_cluster_limits.json')

    def get_largest_cluster_limits(self):
        self.largest_cluster_limits = self.cluster_limits.iloc[self.largest_cluster + 1]
        self.largest_cluster_limits.to_json(r'json_exports/largest_cluster_limits.json')
        print(self.largest_cluster_limits)


# old functions used for finding centroids
"""
    def find_pe_features_mean_core_elements(self):

        This function calculates the average core elements of each cluster, for the synthetic PE-extracted features
        Note: It does not calculate an average core element for the outliers.

        # Exclude rows labeled as outliers
        self.pe_features_mean_core_elements = self.final_dataset.loc[self.final_dataset['Label'] != -1]
        # Find the mean core elements of each cluster
        self.pe_features_mean_core_elements = self.pe_features_mean_core_elements \
            .groupby('Label')[self.pe_features_columns].mean()
        # response = self.calculate_min_distance_centroid()
        # # print(response)
        # if response is not None:
        #     self.check_min_samples_in_eps_or_outlier(min_dist_index=response)
        """
"""
    def fill_with_Nones(self, list_input):
        # This function fills temporarely a list with Nones
        list_temp = list_input.copy()
        for i in range(len(list_temp), self.max_unpacking_code_sequence_lenght):
            list_temp.append(None)
        return list_temp

    def remove_Nones(self, list_input):
        # This function removes temporarely Nones from a list
        temp_list = list_input.copy()
        return [x for x in temp_list if x is not None]

    def extract_row(self, pd):
        for index, row_element in pd.iterrows():
            new_element = row_element
        return new_element
"""
"""
    def find_unpacking_code_sequence_mean_core_elements(self):

        This function calculates the average core elements of each cluster, for the unpacking code sequences
        Note: It does not calculate an average core element for the outliers.

        # Exclude rows labeled as outliers
        self.unpacking_code_sequence_mean_core_elements = self.final_dataset.loc[self.final_dataset['Label'] != -1]

        # Find the mean core elements of each cluster
        self.unpacking_code_sequence_mean_core_elements.groupby('Label')

        temp_unpacking_code_sequence_mean_core_elements = pd.DataFrame(columns=['Unpacking_code_sequence', 'Label'])

        unique_labels = set(self.unpacking_code_sequence_mean_core_elements['Label'])

        # Calculates agressively representative synthetic core points for unpacking code sequences
        for label in unique_labels:
            temp = self.unpacking_code_sequence_mean_core_elements.loc[self.unpacking_code_sequence_mean_core_elements['Label'].isin([label])].copy()
            temp = list(temp['Unpacking_code_sequence'])
            temp = [self.fill_with_Nones(temp[i]) for i in range(len(temp))]
            temp_columns = [[x[i] for x in temp] for i in range(len(temp[0]))]
            temp = [Counter(temp_columns[i]).most_common(1)[0][0] for i in range(len(temp_columns))]

            temp = self.remove_Nones(temp)
            d = {'Unpacking_code_sequence': temp, 'Label': label}
            temp_unpacking_code_sequence_mean_core_elements = temp_unpacking_code_sequence_mean_core_elements .append(d,ignore_index=True)

        self.unpacking_code_sequence_mean_core_elements = temp_unpacking_code_sequence_mean_core_elements

        print(self.unpacking_code_sequence_mean_core_elements)
        # print(self.mean_core_elements)
        # response = self.calculate_min_distance_centroid()
        # # print(response)
        # if response is not None:
        #     self.check_min_samples_in_eps_or_outlier(min_dist_index=response)
"""
"""
    def calculate_min_distance_centroid(self):

        This function identifies the closest mean_core_element to the incoming element
        that has not yet been added to a cluster or considered as outlier.
        An average between Tappered Levenshtein and euclidean_distance is calculated using the both distances as it is described above.

        :returns min_dist_index: if there is a cluster that is closest to the new entry element
        or None if there are no clusters yet.

        min_dist = None
        min_dist_index = None

        new_element = self.extract_row(self.final_dataset.tail(1))

        # Check if there are elements in the core_elements dataframe.
        # In other words if there are clusters created by the DBSCAN algorithm
        if not self.pe_features_mean_core_elements.empty and not self.unpacking_code_sequence_mean_core_elements.empty:
            # Iterate over the pe_features_mean_core_elements dataframe and find the minimum euclidean_distance
            for (index, current_pe_feature_mean_core_element), (index2, current_unpacking_code_sequence_mean_core_element) in zip(self.pe_features_mean_core_elements.iterrows(),self.unpacking_code_sequence_mean_core_elements.iterrows()):
                tmp_dist = (manhattan_distance(element_1=new_element,element_2=current_pe_feature_mean_core_element) + tapered_levenshtein_distance(element_1=new_element,element_2=current_unpacking_code_sequence_mean_core_element))/2
                #tmp_dist = tapered_levenshtein_distance(element_1=new_element, element_2=current_unpacking_code_sequence_mean_core_element)
                if min_dist is None:
                    min_dist = tmp_dist
                    min_dist_index = index
                elif tmp_dist < min_dist:
                    min_dist = tmp_dist
                    min_dist_index = index
            print('Minimum distance is: ', min_dist, ' at cluster ', min_dist_index)
            return min_dist_index
        else:
            return None
"""
"""
    def check_min_samples_in_eps_or_outlier(self, min_dist_index):

        This function checks whether there are at least min_samples in the given radius from the new
        entry element.
        If there are at least min_samples this element will be added to the cluster and the
        mean_core_element of the current cluster has to be re-calculated.
        If not, there are two options.
            1. Check if there are at least min_samples  outliers in the given radius in order to create a new
                cluster, or
            2.  Consider it as a new outlier

        :param min_dist_index: This is the parameter that contains information related to the closest
        mean_core_element to the current element.

        # Use only the elements of the closest cluster from the new entry element
        new_element = self.extract_row(self.final_dataset.tail(1))
        nearest_cluster_elements = self.final_dataset[self.final_dataset['Label'] == min_dist_index]

        min_samples_count = 0
        for index, cluster_element in nearest_cluster_elements.iterrows():

            if ((manhattan_distance(element_1=new_element, element_2=cluster_element) + tapered_levenshtein_distance(element_1=new_element, element_2=cluster_element))/2)  <= self.eps:
            #if tapered_levenshtein_distance(element_1=new_element, element_2=cluster_element) <= self.eps:
                min_samples_count += 1
                if min_samples_count >= self.min_samples:
                    break;


        if min_samples_count >= self.min_samples:
            # The new element  has enough cluster labels in the eps range
            #  and is now considered as a new member of the cluster.
            #  The mean core element of this cluster is re-calculated.
            self.final_dataset.loc[self.final_dataset.index[-1], 'Label'] = min_dist_index
            self.find_pe_features_mean_core_elements()
            self.find_unpacking_code_sequence_mean_core_elements()
        else:
            # The new element is not added to its closest cluster. Now we have to check
            # whether it is going to be considered an outlier or it will form a new cluster
            # with other outliers.
            outliers = self.final_dataset[self.final_dataset['Label'] == -1]
            min_outliers_count = 0
            new_cluster_elements = pd.DataFrame(columns=['Index'])
            for index, outlier in outliers.iterrows():
                if (manhattan_distance(element_1=new_element, element_2=outlier) + tapered_levenshtein_distance(element_1=new_element, element_2=outlier))/2 <= self.eps:
                #if tapered_levenshtein_distance(element_1=new_element, element_2=outlier)<= self.eps:
                    min_outliers_count += 1
                    new_cluster_elements = new_cluster_elements.append({"Index": index}, ignore_index=True)

            if min_outliers_count >= self.min_samples:
                # The new element has enough outliers in its eps radius in order to form a new cluster.
                new_cluster_number = int(self.final_dataset['Label'].max()) + 1
                for new_cluster_element in new_cluster_elements.iterrows():
                    self.final_dataset.loc[self.final_dataset.index[int(new_cluster_element[1])], 'Label'] = new_cluster_number

                print("A new cluster is now formed out of already existing outliers.")

                # The new cluster's mean core element is calculated after the cluster's creation.
                self.find_pe_features_mean_core_elements()
                self.find_unpacking_code_sequence_mean_core_elements()
                print("New cluster found !")

            else:
                # The new element is an outlier.
                # It is not close enough to its closest in order to be added to it,
                # neither has enough outliers close by to form a new cluster.
                self.final_dataset.loc[self.final_dataset.index[-1], 'Label'] = -1
                print("Outlier !")

        print("The new element in the dataset: \n", self.final_dataset.tail(1))
"""