import numpy as np
import sys


tap_matrix_long_name_path = '../distance_matrices/offline_tapered_levenshtein_distances_matrix_' + sys.argv[1] + '.npy'
offline_tapered_levenshtein_distances_matrix = np.load(tap_matrix_long_name_path)
print "offline tapered_levenshtein_distances_matrix shape is:", offline_tapered_levenshtein_distances_matrix.shape
#offline_euclidean_distance_matrix = np.load('../distance_matrices/offline_euclidean_distance_matrix.npy')
#print "offline euclidean_distance_matrix shape is:", offline_euclidean_distance_matrix.shape

manh_long_name_pathman = '../distance_matrices/offline_manhattan_distance_matrix_' + sys.argv[1] + '.npy'
offline_manhattan_distances_matrix = np.load(manh_long_name_pathman)
print "offline manhattan_distance_matrix shape is:", offline_manhattan_distances_matrix.shape
#offline_gower_distances_matrix = np.load('../distance_matrices/offline_gower_distances_matrix.npy')
#print "offline gower_distances_matrix shape is:", offline_gower_distances_matrix.shape
#offline_final_averaged_distances_matrix = (offline_tapered_levenshtein_distances_matrix + offline_euclidean_distance_matrix) / 2
#np.save('../distance_matrices/offline_averaged_distances_matrix.npy', offline_final_averaged_distances_matrix)
#print "offline final_averaged_distances_matrix shape is:", offline_final_averaged_distances_matrix.shape

offline_final_averaged_distances_matrix = (offline_tapered_levenshtein_distances_matrix + offline_manhattan_distances_matrix) / 2

# Code that removes -1 from the lev matrix and replaces it with it corresponding value in the manh distance.
indexes_to_replaced = np.argwhere(offline_tapered_levenshtein_distances_matrix==-1)
for i in range(len(indexes_to_replaced)):
    offline_final_averaged_distances_matrix[indexes_to_replaced[i][0], indexes_to_replaced[i][1]] = 23*offline_manhattan_distances_matrix[indexes_to_replaced[i][0], indexes_to_replaced[i][1]]

matrix_long_name_path = '../distance_matrices/offline_averaged_distances_matrix_' + sys.argv[1] + '.npy'
np.save(matrix_long_name_path, offline_final_averaged_distances_matrix)
print "offline final_averaged_distances_matrix shape is:", offline_final_averaged_distances_matrix.shape