import sys
import numpy as np
from prettytable import PrettyTable
import numpy as np
from sklearn.manifold import MDS


# Load the final averaged similarities matrix
#final_averaged_distances_matrix = np.load('../distance_matrices/offline_averaged_distances_matrix.npy')
final_averaged_distances_matrix = np.load('distance_matrices/offline_averaged_distances_matrix.npy')
final_averaged_distances_matrix = np.float64(final_averaged_distances_matrix)


print("Starts calculating the coordinates with MDS")
mds_2d = MDS(n_components=2, dissimilarity="precomputed", max_iter=3000, eps=1e-9, n_jobs= -1)
mds_3d = MDS(n_components=3, dissimilarity="precomputed", max_iter=3000, eps=1e-9, n_jobs= -1)
final_averaged_distances_matrix_2d = mds_2d.fit_transform(final_averaged_distances_matrix)
final_averaged_distances_matrix_3d = mds_3d.fit_transform(final_averaged_distances_matrix)


#np.save('data/2d_clusters_points_coordinates_1.npy', final_averaged_distances_matrix_2d)
#np.save('data/3d_clusters_points_coordinates.npy', final_averaged_distances_matrix_3d)
np.save('mds_results/2d_clusters_points_coordinates.npy', final_averaged_distances_matrix_2d)
np.save('mds_results/3d_clusters_points_coordinates.npy', final_averaged_distances_matrix_3d)
