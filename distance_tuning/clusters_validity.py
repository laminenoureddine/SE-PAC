import numpy as np
from scipy.spatial.distance import euclidean, cdist
from scipy.sparse.csgraph import minimum_spanning_tree
from scipy.sparse import csgraph
from sklearn.metrics import pairwise_distances
from scipy.spatial.distance import cdist
from hdbscan._hdbscan_linkage import mst_linkage_core
from hdbscan.hdbscan_ import isclose


def clusters_validity(X, labels, dist_function=euclidean):

    score = validity_index(X, labels, metric='precomputed',d=2, per_cluster_scores=True)

    return score




def _core_dist(point, neighbors, dist_function):
    """
    Computes the core distance of a point.
    Core distance is the inverse density of an object.

    Args:
        point (np.array): array of dimensions (n_features,)
            point to compute core distance of
        neighbors (np.ndarray): array of dimensions (n_neighbors, n_features):
            array of all other points in object class
        dist_dunction (func): function to determine distance between objects
            func args must be [np.array, np.array] where each array is a point

    Returns: core_dist (float)
        inverse density of point
    """
    n_features = np.shape(point)[0]
    n_neighbors = np.shape(neighbors)[1]

    distance_vector = cdist(point.reshape(1, -1), neighbors)
    distance_vector = distance_vector[distance_vector != 0]
    numerator = ((1/distance_vector)**n_features).sum()
    core_dist = (numerator / (n_neighbors)) ** (-1/n_features)
    return core_dist


def _mutual_reachability_dist(point_i, point_j, neighbors_i,
                              neighbors_j, dist_function):
    """.
    Computes the mutual reachability distance between points

    Args:
        point_i (np.array): array of dimensions (n_features,)
            point i to compare to point j
        point_j (np.array): array of dimensions (n_features,)
            point i to compare to point i
        neighbors_i (np.ndarray): array of dims (n_neighbors, n_features):
            array of all other points in object class of point i
        neighbors_j (np.ndarray): array of dims (n_neighbors, n_features):
            array of all other points in object class of point j
        dist_dunction (func): function to determine distance between objects
            func args must be [np.array, np.array] where each array is a point

    Returns: mutual_reachability (float)
        mutual reachability between points i and j

    """
    core_dist_i = _core_dist(point_i, neighbors_i, dist_function)
    core_dist_j = _core_dist(point_j, neighbors_j, dist_function)
    dist = dist_function(point_i, point_j)
    mutual_reachability = np.max([core_dist_i, core_dist_j, dist])
    return mutual_reachability


def _mutual_reach_dist_graph(X, labels, dist_function):
    """
    Computes the mutual reach distance complete graph.
    Graph of all pair-wise mutual reachability distances between points

    Args:
        X (np.ndarray): ndarray with dimensions [n_samples, n_features]
            data to check validity of clustering
        labels (np.array): clustering assignments for data X
        dist_dunction (func): function to determine distance between objects
            func args must be [np.array, np.array] where each array is a point

    Returns: graph (np.ndarray)
        array of dimensions (n_samples, n_samples)
        Graph of all pair-wise mutual reachability distances between points.

    """
    n_samples = np.shape(X)[0]
    graph = []
    counter = 0
    for row in range(n_samples):
        graph_row = []
        for col in range(n_samples):
            point_i = X[row]
            point_j = X[col]
            class_i = labels[row]
            class_j = labels[col]
            members_i = _get_label_members(X, labels, class_i)
            members_j = _get_label_members(X, labels, class_j)
            dist = _mutual_reachability_dist(point_i, point_j,
                                             members_i, members_j,
                                             dist_function)
            graph_row.append(dist)
        counter += 1
        graph.append(graph_row)
    graph = np.array(graph)
    return graph


def _mutual_reach_dist_MST(dist_tree):
    """
    Computes minimum spanning tree of the mutual reach distance complete graph

    Args:
        dist_tree (np.ndarray): array of dimensions (n_samples, n_samples)
            Graph of all pair-wise mutual reachability distances
            between points.

    Returns: minimum_spanning_tree (np.ndarray)
        array of dimensions (n_samples, n_samples)
        minimum spanning tree of all pair-wise mutual reachability
            distances between points.
    """
    mst = minimum_spanning_tree(dist_tree).toarray()

    return mst + np.transpose(mst)


def _cluster_density_sparseness(MST, labels, cluster):
    """
    Computes the cluster density sparseness, the minimum density
        within a cluster

    Args:
        MST (np.ndarray): minimum spanning tree of all pair-wise
            mutual reachability distances between points.
        labels (np.array): clustering assignments for data X
        cluster (int): cluster of interest

    Returns: cluster_density_sparseness (float)
        value corresponding to the minimum density within a cluster
    """
    indices = np.where(labels == cluster)[0]
    cluster_MST = MST[indices][:, indices]
    cluster_density_sparseness = np.max(cluster_MST)
    return cluster_density_sparseness


def _cluster_density_separation(MST, labels, cluster_i, cluster_j):
    """
    Computes the density separation between two clusters, the maximum
        density between clusters.

    Args:
        MST (np.ndarray): minimum spanning tree of all pair-wise
            mutual reachability distances between points.
        labels (np.array): clustering assignments for data X
        cluster_i (int): cluster i of interest
        cluster_j (int): cluster j of interest

    Returns: density_separation (float):
        value corresponding to the maximum density between clusters
    """
    indices_i = np.where(labels == cluster_i)[0]
    indices_j = np.where(labels == cluster_j)[0]
    shortest_paths = csgraph.dijkstra(MST, indices=indices_i)
    relevant_paths = shortest_paths[:, indices_j]
    density_separation = np.min(relevant_paths)
    return density_separation


def _cluster_validity_index(MST, labels, cluster):
    """
    Computes the validity of a cluster (validity of assignmnets)

    Args:
        MST (np.ndarray): minimum spanning tree of all pair-wise
            mutual reachability distances between points.
        labels (np.array): clustering assignments for data X
        cluster (int): cluster of interest

    Returns: cluster_validity (float)
        value corresponding to the validity of cluster assignments
    """
    min_density_separation = np.inf
    for cluster_j in np.unique(labels):
        if cluster_j != cluster:
            cluster_density_separation = _cluster_density_separation(MST,
                                                                     labels,
                                                                     cluster,
                                                                     cluster_j)
            if cluster_density_separation < min_density_separation:
                min_density_separation = cluster_density_separation
    cluster_density_sparseness = _cluster_density_sparseness(MST,
                                                             labels,
                                                             cluster)
    numerator = min_density_separation - cluster_density_sparseness
    denominator = np.max([min_density_separation, cluster_density_sparseness])
    cluster_validity = numerator / denominator
    return cluster_validity


def _clustering_validity_index(MST, labels):
    """
    Computes the validity of all clustering assignments for a
    clustering algorithm

    Args:
        MST (np.ndarray): minimum spanning tree of all pair-wise
            mutual reachability distances between points.
        labels (np.array): clustering assignments for data X

    Returns: validity_index (float):
        score in range[-1, 1] indicating validity of clustering assignments
    """
    n_samples = len(labels)
    validity_index = 0
    for label in np.unique(labels):
        fraction = np.sum(labels == label) / float(n_samples)
        cluster_validity = _cluster_validity_index(MST, labels, label)
        validity_index += fraction * cluster_validity
    return validity_index


def _get_label_members(X, labels, cluster):
    """
    Helper function to get samples of a specified cluster.

    Args:
        X (np.ndarray): ndarray with dimensions [n_samples, n_features]
            data to check validity of clustering
        labels (np.array): clustering assignments for data X
        cluster (int): cluster of interest

    Returns: members (np.ndarray)
        array of dimensions (n_samples, n_features) of samples of the
        specified cluster.
    """
    indices = np.where(labels == cluster)[0]
    members = X[indices]
    return members





















def all_points_core_distance(distance_matrix, d=2.0):
    """
    Compute the all-points-core-distance for all the points of a cluster.
    Parameters
    ----------
    distance_matrix : array (cluster_size, cluster_size)
        The pairwise distance matrix between points in the cluster.
    d : integer
        The dimension of the data set, which is used in the computation
        of the all-point-core-distance as per the paper.
    Returns
    -------
    core_distances : array (cluster_size,)
        The all-points-core-distance of each point in the cluster
    References
    ----------
    Moulavi, D., Jaskowiak, P.A., Campello, R.J., Zimek, A. and Sander, J.,
    2014. Density-Based Clustering Validation. In SDM (pp. 839-847).
    """
    distance_matrix[distance_matrix != 0] = (1.0 / distance_matrix[
        distance_matrix != 0]) ** d
    result = distance_matrix.sum(axis=1)
    result /= distance_matrix.shape[0] - 1
    result **= (-1.0 / d)

    return result


def all_points_mutual_reachability(X, labels, cluster_id, metric='euclidean', d=None, **kwd_args):
    """
    Compute the all-points-mutual-reachability distances for all the points of
    a cluster.
    If metric is 'precomputed' then assume X is a distance matrix for the full
    dataset. Note that in this case you must pass in 'd' the dimension of the
    dataset.
    Parameters
    ----------
    X : array (n_samples, n_features) or (n_samples, n_samples)
        The input data of the clustering. This can be the data, or, if
        metric is set to `precomputed` the pairwise distance matrix used
        for the clustering.
    labels : array (n_samples)
        The label array output by the clustering, providing an integral
        cluster label to each data point, with -1 for noise points.
    cluster_id : integer
        The cluster label for which to compute the all-points
        mutual-reachability (which should be done on a cluster
        by cluster basis).
    metric : string
        The metric used to compute distances for the clustering (and
        to be re-used in computing distances for mr distance). If
        set to `precomputed` then X is assumed to be the precomputed
        distance matrix between samples.
    d : integer (or None)
        The number of features (dimension) of the dataset. This need only
        be set in the case of metric being set to `precomputed`, where
        the ambient dimension of the data is unknown to the function.
    **kwd_args :
        Extra arguments to pass to the distance computation for other
        metrics, such as minkowski, Mahanalobis etc.
    Returns
    -------
    mutual_reachaibility : array (n_samples, n_samples)
        The pairwise mutual reachability distances between all points in `X`
        with `label` equal to `cluster_id`.
    core_distances : array (n_samples,)
        The all-points-core_distance of all points in `X` with `label` equal
        to `cluster_id`.
    References
    ----------
    Moulavi, D., Jaskowiak, P.A., Campello, R.J., Zimek, A. and Sander, J.,
    2014. Density-Based Clustering Validation. In SDM (pp. 839-847).
    """
    if metric == 'precomputed':
        if d is None:
            raise ValueError('If metric is precomputed a '
                             'd value must be provided!')
        distance_matrix = X[labels == cluster_id, :][:, labels == cluster_id]
    else:
        subset_X = X[labels == cluster_id, :]
        distance_matrix = pairwise_distances(subset_X, metric=metric,
                                             **kwd_args)
        d = X.shape[1]

    core_distances = all_points_core_distance(distance_matrix.copy(), d=d)
    core_dist_matrix = np.tile(core_distances, (core_distances.shape[0], 1))

    result = np.dstack(
        [distance_matrix, core_dist_matrix, core_dist_matrix.T]).max(axis=-1)

    return result, core_distances


def internal_minimum_spanning_tree(mr_distances):
    """
    Compute the 'internal' minimum spanning tree given a matrix of mutual
    reachability distances. Given a minimum spanning tree the 'internal'
    graph is the subgraph induced by vertices of degree greater than one.
    Parameters
    ----------
    mr_distances : array (cluster_size, cluster_size)
        The pairwise mutual reachability distances, inferred to be the edge
        weights of a complete graph. Since MSTs are computed per cluster
        this is the all-points-mutual-reacability for points within a single
        cluster.
    Returns
    -------
    internal_nodes : array
        An array listing the indices of the internal nodes of the MST
    internal_edges : array (?, 3)
        An array of internal edges in weighted edge list format; that is
        an edge is an array of length three listing the two vertices
        forming the edge and weight of the edge.
    References
    ----------
    Moulavi, D., Jaskowiak, P.A., Campello, R.J., Zimek, A. and Sander, J.,
    2014. Density-Based Clustering Validation. In SDM (pp. 839-847).
    """
    single_linkage_data = mst_linkage_core(mr_distances)
    min_span_tree = single_linkage_data.copy()
    for index, row in enumerate(min_span_tree[1:], 1):
        candidates = np.where(isclose(mr_distances[int(row[1])], row[2]))[0]
        candidates = np.intersect1d(candidates,
                                    single_linkage_data[:index, :2].astype(
                                        int))
        candidates = candidates[candidates != row[1]]
        assert len(candidates) > 0
        row[0] = candidates[0]

    vertices = np.arange(mr_distances.shape[0])[
        np.bincount(min_span_tree.T[:2].flatten().astype(np.intp)) > 1]
    # A little "fancy" we select from the flattened array reshape back
    # (Fortran format to get indexing right) and take the product to do an and
    # then convert back to boolean type.
    edge_selection = np.prod(np.in1d(min_span_tree.T[:2], vertices).reshape(
        (min_span_tree.shape[0], 2), order='F'), axis=1).astype(bool)

    # Density sparseness is not well defined if there are no
    # internal edges (as per the referenced paper). However
    # MATLAB code from the original authors simply selects the
    # largest of *all* the edges in the case that there are
    # no internal edges, so we do the same here
    if np.any(edge_selection):
        # If there are any internal edges, then subselect them out
        edges = min_span_tree[edge_selection]
    else:
        # If there are no internal edges then we want to take the
        # max over all the edges that exist in the MST, so we simply
        # do nothing and return all the edges in the MST.
        edges = min_span_tree.copy()

    return vertices, edges


def density_separation(X, labels, cluster_id1, cluster_id2,
                       internal_nodes1, internal_nodes2,
                       core_distances1, core_distances2,
                       metric='euclidean', **kwd_args):
    """
    Compute the density separation between two clusters. This is the minimum
    all-points mutual reachability distance between pairs of points, one from
    internal nodes of MSTs of each cluster.
    Parameters
    ----------
    X : array (n_samples, n_features) or (n_samples, n_samples)
        The input data of the clustering. This can be the data, or, if
        metric is set to `precomputed` the pairwise distance matrix used
        for the clustering.
    labels : array (n_samples)
        The label array output by the clustering, providing an integral
        cluster label to each data point, with -1 for noise points.
    cluster_id1 : integer
        The first cluster label to compute separation between.
    cluster_id2 : integer
        The second cluster label to compute separation between.
    internal_nodes1 : array
        The vertices of the MST for `cluster_id1` that were internal vertices.
    internal_nodes2 : array
        The vertices of the MST for `cluster_id2` that were internal vertices.
    core_distances1 : array (size of cluster_id1,)
        The all-points-core_distances of all points in the cluster
        specified by cluster_id1.
    core_distances2 : array (size of cluster_id2,)
        The all-points-core_distances of all points in the cluster
        specified by cluster_id2.
    metric : string
        The metric used to compute distances for the clustering (and
        to be re-used in computing distances for mr distance). If
        set to `precomputed` then X is assumed to be the precomputed
        distance matrix between samples.
    **kwd_args :
        Extra arguments to pass to the distance computation for other
        metrics, such as minkowski, Mahanalobis etc.
    Returns
    -------
    The 'density separation' between the clusters specified by
    `cluster_id1` and `cluster_id2`.
    References
    ----------
    Moulavi, D., Jaskowiak, P.A., Campello, R.J., Zimek, A. and Sander, J.,
    2014. Density-Based Clustering Validation. In SDM (pp. 839-847).
    """
    if metric == 'precomputed':
        sub_select = X[labels == cluster_id1, :][:, labels == cluster_id2]
        distance_matrix = sub_select[internal_nodes1, :][:, internal_nodes2]
    else:
        cluster1 = X[labels == cluster_id1][internal_nodes1]
        cluster2 = X[labels == cluster_id2][internal_nodes2]
        distance_matrix = cdist(cluster1, cluster2, metric, **kwd_args)

    core_dist_matrix1 = np.tile(core_distances1[internal_nodes1],
                                (distance_matrix.shape[1], 1)).T
    core_dist_matrix2 = np.tile(core_distances2[internal_nodes2],
                                (distance_matrix.shape[0], 1))

    mr_dist_matrix = np.dstack([distance_matrix,
                                core_dist_matrix1,
                                core_dist_matrix2]).max(axis=-1)

    return mr_dist_matrix.min()


def validity_index(X, labels, metric='euclidean',
                   d=None, per_cluster_scores=False, **kwd_args):
    """
    Compute the density based cluster validity index for the
    clustering specified by `labels` and for each cluster in `labels`.
    Parameters
    ----------
    X : array (n_samples, n_features) or (n_samples, n_samples)
        The input data of the clustering. This can be the data, or, if
        metric is set to `precomputed` the pairwise distance matrix used
        for the clustering.
    labels : array (n_samples)
        The label array output by the clustering, providing an integral
        cluster label to each data point, with -1 for noise points.
    metric : optional, string (default 'euclidean')
        The metric used to compute distances for the clustering (and
        to be re-used in computing distances for mr distance). If
        set to `precomputed` then X is assumed to be the precomputed
        distance matrix between samples.
    d : optional, integer (or None) (default None)
        The number of features (dimension) of the dataset. This need only
        be set in the case of metric being set to `precomputed`, where
        the ambient dimension of the data is unknown to the function.
    per_cluster_scores : optional, boolean (default False)
        Whether to return the validity index for individual clusters.
        Defaults to False with the function returning a single float
        value for the whole clustering.
    **kwd_args :
        Extra arguments to pass to the distance computation for other
        metrics, such as minkowski, Mahanalobis etc.
    Returns
    -------
    validity_index : float
        The density based cluster validity index for the clustering. This
        is a numeric value between -1 and 1, with higher values indicating
        a 'better' clustering.
    per_cluster_validity_index : array (n_clusters,)
        The cluster validity index of each individual cluster as an array.
        The overall validity index is the weighted average of these values.
        Only returned if per_cluster_scores is set to True.
    References
    ----------
    Moulavi, D., Jaskowiak, P.A., Campello, R.J., Zimek, A. and Sander, J.,
    2014. Density-Based Clustering Validation. In SDM (pp. 839-847).
    """
    core_distances = {}
    density_sparseness = {}
    mst_nodes = {}
    mst_edges = {}

    max_cluster_id = labels.max() + 1
    density_sep = np.inf * np.ones((max_cluster_id, max_cluster_id),
                                   dtype=np.float64)
    cluster_validity_indices = np.empty(max_cluster_id, dtype=np.float64)

    for cluster_id in range(max_cluster_id):

        if np.sum(labels == cluster_id) == 0:
            continue

        mr_distances, core_distances[
            cluster_id] = all_points_mutual_reachability(
            X,
            labels,
            cluster_id,
            metric,
            d,
            **kwd_args
        )

        mst_nodes[cluster_id], mst_edges[cluster_id] = \
            internal_minimum_spanning_tree(mr_distances)
        density_sparseness[cluster_id] = mst_edges[cluster_id].T[2].max()

    for i in range(max_cluster_id):

        if np.sum(labels == i) == 0:
            continue

        internal_nodes_i = mst_nodes[i]
        for j in range(i + 1, max_cluster_id):

            if np.sum(labels == j) == 0:
                continue

            internal_nodes_j = mst_nodes[j]
            density_sep[i, j] = density_separation(
                X, labels, i, j,
                internal_nodes_i, internal_nodes_j,
                core_distances[i], core_distances[j],
                metric=metric, **kwd_args
            )
            density_sep[j, i] = density_sep[i, j]

    n_samples = float(X.shape[0])
    result = 0

    for i in range(max_cluster_id):

        if np.sum(labels == i) == 0:
            continue

        min_density_sep = density_sep[i].min()
        cluster_validity_indices[i] = (
            (min_density_sep - density_sparseness[i]) /
            max(min_density_sep, density_sparseness[i])
        )
        cluster_size = np.sum(labels == i)
        result += (cluster_size / n_samples) * cluster_validity_indices[i]

    if per_cluster_scores:
        return result, cluster_validity_indices
    else:
        return result


