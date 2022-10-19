import numpy as np
import sys
k = 0.5
scenario = int(sys.argv[1])
H = float(sys.argv[2])

if scenario == 1:
    eps = 0.0825
    shuffle = 1
else:
    eps = 0.0675
    shuffle = -1

results_path = 'results/'
file_matrix = results_path + "eps_" + str(eps) + "_min_samples_3_scenario_" + str(scenario) + "_shuffle_" + str(shuffle) + "_k_" + str(k) + "_final_matrix.npy"
file_cluster_labels = results_path + "eps_" + str(eps) + "_min_samples_3_scenario_" + str(scenario) + "_shuffle_" + str(shuffle) + "_k_" + str(k) +"_final_clusters_labels.npy"
file_packer_labels = results_path + "eps_" + str(eps) +"_min_samples_3_scenario_" + str(scenario) + "_shuffle_" + str(shuffle) + "_k_" + str(k) + "_packers_families_labels.npy"
file_core_points_indexes = results_path + "eps_" + str(eps) +"_min_samples_3_scenario_" + str(scenario) + "_shuffle_" + str(shuffle) + "_k_" + str(k) +"_core_points.npy"
matrix = np.load(file_matrix)
cluster_labels = list(np.load(file_cluster_labels))
packer_labels = list(np.load(file_packer_labels))
core_points_indexes = list(np.load(file_core_points_indexes))
print("the matrix is: ", matrix)
print("The k is: ", k)
print("cluster_labels: ", cluster_labels)
print("packer_labels: ", packer_labels)
print("core points indexes: ", core_points_indexes)
print("\n-------------------------------------")


clusters = {}

for cluster_label in list(set(cluster_labels)):
    clusters[cluster_label]= {}

for cluster_label in list(set(cluster_labels)):
    #if cluster_label != -1:
    clusters[cluster_label]['packer_labels'] = []
    clusters[cluster_label]['num_samples'] = 0

for i, cluster_label in enumerate(cluster_labels):
    #if cluster_label != -1:
    clusters[cluster_label]['packer_labels'].append(packer_labels[i])

for cluster_label in list(set(cluster_labels)):
    #if cluster_label != -1:
    clusters[cluster_label]['num_samples'] = len(clusters[cluster_label]['packer_labels'])

packers = {}
for packer_label in list(set(packer_labels)):
    packers[packer_label]= {}

for packer_label in list(set(packer_labels)):
    packers[packer_label]['cluster_labels'] = []
    packers[packer_label]['cluster_contents_after_selection'] = []

sorted_packers_labels= ['Armadillo',
     'ASPack',
     'ASProtect',
     'BitShapeYodas',
     'ExeStealth',
     'eXPressor',
     'ezip',
     'FSG',
     'InstallShield',
     'MEW',
     'MoleBox',
     'MPRESS',
     'NeoLite',
     'NsPacK',
     'Nullsoft',
     'Packman',
     'PECompact',
     'PELock',
     'PENinja',
     'PEPACK',
     'Petite',
     'RLPack',
     'tElock',
     'Themida',
     'UPX',
     'WinRAR',
     'WinUpack',
     'WinZip',
     'Wise'
]


clusters_core_points ={}
for i, label in enumerate(list(set(cluster_labels))):
    if label != -1:
        clusters_core_points[label] = []

for i, label in enumerate(cluster_labels):
    if i in core_points_indexes:
        clusters_core_points[label].append(i)

clusters_relevant_samples = {}
for i, label in enumerate(list(set(cluster_labels))):
    if label != -1:
        clusters_relevant_samples[label] = clusters_core_points[label][:]

already_visited = []
density_marker = []
for label in clusters_core_points.keys():
    already_visited = []
    for i in clusters_core_points[label]:
        if i not in already_visited:
            density_point = 1
            for j in clusters_core_points[label]:
                if (j != i) and j not in already_visited:
                    if(matrix[i][j] <= (H*eps)):
                        clusters_relevant_samples[label].remove(j)
                        already_visited.append(j)
                        density_point +=1
            clusters_relevant_samples[label].remove(i)
            already_visited.append(i)
            density_marker.append((label, i, density_point))

clusters_contents_after_selection = {}
for cluster_label in list(set(cluster_labels)):
    clusters_contents_after_selection[cluster_label] = []

for elem in density_marker:
    index = len(clusters_contents_after_selection[elem[0]])+1
    rcp_index = 'RCP_'
    density_marker_percentage = round(((elem[2]) / float(clusters[elem[0]]['num_samples']) * 100), 2)
    rcp_elem = (rcp_index, density_marker_percentage)
    clusters_contents_after_selection[elem[0]].append(rcp_elem)

clusters_lenght_after_selection = []
for cluster_label in list(set(cluster_labels)):
    clusters_contents_after_selection[cluster_label] = sorted(clusters_contents_after_selection[cluster_label], key=lambda tup: tup[1], reverse=True)
    for index, elem in enumerate(clusters_contents_after_selection[cluster_label]):
        rcp_index = 'RCP_' + str(index+1)
        rcp_percentage = str(elem[1]) + "%"
        clusters_contents_after_selection[cluster_label][index] = (rcp_index, rcp_percentage)

    clusters_lenght_after_selection.append((cluster_label, len(clusters_contents_after_selection[cluster_label])))

clusters_lenght_after_selection = sorted(clusters_lenght_after_selection, key=lambda tup: tup[1], reverse=True)

cluster_grouped_by_lenght = {}
for elem in clusters_lenght_after_selection:
    cluster_grouped_by_lenght[elem[1]] = []

for elem in clusters_lenght_after_selection:
    print("Cluster: ", elem[0], " has the lenght: ", elem[1])
    print("Its content after selection is: ")
    print(elem[0], " : ", "{", clusters_contents_after_selection[elem[0]], "}")
    print("-------------------------------------------\n\n")
    cluster_grouped_by_lenght[elem[1]].append(elem[0])
    
for key in cluster_grouped_by_lenght.keys():
    cluster_grouped_by_lenght[key] = sorted(cluster_grouped_by_lenght[key])
    print(cluster_grouped_by_lenght[key], "  :  ", key)

original_len = len(core_points_indexes)
new_len = len(density_marker)

print("Before selection: ", original_len)
print("After selection: ", new_len)
#print(density_marker)

"""
#H=1
#Before selection:  25880
#After selection:  231

#H=2
#Before selection:  25880
#After selection:  176

#H=3
#Before selection:  25880
#After selection:  171
"""
