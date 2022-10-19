import sys
import numpy as np
from sklearn.metrics.cluster import homogeneity_score, completeness_score, v_measure_score
from sklearn.metrics import accuracy_score
from sklearn import metrics
from sklearn.metrics.cluster import normalized_mutual_info_score, adjusted_mutual_info_score, v_measure_score, adjusted_rand_score, v_measure_score
from datetime import datetime
from collections import Counter
#from tabulate import tabulate

scenario = int(sys.argv[1])
if scenario == 1 :
    eps = 0.0825
    sorted_packers_labels = [
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
        "Nullsoft",
        "Packman",
        "PECompact",
        "PEPACK",
        "Petite",
        "RLPack",
        "Themida",
        "UPX",
        "WinUpack",
        "WinZip",
        "Wise"
     ]

else:
    eps = 0.0675
    sorted_packers_labels = [
      "Armadillo",
      "ASPack",
      "BitShapeYodas",
      "eXPressor",
      "ezip",
      "FSG",
      "MEW",
      "MPRESS",
      "NeoLite",
      "Packman",
      "PECompact",
      "PELock",
      "PENinja",
      "Petite",
      "RLPack",
      "tElock",
      "Themida",
      "UPX",
      "WinRAR",
      "WinUpack",
      "WinZip"
      ]

results_path = sys.argv[2]

file_cluster_labels = results_path + "offline_clusters_labels_" + str(scenario) + ".npy"
file_packer_labels = results_path + "offline_packers_families_labels_" + str(scenario) + ".npy"
clustering_labels = list(np.load(file_cluster_labels))
all_labels = list(np.load(file_packer_labels))
all_labels = [label.decode('UTF-8') for label in list(all_labels)]

print("cluster_labels: ", clustering_labels)
print("packer_labels: ", all_labels)
print("\n-------------------------------------")

packers = {}
clusters = {}
for packer_label in list(set(all_labels)):
    packers[packer_label]= {}

for cluster_label in list(set(clustering_labels)):
    clusters[cluster_label]= {}

for packer_label in list(set(all_labels)):
    packers[packer_label]['cluster_labels'] = []
    packers[packer_label]['num_clusters'] = 0
    packers[packer_label]['noise'] = 0
    packers[packer_label]['cluster_contents'] = []


for cluster_label in list(set(clustering_labels)):
    #if cluster_label != -1:
    clusters[cluster_label]['packer_labels'] = []
    clusters[cluster_label]['num_samples'] = 0

for i, packer_label in enumerate(all_labels):
    packers[packer_label]['cluster_labels'].append(clustering_labels[i])
    #if INC_clustering_labels[i] != -1:
    #    packers[packer_label]['cluster_labels'].append(INC_clustering_labels[i])
    #else:
    #    packers[packer_label]['noise'] +=1

for packer_label in list(set(all_labels)):
    print(packer_label)
    c = Counter(packers[packer_label]['cluster_labels'])
    packers[packer_label]['cluster_contents'].extend(c.most_common())
    #packers[packer_label]['cluster_labels'] = list(set(packers[packer_label]['cluster_labels']))

for i, cluster_label in enumerate(clustering_labels):
    #if cluster_label != -1:
    clusters[cluster_label]['packer_labels'].append(all_labels[i])

for cluster_label in list(set(clustering_labels)):
    #if cluster_label != -1:
    cluster_ami = adjusted_mutual_info_score(clusters[cluster_label]['packer_labels'], [cluster_label]*len(clusters[cluster_label]['packer_labels']))
    cluster_homogenity = homogeneity_score(clusters[cluster_label]['packer_labels'], [cluster_label]*len(clusters[cluster_label]['packer_labels']))
    clusters[cluster_label]['num_samples'] = len(clusters[cluster_label]['packer_labels'])

for packer_label in sorted_packers_labels:
    print("The packer ", packer_label, "has the following details:")
    packers[packer_label]['cluster_contents'] = sorted(packers[packer_label]['cluster_contents'], key=lambda x: x[0])
    if packers[packer_label]['cluster_contents'][0][0] == -1:
        packers[packer_label]['cluster_contents'].append(packers[packer_label]['cluster_contents'].pop(0))
    print("Content of its clusters are:", packers[packer_label]['cluster_contents'])
    print("-----------------------------------------------------------")


for i, packer_label_1 in enumerate(sorted_packers_labels):
    for j, packer_label_2 in enumerate(sorted_packers_labels):
        if packer_label_1 != packer_label_2:
            for k1, content in enumerate(packers[packer_label_1]['cluster_contents']):
                for k2, content in enumerate(packers[packer_label_2]['cluster_contents']):
                    if packers[packer_label_1]['cluster_contents'][k1][0] == packers[packer_label_2]['cluster_contents'][k2][0] and packers[packer_label_1]['cluster_contents'][k1][0] != -1:
                        print("Bad merge for the packers :", packer_label_1, "   ", packer_label_2)
                        print("Cluster content of the packers in bad merge are :")
                        print(packers[packer_label_1]['cluster_contents'][k1])
                        print(packers[packer_label_2]['cluster_contents'][k2])

"""
shape2 = len(list(set(all_labels))) + 1
shape1 = len(list(set(INC_clustering_labels)))

packers_latex_table = np.zeros((shape1, shape2), dtype=int)

for i, packer_label in enumerate(sorted_packers_labels):
    for elem in packers[packer_label]['cluster_contents']:
            packers_latex_table[elem[0]][i+1] = int(elem[1])

sorted_packers_labels.insert(0, None)
packers_latex_table= list(packers_latex_table)
packers_latex_table.insert(0,sorted_packers_labels)

print(len(packers_latex_table))
input()

for j in range(0, len(packers_latex_table)-1):
    packers_latex_table[j+1][0]= j-1

packers_latex_table = tabulate(packers_latex_table, tablefmt="latex")

text_file = open("packers_latex_table.txt", "w")
n = text_file.write(packers_latex_table)
text_file.close()
"""
