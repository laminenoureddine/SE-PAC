import sys
import numpy as np
from sklearn.metrics.cluster import homogeneity_score, completeness_score, v_measure_score
from sklearn.metrics import accuracy_score
from sklearn import metrics
from sklearn.metrics.cluster import normalized_mutual_info_score, adjusted_mutual_info_score, v_measure_score, adjusted_rand_score, v_measure_score
from datetime import datetime
from collections import Counter
#from tabulate import tabulate

k = 0.5
scenario = int(sys.argv[1])
if scenario == 1 :
    eps = 0.0825
    shuffle = 1
    DBCV_score =   list(np.array([ 0.52897213,  0.33456757, -0.87287769,  0.76454903,  0.602958  ,
                   1.        , -0.60477637,  0.36284615,  0.99318326,  0.84405031,            
                   0.73281817,  0.74443497,  0.69837028, -0.94781328, -0.99646908,            
                   0.66489067,  1.        ,  1.        ,  1.        ,  0.71339898,            
                   0.651122  ,  0.97549777, -0.69448848,  0.80051907,  0.99974657,            
                   0.99665591,  0.58692927,  0.9944188 , -0.75567572,  0.94138019,            
                  -0.94331188,  0.57733729,  0.45414972, -0.39982754,  0.89638948,            
                  -0.58595733, -0.40863553, -0.69410476, -0.53018474,  0.98121156,            
                  -0.99998202, -0.99998151, -0.99998404,  0.        , -0.95716796,            
                   1.        , -0.44080923, -0.1870792 ,  0.96114045, -0.2937451 ,            
                        -0.0888233 , -0.07346939,  0.55453502, -0.1602514 ]))                 
else:
    eps = 0.0675
    shuffle = -1
    DBCV_score =  list(np.array([ 0.61259253, -0.48273652, -0.08960689,  0.04939745,  0.99943187, 
                    0.81040948,  0.44283346,  0.99801715,  0.96595082,  0.48010706,            
                   -0.86989461, -0.96780094,  0.95728535,  0.30539559, -0.99739524,            
                    0.8087439 ,  0.81399878,  0.58057002,  0.94795467,  0.94484045,            
                    0.53815716,  0.98045495, -0.99947009,  1.        ,  0.99669957,            
                    1.        ,  0.29114862, -0.77689908, -0.88950362,  0.99999995,            
                    0.99999999, -0.82650096, -0.45281349, -0.98994   ,  0.38692789,            
                   -0.75414934,  0.99999851, -0.5555218 ,  0.68451929,  0.99872335,            
                    1.        ,  0.99107009,  0.        ,  0.        , -0.87523674,            
                    0.73374065, -0.25062119,  0.03139542,  0.2443147 , -0.39024512,            
                                0.99999999,  0.99999836,  0.99999836]))


DBCV_score = ["%.3f"%x for x in DBCV_score]

results_path = sys.argv[2]

file_cluster_labels = results_path + "eps_" + str(eps) + "_min_samples_3_scenario_" + str(scenario) + "_shuffle_" + str(shuffle) + "_k_" + str(k) +"_final_clusters_labels.npy"
file_packer_labels = results_path + "eps_" + str(eps) +"_min_samples_3_scenario_" + str(scenario) + "_shuffle_" + str(shuffle) + "_k_" + str(k) + "_packers_families_labels.npy"
INC_clustering_labels = list(np.load(file_cluster_labels))
all_labels = list(np.load(file_packer_labels))
print("The k is: ", k)
print("cluster_labels: ", INC_clustering_labels)
print("packer_labels: ", all_labels)
print("DBCV score:", len(DBCV_score))
print("\n-------------------------------------")

packers = {}
clusters = {}
for packer_label in list(set(all_labels)):
    packers[packer_label]= {}

for cluster_label in list(set(INC_clustering_labels)):
    clusters[cluster_label]= {}

for packer_label in list(set(all_labels)):
    packers[packer_label]['cluster_labels'] = []
    packers[packer_label]['ami'] = []
    packers[packer_label]['homogeinity'] = []
    packers[packer_label]['num_clusters'] = 0
    packers[packer_label]['noise'] = 0
    packers[packer_label]['cluster_contents'] = []
    packers[packer_label]['DBCV'] = []

for cluster_label in list(set(INC_clustering_labels)):
    #if cluster_label != -1:
    clusters[cluster_label]['packer_labels'] = []
    clusters[cluster_label]['ami'] = 0
    clusters[cluster_label]['homogeinity'] = 0
    clusters[cluster_label]['num_samples'] = 0
    clusters[cluster_label]['DBCV'] = 0

for i, packer_label in enumerate(all_labels):
    packers[packer_label]['cluster_labels'].append(INC_clustering_labels[i])
    #if INC_clustering_labels[i] != -1:
    #    packers[packer_label]['cluster_labels'].append(INC_clustering_labels[i])
    #else:
    #    packers[packer_label]['noise'] +=1

for packer_label in list(set(all_labels)):
    print(packer_label)
    c = Counter(packers[packer_label]['cluster_labels'])
    packers[packer_label]['cluster_contents'].extend(c.most_common())
    #packers[packer_label]['cluster_labels'] = list(set(packers[packer_label]['cluster_labels']))

for i, cluster_label in enumerate(INC_clustering_labels):
    #if cluster_label != -1:
    clusters[cluster_label]['packer_labels'].append(all_labels[i])

for cluster_label in list(set(INC_clustering_labels)):
    #if cluster_label != -1:
    cluster_ami = adjusted_mutual_info_score(clusters[cluster_label]['packer_labels'], [cluster_label]*len(clusters[cluster_label]['packer_labels']))
    cluster_homogenity = homogeneity_score(clusters[cluster_label]['packer_labels'], [cluster_label]*len(clusters[cluster_label]['packer_labels']))
    clusters[cluster_label]['ami'] = cluster_ami
    clusters[cluster_label]['homogeinity'] = cluster_homogenity
    clusters[cluster_label]['num_samples'] = len(clusters[cluster_label]['packer_labels'])

for i, dbcv_score in enumerate(DBCV_score):
    clusters[i]['DBCV'] = dbcv_score
    print(clusters[i]['DBCV'])

for packer_label in list(set(all_labels)):
    for elem in packers[packer_label]['cluster_contents']:
        packers[packer_label]['homogeinity'].append((elem[0], clusters[elem[0]]['homogeinity']))
        packers[packer_label]['DBCV'].append((elem[0], clusters[elem[0]]['DBCV']))

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

for packer_label in sorted_packers_labels:
    print("The packer ", packer_label, "has the following details:")
    packers[packer_label]['cluster_contents'] = sorted(packers[packer_label]['cluster_contents'], key=lambda x: x[0])
    if packers[packer_label]['cluster_contents'][0][0] == -1:
        packers[packer_label]['cluster_contents'].append(packers[packer_label]['cluster_contents'].pop(0))
    packers[packer_label]['DBCV'] = sorted(packers[packer_label]['DBCV'], key=lambda x: x[0])
    print("Content of its clusters are:", packers[packer_label]['cluster_contents'])
    print("the quality of its clusters are:", packers[packer_label]['DBCV'])
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

text_file = open("packers_latex_table,.txt", "w")
n = text_file.write(packers_latex_table)
text_file.close()
"""
