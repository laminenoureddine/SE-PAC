import sys
import numpy as np
precisions = []
times = []
homos = []
ns_clusters = []
K = [0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 5, 10, 50, 100]

results_path = 'results/'
scenario = int(sys.argv[1])

for k in K:
    if scenario ==1:
        eps = 0.0825
        shuffle = 1
    else:
        eps = 0.0675
        shuffle = -1

    file_time = results_path + "eps_" + str(eps) + "_min_samples_3_scenario_" + str(scenario) + "_shuffle_" + str(shuffle) + "_k_" + str(k) +"_INC_avg_time_per_sample.npy"
    file_precision = results_path + "eps_" + str(eps) +"_min_samples_3_scenario_" + str(scenario) + "_shuffle_" + str(shuffle) + "_k_" + str(k) +"_INC_adjusted_nmi.npy"
    file_homo = results_path + "eps_" + str(eps) +"_min_samples_3_scenario_" + str(scenario) + "_shuffle_" + str(shuffle) + "_k_" + str(k) +"_INC_homogeneity.npy"
    file_n_clusters = results_path + "eps_" + str(eps) +"_min_samples_3_scenario_" + str(scenario) + "_shuffle_" + str(shuffle) + "_k_" + str(k) +"_INC_n_clusters.npy"
    time = np.load(file_time)
    precision = np.load(file_precision)
    homo = np.load(file_homo)
    n_clusters = np.load(file_n_clusters)
    print("the scenario is: ", scenario)
    print("The k is: ", k)
    print("precision: ", precision[-1])
    print("time: ", time)
    print("\n-------------------------------------")
    precisions.append(0.01*precision[-1])
    times.append(time[-1])
    homos.append(0.01*homo[-1])
    ns_clusters.append(n_clusters[-1])

precisions = ["%.3f"%x for x in precisions]
times = ["%.3f"%x for x in times]
homos = ["%.3f"%x for x in homos]

print("Precisions are: ")
print(precisions)
print("Times are: ")
print(times)
print("Homos are: ")
print(homos)
print("Num clusters are: ")
print(ns_clusters)
