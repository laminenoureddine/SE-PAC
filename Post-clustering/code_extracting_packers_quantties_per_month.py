import sys
import os
import numpy

if int(sys.argv[1])==1:
    index_start = 1 
    index_end = 9
    s2 = [
        "consensus_Armadillo",
        "consensus_ASPack",
        "consensus_BitShapeYodas",
        "consensus_eXPressor",
        "consensus_ezip",
        "consensus_FSG",
        "consensus_MEW",
        "consensus_MPRESS",
        "consensus_NeoLite",
        "consensus_Packman",
        "consensus_PECompact",
        "consensus_PELock",
        "consensus_PENinja",
        "consensus_Petite",
        "consensus_RLPack",
        "consensus_tElock",
        "consensus_Themida",
        "consensus_UPX",
        "consensus_WinRAR",
        "consensus_WinUpack",
        "consensus_WinZip"
            ]

else:
    index_start = 2 
    index_end = 11
    s2 = [
        "consensus_ASPack",
        "consensus_ASProtect",
        "consensus_ExeStealth",
        "consensus_eXPressor",
        "consensus_FSG",
        "consensus_InstallShield",
        "consensus_MEW",
        "consensus_MoleBox",
        "consensus_NeoLite",
        "consensus_NsPacK",
        "consensus_Nullsoft",
        "consensus_Packman",
        "consensus_PECompact",
        "consensus_PEPACK",
        "consensus_Petite",
        "consensus_RLPack",
        "consensus_Themida",
        "consensus_UPX",
        "consensus_WinUpack",
        "consensus_WinZip",
        "consensus_Wise"
            ]


fixed_path = sys.argv[2]

for k in range(index_start,index_end):
         OK= False
         to_print = [0]*len(s2)
         scores = []
         incoming_packers = []
         if k ==10:
             path = fixed_path+str(k)+"/"
         else:
             path = fixed_path+str(0) + str(k)+"/"
         consensus_files = os.listdir(path)
         for consensus_file in consensus_files:
             count=0
             with open(path+consensus_file) as fd:
                 for line in fd:
                    if line.split() != []:
                        count+=1
             fd.close()
             incoming_packers.append(consensus_file)
             scores.append(count)
         for i,packer in enumerate(s2):
             for j, incoming_packer in enumerate(incoming_packers):
                 if packer == incoming_packer:
                     to_print[i] = scores[j]
                     OK = True
             if OK==False: 
                 to_print[i]=0
         
         print("For K=", k , "The values are: ")
         for score in to_print:
             print(score)
