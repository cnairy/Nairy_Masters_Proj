import os
import shutil

# FL3
C1 = '/home/christian.nairy/capeex19/Aircraft/CitationII_N555DS/FlightData/20190803_142455/PHIPS_Data/FL4_20210824/chains_gt450um/C1/'
C2 = '/home/christian.nairy/capeex19/Aircraft/CitationII_N555DS/FlightData/20190803_142455/PHIPS_Data/FL4_20210824/chains_gt450um/C2/'
dest1_C1_C2 = '/home/christian.nairy/capeex19/Aircraft/CitationII_N555DS/FlightData/20190803_142455/PHIPS_Data/FL4_20210824/chains_gt450um/C1-C2/'
for f in os.listdir(C1):
    # print(f)
    for p in os.listdir(C2):
        if f[33:39] == p[33:39]:
            print(f)
            command = "montage " + C1 + f + " " + C2 + p + " -tile 2x1 -geometry +10+0 out" + f[33:39] + ".png"
            print(command)
            os.system(command)
            
            command = "mv out" + f[33:39] + ".png " + " " + dest1_C1_C2 + f[0:56] + ".C1-C2.png"
            os.system(command)
            
            
            