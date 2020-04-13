""" SAMPLE
In this file you can find sample data which could be used
into the TrafficFlowMod class in model.py file
"""

#Sixonfall network
# Graph represented by directed dictionary
# In order: first ("5", "7"), second ("5", "9"), third ("6", "7")...
graph = [
    ("1", ["2", "3"]),
    ("2", ["1", "6"]),
    ("3", ["1", "4", "12"]),
    ("4", ["3", "5", "11"]),
    ("5", ["4", "6", "9"]),
    ("6", ["2", "5", "8"]),
    ("7", ["8", "18"]),
    ("8", ["6", "7", "9", "16"]),
    ("9", ["5", "8", "10"]),
    ("10", ["9", "11", "15", "16", "17"]),
    ("11", ["4", "10", "12", "14"]),
    ("12", ["3", "11", "13"]),
    ("13", ["12", "24"]), 
    ("14", ["11", "15", "23"]), 
    ("15", ["10", "14", "19", "22"]), 
    ("16", ["8", "10", "17", "18"]), 
    ("17", ["10", "16", "19"]), 
    ("18", ["7", "16", "20"]), 
    ("19", ["15", "17", "20"]), 
    ("20", ["18", "19", "21", "22"]), 
    ("21", ["20", "22", "24"]), 
    ("22", ["15", "20", "21", "23"]), 
    ("23", ["14", "22", "24"]), 
    ("24", ["13", "21", "23"])
]
      
# Origin-destination pairs
origins = ["1", "1", "3", "4", "5", "18"]
destinations = ["20", "24", "24", "20", "14", "20"]


# Generated ordered OD pairs: 
# first ("5", "15"), second ("5", "17"), third ("6", "15")...


# Demand between each OD pair (Conjugated to the Cartesian 
# product of Origins and destinations with order)
import pandas as pd
import numpy as np

Data1 = np.array(pd.read_csv(r"C:\Users\10624\Desktop\User-Equilibrium-Solution - Copy\Sixon_network_test\Sixon_network_Capacity.csv"))
demand = [20000, 40000, 30000, 20000, 30000, 10000]
ODmatrix = np.array(pd.read_csv(r"C:\Users\10624\Desktop\User-Equilibrium-Solution - Copy\Sixon_network_test\Sixon_network_ODpair.csv"))
capacity = list(Data1[:, 2])
free_time = list(Data1[:, 4])

