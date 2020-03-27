""" SAMPLE
In this file you can find sample data which could be used
into the TrafficFlowMod class in model.py file
"""
#network used as an example in my couple transportation and power network
# Graph represented by directed dictionary
# In order: first ("5", "7"), second ("5", "9"), third ("6", "7")...
graph = [
    ("1", ["2", "12"]),
    ("2", ["3"]),
    ("3", ["4", "10"]),
    ("4", ["5"]),
    ("5", ["6", "8"]),
    ("6", []),
    ("7", ["6", "13"]),
    ("8", ["7"]),
    ("9", ["4", "8"]),
    ("10", ["9"]),
    ("11", ["2", "3", "10", "14", "15"]),
    ("12", ["2", "11", "15"]),
    ("13", []),
    ("14", ["7", "10", "13"]),
    ("15", ["14"]),
    ("16", ["12", "15"]),
]



# Capacity of each link (Conjugated to Graph with order)
# Here all the 19 links have the same capacity
capacity = [300, 200, 200, 200, 350, 400, 500, 250, 250, 300, 500, 550, 200, 400, 300, 300, 200, 300, 200,\
            200, 400, 300, 500, 600, 200, 500, 400, 350]

# Free travel time of each link (Conjugated to Graph with order)
free_time = [7, 9, 9, 12, 3, 9, 5, 13, 5, 9, 9, 10, 9, 6, 9, 8, 7, 14, 11,\
             5, 8, 10, 9, 12, 8, 11, 10, 10]

# Origin-destination pairs
origins = ["1", "1", "16", "16"]
destinations = ["6", "13", "6", "13"]
# Generated ordered OD pairs: 
# first ("5", "15"), second ("5", "17"), third ("6", "15")...


# Demand between each OD pair (Conjugated to the Cartesian 
# product of Origins and destinations with order)
demand = [660, 800, 800, 600]

