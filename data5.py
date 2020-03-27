""" SAMPLE
In this file you can find sample data which could be used
into the TrafficFlowMod class in model.py file
"""
#network used as an example in my couple transportation and power network
# Graph represented by directed dictionary
# In order: first ("5", "7"), second ("5", "9"), third ("6", "7")...
graph = [
    ("1", ["5", "12"]),
    ("2", ["8", "11"]),
    ("3", ["11", "13"]),
    ("4", ["5", "9"]),
    ("5", ["1", "4", "6", "9"]),
    ("6", ["5", "7", "10", "12"]),
    ("7", ["6", "8", "11"]),
    ("8", ["2", "7", "12"]),
    ("9", ["4", "5", "10", "13"]),
    ("10", ["6", "9", "11"]),
    ("11", ["2", "3", "7", "10"]),
    ("12", ["1", "6", "8"]),
    ("13", ["3", "9"]),
]
# Free travel time of each link (Conjugated to Graph with order)
free_time = [7, 9,\
             9, 9, \
             8, 11, \
             9, 12, \
             7, 9, 3, 9,\
             3, 5, 13, 7,\
             5, 5, 9,\
             9, 5, 14,\
             12, 9, 10, 9,\
             13, 10, 6,\
             9, 8, 9, 6,\
             9, 7, 14,\
             11, 9]
             

# Capacity of each link (Conjugated to Graph with order)
# Here all the 19 links have the same capacity
capacity = [300, 200, \
            500, 300, \
            300, 200, \
            200, 200, \
            300, 200, 350, 400, \
            350, 500, 250, 200, \
            500, 250, 300, \
            500, 250, 300, \
            200, 400, 550, 200, \
            250, 550, 400, \
            300, 300, 300, 400, \
            200, 200, 300, \
            200, 200]
            


# Origin-destination pairs
origins = ["1", "1", "1", "1", "4", "4", "4", "4", "4", "4"]
destinations = ["2", "3", "10", "11", "2", "3", "8", "9", "10", "13"]
# Generated ordered OD pairs: 
# first ("5", "15"), second ("5", "17"), third ("6", "15")...


# Demand between each OD pair (Conjugated to the Cartesian 
# product of Origins and destinations with order)
demand = [660, 800, 800, 600, 412.5, 495, 700, 300, 300, 600]

