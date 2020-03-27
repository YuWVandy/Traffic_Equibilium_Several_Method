from model import TrafficFlowModel
import data4 as dt

# Initialize the model by data
mod = TrafficFlowModel(dt.graph, dt.origins, dt.destinations, 
dt.demand, dt.free_time, dt.capacity)

#mod = TrafficFlowModel(dt.graph, dt.origins, dt.destinations, 
#dt.demand, dt.free_time, dt.capacity)

# Change the accuracy of solution if necessary
mod._conv_accuracy = 1e-10

# Display all the numerical details of
# each variable during the iteritions
mod.disp_detail()

# Set the precision of display, which influences
# only the digit of numerical component in arrays
mod.set_disp_precision(10)

# Solve the model by Frank-Wolfe Algorithm
#mod.FW_solver_linesearch(epsilon = 1e-4, accuracy = 1e-7, criteria_type = 3, Type = "newton")
#mod.solve_CFW(epsilon = 1e-6, accuracy = 1e-8, criteria_type = 2, delta = 1e-2)
#mod.solve_BFW(epsilon = 1e-8, accuracy = 1e-8, criteria_type = 2, delta = 1e-2)
mod.solve_GP(epsilon = 1e-8, accuracy = 1e-8, criteria_type = 2, delta = 1e-2, gama = 0.25)

# Generate report to console
mod.report()

# Return the solution if necessary
mod._formatted_solution()

