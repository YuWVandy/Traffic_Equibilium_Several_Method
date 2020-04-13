from model import TrafficFlowModel
import data2 as dt

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

#Initialize the time and performance array
performance = []
time = []
accuracy = 1e-6
epsilon = 1e-6
criteria_type = 2

# Solve the model by Frank-Wolfe Algorithm with newton line search
mod.performance = []
mod.time = []
mod.FW_solver_linesearch(epsilon, accuracy, criteria_type, Type = "newton")
performance.append(mod.performance)
time.append(mod.time)

# Solve the model by Frank-Wolfe Algorithm with bisection line search
mod.performance = []
mod.time = []
mod.FW_solver_linesearch(epsilon, accuracy, criteria_type, Type = "bisection")
performance.append(mod.performance)
time.append(mod.time)

# Solve the model by Frank-Wolfe Algorithm with golden section line search
mod.performance = []
mod.time = []
mod.FW_solver_linesearch(epsilon, accuracy, criteria_type, Type = "gold_section")
performance.append(mod.performance)
time.append(mod.time)

# Solve the model by Conjugate Frank_Wolfe Algorithm with newton line search
mod.performance = []
mod.time = []
mod.solve_CFW(epsilon, accuracy, criteria_type, delta = 1e-2)
performance.append(mod.performance)
time.append(mod.time)

# Solve the model by Biconjugate Frank_Wolfe Algorithm with newton line search
mod.performance = []
mod.time = []
mod.solve_BFW(epsilon, accuracy, criteria_type, delta = 1e-2)
performance.append(mod.performance)
time.append(mod.time)

# Solve the model by Gradient projection with golden section line search
mod.performance = []
mod.time = []
mod.solve_GP(epsilon, accuracy, criteria_type, delta = 1e-2, gama = 0.25)
performance.append(mod.performance)
time.append(mod.time)


fig1 = plt.figure(figsize = (10,5))
Color = ['blue', 'orange', 'green', 'red', 'purple', 'brown']
Label = ['FW-Newton', 'FW-Bisection', 'FW-GoldenSection', 'CFW', 'BFW', "GP"]
#Label = ['CFW', 'BFW', "GP"]
#Color = ['red', 'mediumpurple', 'brown']
for i in range(len(performance)):
    plt.plot(time[i], np.log10(performance[i]), label = Label[i], color = Color[i])
plt.legend(bbox_to_anchor=(1, 1), loc='upper left', ncol=1, frameon = 0)
locs, labels = plt.yticks()

new_label = []
for i in range(len(labels)):
    if(i != len(labels) - 1):
        text = labels[i].get_text()
        new_label.append("10e"+text)
fig1 = plt.figure(figsize = (7,5))
for i in range(len(performance)):
    plt.plot(time[i], np.log10(performance[i]), label = Label[i], color = Color[i])
plt.yticks(locs, new_label)
plt.xlabel('Time(secs)')
plt.ylabel('RGAP')
plt.legend(bbox_to_anchor=(1, 1), loc='upper left', ncol=1, frameon = 0)



# Generate report to console
mod.report()

# Return the solution if necessary
mod._formatted_solution()

