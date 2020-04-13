from graph import TrafficNetwork, Graph
import numpy as np


class TrafficFlowModel:
    ''' TRAFFIC FLOW ASSIGN MODEL
        Inside the Frank-Wolfe algorithm is given, one can use
        the method `solve` to compute the numerical solution of
        User Equilibrium problem.
    '''
    def __init__(self, graph= None, origins= [], destinations= [], 
    demands= [], link_free_time= None, link_capacity= None):

        self.__network = TrafficNetwork(graph= graph, O= origins, D= destinations)

        # Initialization of parameters
        self.__link_free_time = np.array(link_free_time)
        self.__link_capacity = np.array(link_capacity)
        self.__demand = np.array(demands)


        # Alpha and beta (used in performance function)
        self._alpha = 0.15
        self._beta = 4


        # Convergent criterion
        self._conv_accuracy = 1e-5

        # Boolean varible: If true print the detail while iterations
        self.__detail = False

        # Boolean varible: If true the model is solved properly
        self.__solved = False

        # Some variables for contemporarily storing the
        # computation result
        self.__final_link_flow = None
        self.__iterations_times = None

    def __insert_links_in_order(self, links):
        ''' Insert the links as the expected order into the
            data structure `TrafficFlowModel.__network`
        '''
        first_vertice = [link[0] for link in links]
        for vertex in first_vertice:
            self.__network.add_vertex(vertex)
        for link in links:
            self.__network.add_edge(link)
    
    def FW_solver_linesearch(self, epsilon, accuracy, criteria_type, Type):
        '''Solve the traffic flow assignment model (UE) by Frank-Wolfe algorithm
        '''
        
        import time
        start = time.time()
#        if self.__detail:
#            print(self.__dash_line())
#            print("TRAFFIC FLOW ASSIGN MODEL (USER EQUILIBRIUM) \nFRANK-WOLFE ALGORITHM - DETAIL OF ITERATIONS")
#            print(self.__dash_line())
#            print(self.__dash_line())
#            print("Initialization")
#            print(self.__dash_line())
        
        # Step 0: based on the zero flow, perform all_or_nothing assign to generate the initial feasible flow
        empty_flow = np.zeros(self.__network.num_of_links())
        self.link_flow = self.all_or_nothing_assign(empty_flow)
        
        counter = 0
        while True:
#            if self.__detail:
#                print(self.__dash_line())
#                print("Iteration %s" % counter)
#                print(self.__dash_line())
#                print("Current link flow:\n%s" % self.link_flow)

            # Step 1 & Step 2: Use the link flow matrix -x to generate the time, then generate the auxiliary link flow matrix -y
            self.auxiliary_link_flow = self.all_or_nothing_assign(self.link_flow)

            # Step 3: Linear Search： bisection search, golden_section search, newton search
            if(Type == "gold_section"):
                opt_theta = self.__golden_section(self.link_flow, self.auxiliary_link_flow, epsilon)
            if(Type == "bisection"):
                opt_theta = self.__bisection(self.link_flow, self.auxiliary_link_flow, epsilon)
            if(Type == "newton"):
                opt_theta = self.__newton(self.link_flow, self.auxiliary_link_flow, epsilon)
            
            # Step 4: Using optimal theta to update the link flow matrix
            self.new_link_flow = (1 - opt_theta) * self.link_flow + opt_theta * self.auxiliary_link_flow

            # Print the detail if necessary
#            if self.__detail:
#                print("Optimal theta: %.8f" % opt_theta)
#                print("Auxiliary link flow:\n%s" % self.auxiliary_link_flow)

            # Step 5: Check the Convergence, if FALSE, then return to Step 1
            con, result = self.convergence(self.link_flow, self.new_link_flow, criteria_type, accuracy)
            end = time.time()
            self.performance.append(con)
            self.time.append(end - start)
            if result:
                if self.__detail:
                    print(self.__dash_line())
                self.__solved = True
                self.__final_link_flow = self.new_link_flow
                self.__iterations_times = counter
                break
            else:
                self.link_flow = self.new_link_flow
                counter += 1
                
    def solve_CFW(self, epsilon, accuracy, criteria_type, delta):
        ''' Solve the traffic flow assignment model (user equilibrium)
            by Conjugate-Frank-Wolfe algorithm, all the necessary data must be 
            properly input into the model in advance. 

            (Implicitly) Return
            ------
            self.__solved = True
        '''
        import time
        start = time.time()
#        if self.__detail:
#            print(self.__dash_line())
#            print("TRAFFIC FLOW ASSIGN MODEL (USER EQUILIBRIUM) \nFRANK-WOLFE ALGORITHM - DETAIL OF ITERATIONS")
#            print(self.__dash_line())
#            print(self.__dash_line())
#            print("Initialization")
#            print(self.__dash_line())
        
        # Step 0: based on the x0, generate the x1
        empty_flow = np.zeros(self.__network.num_of_links())
        self.link_flow = self.all_or_nothing_assign(empty_flow)

        
        counter = 0
        while True:
#            if self.__detail:
#                print(self.__dash_line())
#                print("Iteration %s" % counter)
#                print(self.__dash_line())
#                print("Current link flow:\n%s" % self.link_flow)

            # Step 1 & Step 2: Use the link flow matrix -x to generate the time, then generate the auxiliary link flow matrix -y
            self.auxiliary_link_flow = self.all_or_nothing_assign(self.link_flow)
            if(counter == 0):
                self.conjugate_flow = self.auxiliary_link_flow
            
            #step3 : find the conjugate direction
            if(counter != 0):
                lamb = self.__conjugate_direction(self.link_flow, self.auxiliary_link_flow, self.conjugate_flow, delta)
                self.conjugate_flow = lamb*self.conjugate_flow + (1 - lamb)*self.auxiliary_link_flow
            
            # Step 3: Linear Search
            opt_theta = self.__bisection(self.link_flow, self.conjugate_flow, epsilon)
            
            # Step 4: Using optimal theta to update the link flow matrix
            self.new_link_flow = (1 - opt_theta) * self.link_flow + opt_theta * self.conjugate_flow

            # Print the detail if necessary
            if self.__detail:
                print("Optimal theta: %.8f" % opt_theta)
#                print("Auxiliary link flow:\n%s" % self.auxiliary_link_flow)
            
            # Step 5: Check the Convergence, if FALSE, then return to Step 1
            con, result = self.convergence(self.link_flow, self.new_link_flow, criteria_type, accuracy)
            end = time.time()
            self.performance.append(con)
            self.time.append(end - start)
            
            if result:
                if self.__detail:
                    print(self.__dash_line())
                self.__solved = True
                self.__final_link_flow = self.new_link_flow
                self.__iterations_times = counter
                break
            else:
                self.link_flow = self.new_link_flow
                counter += 1
                
    def solve_BFW(self, epsilon, accuracy, criteria_type, delta):
        ''' Solve the traffic flow assignment model (user equilibrium)
            by Bi-Conjugate-Frank-Wolfe algorithm, all the necessary data must be 
            properly input into the model in advance. 

            (Implicitly) Return
            ------
            self.__solved = True
        '''
        import time
        start = time.time()
#        if self.__detail:
#            print(self.__dash_line())
#            print("TRAFFIC FLOW ASSIGN MODEL (USER EQUILIBRIUM) \nFRANK-WOLFE ALGORITHM - DETAIL OF ITERATIONS")
#            print(self.__dash_line())
#            print(self.__dash_line())
#            print("Initialization")
#            print(self.__dash_line())
        
        # Step 0: based on the x0, generate the x1
        empty_flow = np.zeros(self.__network.num_of_links())
        self.link_flow = self.all_or_nothing_assign(empty_flow)
        ##first calculate the 1st 2 points s_{k-1}^BFW, s_{k-2}^BFW
        self.auxiliary_link_flow = self.all_or_nothing_assign(self.link_flow)
        self.s1 = self.auxiliary_link_flow
        self.d1 = self.s1 - self.link_flow
        
        opt_theta1 = self.__bisection(self.link_flow, self.s1, epsilon)
        self.link_flow2 = (1 - opt_theta1) * self.link_flow + opt_theta1 * self.s1
        self.auxiliary_link_flow = self.all_or_nothing_assign(self.link_flow2)
        lamb = self.__conjugate_direction(self.link_flow2, self.auxiliary_link_flow, self.s1, delta)
        self.s2 = lamb*self.s1+ (1 - lamb)*self.auxiliary_link_flow
        self.d2 = self.s2 - self.link_flow2
        
        opt_theta2 = self.__bisection(self.link_flow2, self.s2, epsilon)
        self.link_flow = (1 - opt_theta2) * self.link_flow2 + opt_theta2 * self.s2

        counter = 0
        while True:
            self.auxiliary_link_flow = self.all_or_nothing_assign(self.link_flow)
            self.d = self.auxiliary_link_flow - self.link_flow
            d2_bar = self.s2 - self.link_flow
            d1_bar = opt_theta2*self.s2 - self.link_flow + (1 - opt_theta2)*self.s1
            Hessian = self.__Hessian(self.link_flow, self.__link_free_time, self.__link_capacity)
            
            u_numerator = -np.matmul(np.matmul(d1_bar, Hessian), self.d)
            u_denomintor = np.matmul(np.matmul(d1_bar, Hessian), self.s1 - self.s2)
            
#            v = -np.matmul(np.matmul(d2_bar, Hessian), self.d)/np.matmul(np.matmul(d2_bar, Hessian), d2_bar) + u*opt_theta2/(1 - opt_theta2)
            v_denomintor1 = np.matmul(np.matmul(d2_bar, Hessian), d2_bar)
            v_denomintor2 = 1 - opt_theta2
            
            if(u_denomintor == 0):
                u = 0
            else:
                u = u_numerator/u_denomintor
                if(u <= 0 or u > 1 - delta):
                    u = 1 - delta                    
                    
            if(v_denomintor1 == 0 or v_denomintor2 == 0):
                v = 0
            else:
                v = -np.matmul(np.matmul(d2_bar, Hessian), self.d)/np.matmul(np.matmul(d2_bar, Hessian), d2_bar) + u*opt_theta2/(1 - opt_theta2)
                if(v <= 0 or v > 1 - delta):
                    v = 1 - delta
            
            beta0 = 1/(1 + u + v)
            beta1 = v*beta0
            beta2 = u*beta0

            
            temp_s = beta0*self.auxiliary_link_flow + beta1*self.s2 + beta2*self.s1
            temp_theta = self.__bisection(self.link_flow, temp_s, epsilon)
            self.new_link_flow = (1 - temp_theta) * self.link_flow + temp_theta * temp_s
            
            con, result = self.convergence(self.link_flow, self.new_link_flow, criteria_type, accuracy)
            end = time.time()
            self.performance.append(con)
            self.time.append(end - start)
            
            if result:
                if self.__detail:
                    print(self.__dash_line())
                self.__solved = True
                self.__final_link_flow = self.new_link_flow
                self.__iterations_times = counter
                break
            else:
                self.link_flow = self.new_link_flow
                self.s1 = self.s2
                self.s2 = temp_s
                opt_theta1 = opt_theta2
                opt_theta2 = temp_theta
                counter += 1
                
    def solve_GP(self, epsilon, accuracy, criteria_type, delta, gama):
        '''Solve the user equibilium problem using path-based method based on Gaussian Projection
        '''
        import time
        start = time.time()
        
        self.lp = self.__network.generate_LP_matrix()
        self.paths, self.paths_category = self.__network.generate_paths_by_demands()
        self.OD_paths = self.__network.OD_paths(self.paths_category)
        
        #initialization
        empty_flow = np.zeros(self.__network.num_of_links())
        self.link_time = self.link_flow_to_link_time(empty_flow)
        self.paths_time = np.matmul(self.link_time, self.lp)
        self.paths_flow = np.zeros(len(self.paths))
        self.Initial_Kp_time()
        
        self.link_flow = self.all_or_nothing_assign(empty_flow)
        self.link_time = self.link_flow_to_link_time(self.link_flow)

#        self.paths_time = np.matmul(self.link_time, self.lp)
#        self.paths_flow = np.zeros(len(self.paths))
#        self.Initial_Kp_time()

        counter = 0
        while(1):
            counter += 1
            for i in range(len(self.OD_paths)):
                self.update_sp_cost(i)
                flag = self.improve_Kp(i) 
    
                if(flag or len(self.Kp[i]) > 1):
                    self.GP_projection(i, gama)
    
                    temp1 = np.matmul(self.lp, self.paths_flow)
                    temp2 = self.link_flow
                    self.link_flow = temp1
                    self.link_time = self.link_flow_to_link_time(self.link_flow)
                    self.remove_paths(i)
                            
            con, result = self.convergence(temp2, temp1, criteria_type, accuracy)
            end = time.time()
            self.performance.append(con)
            self.time.append(end - start)
            
            if result:
                if self.__detail:
                    print(self.__dash_line())
                self.__solved = True
                self.__final_link_flow = self.link_flow
                self.__iterations_times = counter
                break
    
    def update_sp_cost(self, i):
        import math
        temp = math.inf
        
        for j in range(len(self.Kp[i])):
            temp_path_time = np.matmul(self.link_time, self.lp[:, self.Kp[i][j]])
            self.paths_time[self.Kp[i][j]] = temp_path_time
            if(temp_path_time < temp):
                temp = temp_path_time
                index = self.Kp[i][j]
        
        self.path_time_min[i] = temp
        self.path_time_min_index[i] = index
        
        
        
    def GP_projection(self, i, gama):
        temp2 = 0
        for j in range(len(self.Kp[i])):
            if(self.Kp[i][j] != self.path_time_min_index[i]):
                intersect_link = self.link_intersection(self.path_time_min_index[i], self.Kp[i][j])

                temp = self._performance_derivative(self.link_flow[intersect_link], self.__link_capacity[intersect_link], self.__link_free_time[intersect_link])

                delta = (self.paths_time[self.Kp[i][j]] - self.paths_time[self.path_time_min_index[i]])/np.sum(temp)

                self.paths_flow[self.Kp[i][j]] = self.paths_flow[self.Kp[i][j]] - min(gama*delta, self.paths_flow[self.Kp[i][j]])
                temp2 += self.paths_flow[self.Kp[i][j]]
                
        self.paths_flow[self.path_time_min_index[i]] = self.__demand[i] - temp2
    
    def link_intersection(self, path1, path2):
        links = []
        link_index1 = self.lp[:, path1]
        link_index2 = self.lp[:, path2]
        share_link = link_index1*link_index2
        for i in range(len(link_index1)):
            if(share_link[i] == 0):
                if(link_index1[i] == 1 or link_index2[i] == 1):
                    links.append(i)
        return links
        
    def update_link_time(self):
        self.link_flow = np.matmul(self.lp, self.paths_flow)
        self.link_time = self.link_flow_to_link_time(self.link_flow)
        
        
    def Initial_Kp_time(self):
        import math
        
        self.Kp = [None]*len(self.__demand)
        self.path_time_min = [None]*len(self.__demand)
        self.path_time_min_index = [None]*len(self.__demand)
        for i in range(len(self.OD_paths)):
            temp = math.inf
            self.Kp[i] = []
            for j in range(len(self.OD_paths[i])):
                if(self.paths_time[self.OD_paths[i][j]] < temp):
                    temp = self.paths_time[self.OD_paths[i][j]]
                    index = self.OD_paths[i][j]
            self.Kp[i].append(index)
            self.path_time_min[i] = temp
            self.path_time_min_index[i] = index
            self.paths_flow[index] = self.__demand[i]
            
    def improve_Kp(self, i):
        import math
        
        temp = math.inf
        for j in range(len(self.OD_paths[i])):
            self.paths_time[self.OD_paths[i][j]] = np.matmul(self.link_time, self.lp[:, self.OD_paths[i][j]])
            if(self.paths_time[self.OD_paths[i][j]] < temp):
                temp = self.paths_time[self.OD_paths[i][j]]
                index = self.OD_paths[i][j]

        if(temp < self.path_time_min[i]):
            self.Kp[i].append(index)
            self.path_time_min[i] = temp
            self.path_time_min_index[i] = index
            return 1
    
    def remove_paths(self, i):
        temp = []
        for j in range(len(self.Kp[i])):
            if(self.paths_flow[self.Kp[i][j]] != 0):
                temp.append(self.Kp[i][j])
        self.Kp[i] = temp            
        
    def __conjugate_direction(self, flow, auxiliary_flow, conjugate_flow, delta):
        '''Calculate the direction to get the conjugate flow
        '''
        temp1 = auxiliary_flow - flow
        temp2 = conjugate_flow - flow
        temp3 = temp1 - temp2

        Hessian = self.__Hessian(flow, self.__link_free_time, self.__link_capacity)
        
        N = np.matmul(np.matmul(temp2, Hessian), temp1)
        D = np.matmul(np.matmul(temp2, Hessian), temp3)

        if(D != 0):
            temp = N/D
            if(temp >= 0 and temp <= 1 - delta):
                theta = N/D
            else:
                theta = 1 - delta
        else:
            theta = 0

        return theta
        
    def __Hessian(self, flow, t0, capacity):
        '''Calculate the Hessian Matrix
        '''
        Hessian = np.zeros([len(flow), len(flow)])
        for i in range(len(flow)):
            temp = self._alpha*self._beta*t0[i]/capacity[i]
            Hessian[i, i] = temp*(flow[i]/capacity[i])**(self._beta - 1)
        
        return Hessian
    
    def convergence(self, link_flow, new_link_flow, criterian_type, accuracy):
        """Decide which convergence we will use
        """
#        con1, result1 = self.__is_convergent1(link_flow, new_link_flow, accuracy)
        con2, result2 = self.__is_convergent2(new_link_flow, accuracy)
#        con3, result3 = self.__is_convergent3(new_link_flow, accuracy)
#        print("Criteria 1", con1, result1)
        print("Criteria 2", con2, result2)
#        print("Criteria 3", con3, result3)
#        if(criterian_type == 1):
#            return result1
        if(criterian_type == 2):
            return con2, result2
#        if(criterian_type == 3):
#            return result3
        
    def __is_convergent1(self, flow1, flow2, accuracy):
        ''' Regard those two link flows lists as the point
            in Euclidean space R^n, then judge the convergence
            under given accuracy criterion.
            Here the formula
                ERR = || x_{k+1} - x_{k} || / || x_{k} ||
            is recommended.
        '''
        err = np.linalg.norm(flow1 - flow2) / np.linalg.norm(flow1)

        if err < accuracy:
            return err, True
        else:
            return err, False
        
    def __is_convergent2(self, flow, accuracy):
        '''Check whether the current flow is the minimum flow using TSTT and SPTT
        '''
        tstt = self.TSTT(flow)
        sptt = self.SPTT(flow)
        self.flow = flow
        self.tstt = tstt
        self.sptt = sptt
        relative_gap = tstt/sptt - 1

        if(relative_gap < accuracy):
            return relative_gap, True
        else:
            return relative_gap, False
        
    def __is_convergent3(self, flow, accuracy):
        '''Check whether the current flow is the minimum flow using criterian 3
        '''
        link_time = self.link_flow_to_link_time(flow)
        
        denomintor = np.sum(link_time*flow)
        
        self.all_or_nothing_assign(flow)
        numerator = np.sum(self.__demand*np.array(self.path_time_min_OD))
        
        relative_gap = (denomintor - numerator)/denomintor

        if(relative_gap < accuracy):
            return relative_gap, True
        else:
            return relative_gap, False
        
    def TSTT(self, flow):
        ''' Calculate the sum of the total time each user using certain links over all links
        '''
        return (np.sum(self.link_time*flow))

    def SPTT(self, flow):
        '''
        '''
        new_flow = self.all_or_nothing_assign(flow)
        return (np.sum(self.link_time*new_flow))

    def __bisection(self, link_flow, auxiliary_link_flow, epsilon):
        '''Bisection method to calculate the optimal step
        '''
        theta_h = 1
        theta_l = 0

        while(theta_h - theta_l > epsilon):    
            theta = 0.5*(theta_l + theta_h)
            new_flow = theta*auxiliary_link_flow + (1 - theta)*link_flow
            new_time = self.link_flow_to_link_time(new_flow)
            total_time = np.sum(new_time*(auxiliary_link_flow - link_flow))
            if(total_time > 0):
                theta_h = theta
            else:
                theta_l = theta
        return theta
    
    def __newton(self, link_flow, auxiliary_link_flow, epsilon):
        '''Newton method to calculate the optimal step
        '''
        theta = 0.5
        while(1):
            new_flow = theta*auxiliary_link_flow + (1 - theta)*link_flow
            new_time = self.link_flow_to_link_time(new_flow)
            f_val = np.sum(new_time*(auxiliary_link_flow - link_flow))
            temp = self._performance_derivative(new_flow, self.__link_capacity, self.__link_free_time)
            f_prime_val = np.sum((auxiliary_link_flow - link_flow)**2*temp)
            theta = theta - f_val/f_prime_val
            if(theta >= 1 or theta <= 0):
                theta = np.random.rand()
            if(f_val <= epsilon):
                break
        return theta
    
    def _performance_derivative(self, new_flow, link_capacity, link_free_time):
        """Calculate the derivative of the transportation link function
        compute the value of the derivative at the new_flow point
        t = t0 * (1 + alpha * (flow / capacity)^beta)
        t' = t0*alpha/(capacity^beta)*beta*flow^(beta - 1)
        """
        value = np.zeros(len(new_flow))
        for i in range(len(new_flow)):
            value[i] = link_free_time[i]*self._alpha*self._beta*new_flow[i]**(self._beta - 1)/(int(link_capacity[i])**int(self._beta))
        return value
        
        
    def _formatted_solution(self):
        ''' According to the link flow we obtained in `solve`,
            generate a tuple which contains four elements:
            `link flow`, `link travel time`, `path travel time` and
            `link vehicle capacity ratio`. This function is exposed 
            to users in case they need to do some extensions based 
            on the computation result.
        '''
        if self.__solved:
            link_flow = self.__final_link_flow
            link_time = self.link_flow_to_link_time(link_flow)
            path_time = self.link_time_to_path_time(link_time)
            link_vc = link_flow / self.__link_capacity
            return link_flow, link_time, path_time, link_vc
        else:
            return None

    def report(self):
        ''' Generate the report of the result in console,
            this function can be invoked only after the
            model is solved.
        '''
        if self.__solved:
            # Print the input of the model
            print(self)
            
            # Print the report
            
            # Do the computation
            link_flow, link_time, path_time, link_vc = self._formatted_solution()

            print(self.__dash_line())
            print("TRAFFIC FLOW ASSIGN MODEL (USER EQUILIBRIUM) \nFRANK-WOLFE ALGORITHM - REPORT OF SOLUTION")
            print(self.__dash_line())
            print(self.__dash_line())
            print("TIMES OF ITERATION : %d" % self.__iterations_times)
            print(self.__dash_line())
            print(self.__dash_line())
            print("PERFORMANCE OF LINKS")
            print(self.__dash_line())
            for i in range(self.__network.num_of_links()):
                print("%2d : link= %12s, flow= %8.2f, time= %8.3f, v/c= %.3f" % (i, self.__network.edges()[i], link_flow[i], link_time[i], link_vc[i]))
            print(self.__dash_line())
            print("PERFORMANCE OF PATHS (GROUP BY ORIGIN-DESTINATION PAIR)")
            print(self.__dash_line())
            counter = 0
            for i in range(self.__network.num_of_paths()):
                if counter < self.__network.paths_category()[i]:
                    counter = counter + 1
                    print(self.__dash_line())
                print("%2d : group= %2d, time= %8.3f, path= %s" % (i, self.__network.paths_category()[i], path_time[i], self.__network.paths()[i]))
            print(self.__dash_line())
        else:
            raise ValueError("The report could be generated only after the model is solved!")

    def all_or_nothing_assign(self, link_flow):
        ''' Perform the all-or-nothing assignment of
            Frank-Wolfe algorithm in the User Equilibrium
            Traffic Assignment Model.
            This assignment aims to assign all the traffic
            flow, within given origin and destination, into
            the least time consuming path

            Input: link flow -> Output: new link flow
            The input is an array.
        '''
        # LINK FLOW -> LINK TIME
        link_time = self.link_flow_to_link_time(link_flow)
        self.link_time = link_time
        # LINK TIME -> PATH TIME
        path_time = self.link_time_to_path_time(link_time)

        # PATH TIME -> PATH FLOW
        # Find the minimal traveling time within group 
        # (splited by origin - destination pairs) and
        # assign all the flow to that path
        self.path_time_min_OD = []
        path_flow = np.zeros(self.__network.num_of_paths())
        for OD_pair_index in range(self.__network.num_of_OD_pairs()):
            indice_grouped = []
            for path_index in range(self.__network.num_of_paths()):
                if self.__network.paths_category()[path_index] == OD_pair_index:
                    indice_grouped.append(path_index)
            sub_path_time = [path_time[ind] for ind in indice_grouped]
            min_in_group = min(sub_path_time)
            self.path_time_min_OD.append(min_in_group)
            ind_min = sub_path_time.index(min_in_group)
            target_path_ind = indice_grouped[ind_min]
            path_flow[target_path_ind] = self.__demand[OD_pair_index]
#        if self.__detail:
#            print("Link time:\n%s" % link_time)
#            print("Path flow:\n%s" % path_flow)
#            print("Path time:\n%s" % path_time)
        
        # PATH FLOW -> LINK FLOW
        new_link_flow = self.path_flow_to_link_flow(path_flow)

        return new_link_flow
        
    def link_flow_to_link_time(self, link_flow):
        ''' Based on current link flow, use link 
            time performance function to compute the link 
            traveling time.
            The input is an array.
        '''
        n_links = self.__network.num_of_links()
        link_time = np.zeros(n_links)
        for i in range(n_links):
            link_time[i] = self.link_time_performance(i, link_flow[i], self.__link_free_time[i], self.__link_capacity[i])
        return link_time

    def link_time_to_path_time(self, link_time):
        ''' Based on current link traveling time,
            use link-path incidence matrix to compute 
            the path traveling time.
            The input is an array.
        '''
        path_time = link_time.dot(self.__network.LP_matrix())
        return path_time
    
    def path_flow_to_link_flow(self, path_flow):
        ''' Based on current path flow, use link-path incidence 
            matrix to compute the traffic flow on each link.
            The input is an array.
        '''
        link_flow = self.__network.LP_matrix().dot(path_flow)
        return link_flow

    def get_path_free_time(self):
        ''' Only used in the final evaluation, not the recursive structure
        '''
        path_free_time = self.__link_free_time.dot(self.__network.LP_matrix())
        return path_free_time

    def link_time_performance(self, i, link_flow, t0, capacity):
        ''' Performance function, which indicates the relationship
            between flows (traffic volume) and travel time on 
            the same link. According to the suggestion from Federal
            Highway Administration (FHWA) of America, we could use
            the following function:
                t = t0 * (1 + alpha * (flow / capacity)^beta)
        '''
        value = t0 * (1 + self._alpha * ((link_flow/capacity)**self._beta))
        return value
    
    def link_cost(self, k, flow):
        value = self.__link_free_time[k] * (1 + self._alpha * ((flow/self.__link_capacity[k])**self._beta))
    
        return value

    def link_time_performance_integrated(self, link_flow, t0, capacity):
        ''' The integrated (with repsect to link flow) form of
            aforementioned performance function.
        '''
        val1 = t0 * link_flow
        # Some optimization should be implemented for avoiding overflow
        val2 = (self._alpha * t0 * link_flow / (self._beta + 1)) * (link_flow / capacity)**self._beta
        value = val1 + val2
        return value

    def __object_function(self, mixed_flow):
        ''' Objective function in the linear search step 
            of the optimization model of user equilibrium 
            traffic assignment problem, the only variable
            is mixed_flow in this case.
        '''
        val = 0
        for i in range(self.__network.num_of_links()):
            val += self.link_time_performance_integrated(link_flow = mixed_flow[i], t0 = self.__link_free_time[i], capacity = self.__link_capacity[i])
        return val

    def __golden_section(self, link_flow, auxiliary_link_flow, epsilon):
        ''' The golden-section search is a technique for 
            finding the extremum of a strictly unimodal 
            function by successively narrowing the range
            of values inside which the extremum is known 
            to exist. The accuracy is suggested to be set
            as 1e-8. For more details please refer to:
            https://en.wikipedia.org/wiki/Golden-section_search
        '''
        # Initial params, notice that in our case the
        # optimal theta must be in the interval [0, 1]
        LB = 0
        UB = 1
        goldenPoint = 0.618
        leftX = LB + (1 - goldenPoint) * (UB - LB)
        rightX = LB + goldenPoint * (UB - LB)
        while True:
            val_left = self.__object_function((1 - leftX) * link_flow + leftX * auxiliary_link_flow)
            val_right = self.__object_function((1 - rightX) * link_flow + rightX * auxiliary_link_flow)
            if val_left <= val_right:
                UB = rightX
            else:
                LB = leftX
            if abs(LB - UB) < epsilon:
                opt_theta = (rightX + leftX) / 2.0
                return opt_theta
            else:
                if val_left <= val_right:
                    rightX = leftX
                    leftX = LB + (1 - goldenPoint) * (UB - LB)
                else:
                    leftX = rightX
                    rightX = LB + goldenPoint*(UB - LB)
 
    def disp_detail(self):
        ''' Display all the numerical details of each variable
            during the iteritions.
        '''
        self.__detail = True

    def set_disp_precision(self, precision):
        ''' Set the precision of display, which influences only
            the digit of numerical component in arrays.
        '''
        np.set_printoptions(precision= precision)

    def __dash_line(self):
        ''' Return a string which consistently 
            contains '-' with fixed length
        '''
        return "-" * 80
    
    def __str__(self):
        string = ""
        string += self.__dash_line()
        string += "\n"
        string += "TRAFFIC FLOW ASSIGN MODEL (USER EQUILIBRIUM) \nFRANK-WOLFE ALGORITHM - PARAMS OF MODEL"
        string += "\n"
        string += self.__dash_line()
        string += "\n"
        string += self.__dash_line()
        string += "\n"
        string += "LINK Information:\n"
        string += self.__dash_line()
        string += "\n"
        for i in range(self.__network.num_of_links()):
            string += "%2d : link= %s, free time= %.2f, capacity= %s \n" % (i, self.__network.edges()[i], self.__link_free_time[i], self.__link_capacity[i])
        string += self.__dash_line()
        string += "\n"
        string += "OD Pairs Information:\n"
        string += self.__dash_line()
        string += "\n"
        for i in range(self.__network.num_of_OD_pairs()):
            string += "%2d : OD pair= %s, demand= %d \n" % (i, self.__network.OD_pairs()[i], self.__demand[i])
        string += self.__dash_line()
        string += "\n"
        string += "Path Information:\n"
        string += self.__dash_line()
        string += "\n"
        for i in range(self.__network.num_of_paths()):
            string += "%2d : Conjugated OD pair= %s, Path= %s \n" % (i, self.__network.paths_category()[i], self.__network.paths()[i])
        string += self.__dash_line()
        string += "\n"
        string += "Link - Path Incidence Matrix:\n"
        string += self.__dash_line()
        string += "\n"
        string += str(self.__network.LP_matrix())
        return string