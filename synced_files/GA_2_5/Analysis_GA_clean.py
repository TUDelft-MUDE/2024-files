import pandas as pd
import numpy as np
import gurobipy as gp
import matplotlib.pyplot as plt

from pymoo.algorithms.soo.nonconvex.ga import GA
from pymoo.core.problem import ElementwiseProblem
from pymoo.optimize import minimize
from pymoo.operators.sampling.rnd import BinaryRandomSampling
from pymoo.operators.crossover.hux import HalfUniformCrossover 
from pymoo.operators.mutation.bitflip import BitflipMutation
from pymoo.operators.crossover.pntx import PointCrossover 

import os
import time

from utils.read import read_cases
from utils.read import read_net
from utils.read import read_od
from utils.read import create_nd_matrix

extension_factor = 2.5 

networks = ['SiouxFalls']
networks_dir = os.getcwd() +'/input/TransportationNetworks'

net_dict, ods_dict = read_cases(networks, networks_dir)
net_data, ods_data = net_dict[networks[0]], ods_dict[networks[0]]

links = list(net_data['capacity'].keys())
nodes = np.unique([list(edge) for edge in links])
fftts = net_data['free_flow']

from utils.network_visualization import network_visualization
from utils.network_visualization_highlight_link import network_visualization_highlight_links
from utils.network_visualization_upgraded import network_visualization_upgraded

coordinates_path = 'input/TransportationNetworks/SiouxFalls/SiouxFallsCoordinates.geojson'

G, pos = network_visualization(link_flow = fftts,coordinates_path= coordinates_path) # the network we create here will be used later for further visualizations!

def ta_qp(dvs, net_data=net_data, ods_data=ods_data, extension_factor=2.5):

    # variables
    beta = 2
    links = list(net_data['capacity'].keys())
    nodes = np.unique([list(edge) for edge in links])
    fftts = net_data['free_flow']
    links_selected = dict(zip(links, dvs))

    # capacity
    cap_normal = {(i, j): cap for (i, j), cap in net_data['capacity'].items()}
    cap_extend = {(i, j): cap * extension_factor for (i, j), cap in net_data['capacity'].items()}
    capacity = {(i, j): cap_normal[i, j] * (1 - links_selected[i, j]) + cap_extend[i, j] * links_selected[i, j]
                for (i, j) in links}

    dests = np.unique([dest for (orig, dest) in list(ods_data.keys())])
    origs = np.unique([orig for (orig, dest) in list(ods_data.keys())])

    demand = create_nd_matrix(ods_data, origs, dests, nodes)

    ## create a gurobi model object
    model = gp.Model()
    model.Params.LogToConsole = 0

    ## decision variables:
    link_flow = model.addVars(links, vtype=gp.GRB.CONTINUOUS, name='x')
    dest_flow = model.addVars(links, dests, vtype=gp.GRB.CONTINUOUS, name='xs')

    ## constraints
    model.addConstrs(
        gp.quicksum(dest_flow[i, j, s] for j in nodes if (i, j) in links) -
        gp.quicksum(dest_flow[j, i, s] for j in nodes if (j, i) in links) == demand[i, s]
        for i in nodes for s in dests
    )

    model.addConstrs(gp.quicksum(dest_flow[i, j, s] for s in dests) == link_flow[i, j] for (i, j) in links)

    model.setObjective(
        gp.quicksum(link_flow[i, j] * (fftts[i, j] * (1 + (beta * link_flow[i, j]/capacity[i, j]))) for (i, j) in links))

    ## solve
    model.update()
    start_solve = time.time()
    model.optimize()
    solve_time = (time.time() - start_solve)

    # fetch optimal DV and OF values
    link_flows = {(i, j): link_flow[i, j].X for (i, j) in links}
    total_travel_time = model.ObjVal

    return total_travel_time, capacity, link_flows, links_selected

class NDP(ElementwiseProblem):

    def __init__(self, budget):

        super().__init__(n_var=len(links),       
                         n_obj=1,               
                         n_constr=1,            
                         vtype=bool,           
                        )
        self.budget = budget

    def _evaluate(self, decision_vars, out, *args, **kwargs):
    
        total_travel_time,capacity, link_flows, links_selected = ta_qp(decision_vars)
        g = sum(decision_vars) - self.budget

        out["F"] = total_travel_time
        out["G"] = g

extension_max_no = 40
pop_size = 200

problem = NDP(budget=extension_max_no)

method = GA(pop_size=pop_size,
            sampling=BinaryRandomSampling(),
            mutation=BitflipMutation(),
            crossover=HalfUniformCrossover())

opt_results = minimize(problem,
               method,
               termination=("time", "00:03:00"), #this is maximum computation time
               seed=7,
               save_history=True,
               verbose=True,
               )

print("Best Objective Function value: %s" % opt_results.F)
print("Constraint violation: %s" % opt_results.CV)
print("Best solution found: %s" % opt_results.X)

def get_results(opt_results):

    number_of_individuals = [] # The number of individuals in each generation
    optimal_values_along_generations = []  # The optimal value found in each generation

    for generation_status in opt_results.history:

        # retrieve the optimum from the algorithm
        optimum = generation_status.opt

        # filter out only the feasible solutions and append and objective space values
        try:
            feas = np.where(optimum.get("feasible"))[0]
            optimal_values_along_generations.append(optimum.get("F")[feas][0][0])
            # store the number of function evaluations
            number_of_individuals.append(generation_status.evaluator.n_eval)
        except:
            #In case a generation does not have any feasible solutions, it will be ignored.
            pass

    return number_of_individuals, optimal_values_along_generations

def plot_results(number_of_individuals, optimal_values_along_generations):

    # Create a scatter plot with enhanced styling
    plt.figure(figsize=(8, 6))  # Set the figure size

    # Create a scatter plot
    plt.scatter(number_of_individuals, optimal_values_along_generations, label='Best objective function', color='blue', marker='o', s=100, alpha=0.7, edgecolors='black', linewidths=1.5)

    # Add labels and a legend with improved formatting
    plt.xlabel('Function evaluations', fontsize=14, fontweight='bold')
    plt.ylabel('Total Travel Time', fontsize=14, fontweight='bold')
    plt.title('Best solution evolution', fontsize=16, fontweight='bold')
    plt.legend(loc='upper right', fontsize=12)

    # Customize the grid appearance
    plt.grid(True, linestyle='--', alpha=0.5)

    # Customize the tick labels
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)

    # Add a background color to the plot
    plt.gca().set_facecolor('#f2f2f2')

    # Show the plot
    plt.show()

number_of_individuals, optimal_values_along_generations = get_results(opt_results)

plot_results(number_of_individuals, optimal_values_along_generations)

travel_time, capacity, link_flows, links_selected= ta_qp(dvs=opt_results.X, net_data=net_data, ods_data=ods_data, extension_factor=2.5)

network_visualization_upgraded (G = G, pos=pos, link_flow=link_flows, capacity_new=capacity ,link_select=links_selected, labels='off')

