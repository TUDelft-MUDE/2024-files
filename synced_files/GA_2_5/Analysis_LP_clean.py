# ----------------------------------------
import gurobipy as gp
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

# ----------------------------------------
# import required packages
import os
import time

# import required functions
from utils.read import read_cases
from utils.read import read_net
from utils.read import read_od
from utils.read import create_nd_matrix

# ----------------------------------------
# For visualization
import networkx as nx
import json
from matplotlib.lines import Line2D 
from utils.network_visualization import network_visualization
from utils.network_visualization_highlight_link import network_visualization_highlight_links
from utils.network_visualization_upgraded import network_visualization_upgraded

# ----------------------------------------
# We are using the SiouxFalls network which is one of the most used networks in transportation reseach: https://github.com/bstabler/TransportationNetworks/blob/master/SiouxFalls/Sioux-Falls-Network.pdf
networks = ['SiouxFalls']
networks_dir = 'input/TransportationNetworks'

net_dict, ods_dict = read_cases(networks, networks_dir)
net_data, ods_data = net_dict[networks[0]], ods_dict[networks[0]]

## now let's prepare the data in a format readable by gurobi
links = list(net_data['capacity'].keys())
nodes = np.unique([list(edge) for edge in links])
fftts = net_data['free_flow']

# Visualise
coordinates_path = 'input/TransportationNetworks/SiouxFalls/SiouxFallsCoordinates.geojson'
G, pos = network_visualization(link_flow = fftts,coordinates_path= coordinates_path) 

# ----------------------------------------
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

data = ods_data

max_origin = max(key[0] for key in data.keys())
max_destination = max(key[1] for key in data.keys())

od_matrix = pd.DataFrame(index=range(1, max_origin + 1), columns=range(1, max_destination + 1))

for key, value in data.items():
    od_matrix.loc[key[0], key[1]] = value

print("Origin-Destination Matrix:")
print(od_matrix.head(5))

# ----------------------------------------
# Convert to numeric and handle NaN
od_matrix = od_matrix.fillna(0).astype(float)

# Plotting the Heatmap
plt.figure(figsize=(8, 6))
plt.title("Origin-Destination Heatmap")
plt.xlabel("Destination")
plt.ylabel("Origin")
plt.imshow(od_matrix.values, cmap='RdYlGn_r', aspect="auto", interpolation="nearest")
plt.colorbar(label="Flow Values")
plt.xticks(ticks=np.arange(od_matrix.shape[1]), labels=od_matrix.columns)
plt.yticks(ticks=np.arange(od_matrix.shape[0]), labels=od_matrix.index)
plt.show()

# ----------------------------------------
# define parameters
extension_factor = 2.5  # capacity after extension
extension_max_no = 40  # simplified budget limit
timelimit = 300  # seconds
beta = 2  # explained later

# auxiliary parameters 
cap_normal = {(i, j): cap for (i, j), cap in net_data['capacity'].items()}
cap_extend = {(i, j): cap * extension_factor for (i, j), cap in net_data['capacity'].items()}

# origins and destinations
origs = np.unique([orig for (orig, dest) in list(ods_data.keys())])
dests = np.unique([dest for (orig, dest) in list(ods_data.keys())])

# OD-matrix is built
demand = create_nd_matrix(ods_data, origs, dests, nodes)

# ----------------------------------------
## create a gurobi model object
model = gp.Model()

model.params.TimeLimit = timelimit  
model.params.NonConvex = 2 
model.params.PreQLinearize = 1

# ----------------------------------------
# decision variables:

link_selected = model.addVars(links, vtype=gp.GRB.BINARY, name='y')
link_flow = model.addVars(links, vtype=gp.GRB.CONTINUOUS, name='x')
dest_flow = model.addVars(links, dests, vtype=gp.GRB.CONTINUOUS, name='xs')

link_flow_sqr = model.addVars(links, vtype=gp.GRB.CONTINUOUS, name='x2')


# ----------------------------------------
# objective function (total travel time)

model.setObjective(
    gp.quicksum(fftts[i, j] * link_flow[i, j] +
                fftts[i, j] * (beta/cap_normal[i, j]) * link_flow_sqr[i, j] -
                fftts[i, j] * (beta/cap_normal[i, j]) * link_flow_sqr[i, j] * link_selected[i, j] +
                fftts[i, j] * (beta/cap_extend[i, j]) * link_flow_sqr[i, j] * link_selected[i, j]
                for (i, j) in links))

# ----------------------------------------
# budget constraint
c_bgt = model.addConstr(gp.quicksum(link_selected[i, j] for (i, j) in links) <= extension_max_no)

# ----------------------------------------
# link flow conservation
c_lfc = model.addConstrs(gp.quicksum(dest_flow[i, j, s] for s in dests) == link_flow[i, j] for (i, j) in links)

# ----------------------------------------
# node flow conservation
c_nfc = model.addConstrs(
    gp.quicksum(dest_flow[i, j, s] for j in nodes if (i, j) in links) -
    gp.quicksum(dest_flow[j, i, s] for j in nodes if (j, i) in links) == demand[i, s]
    for i in nodes for s in dests
)

# ----------------------------------------
# dummy constraints for handling quadratic terms
c_qrt = model.addConstrs(link_flow_sqr[i, j] == link_flow[i, j] * link_flow[i, j] for (i, j) in links)

# ----------------------------------------
# constrain the vehicles to the capacity of the road:
# c_new = YOUR_CODE_HERE

# ----------------------------------------
#Next we are ready to solve the model
model.optimize()

# ----------------------------------------
# fetch optimal decision variables and Objective Function values
link_flows = {(i, j): link_flow[i, j].X for (i, j) in links}
links_selected = {(i, j): link_selected[i, j].X for (i, j) in links}
total_travel_time = model.ObjVal

# Let's print right now the objective function
print("Optimal Objective function Value", model.objVal)

# Let's print right now the decision variables
for var in model.getVars():
    print(f"{var.varName}: {round(var.X, 3)}")  # print the optimal decision variable values.

# ----------------------------------------
network_visualization_highlight_links (G, pos, link_select=links_selected)

# ----------------------------------------
# Define new capacity after expansion
cap_normal = {(i, j): cap for (i, j), cap in net_data['capacity'].items()}
cap_extend = {(i, j): cap * extension_factor for (i, j), cap in net_data['capacity'].items()}
capacity = {(i, j): cap_normal[i, j] * (1 - links_selected[i, j]) + cap_extend[i, j] * links_selected[i, j]
            for (i, j) in links}

# ----------------------------------------
# To see flow and capacity for the entire network
network_visualization_upgraded (G = G, pos=pos, link_flow=link_flows, capacity_new=capacity ,link_select=links_selected, labels='on')

