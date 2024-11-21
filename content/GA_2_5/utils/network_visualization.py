#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 22 19:23:56 2023

@author: mmovaghar
"""

import networkx as nx
import json
import matplotlib.pyplot as plt


coordinates_path = 'input/TransportationNetworks/SiouxFalls/SiouxFallsCoordinates.geojson'

def network_visualization(link_flow,coordinates_path):

    plt.figure(figsize=(30, 30))

    # Read the JSON file containing node coordinates
    # Load the JSON data
    with open(coordinates_path, 'r') as f:
        data_cor = json.load(f)
        
    # Extract node coordinates
    node_coordinates = {}
    for feature in data_cor['features']:
        node_id = feature['properties']['id']
        x_coord = feature['properties']['x']
        y_coord = feature['properties']['y']
        node_coordinates[node_id] = {'x': x_coord, 'y': y_coord}
        
        
    # Create the graph with node positions based on coordinates from the JSON file
    G = nx.DiGraph()
    G.add_edges_from(link_flow.keys())  # Assuming link_flow contains edges

    # Set node positions based on coordinates from the JSON file
    pos = {node: (node_coordinates[node]['x'], node_coordinates[node]['y']) for node in G.nodes()}
    
    # Draw all edges
    nx.draw(G, pos, with_labels=True, node_size=4000, node_color='lightblue', font_size=40, connectionstyle='arc3, rad = 0.1',arrowsize=50)

    # Add title and grid
    plt.title("Network Visualization", fontsize=70)  # Add a title
    plt.grid(True)
    plt.grid('minor') # Turn the grids on
    plt.axis('on')
    # Show the plot
    plt.show()
    return G, pos
    
# To Use:
# coordinates_path = 'input/TransportationNetworks/SiouxFalls/SiouxFallsCoordinates.geojson'
# G, pos = network_visualization(link_flow = fftts,coordinates_path= coordinates_path) # the network we create here will be used later for further visualizations!