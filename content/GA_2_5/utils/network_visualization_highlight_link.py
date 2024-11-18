#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 22 19:44:48 2023

@author: mmovaghar
"""

import networkx as nx
import matplotlib.pyplot as plt




def network_visualization_highlight_links(G, pos, link_select):

    plt.figure(figsize=(30, 30))

    # Draw all edges
    nx.draw(G, pos, with_labels=True, node_size=4000, node_color='lightblue', font_size=40, connectionstyle='arc3, rad = 0.1',arrowsize=50)
    # Create a subgraph based on the selected links
    subselected_links = {key: value for key, value in link_select.items() if value != 0.0}
    subgraph = G.edge_subgraph(subselected_links)

    # Draw selected edges with a different color or style
    nx.draw(subgraph, pos, with_labels=True, node_size=4000, node_color='pink',edge_color='red', font_size=40, connectionstyle='arc3, rad = 0.1',arrowsize=50)

    # Add title and grid
    plt.title("Network Visualization with Highlighted Selected Links", fontsize=70)  # Add a title
    plt.grid(True)
    plt.grid('minor') # Turn the grids on
    plt.axis('on')
    # Show the plot
    plt.show()
# To Use:
# coordinates_path = 'input/TransportationNetworks/SiouxFalls/SiouxFallsCoordinates.geojson'
# G, pos = network_visualization_highlight_links (link_flow=link_flows,link_select=links_selected, coordinates_path=coordinates_path)