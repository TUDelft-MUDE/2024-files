#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 22 19:46:09 2023

@author: mmovaghar
"""

import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.lines import Line2D


def network_visualization_upgraded(G, pos, link_flow, capacity_new, link_select, labels):

    plt.figure(figsize=(30, 30))


    # Set edge attributes
    nx.set_edge_attributes(G, capacity_new, "capacity")
    nx.set_edge_attributes(G, link_flow, "flow")
    
    subselected_links = {key: value for key, value in link_select.items() if value != 0.0}


    if link_select is not None:
        subselected_links = {key: value for key, value in link_select.items() if value != 0.0}
        subgraph = G.edge_subgraph(subselected_links)
    else:
        subselected_links = {}

    # Create a dictionary to map edges to colors based on some criteria
    edge_colors = ['red', 'darkorange', 'lightgreen','yellow', 'lime']
    bb = pd.DataFrame(link_flow.values())/pd.DataFrame(capacity_new.values())
    boundries_color = (bb[0].max()-bb[0].min())/4
    edge_to_color = {}
    for e in subgraph.edges:
        if subgraph[e[0]][e[1]]['flow']/subgraph[e[0]][e[1]]['capacity'] >= (bb[0].min()+3* boundries_color):
            edge_to_color[e] = edge_colors[0]
        elif (bb[0].min()+2*boundries_color)<= subgraph[e[0]][e[1]]['flow']/subgraph[e[0]][e[1]]['capacity'] <(bb[0].min()+3*boundries_color):
            edge_to_color[e] = edge_colors[1]
        elif (bb[0].min()+1*boundries_color)<= subgraph[e[0]][e[1]]['flow']/subgraph[e[0]][e[1]]['capacity'] <(bb[0].min()+2*boundries_color):
            edge_to_color[e] = edge_colors[2]
        elif subgraph[e[0]][e[1]]['flow']/subgraph[e[0]][e[1]]['capacity'] <= (bb[0].min()+1*boundries_color):
            edge_to_color[e] = edge_colors[3]
        else:
            edge_to_color[e] = edge_colors[4]

    # Define edge widths (thickness) based on some criteria for directed graph
    boundries_width = (max(link_flow.values())-min(link_flow.values()))/4
    edge_widths = []
    for e in subgraph.edges:
        if subgraph[e[0]][e[1]]['flow'] >= (min(link_flow.values())+3*boundries_width):
            edge_widths.append(5)  # Set the width to 5 for these edges
        elif (min(link_flow.values())+2*boundries_width)<= subgraph[e[0]][e[1]]['flow']< (min(link_flow.values())+3*boundries_width):
            edge_widths.append(10)  # Set the width to 10 for these edges
        elif (min(link_flow.values())+1*boundries_width)<= subgraph[e[0]][e[1]]['flow']< (min(link_flow.values())+2*boundries_width):
            edge_widths.append(15)  # Set the width to 15 for these edges
        elif subgraph[e[0]][e[1]]['flow'] <= (min(link_flow.values())+1*boundries_width):
            edge_widths.append(20)  # Set the width to 20 for these edges
        else:
            edge_widths.append(25)  # Set the width to 25 for these edges

                

    # Create labels for selected edges with flow and capacity information
    if labels=='on':
        edge_labels = {(u, v): f"F{u,v} {G[u][v]['flow']:.2f} \nC: {G[u][v]['capacity']:.2f}" for u, v in G.edges()}
    elif labels=='off': 
        edge_labels = {(u, v): f"F{u,v} {subgraph[u][v]['flow']:.2f} \nC: {subgraph[u][v]['capacity']:.2f}" for u, v in subgraph.edges()}
    

    # Draw nodes and edges using positions from the JSON file
    nx.draw(G, pos, with_labels=True, node_size=4000, node_color='lightblue', font_size=40, connectionstyle='arc3, rad = 0.1',arrowsize=50)
    
    nx.draw_networkx_edges(subgraph, pos,edgelist=edge_to_color.keys(), width=edge_widths, 
                           edge_color=[edge_to_color[e] for e in edge_to_color.keys()], style='solid',
                           arrowstyle=None, arrowsize=50, 
                           edge_cmap='Spectral', arrows=True, label= None, 
                           nodelist=link_select, node_shape='o', connectionstyle='arc3, rad = 0.1')
    
    nx.draw_networkx_nodes(subgraph, pos, node_size=4000, node_color='pink', 
                           node_shape='o', linewidths=None, label=None)
    
     # Draw edge labels for selected edges
    nx.draw_networkx_edge_labels(subgraph, pos, edge_labels=edge_labels, 
                                 font_size=20, font_color='black',label_pos=0.3,rotate=False, 
                                bbox=dict(facecolor='none', edgecolor='none', boxstyle='round,pad=0.1'),
                                clip_on=True)
    
    
    # Create a custom legend
    legend_labels = ['>='"{:.2f}".format(bb[0].min()+3* boundries_color)
                     ,'>='"{:.2f}".format(bb[0].min()+2* boundries_color)
                     ,'>='"{:.2f}".format(bb[0].min()+1* boundries_color)
                     ,'>='"{:.2f}".format(bb[0].min()+0* boundries_color)]
    legend_colors = edge_colors[:len(legend_labels)]
    legend_elements = [Line2D([0], [0], color=color, lw=4, label=label) for color, label in zip(legend_colors, legend_labels)]
    
    # Show the legend with adjusted title fontsize
    plt.legend(handles=legend_elements, title="Flow/Capacity", title_fontsize=40, loc='best', fontsize=40)
    
    # Add title and grid
    plt.title("Network Visualization with Upgraded Links", fontsize=70)  # Add and customize the title
    plt.grid(True)
    plt.grid('minor')
    plt.axis('on')

    # Show the plot
    plt.ion() # to make the plot interactice by clicking on. The numbers will be more clear. 
    plt.show()

# To Use: 
# network_visualization_upgraded (G = G, pos=pos, link_flow=link_flows, capacity_new=capacity ,link_select=links_selected)