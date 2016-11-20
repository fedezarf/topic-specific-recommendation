"""
Recommendation System.
"""

# Author: Federico Zarfati <fz234@cornell.edu>

import csv
import networkx as nx
 

def create_graph_set_of_users_set_of_items(user_item_ranking_file):
    """Parse the input file.
        
        Creation of a dictionary containing the NetworkX graph object, the list
        of the users and the list of the items.
    
    """
    graph_users_items = {}
    all_users_id = set()
    all_items_id = set()
    g = nx.DiGraph()
    input_file = open(user_item_ranking_file, 'r')
    input_file_csv_reader = csv.reader(input_file, delimiter='\t', 
                                       quotechar='"', quoting=csv.QUOTE_NONE)
    for line in input_file_csv_reader:
        user_id = int(line[0])
        item_id = int(line[1])
        rating = int(line[2])
        g.add_edge(user_id, item_id, weight=rating)
        all_users_id.add(user_id)
        all_items_id.add(item_id)
    input_file.close()
    graph_users_items['graph'] = g
    graph_users_items['users'] = all_users_id
    graph_users_items['items'] = all_items_id
    return graph_users_items