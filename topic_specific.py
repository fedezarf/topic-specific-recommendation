
"""
Created on Sat Nov 19 14:48:44 2016

@author: Federico
"""

import time
import sys


from TopicSpecificRecommendation import TopicSpecificRecommendation as TSR
from Utils import create_graph_set_of_users_set_of_items as txt_to_graph


def main():
    if len(sys.argv) < 2:
        print "Wrong number of arguments."
        return
    
    print
    print "Start Personal Recommandation for each user in the Test Set."
    print "Current time: " + str(time.asctime(time.localtime()))
    
    train_graph_filename = str(sys.argv[1])
    test_graph_filename = str(sys.argv[2])
    
    
    print "              " + train_graph_filename
    print "              " + test_graph_filename
    print
    
    
    train_graph = txt_to_graph(train_graph_filename)
    test_graph = txt_to_graph(test_graph_filename)
    
    print
    print
    print "-----------------------------------------------"
    print "Current data: "
    print "              " + train_graph_filename
    print "              " + test_graph_filename
    print
    
    print "Current time: " + str(time.asctime(time.localtime()))
    
    print " #Users in Test Graph= " + str(len(test_graph['users']))
    print " #Items in Test Graph= " + str(len(test_graph['items']))
    print " #Nodes in Test Graph= " + str(len(test_graph['graph']))
    print " #Edges in Test Graph= " + str(test_graph['graph'].number_of_edges())
    print "Current time: " + str(time.asctime(time.localtime()))
    print
    print
    
    print " #Users in Test Graph= " + str(len(train_graph['users']))
    print " #Items in Test Graph= " + str(len(train_graph['items']))
    print " #Nodes in Test Graph= " + str(len(train_graph['graph']))
    print " #Edges in Test Graph= " + str(train_graph['graph'].number_of_edges())
    print "Current time: " + str(time.asctime(time.localtime()))
    print
    print
    
    print "Create Item-Item-Weighted Graph."
    print "Current time: " + str(time.asctime(time.localtime()))
    
    Recommender = TSR(train_graph)
    
    print " #Nodes in Item-Item Graph= " + str(len(Recommender.item_item_graph))
    print " #Edges in Item-Item Graph= " + str(Recommender.item_item_graph.number_of_edges())
    print "Current time: " + str(time.asctime(time.localtime()))
    print
    print
    
    print
    print "Start Personal Recommandation for each user in the Test Set."
    print "Current time: " + str(time.asctime(time.localtime()))
    
    aNDCG_LOWER_BOUND, aNDCG_for_PERSONAL_recommendation = Recommender.calculate_dcg_metrics(test_graph)
    
    print "Current time: " + str(time.asctime(time.localtime()))
    print "Done."
    
    print
    print
    print "  average_normalized_DCG_LOWER_BOUND_over_all_training_set_test_set_couples                 = " + str(aNDCG_LOWER_BOUND)
    print "  average_normalized_DCG_for_PERSONAL_recommendation_over_all_training_set_test_set_couples = " + str(aNDCG_for_PERSONAL_recommendation)
    print
    
    
    
    

if __name__ == "__main__":
    main()