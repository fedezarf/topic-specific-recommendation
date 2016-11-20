"""
Recommendation System.
"""

# Author: Federico Zarfati <fz234@cornell.edu>


import networkx as nx
import scipy.sparse
import numpy as np
import scipy.sparse
from networkx.exception import NetworkXError
from networkx.algorithms import bipartite


class TopicSpecificRecommendation():
    """Topic Specific Recommendation based on Topic-Specific PageRank.
    
    Performs graph transformation and uses Topic-Specific-PageRank on the 
    transformed graph to obtain recommendations for every user in the graph.
    
    Tested on MovieLens 100K Dataset. 
    The dataset contains 100000 ratings of 943 users on 1682 movies. 
    Each rating is a natural number in [1, 5].
    
    For more info on the dataset:
    http://grouplens.org/datasets/movielens/100k/
    
    
    Parameters
    ----------
    graph_users_items : 
        The graph represented with a dictionary where 
        graph_users_items['graph'] is a NetworkX object, 
        graph_users_items['users'] = all_users_id is a set of users IDs
        graph_users_items['items'] = all_items_id is a set of items IDs
        
    Attributes
    ----------
    item_item_graph : 
        Bipartite weighted projected graph generated from the graph_users_items
        graph. 
    
    M : 
        A scipy sparse matrix of the item_item graph to speed up the Page-Rank 
        computation.
        
    
    """
    def __init__(self, graph_users_items):
        self.graph_users_items = graph_users_items
        self.item_item_graph = self.create_item_item_graph()
        self.M = nx.to_scipy_sparse_matrix(self.item_item_graph, 
                                           nodelist=self.item_item_graph.nodes(), 
                                           weight='weight', dtype=float)
            
    def create_item_item_graph(self):
        """Generate the item-item graph.
        
        Wrapper of the NetworkX function to generate the bipartite 
        weighted projected graph. 
    
        """
        
        item_item_graph = nx.Graph()
        
        item_item_graph = bipartite.weighted_projected_graph(
                            self.graph_users_items["graph"].to_undirected(), 
                            self.graph_users_items["items"])
                            
        
        
        return item_item_graph 
        
        
    def pagerank_scipy_mod(self, alpha=0.85, personalization=None,
                   max_iter=100, tol=1.0e-6, weight='weight',
                   dangling=None):
        """Return the PageRank of the nodes in the graph.
    
        ADAPTED FROM THE NETWORKX LIBRARY.
        
        https://networkx.github.io/documentation/networkx-1.10/
        _modules/networkx/algorithms/link_analysis/pagerank_alg.html#pagerank
        
        Given that for our purposes we need to call the pagerank function for
        every user in the graph the function was modified in order compute the 
        scipy matrix of the graph outside the pagerank function. This leads to
        a significant increase in performance.
        
        
        PageRank computes a ranking of the nodes in the graph G based on
        the structure of the incoming links. It was originally designed as
        an algorithm to rank web pages.
    
        Parameters
        ----------
    
        alpha : float, optional
          Damping parameter for PageRank, default=0.85.
    
        personalization: dict, optional
           The "personalization vector" consisting of a dictionary with a
           key for every graph node and nonzero personalization value for each
           node. By default, a uniform distribution is used.
    
        max_iter : integer, optional
          Maximum number of iterations in power method eigenvalue solver.
    
        tol : float, optional
          Error tolerance used to check convergence in power method solver.
    
        weight : key, optional
          Edge data key to use as weight.  If None weights are set to 1.
    
        dangling: dict, optional
          The outedges to be assigned to any "dangling" nodes, i.e., nodes without
          any outedges. The dict key is the node the outedge points to and the dict
          value is the weight of that outedge. By default, dangling nodes are given
          outedges according to the personalization vector (uniform if not
          specified) This must be selected to result in an irreducible transition
          matrix (see notes under google_matrix). It may be common to have the
          dangling dict to be the same as the personalization dict.
    
        Returns
        -------
        pagerank : dictionary
           Dictionary of nodes with PageRank as value
    
    
        Notes
        -----
        The eigenvector calculation uses power iteration with a SciPy
        sparse matrix representation.
    
        This implementation works with Multi(Di)Graphs. For multigraphs the
        weight between two nodes is set to be the sum of all edge weights
        between those nodes.
    
        See Also
        --------
        pagerank, pagerank_numpy, google_matrix
    
        References
        ----------
        .. [1] A. Langville and C. Meyer,
           "A survey of eigenvector methods of web information retrieval."
           http://citeseer.ist.psu.edu/713792.html
        .. [2] Page, Lawrence; Brin, Sergey; Motwani, Rajeev and Winograd, Terry,
           The PageRank citation ranking: Bringing order to the Web. 1999
           http://dbpubs.stanford.edu:8090/pub/showDoc.Fulltext?lang=en&doc=1999-66&format=pdf
        """
    
        G = self.item_item_graph
        N = len(G)
        if N == 0:
            return {}
    
        nodelist = G.nodes()
        
        # the scipy matrix computed in the initialization of the object
        M = self.M
        
        S = scipy.array(M.sum(axis=1)).flatten()
        S[S != 0] = 1.0 / S[S != 0]
        Q = scipy.sparse.spdiags(S.T, 0, *M.shape, format='csr')
        M = Q * M
    
        # initial vector
        x = scipy.repeat(1.0 / N, N)
    
        # Personalization vector
        if personalization is None:
            p = scipy.repeat(1.0 / N, N)
        else:
            missing = set(nodelist) - set(personalization)
            if missing:
                raise NetworkXError('Personalization vector dictionary '
                                    'must have a value for every node. '
                                    'Missing nodes %s' % missing)
            p = scipy.array([personalization[n] for n in nodelist],
                            dtype=float)
            p = p / p.sum()
    
        # Dangling nodes
        if dangling is None:
            dangling_weights = p
        else:
            missing = set(nodelist) - set(dangling)
            if missing:
                raise NetworkXError('Dangling node dictionary '
                                    'must have a value for every node. '
                                    'Missing nodes %s' % missing)
            # Convert the dangling dictionary into an array in nodelist order
            dangling_weights = scipy.array([dangling[n] for n in nodelist],
                                           dtype=float)
            dangling_weights /= dangling_weights.sum()
        is_dangling = scipy.where(S == 0)[0]
    
        # power iteration: make up to max_iter iterations
        for _ in range(max_iter):
            xlast = x
            x = alpha * (x * M + sum(x[is_dangling]) * dangling_weights) + \
                (1 - alpha) * p
            # check convergence, l1 norm
            err = scipy.absolute(x - xlast).sum()
            if err < N * tol:
                return dict(zip(nodelist, map(float, x)))
        raise NetworkXError('pagerank_scipy: power iteration failed to converge '
                            'in %d iterations.' % max_iter)
    
    
        
    def create_preference_vector_for_teleporting(self, user_id):
        """Compute the preference vector for a given user.
        
        To increase the personalization of the recommendation, the teleporting 
        probability distribution among all nodes in the virtual topic must be 
        biased using the rating values that the input user has associated to 
        each rated items.
        For example, the teleporting probability distribution for user_3 on the 
        nodes of the reduced Item-Item graph is the following: 
        
        

        Args:
            user_id: The id of the user.

        Returns:
            A dictinary representing the preference vector of the given user. 

        """
        # Initialization of the preference vector with 0 as value
        preference_vector = {item : 0 for item in 
                                    self.graph_users_items['items']}
        
        # Function to insert numerator in the vector
        def try_to_rate(chosen_item):
            try:
                value = self.graph_users_items['graph'][user_id][chosen_item]['weight']
            except:
                value = 0
            
            return value  
           
        # Insert numerator for every user using try to rate  
        preference_vector = {item : try_to_rate(item) 
                                    for item in preference_vector}
         
        # Denominator to normalize
        den = sum(preference_vector.values())
        
        # Every element of the vector is computed using numerator and 
        # denominator of every item for the user
        preference_vector = {item : preference_vector[item]*1.0/den 
                                    for item in preference_vector}
    	
        return preference_vector 
        
        
    def create_ranked_list_of_recommended_items(self,page_rank_vector_of_items, 
                                               user_id):
        """Ranked list from the resulting page-rank vector for a given user.

        sorted list of items in a descending order of Topic-Specific-PageRank score. 
        Of course, this sorted list of items must not contain the items already 
        rated by the input user. This sorted list is the output of the 
        recommendation process. The list of items are sorted in descending order. 
        Most recommended first.


        Args:
            page_rank_vector_of_items: Resulting vector from page-rank.
            user_id: The id of the user.

        Returns:
            True if successful, False otherwise.

        """
        # This is a list of 'item_id' sorted in descending order of score.
        sorted_list_of_recommended_items = []
        
        # You can produce this list from a list of [item, score] 
        # couples sorted in descending order of score.
        
        # Delete movies already seen by the user
        for elem in self.graph_users_items["graph"].neighbors(user_id):
            del page_rank_vector_of_items[elem]
            
        # Create the list with elements [movie,rating]
        sorted_list_of_recommended_items = [[elem, 
                                             page_rank_vector_of_items[elem]] 
                                        for elem in page_rank_vector_of_items]
                                            
        # Sorting list in descending order based on rating
        sorted_list_of_recommended_items.sort(key=lambda x: (x[1], 
                                                             x[0]), 
                                                            reverse=True)
                                                            
        # Return a list of recommended items (movies)
        sorted_list_of_recommended_items = [elem[0] 
                                for elem in sorted_list_of_recommended_items]
    
        return sorted_list_of_recommended_items
    
    
    def recommended_for_user(self, user_id):
        """Recommendation for a given user.

        Given a user id we compute:
            - The preference vector of the given user.
            - The page-rank vector of the given user.
            - The list of items recommended for the user.

        Args:
            user_id: The id of the user.
            

        Returns:
            The list of items sorted in descending order. Most recommended
            first.

        """
        preference_vector = self.create_preference_vector_for_teleporting(user_id)
        personalized_pagerank_vector_of_items = self \
                                                .pagerank_scipy_mod(personalization=preference_vector)
        return self.create_ranked_list_of_recommended_items(personalized_pagerank_vector_of_items, user_id)
           
           
    @staticmethod
    def discounted_cumulative_gain(user_id, sorted_list_of_recommended_items, test_graph_users_items):
        """Compute the Discounted Comulative Gain for a given user.

        
        
        https://en.wikipedia.org/wiki/Discounted_cumulative_gain

        Args:
            user_id1: The id of the user.
            sorted_list_of_recommended_items: Output of the recommendation system.
            test_graph_users_items: The test graph.

        Returns:
            The DCG metric.

        """
        dcg = 0.
        
        
        # List of [movie,rating] for user
        item_rating = [(elem[1],elem[2]["weight"]) 
                        for elem in test_graph_users_items["graph"] \
                        .edges(user_id, data=True) 
                        if elem[1] in sorted_list_of_recommended_items]
        # Dictionary to sort
        dic = dict([(elem[0], elem[1]) for elem in item_rating])
        # Sorting based on sorted list
        sorted_items = sorted(dic.items(), key=lambda i:sorted_list_of_recommended_items.index(i[0]))
        # List of rankings
        r = [int(i[1]) for i in sorted_items]
        # Dcg computation based on rankings list
        dcg = r[0] + np.sum(r[1:] / np.log2(np.arange(2, len(r)+1)))
    
    
        return dcg
   
   
    @staticmethod
    def minimum_discounted_cumulative_gain(user_id, test_graph_users_items):
        """Minimum Discounted Comulative Gain.

        It is also important to have a lower bound on the nDCG value. 
        For this reason, we must compute the minimum DCG. 

        Args:
            user_id: The id of the user.
            test_graph_users_items: The test graph.

        Returns:
            The Minimum DCG.

        """
        dcg = 0.
        
        # List of [movie,rating] for user
        item_rating = [(elem[1],elem[2]["weight"]) 
                        for elem in test_graph_users_items["graph"].edges(user_id, data=True)]
        # Order reverse false for minimum dcg
        sorted_rank = sorted(item_rating, key=lambda x: (x[1], x[0]), reverse=False)
        sorted_rank = [int(i[1]) for i in sorted_rank]
        # Min dcg
        dcg = sorted_rank[0] + np.sum(sorted_rank[1:] / np.log2(np.arange(2, len(sorted_rank) + 1)))
        
        return dcg



    @staticmethod
    def maximum_discounted_cumulative_gain(user_id, test_graph_users_items):
        """Maximum Discounted Comulative Gain.

        To obtain a normalized version of DCG you have to divide the DCG of 
        a user by the maximum DCG for that user.
 

        Args:
            user_id: The id of the user.
            test_graph_users_items: The test graph.

        Returns:
            The Maximum DCG.

        """
        dcg = 0.
        
        # Reorder
        item_rating = [(elem[1],elem[2]["weight"]) 
                        for elem in test_graph_users_items["graph"].edges(user_id, data=True)]
        # Reorder reverse for maximum dcg
        sorted_rank = sorted(item_rating, key=lambda x: (x[1], x[0]), reverse=True)
        sorted_rank = [int(i[1]) for i in sorted_rank]
        # Max dcg
        dcg = sorted_rank[0] + np.sum(sorted_rank[1:] / np.log2(np.arange(2, len(sorted_rank) + 1)))
        return dcg
        
        
    def calculate_dcg_metrics(self, test_graph_users_items):
        """Perform the recommendation for every user and calculate the DCG.

        For every user in the training graph the recommendation is computed.
        Then, the recommendation is tested with the test graph and the metrics
        computed.
        

        Args:
            test_graph_users_items: The test graph.
            

        Returns:
            Tuple: (avgDCG lower bound, avgNDCG) 

        """
        sum_of_normalizedDCG_for_PERSONAL_recommendation = 0.
        sum_of_normalizedDCG_LOWER_BOUND                 = 0.
        
        
        for user_id in test_graph_users_items['users']:
            
            
            # For every user we calculate the preference vector and PageRank
            recommended_items = self.recommended_for_user(user_id)
            current_dcg_for_PERSONAL_recommendation = self.discounted_cumulative_gain(user_id, 
                                                                                      recommended_items, 
                                                                                      test_graph_users_items)
		 
    		 # Minimum Discounted Cumulative Gain.
            current_minimum_dcg = self.minimum_discounted_cumulative_gain(user_id, 
                                                                          test_graph_users_items)
		 
		 # Maximum Discounted Cumulative Gain.
            current_MAXIMUM_dcg = self.maximum_discounted_cumulative_gain(user_id, 
                                                                          test_graph_users_items)
		
		
            sum_of_normalizedDCG_LOWER_BOUND += current_minimum_dcg/float(current_MAXIMUM_dcg)
            sum_of_normalizedDCG_for_PERSONAL_recommendation += current_dcg_for_PERSONAL_recommendation/float(current_MAXIMUM_dcg)
            
        avg_normalized_DCG_LOWER_BOUND                 = sum_of_normalizedDCG_LOWER_BOUND / float(len(test_graph_users_items['users']))
        avg_normalized_DCG_for_PERSONAL_recommendation = sum_of_normalizedDCG_for_PERSONAL_recommendation / float(len(test_graph_users_items['users']))
        
        return (avg_normalized_DCG_LOWER_BOUND, avg_normalized_DCG_for_PERSONAL_recommendation)
	
            
