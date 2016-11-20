# Topic Specific Recommendation using Topic-Specific PageRank

In this project we are going to solve recommendation problems using link-analysis techniques. In particular, we will perform graph transformation and use Topic-Specific-PageRank on the transformed graph to obtain recommendations.
We will test our methods on MovieLens 100K Dataset. This dataset contains 100000 ratings of 943 users on 1682 movies. Each rating is a natural number in [1, 5]. 

## Using the code

The code expects in input two files: a training graph and a test graph. The two files must be in the format described here:
http://grouplens.org/datasets/movielens/100k/

The files uploaded in the respository were converted from the provided format to .txt files. The .txt files can be used to run the recommendation for every user and compute the evaluation metrics.

```
python topic_specific.py train_graph.txt test_graph.txt
```

## Description of the operations

In this project we will implement a particular link-analysis procedure for movie recommendation. 
First of all, it is fundamental to notice that we can model recommendation system data as a weighted bipartite graph, where nodes are users and items (movies for MovieLens dataset), and where an edge between nodes represents a rating of a user for an item.
In the picture we can see an example of this type of graph:

![User-Item](/images/graph1.png)

### First Step - User Independent part

The first step of the method consists on the creation of a reduced version of this bipartite graph. This reduced graph is a weighted graph containing only item nodes of the original bipartite graph (it is reduced in the number of nodes). 
An edge between two item nodes in the reduced graph is present when they are both connected to the same user node in the original bipartite graph. In other words, this means that in the original bipartite graph there is at least one user node that has an edge to both these item nodes. Equivalently, this means that in the dataset there is at least one user that has rated both items. This reduced graph is a weighted graph, where the weight on edges represents the number of distinct user nodes connected to both item nodes in the original bipartite graph.
This reduced graph is called Item-Item graph.
In the picture is represented the corresponding Item-Item graph of the previous bipartite graph:

![Item-Item](/images/graph2.png)


We must pay attention at the fact that the value of the ratings is not involved in the reduced Item-Item graph construction process.



https://en.wikipedia.org/wiki/Bipartite_graph

### Second Step - User Dependent part

The second step consists on performing Topic-Specific-PageRank on the reduced Item-Item graph, considering as nodes of the topic all items rated by the input user. To increase the personalization of the recommendation, the teleporting probability distribution among all nodes in the virtual topic must be biased using the rating values that the input user has associated to each rated item.
For example, the teleporting probability distribution for user_3 on the nodes of the reduced Item-Item graph is the following: {‘item_1’: 5/(5+1), ‘item_2’: 1/(5+1), ‘item_3’: 0}.

A method that creates the teleporting distribution vector has been implemented in the function: “create_preference_vector_for_teleporting(user_id, graph_users_items)” 

### Third Step - Refinement

This step consists in producing a sorted list of items in a descending order of Topic-Specific-PageRank score. Of course, this sorted list of items must not contain the items already rated by the input user. This sorted list is the output of the recommendation process.
“create_ranked_list_of_recommended_items(page_rank_vector_of_items, user_id, training_graph_users_items)”

### Fourth Step - Metrics

To test the quality of the recommendation method, we will use the Average Normalized Discounted Cumulative Gain metric (average nDCG).
First of all, to evaluate the nDCG for a particular user we need to compute the NCG for that user, and after, normalize the value dividing by the maximum DCN for that user. More formally:

![nDCG](/images/dcg.png)


In the experiment we will compute the DCG on all items in the test set of the user, sorted according to the ranking computed by the recommendation algorithm. 

Let’s clarify this with an example. 
Let’s assume now that, for a generic user ‘u’, we have in the test_set only these (item, rating) data: set([(4, 5), (7, 5), (23, 3), (1, 2), (5, 2)]). 

Let’s assume also that, always for the user ‘u’, the recommendation system has produced this recommended sorted list of items:
“[38, 5, 40, 29, 1, 15, 7, 9, 23, 4, 15, 6, 43, 21, 34, ...]”

(the length of this list is equals to the total number of item nodes in the training graph minus the number of items associated to the user ‘u’ in the training set). 

As described before, we have now to sort all the (item, rating) elements associated to the user ‘u’ in the test set, according to the order imposed by the recommended list of items:

[(5, 2), (1, 2), (7, 5), (23, 3), (4, 5)]. 

This final list is that one to consider for the DCG evaluation, let’s call this list “evaluation_list”.

The DCG of a particular user is defined as follow:

![DCG](/images/dcg1.png)


Where the index of an item is the position of the item itself in the evaluation_list.
From the previous example, we have:
DCG(‘u’)= 2 + 2 + 5/lg(3) + 3/lg(4) + 5/lg(5).

“discounted_cumulative_gain(user_id, sorted_list_of_recommended_items, test_graph_users_items)”

As explained before, to obtain a normalized version of DCG we have to divide the DCG of a user by the maximum DCG for that user.
“maximum_discounted_cumulative_gain(user_id, test_graph_users_items)” 


The nDCG gives automatically an idea about how far the predictor is form a perfect predictor, because a perfect predictor has an nDCG equals to one. It is also important to have a lower bound on the nDCG value. For this reason, the the python function “minimum_discounted_cumulative_gain(user_id, test_graph_users_items)”  has been implemented. 

### References

[1] A. Langville and C. Meyer,
           "A survey of eigenvector methods of web information retrieval."
           http://citeseer.ist.psu.edu/713792.html
           
[2] Page, Lawrence; Brin, Sergey; Motwani, Rajeev and Winograd, Terry,
           The PageRank citation ranking: Bringing order to the Web. 1999
           http://dbpubs.stanford.edu:8090/pub/showDoc.Fulltext?lang=en&doc=1999-66&format=pdf
           
Topic-specific PageRank
http://nlp.stanford.edu/IR-book/html/htmledition/topic-specific-pagerank-1.html
           
        









