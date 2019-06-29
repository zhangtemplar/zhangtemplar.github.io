---
layout: post
title: Graph Embedding
tags: deep-learning graph embedding deep-walk
---

![](https://pic3.zhimg.com/80/v2-ad4da5b234e34bf4cf19aa1ea511d496_hd.jpg)

Graphs are commonly used in different real-world applications, e.g., ocial networks are large graphs of people that follow each other. Graph embeddings are the transformation of property graphs to a vector or a set of vectors. Embedding should capture the graph topology, vertex-to-vertex relationship, and other relevant information about graphs, subgraphs, and vertices. More properties embedder encode better results can be retrieved in later tasks.

# [DeepWalk](https://arxiv.org/abs/1403.6652)

![](https://pic3.zhimg.com/80/v2-6c548cc39af4400988d04ed1104bb3c2_hd.jpg)

DeepWalk: Online Learning of Social Representations proposed in 2014 was one of most important papers for graph embedding idea. The idea is that:
- starting from a node, start a random walk on the graph;
- feed the sequence of nodes visited on the random walk into sequence embedding method, e.g., word2vec;

The most important step is the first one, more specifically, how to define the random walk. In DeepWalk, the transition probability from Node a to Node b is proportional to the the weight between a and b.

# [Node2vec](https://cs.stanford.edu/~jure/pubs/node2vec-kdd16.pdf)

![](https://pic2.zhimg.com/80/v2-20a6b345cfe45706b43db91a78ee5b69_hd.jpg)

Compared with DeepWalk, node2vec: Scalable Feature Learning for Networks improves homophily and structural equivalence by using a different transition probability:
- homophily: the nodes close to each other in the graph should have similar embedding vector;
- structural equivalence: the nodes which has similar neighbor structure should have similar embedding vector;

BFS prefers homophily and DFS prefers structural equivalence, which could be controlled by transition probablity. Especially, the transition probablity is computed as:
![](https://pic2.zhimg.com/80/v2-61287731efe14d38a7084fa2f77ec3c1_hd.jpg)

![](https://pic2.zhimg.com/80/v2-481056c49b3619ff679fe10ee38c24c1_hd.jpg)
where $w_vc$ is the weight of the edge between v and c; $d_tx$ is the distance between t and x. Smaller p encourages BFS; Smaller q encourages DFS.

# [EGES](https://arxiv.org/pdf/1803.02349)

![](https://pic2.zhimg.com/80/v2-740642a04298d289d19cd4225d062b5d_hd.jpg)

Enhanced Graph Embedding with Side Information improves DeepWalk by incorporating side information. In the recommendation system, code start is a chanllenging but common problem, where a new vertex has few connections to the existing vertices in the graph. DeepWalk cannot generate meaningful embedding for the new vertex. EGES propose to use side information to address such problem.

The idea of EGES is intuitive: using different side information to generate different embeddings; then applying average pool to combine those embeddings.
