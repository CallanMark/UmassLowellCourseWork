# [Author: Mark Callan]


import networkx as nx
import pandas as pd
import scipy.stats as stats 
from scipy.stats import pearsonr
import numpy as np
import matplotlib.pyplot as plt
#uncomment the following imports for the extra credit part
#from gensim.models.ldamodel import LdaModel
#from gensim.corpora.dictionary import Dictionary
#import nltk
#from nltk.tokenize import word_tokenize
#nltk.download('punkt') # uncomment if needed.


# part 1: weak tie analysis
def weaktie_analysis(LCC):
    # [Fill in to identify tie strength in LCC]

    degree_centrality = nx.degree_centrality(LCC)
    centrality_values = list(degree_centrality.values())
    sorted_degree_centrality = sorted(degree_centrality.items(), key=lambda x: x[1])
    
    threshold_weak = np.percentile(centrality_values,5) # Adjust to change weaktie threshold 
    threshold_strong = np.percentile(centrality_values,95) # Adjust to change threshold for strong ties 

    weak_ties = [node for node , centrality in degree_centrality.items() if centrality <= threshold_weak]
    strong_ties = [node for node , centrality in degree_centrality.items()if centrality >= threshold_strong] 

    # Plot orginal LCC here 

    LCC_weak_removed = LCC.copy()
    LCC_weak_removed.remove_nodes_from(weak_ties)
    LCC_weak_strong_removed = LCC_weak_removed.copy()
    LCC_weak_strong_removed.remove_nodes_from(strong_ties) # No nodes 
    
    print("-------------------------------------------------------------------------")
    print("                      Weak and Strong ties                                     ")    
    print("LCC length before removal : ", LCC) 
    print("LCC length after weak tie removal : " , LCC_weak_removed)
    print("LCC length after weak and strong tie removal " , LCC_weak_strong_removed)
    print(f"Lowest Degree Centrality Nodes:", sorted_degree_centrality[:5])  # First 5 (lowest)
    print("Highest Degree Centrality Nodes:", sorted_degree_centrality[-5:])  # Last 5 (highest)
    # [Fill in to plot the the effect of weak/strong tie removal on the size of LCC]

    plt.figure(figsize=(15,5)) 
    plt.subplot(1, 3, 1)
    nx.draw(LCC, node_size=2, edge_color="gray", alpha=0.6)
    plt.title("Original Network")

    plt.subplot(1, 3, 2)
    nx.draw(LCC_weak_removed, node_size=2, edge_color="gray", alpha=0.6)
    plt.title("After Weak Tie Removal")
    
    plt.subplot(1, 3, 3)  
    nx.draw(LCC_weak_strong_removed, node_size=2,edge_color="gray", alpha = 0.6)
    plt.title("After weak and strong tie removal ")
    
    plt.tight_layout() 
    plt.show()


    return


# Part 2: centrality
def centrality_analysis(LCC):
    degree = nx.degree_centrality(LCC)
    closeness = nx.closeness_centrality(LCC)
    betweenness = nx.betweenness_centrality(LCC)

    degree_vals = list(degree.values())
    closeness_vals = list(closeness.values())
    betweenness_vals = list(betweenness.values())

    # [Fill in to compute Pearson correlation coefficients between each pair of centrality measures]
    
    pearson_degree_closeness = pearsonr(degree_vals,closeness_vals)
    pearsonr_degree_betweeness = pearsonr(degree_vals,betweenness_vals)
    pearson_closeness_betweeness = pearsonr(closeness_vals,betweenness_vals)

    print("-------------------------------------------------------------------------")
    print("                      Pearson values                                     ")
    print("Pearson coefficent : (degree and closeness) :" ,pearson_degree_closeness[0])
    print("Pearson coefficent P value : (degree and closeness) :" ,pearson_degree_closeness[1])
    print("Pearson coefficent (degree and betweeness) :", pearsonr_degree_betweeness[0])
    print("Pearson coefficent P value (degree and betweeness) :", pearsonr_degree_betweeness[1])
    print("Pearson coefficent (closenesss and betweenesss) : ", pearson_closeness_betweeness[0])
    print("Pearson coefficent P value(closenesss and betweenesss) : ", pearson_closeness_betweeness[1])



    return


# part 3: optional: research evolution
def research_evolution_analysis(G):
    # [Fill in for extra credit task]
    # example code to apply LDA
    # texts = [word_tokenize(text) for text in papers]
    # dictionary = Dictionary(texts)
    # corpus = [dictionary.doc2bow(text) for text in texts]
    # lda = LdaModel(corpus=corpus, id2word=dictionary, num_topics=10)

    pass


def main():
    # loading the network and node embeddigns
    G = nx.read_graphml('aclbib.graphml')
    embeddings = pd.read_csv('embeddings.csv', index_col=0)

    # retrieving the LCC in G
    LCC = G.subgraph(max(nx.connected_components(G), key=len))

    # weak tie analysis
    weaktie_analysis(LCC)  

    # centrality analysis
    centrality_analysis(LCC)
    # [Uncomment the next lines if you're doing the extra credit task]
    # # research evolution analysis
    # research_evolution_analysis(G)


if __name__ == '__main__':
    main()
