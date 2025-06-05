## Mark Callan
import networkx as nx
import numpy as np
from node2vec import Node2Vec # type: ignore
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
from sklearn.preprocessing import StandardScaler

# Load edge list
G = nx.read_edgelist("edges.txt", nodetype=str)
print("Loaded Graph with ", G.number_of_nodes(), "nodes and " ,G.number_of_edges() ," edges" )

####[optional] Your code here: add node features

try:
    # Read in feat.txt 
    features_df = pd.read_csv("feat.txt", sep='\s+', header=None, index_col=0)
    features_df.index = features_df.index.astype(str)
    
    # Make features numeric so we can work with them 
    features_df = features_df.fillna(0).astype(float)
    
    # Create features as dictionary for nodes in the graph
    features = {node: features_df.loc[node].values for node in features_df.index if node in G.nodes()}
    
    # Normalize oour features
    scaler = StandardScaler()
    feature_matrix_scaled = scaler.fit_transform(list(features.values()))
    features = dict(zip(features.keys(), feature_matrix_scaled))
    
    # load feature names
    try:
        with open("feat_names.txt", "r") as f:
            feature_names = [line.strip() for line in f]
            print(f"Loaded {len(feature_names)} feature names")
    except:
        feature_names = None
        print("feat_names.txt not found")
except (FileNotFoundError, ValueError) as e:
    features = None
    print(f"Error loading feat.txt: {e}. Using structure-only embeddings.")


# Node2Vec
#### modify Node2Vec parameters to get better high quality embeddings. 
#### The values you see below are randomly chosen
#### the workers parameter is not part of node2vec, it controls how many CPU 
#### cores (threads) are used in parallel. Increasing it speeds up the training. 
node2vec = Node2Vec(G, dimensions=128, walk_length=80, num_walks=10,p =0.25 , q=0.25, workers=4)

# Fit model
#### modify the parameters of fit to get better high quality embeddings. 
#### window=5 means that for each node in a walk, the model tries to predict 
#### neighboring nodes within 5 steps (before and after) in the walk.
#### what's the effect of larger or smaller window sizes?
model = node2vec.fit(window=10, min_count=1, batch_words=4)

# Save embeddings to CSV
embeddings = {node: model.wv[node] for node in G.nodes()}

# Factor in node features 
if features:
    feature_dim = next(iter(features.values())).shape[0]
    for node in embeddings:
        if node in features:
            # Concatenate node2vec embedding with normalized features
            embeddings[node] = np.concatenate([embeddings[node], features[node]])
        else:
            # Pad with zeros for missing features
            embeddings[node] = np.concatenate([embeddings[node], np.zeros(feature_dim)])
df = pd.DataFrame.from_dict(embeddings, orient='index')
df.index.name = 'node_id'
df.to_csv("node2vec_embeddings.csv")


# Load target node pairs
similarity_output = []
with open("target_nodes.txt", "r") as f:
    for line in f:
        node1, node2 = line.strip().split()
        if node1 in model.wv and node2 in model.wv:
            sim = cosine_similarity([model.wv[node1]], [model.wv[node2]])[0][0]
            similarity_output.append(f"{node1} - {node2}: {sim:.4f}")
        else:
            similarity_output.append(f"{node1} - {node2}: One or both nodes missing from embedding")
            

# Save similarity results
with open("similarity_results.txt", "w") as out:
    out.write("\n".join(similarity_output))
