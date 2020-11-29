import matplotlib.pyplot as plt
import numpy as np
import pickle
import os
import re
from scipy.spatial import distance


def get_APs(queries):
    """
    Queries are a list of sorted by distance metrics. Each element is a pair containing 
    the distance value and a boolean value specifying if it's a relevant match or not. 

    Returns the mean average precesion (mAP). 
    """
    APs = []
    for q in queries:
        p_at_k = []
        count_positive = 0
        k = 0
        while k < len(q):
            if q[k][1]:
                count_positive += 1
                p_at_k.append(count_positive/(k+1))
            k += 1
        
        APs.append(np.mean(np.array(p_at_k)))

    return APs

def get_mAP(queries):
    """
    Queries are a list of sorted by distance metrics. Each element is a pair containing 
    the distance value and a boolean value specifying if it's a relevant match or not. 

    Returns the mean average precesion (mAP). 
    """

    APs = get_APs(queries)

    APs = np.array(APs)
    return np.array([ np.mean(APs), np.var(APs)]), np.quantile(APs, [0.2,0.5,0.8])

def parse_output_data(path):
    """
    Parse output data generated from metric learning model.
    Expecting a directory containing pickel files with 
    """
    def atoi(text):
        return int(text) if text.isdigit() else text
    def natural_keys(text):
        return [ atoi(c) for c in re.split('(\d+)',text) ]

    epochs = []
    for root, dirs, files in os.walk(path):
        files.sort(key=natural_keys)
        for name in files:
            with open(os.path.join(root, name), 'rb') as pickle_file:
                vals = pickle.load(pickle_file)
            if len(vals) > 0 and "data" in vals.keys():
                vals = [val.get('data') for val in vals]
            epochs.append(vals)
    
    return epochs

def generate_ordered_queries(epochs, with_filepath=False):
    """
    Iterate through all examples to compare them all to each other. Each feature vector 
    will be the query and compared to all other idx columns. The 'idx' represents the number
    of examples per class and the 'key' represents the class (or instance) id name

    One epoch is a dictionnary of feature vector lists for each instance:
              idx
            --->
    key  |  a1, a2, a3, ...
         |  b1, b2, b3, ...
         v  ...
    """
    def sort_val(data):
        return data[0]

    epoch_queries = []
    for ep in epochs:
        accuracy = 0
        queries = []
        for query_key, q_vals in ep.items():
            for query_idx in range(len(q_vals)): 
                query_val_list = []
                query = q_vals[query_idx]
                for feat_key, f_vals in ep.items():
                    for feat_idx in range(len(f_vals)):
                        if feat_idx != query_idx:
                            feat_vec = ep.get(feat_key)[feat_idx]
                            is_same = feat_key == query_key
                            if with_filepath:
                                similitude = np.matmul(query[0], feat_vec[0].T)
                                euc_dist = distance.euclidean(query[0], feat_vec[0])
                                query_val_list.append((similitude, is_same, (query[1], feat_vec[1]), (query_idx, feat_idx), euc_dist))
                            else:
                                similitude = np.matmul(query, feat_vec.T)
                                euc_dist = distance.euclidean(query, feat_vec)
                                query_val_list.append((similitude, is_same, euc_dist))
            
                query_val_list.sort(key=sort_val, reverse=True)
                queries.append(query_val_list)

        epoch_queries.append(queries)

    return epoch_queries


def test_mAP_outputs(folder_path=None, epochs=None, with_filepath=False):
    """
    Provide a folder_path to build a list of ordered queries from files, OR provide directly the epochs outputs from,
    for example, run_test_model to calculate mAP. The threshold is a placement threshold in the ordered list.
    
    Returns a list of mAP per epoch. 
    """
    if folder_path is not None:
        epochs = parse_output_data(folder_path)
    epoch_queries = generate_ordered_queries(epochs, with_filepath)
    epoch_mAPs = []
    quantiles = []
    for ep in epoch_queries:
        mAP_o, quant_o = get_mAP(ep)
        epoch_mAPs.append(mAP_o)
        quantiles.append(quant_o)

    return epoch_mAPs, quantiles

def display_relevantcy_barchart(folder_path=None, epochs=None, with_filepath=True):
    """
    Display a histogram chart for each epoch of both positive and negative distances. 
    """
    if folder_path is not None:
        epochs = parse_output_data(folder_path)
    epoch_queries = generate_ordered_queries(epochs, with_filepath)
    for queries in epoch_queries:
        positives_dist = []
        negatives_dist = []
        positives_similitude = []
        negatives_similitude = []
        for q in queries:
            for img in q:
                if img[1]:
                    positives_similitude.append(img[0])
                    positives_dist.append(img[-1])
                else:
                    negatives_similitude.append(img[0])
                    negatives_dist.append(img[-1])

        fig, (dist, eucl) = plt.subplots(nrows=1, ncols=2)            
        n, bins, patches = dist.hist(positives_similitude, 50, facecolor='g', alpha=0.5, label="Relevent")
        n, bins, patches = dist.hist(negatives_similitude, 50, facecolor='r', alpha=0.5, label="Non-Relevent")
        dist.set_xlabel("Similitude score")
        dist.set_ylabel("Number of examples")

        n, bins, patches = eucl.hist(positives_dist, 50, facecolor='g', alpha=0.5, label="Relevent")
        n, bins, patches = eucl.hist(negatives_dist, 50, facecolor='r', alpha=0.5, label="Non-Relevent")
        eucl.set_xlabel("Euclidean distance")
        eucl.set_ylabel("Number of examples")

        plt.legend()
        plt.grid(True)
        plt.show()


def display_best_and_worst_ap(folder_path=None, epochs=None, nb_images=19):
    """
    This function display the 8 first image of the best and worst queries of each epoch.

    Provide a folder_path to build a list of ordered queries from files, OR provide directly the epochs outputs from,
    for example, run_test_model to calculate mAP. The threshold is a placement threshold in the ordered list.
    
    Returns a list of mAP per epoch. 
    """
    import matplotlib.image as mpimg

    if folder_path is not None:
        epochs = parse_output_data(folder_path)
    epoch_queries = generate_ordered_queries(epochs, with_filepath=True)
    best_ap = None
    worst_ap = None
    for ep in epoch_queries:
        APs = get_APs(ep)
        for ap_idx in range(len(APs)):
            if best_ap is None or APs[best_ap] < APs[ap_idx]:
                best_ap = ap_idx
            if worst_ap is None or APs[worst_ap] > APs[ap_idx]:
                worst_ap = ap_idx

        path_idx = 2
        compared_idx = 1  
        f, axarr = plt.subplots(2,nb_images+1) 
        img = mpimg.imread(ep[best_ap][0][path_idx][0])
        axarr[0][0].imshow(img)
        img = mpimg.imread(ep[worst_ap][0][path_idx][0])
        axarr[1][0].imshow(img)
        for i in range(nb_images):
            img = mpimg.imread(ep[best_ap][i][path_idx][compared_idx])
            axarr[0][i+1].imshow(img)
        for i in range(nb_images):
            img = mpimg.imread(ep[worst_ap][i][path_idx][compared_idx])
            axarr[1][i+1].imshow(img)
        plt.axis('off')
        plt.show()

    return None, None
