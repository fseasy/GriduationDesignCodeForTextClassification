#/usr/bin/env python
#coding=utf-8

import sys
import argparse
import logging
import numpy as np
try :
    import cPickle as pickle
except :
    import pickle
    
from dict_data_handler import load_dict_data

logging.basicConfig(level=logging.INFO)

def load_data_under_dictdata(dict_data_f) :
    '''
    from dict data load the gram vector list
    input > dict_data_f : file object to dict data
    return > gram_vecs : dataset , matrix format
    '''
    logging.info("loading dict data from file '%s'" %(dict_data_f.name))
    words , vec_list , r , gram_n = load_dict_data(dict_data_f)
     
    gram_mat = np.mat(vec_list)
    logging.info("loading dict data done .")
    return gram_mat

def load_data_under_file(fpi) :
    dataset = []
    for line in fpi :
        parts = line.strip().split()
        flt_parts = map(float , parts)
        dataset.append(flt_parts)
    return np.mat(dataset)

def get_random_centroid(dataset , k) :
    dimension = dataset.shape[1]
    rand_centroid = np.mat(np.zeros((k,dimension)))
    for j in range(0,dimension) :
        minJ = np.min(dataset[: , j])
        maxJ = np.max(dataset[: , j])
        dist = maxJ - minJ 
        rand_centroid[: , j] = minJ + dist * np.random.rand(k,1) # random give the value in every y-aix 
    return rand_centroid

def dist_euclidean(v_a , v_b) :
    '''
    calc the distance of 2 vectors( Matrix format ) using euclidean distance
    input > v_a , v_b : 2 vectors in matrix format
    return > distance
    '''
    return np.sqrt(np.power((v_a - v_b) , 2).sum())

def kmeans(dataset , k , dist_func=dist_euclidean) :
    '''
    KMeans 
    input > dataset : dataset in numpy.Matrix format
            k : cluster num
            dist_func : distance function 
    '''
    logging.info("kmeans process ...")
    rand_centroids = get_random_centroid(dataset , k)
    data_num = dataset.shape[0]
    data_cluster_info = np.mat(np.zeros((data_num , 2))) # 2 cols , First is the cluster id , behind is the SE (square error)
    has_converged = False
    cluster_centroids = rand_centroids 
    ite_num = 1
    while not has_converged :
        
        logging.debug("Kmeans iterate %d ." %(ite_num))
        ite_num += 1
        
        has_converged = True
        # arrange the data to the class
        for data_idx in range(data_num) :
            min_dist = np.inf
            min_dist_centroid_idx = -1
            for centroid_idx in range(k) :
                dist = dist_func(dataset[data_idx,:] , cluster_centroids[centroid_idx,:])
                if min_dist > dist :
                    min_dist = dist
                    min_dist_centroid_idx = centroid_idx
            if data_cluster_info[data_idx,0] != min_dist_centroid_idx :
                has_converged = False
            data_cluster_info[data_idx,:] = min_dist_centroid_idx , min_dist**2 # square error is the distance square
        # get the new centroids
        #print cluster_centroids
        for centroid_idx in range(k) :
            data_in_this = dataset[ np.nonzero(data_cluster_info[:,0].A == centroid_idx)[0] ] # using filter to get the datasets of this cluster
            cluster_centroids[centroid_idx,:] = np.mean(data_in_this , axis=0) # get new centroid by get mean
    logging.info("kmeans porcess done .")
    return cluster_centroids , data_cluster_info

def bi_kmeans(dataset , k , dist_func=dist_euclidean) :
    logging.info('bisicting kmeans process ... ')
    data_num = dataset.shape[0]
    data_cluster_info = np.mat(np.zeros((data_num , 2)))
    #init cnetroid  and square error
    init_centroid = np.mean(dataset , axis=0).tolist()[0]
    cluster_centroids = [init_centroid ,]
    for data_idx in range(data_num) :
        data_cluster_info[data_idx , 1] = dist_func(dataset[data_idx , :] , init_centroid) **2
    while len(cluster_centroids) < k :
        min_sse = np.inf
        min_sse_split_idx = -1
        best_split_centroids = None
        best_split_cluster_info = None
        for i in range(len(cluster_centroids)) :
            data_in_this = dataset[ np.nonzero(data_cluster_info[:,0].A == i)[0] ]
            
            new_centroids , new_cluster_info = kmeans(data_in_this , 2 , dist_func)
            # calc sse
            splited_sse = np.sum(new_cluster_info[:,1])
            not_splited_cluster_info = data_cluster_info[ np.nonzero(data_cluster_info[:,0].A != i)[0] ]
            others_sse = np.sum(not_splited_cluster_info[:,1] )
            total_sse = splited_sse + others_sse
            if total_sse < min_sse :
                min_sse = total_sse
                min_sse_split_idx = i
                best_split_centroids = new_centroids
                best_split_cluster_info = new_cluster_info.copy()
        # has get the best split . now update the total centroid
        ## one of the new centroids (2 centroids) should be placed at the origin palce , and the other should be placed at the last of
        ## the cluster_centroids . It has the min modify 
        
        #print "min sse split idx " , min_sse_split_idx
        #print 'best centroids : ' , best_split_centroids
        cluster_centroids[min_sse_split_idx] = best_split_centroids[0].tolist()[0]
        cluster_centroids.append(best_split_centroids[1].tolist()[0])

        first_idx =  np.nonzero(best_split_cluster_info[:,0].A == 0 )[0]
        second_idx = np.nonzero(best_split_cluster_info[:,0].A == 1 )[0]
        best_split_cluster_info[ first_idx , 0] = min_sse_split_idx
        best_split_cluster_info[ second_idx , 0] = len(cluster_centroids) -1
        
        data_cluster_info[ np.nonzero(data_cluster_info[:,0].A == min_sse_split_idx)[0] ] = best_split_cluster_info
        #print cluster_centroids
        #print data_cluster_info
    logging.info("bisecting kmeans process done .")
    return np.mat(cluster_centroids) , data_cluster_info

def save_kmeans_rst(fpo , centroids , cluster_info) :
    logging.info("save kmeans result to '%s'" %(fpo.name))
    pickle.dump(centroids , fpo)
    pickle.dump(cluster_info , fpo)
    logging.info("done .")

def load_kmeands_rst(fpi) :
    logging.info("load kmeans result from '%s'" %(fpi.name))
    centroids = pickle.load(fpi)
    cluster_info = pickle.load(fpi)
    logging.info("done .")
    return centroids , cluster_info

def main(dict_data_f , k , out) :
    dataset = load_data_under_dictdata(dict_data_f)
    #dataset = load_data_under_file(dict_data_f)
    rand_centroid = get_random_centroid(dataset , k)
    #print rand_centroid
    #print dist_euclidean(rand_centroid[0,:] , rand_centroid[1,:])
    #centroids , cluster_info = kmeans(dataset , k)
    centroids , cluster_info = bi_kmeans(dataset,k)
    save_kmeans_rst(out , centroids , cluster_info)

if __name__ == "__main__" :
    argparser = argparse.ArgumentParser(description="kmeans approach under dict data")
    argparser.add_argument("--dictpath" , help="dict data pickle path" , default=sys.stdin ,  type=argparse.FileType('r'))
    argparser.add_argument("--k" , help="num of clusters " , default="10" , type=int)
    argparser.add_argument("--out" , help="cluster result output path" , default=sys.stdout , type=argparse.FileType('w'))
    
    args = argparser.parse_args()
    main(args.dictpath , args.k , args.out)
    args.dictpath.close()
    args.out.close()

