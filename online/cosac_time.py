import numpy as np
from geom_tm_puncture import geom_tm
from scipy.sparse import csr_matrix
import time
import os
import pickle

from sklearn.preprocessing import normalize
from sklearn.base import BaseEstimator, ClusterMixin


def load_sparse_csr(filename):
    loader = np.load(filename)
    return csr_matrix((loader['data'], loader['indices'], loader['indptr']),
                      shape=loader['shape'])


def print_topics(beta, vocab, ind=False, top_n=20):
    top_words = []
    top_ind = []
    K = beta.shape[0]
    for k in range(K):
        k_ind = np.argsort(-beta[k, :])[:top_n]
        top_ind.append(k_ind)
        top_words.append(vocab[k_ind].tolist())
    if ind:
        return top_ind
    else:
        return top_words


def save_topics(topics, path):
    with open(path, 'w') as f:
        for t in topics:
            f.write(' '.join([str(x).strip() for x in t]) + '\n')


def get_group_id(path_to_group):
    group_id = path_to_group.split('.')[-2].split('_')[-1]
    return int(group_id)


###################################
# Class Wrapper of the Algorithms #
###################################
class CosacParallel(BaseEstimator, ClusterMixin):
    def __init__(self, path_to_save, delta=0.6, prop_discard=0.5, prop_n=0.001):

        # cone angle (\omega in the paper)
        self.delta = delta
        # quantile to compute \mathcal{R}
        self.prop_discard = prop_discard
        # proportion of data to be used as outlier threshold - \lambda
        self.prop_n = prop_n
        self.path_to_save = path_to_save
        # Variables to be computed
        self.cent = None
        self.M = None
        self.cosac = None

    def save_cosac(self, year):
        path = self.path_to_save + 'cosac_topics'
        if not os.path.exists(path):
            print 'making directory ' + path
            os.makedirs(path)
            
        path = path + '/' + year + '.cosac'

        # Save the output as object
        with open(path, "wb") as fp:
            pickle.dump([self.cosac.sph_betas_, self.M, self.cent], fp)
        
        print 'Dumped ' + path + '\n'
        
    def process_group(self, time_group_path):
        """path to group at time
        """
        wdf = load_sparse_csr(time_group_path)
        # Normalize & Center
        wdfn = normalize(wdf, 'l1')
        self.cent = wdfn.mean(axis=0).A.flatten()
        self.M = wdf.shape[0]
        
        # Fit CoSAC
        t_s = time.time()
        self.cosac = geom_tm(toy=False, verbose=False, delta=self.delta, prop_n=self.prop_n, 
                             prop_discard=self.prop_discard, max_discard=wdf.shape[0])
        if self.M > 100:
            self.cosac.fit_all(wdfn, self.cent, it=30)
            t_e = time.time()
            print 'For path ' + time_group_path + ' CoSAC took %f seconds and found %d topics in %d documents' % \
                  (t_e-t_s, self.cosac.K_, self.M)
            
        else:
            self.cosac.K_ = 0
            print 'Ignoring: Only %d documents at path ' % (self.M) + time_group_path 
        
        if self.cosac.K_ == 0:
                self.cosac.sph_betas_ = []
        
        year = time_group_path.split('/')[-1].split('.')[0]
        
        self.save_cosac(year)