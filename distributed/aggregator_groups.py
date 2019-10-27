import numpy as np
from group_matching import matching_upd
from scipy.sparse import csr_matrix
import time
import glob
import pickle
from sklearn.base import BaseEstimator, ClusterMixin
import os


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
class StreamGroupCosac(BaseEstimator, ClusterMixin):
    def __init__(self, path_cosac, save_path='./', tau0=3., tau1=3., gamma=1.,
                     verbose=True, vocab=None, init_topics=None, init_counts=None, init_cent=None, init_M=None):
        
        self.gamma = gamma
        self.verbose = verbose
        self.tau0 = tau0
        self.tau1 = tau1
        self.path_cosac = path_cosac + 'cosac_topics/'
        self.save_path = save_path + 'global_topics'
        
        if not os.path.exists(self.save_path):
            print 'making directory ' + self.save_path
            os.makedirs(self.save_path)
        
        self.save_path = self.save_path + '/'
        
        if vocab is not None:
            self.vocab = np.array(vocab)
        else:
            print 'No vocabulary passed'
            
        self.topic_counts_ = init_counts
        self.global_topics_ = init_topics
        
        self.M = init_M
        self.cent = init_cent
        
        self.global_topics_path_ = []
        self.topic_counts_path_ = []
        self.K_path_ = []

        self.cosac_topics = []
        self.cosac_centers = []
        self.document_sizes = []

    def path_and_save(self):
        
        # Update Path
        cur_K = self.global_topics_.shape[0]
        self.K_path_.append(cur_K)

        self.global_topics_path_.append(self.global_topics_)
        self.topic_counts_path_.append(self.topic_counts_)
        
        path_words = self.save_path + 'top_words'
        
        if not os.path.exists(path_words):
            print 'making directory ' + path_words
            os.makedirs(path_words)
            
        path_words = path_words + '/'
            
        # Save Topics
        if self.vocab is not None:
            print 'Saving topics '
            topics_a = print_topics(self.global_topics_, self.vocab)
            save_topics(topics_a, path_words + '_topics_%d' % len(topics_a))

        # Save the output as the object
        with open(self.save_path + 'topics_counts_M_cent', "wb") as fp:
            pickle.dump([self.global_topics_, self.topic_counts_, self.M, self.cent], fp)
            
    def process_group(self, group_path): # path to group at time
        
        group = group_path.split('/')[-1].split('.')[0]
        
        cosac_path = self.path_cosac + group + '.cosac'
        
        flag = True
        while flag:
            try:
                with open(cosac_path, "rb") as fp:
                    topics, M, cent = pickle.load(fp)
                    if type(topics) is list:
                        topics = np.array(topics)
                flag = False
            except:
                print 'Waiting for \n' + group_path
                time.sleep(2)
        
        print 'Loaded \n' + cosac_path
        
        return topics, cent, M
    
    def process_all_groups(self, path, it):
        """path to time
        """
        group_files = glob.glob(path + '/*.npz')
        group_files.sort(key=lambda x: get_group_id(x))
        
        self.cosac_topics = []
        self.cosac_centers = []
        self.document_sizes = []

        for group_path in group_files:
            cosac_topics_j, cent_j, M_j = self.process_group(group_path)
            if len(cosac_topics_j) > 0:
                self.cosac_topics.append(cosac_topics_j)
                self.cosac_centers.append(cent_j)
                self.document_sizes.append(M_j)
        
        # Update Global Center
        if self.cent is None:
            self.M = sum(self.document_sizes)
            self.cent = sum([c*m for c, m in zip(self.cosac_centers, self.document_sizes)])/self.M
        else:
            self.cent = (self.cent*self.M +
                         sum([c*m for c,m in
                              zip(self.cosac_centers, self.document_sizes)]))/(self.M + sum(self.document_sizes))
            self.M += sum(self.document_sizes)

        print 'Starting aggregation'
        
        # Update Global Topics
        t_s = time.time()
        self.global_topics_, self.topic_counts_ = matching_upd(self.cosac_topics, self.cent,
                                                               self.tau1, self.gamma, it=it)
        t_e = time.time()
        print 'Aggregation is done and took %f seconds; total of %d global topics\n' % \
              (t_e-t_s, self.global_topics_.shape[0])
        print '--------------------------------------------------------------------'

        if self.verbose:
            self.path_and_save()
