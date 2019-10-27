import glob
import pickle
import numpy as np
import json
import sys
from scipy.sparse import csr_matrix, vstack
from sklearn.preprocessing import normalize
from optparse import OptionParser
from all_func import perplexity


def parse_args():
    parser = OptionParser()
    parser.set_defaults(test_path=None, topics_path=None, algorithm=None)
    
    parser.add_option("--test_path", type="string", dest="test_path",
                      help="path to test data")
    parser.add_option("--topics_path", type="string", dest="topics_path",
                      help="path storing the topics pre-trained")
    parser.add_option("--algorithm", type="string", dest="algorithm",
                      help="algorithm used to train the topics")

    (options, args) = parser.parse_args()
    
    return options


def check_args(test_path, topics_path, algorithm):
    if test_path is None:
         raise ValueError('<test_path> is NOT specified')
    if topics_path is None:
         raise ValueError('<topics_path> is NOT specified' )
    if algorithm is None:
         raise ValueError('<algorithm> is NOT specified')


def clean_topics(topics):
    topics[topics < 0] = 0
    topics = normalize(topics, 'l1')
    return topics


def load_sparse_csr(filename):
    loader = np.load(filename)
    return csr_matrix((loader['data'], loader['indices'], loader['indptr']),
                      shape=loader['shape'])


def main():
    options = parse_args()
    print options
    
    test_path = options.test_path
    topics_path = options.topics_path
    algorithm = options.algorithm
    
    check_args(test_path, topics_path, algorithm)
   
    # Load test data
    test_files = glob.glob(test_path + '*')
    wdf_test = []
    for g in test_files:
        wdf_test.append(load_sparse_csr(g))

    wdf_test = vstack(wdf_test).toarray()

    if algorithm == 'sdm':
        sdm_topic_files = glob.glob(topics_path + '*topics*')
        sdm_topic_files.sort()
        sdm_topics_time = []

        for topic_path in sdm_topic_files:
             with open(topic_path, "rb") as fp:   # Unpickling
                 topic, _, _, _ = pickle.load(fp)
             sdm_topics_time.append(topic)

        sdm_topic_time_last = sdm_topics_time[-1]
        sdm_topic_time_last = clean_topics(sdm_topic_time_last)
        topics_nums = [topics_t.shape[0] for topics_t in sdm_topics_time] 
        topics_num_str = ''
        for i in topics_nums:
            topics_num_str += str(i) + ','
        topics_num_str = topics_num_str[:len(topics_num_str)-1]
        print('topic_num: %s' % topics_num_str)
        print('perplexity:%d' % perplexity(wdf_test, sdm_topic_time_last, scale=True, smooth=False))        
    elif algorithm == 'dm':
        dm_topics_file = glob.glob(topics_path + '*topics*')[0]
        with open(dm_topics_file, "rb") as fp:   # Unpickling
            dm_topics, _, _, _ = pickle.load(fp)
        dm_topics = clean_topics(dm_topics)
        print('topic_num: %d' % dm_topics.shape[0])
        print('perplexity:%d' % perplexity(wdf_test, dm_topics, scale=True, smooth=False))
    elif algorithm == 'sddm':
        sddm_topic_files = glob.glob(topics_path + '*topics*')
        sddm_topic_files.sort()
        sddm_topics_time = []

        for topic_path in sddm_topic_files:
            with open(topic_path, "rb") as fp:   # Unpickling
                 topic, _, _, _ = pickle.load(fp)
            sddm_topics_time.append(topic)

        sddm_topic_time_last = sddm_topics_time[-1]
        sddm_topic_time_last = clean_topics(sddm_topic_time_last)
        topics_nums = [topics_t.shape[0] for topics_t in sddm_topics_time]
        topics_num_str = ''
        for i in topics_nums:
            topics_num_str += str(i) + ','
        topics_num_str = topics_num_str[:len(topics_num_str)-1]
        print('topic_num: %s' % topics_num_str)
        print('perplexity:%d' % perplexity(wdf_test, sddm_topic_time_last, scale=True, smooth=False))
    else:
         raise ValueError('%s is NOT a known or supported algorithm' % (algorithm)) 


if __name__ == '__main__':
    main()    
