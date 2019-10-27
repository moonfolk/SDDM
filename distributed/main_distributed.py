from cosac_groups import CosacParallel
from aggregator_groups import StreamGroupCosac
import glob
from joblib import Parallel, delayed
from optparse import OptionParser
import time
import os


def parse_args():
    parser = OptionParser()
    parser.set_defaults(data_path="./", path_to_save="./", global_topics_path="./",meta_path="./",
                        it=1, tau1=5., gamma=1., num_cores=1, delta=0.6, 
                        prop_discard=0.5, prop_n=0.001)

    parser.add_option("--data_path", type="string", dest="data_path",
                      help="path to data")
    parser.add_option("--path_to_save", type="string", dest="path_to_save",
                      help="where to store batch topics")
    parser.add_option("--global_topics_path", type="string", dest="global_topics_path",
                      help="where to store global topics")
    parser.add_option("--meta_path", type="string", dest="meta_path",
                      help="path to meta files")
    parser.add_option("--it", type="int", dest="it",
                      help="number of Hungarian iterations")
    parser.add_option("--tau1", type="float", dest="tau1",
                      help="tau1")
    parser.add_option("--gamma", type="float", dest="gamma",
                      help="gamma")
    parser.add_option("--num_cores", type="int", dest="num_cores",
                      help="number of cores")
    parser.add_option("--delta", type="float", dest="delta",
                      help="cone cosine radius for CoSAC")
    parser.add_option("--prop_discard", type="float", dest="prop_discard",
                      help="quantile for R calculation of CoSAC")
    parser.add_option("--prop_n", type="float", dest="prop_n",
                      help="Quantile for outlier threshold of CoSAC")
    
    (options, args) = parser.parse_args()

    return options


def wrap_cosac(time_group_path, path_to_save, delta=0.6, prop_discard=0.5, prop_n=0.001):
    cosac = CosacParallel(path_to_save, delta=delta, prop_discard=prop_discard, prop_n=prop_n)
    cosac.process_group(time_group_path)
    return time_group_path


def main():
    
    options = parse_args()
    print options
    
    data_path = options.data_path
    path_to_save = options.path_to_save
    global_topics_path = options.global_topics_path
    meta_path = options.meta_path
    it = options.it
    tau1 = options.tau1
    gamma = options.gamma
    num_cores = options.num_cores
    delta = options.delta
    prop_discard = options.prop_discard
    prop_n = options.prop_n
    
    with open(meta_path + '/' + 'vocabulary', 'r') as f:
        vocab = f.readlines()
        
    t_g_s = time.time()
    if os.path.exists(path_to_save + 'cosac_topics'):
        print 'CoSAC already trained'       
    else:
        print 'Starting CoSACs'
        group_files = glob.glob(data_path + '*')
        group_files.sort(key=lambda x: int(x.split('_')[-1].split('.')[0]))
        
        t_s = time.time()
        Parallel(n_jobs=num_cores)(delayed(wrap_cosac)
                                   (group_path, path_to_save, delta, prop_discard, prop_n)
                                   for group_path in group_files)
        t_e = time.time()
        
        print 'Done with all batches. Took %f seconds' % (t_e-t_s)
    
    ejc_groups = StreamGroupCosac(path_cosac=path_to_save, save_path=global_topics_path, tau1=tau1,
                                  gamma=gamma, vocab=vocab)
    ejc_groups.process_all_groups(data_path, it)
    
    t_g_e = time.time()
    
    print 'Distributed CoSAC took %f seconds' % (t_g_e - t_g_s)
    print tau1, gamma


if __name__ == '__main__':
    main()
