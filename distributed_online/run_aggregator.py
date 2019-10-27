from aggregator import StreamGroupCosac
import glob
from optparse import OptionParser
import time


def parse_args():
    parser = OptionParser()
    parser.set_defaults(data_path="./", path_to_save="./", meta_path="./", cosac_path="./",
                        gamma=1., tau0=4, tau1=4, it=10)

    parser.add_option("--data_path", type="string", dest="data_path",
                      help="path to data")
    parser.add_option("--path_to_save", type="string", dest="path_to_save",
                      help="where to store batch topics")
    parser.add_option("--meta_path", type="string", dest="meta_path",
                      help="path to vocabulary and category map")
    parser.add_option("--cosac_path", type="string", dest="cosac_path",
                      help="path to where cosac will be stored")
    parser.add_option("--tau0", type="float", dest="tau0",
                      help="tau0")
    parser.add_option("--tau1", type="float", dest="tau1",
                      help="tau1")
    parser.add_option("--gamma", type="float", dest="gamma",
                      help="gamma")
    parser.add_option("--it", type="int", dest="it",
                      help="Hungarian iterations count")
    
    (options, args) = parser.parse_args()

    return options

def main():
    
    options = parse_args()
    print options
    
    data_path = options.data_path
    path_to_save = options.path_to_save
    meta_path = options.meta_path
    cosac_path = options.cosac_path
    tau0 = options.tau0
    tau1 = options.tau1
    gamma = options.gamma
    it = options.it
    
    with open(meta_path + '/' + 'vocabulary', 'r') as f:
        vocab = f.readlines()
        
    with open(meta_path + '/' + 'group_id_map', 'r') as f:
        group_map = f.readlines()
        
    total_groups = len(group_map)
    
    all_files = glob.glob(data_path + '/*')
    all_files.sort()
    
    t_s = time.time()
    
    ejc_stream = StreamGroupCosac(cosac_path, total_groups, save_path=path_to_save, tau0=tau0,
                                  tau1=tau1, gamma=gamma, vocab=vocab, it=it)
    
    ejc_stream.stream(all_files)
    
    t_e = time.time()
    
    print 'SDDM took %f seconds' % (t_e - t_s)


if __name__ == '__main__':
    main()