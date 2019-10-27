from cosac_group_time import CosacParallel
import glob
from joblib import Parallel, delayed
from optparse import OptionParser
import time


def parse_args():
    parser = OptionParser()
    parser.set_defaults(data_path="./", path_to_save="./", num_cores=1, delta=0.6, prop_discard=0.5, prop_n=0.001)

    parser.add_option("--data_path", type="string", dest="data_path",
                      help="path to data")
    parser.add_option("--path_to_save", type="string", dest="path_to_save",
                      help="where to store batch topics")
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
    num_cores = options.num_cores
    delta = options.delta
    prop_discard = options.prop_discard
    prop_n = options.prop_n
    
    year_files = glob.glob(data_path + '*')
    year_files.sort(key=lambda x: int(x[-4:]))
    
    all_paths = []
    for year in year_files:
        year_groups = glob.glob(year + '/*')
        year_groups.sort(key=lambda x: int(x.split('_')[-1].split('.')[0]))
        all_paths += year_groups
    
    t_s = time.time()
    Parallel(n_jobs=num_cores)(delayed(wrap_cosac)
                               (time_group_path, path_to_save, delta, prop_discard, prop_n)
                               for time_group_path in all_paths)
    t_e = time.time()
    
    print 'Done with all batches. Took %f seconds' % (t_e - t_s)


if __name__ == '__main__':
    main()
