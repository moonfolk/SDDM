import numpy as np
from numpy.linalg import norm
from scipy.special import gammaln
from scipy.optimize import linear_sum_assignment
from sklearn.preprocessing import normalize


# log ratio of vMF constants. V is dimension of directions (i.e. vocabulary - 1)
def bessel(V):
    V = 1.*V/2
    g = gammaln(V)
    b = (V-1)*(1 - np.log(V-1))
    return g + b


def puncture(new_t, cent):
    mui = cent/(-new_t)
    c = mui[mui>0]
    
    if len(c) == 0:
        print 'Warning: can\'t puncture'
        new_t = new_t + cent
        new_t[new_t<0] = 0
        new_t = normalize(new_t, 'l1')
        return new_t
    
    c = c.min()

    return new_t*c + cent


def cosine(a,b):
    return np.dot(a,vb)/(norm(a)*norm(b))


def matching_upd_j(cosac_topics_j, global_topics, topic_counts, gamma, tau1, Jt, T, n_groups_seen, j_idx):
        
    K = global_topics.shape[0]
    
    param_cost = np.zeros((cosac_topics_j.shape[0], K))
    
    for i in range(global_topics.shape[0]):
        param_cost[:,i] = norm(global_topics[i] + tau1 * cosac_topics_j, axis=1) - norm(global_topics[i]) + \
                          np.log(1 + min(topic_counts[j_idx][i],10)) - np.log(T-min(topic_counts[j_idx][i],10))
    
    # Nonparametric cost
    K = global_topics.shape[0]
    Kj = cosac_topics_j.shape[0]
    max_new = min(Kj, max(250-K, 1))
    nonparam_cost = np.ones((Kj, max_new))*tau1
    nonparam_cost -= np.log(np.arange(1, max_new+1))
    nonparam_cost += np.log(gamma/Jt)
    
    cost = np.hstack((param_cost, nonparam_cost))
    
    row_ind, col_ind = linear_sum_assignment(-cost)
    
    assignment_j = []
    new_K = K
    
    for k, i in zip(row_ind, col_ind):
        if i < K:
            global_topics[i] += tau1*cosac_topics_j[k]
            topic_counts[j_idx][i] += 1
            assignment_j.append(i)
        # New Topic
        else:
            global_topics = np.vstack((global_topics, tau1*cosac_topics_j[k]))
            for j in range(len(topic_counts)):
                topic_counts[j] += [int(j == j_idx)]
            assignment_j.append(new_K)
            new_K += 1
            
    return global_topics, topic_counts, assignment_j


def init_matching(cosac_topics, groups_ids, tau1, gamma, topic_counts, global_topics, J, n_groups_seen, T_groups):
    
    group_order = sorted(range(len(groups_ids)), key = lambda x: -cosac_topics[x].shape[0])
    
    start_ind = 0
    assignment = [[] for _ in range(len(cosac_topics))]
    
    if global_topics is None:
        global_topics = tau1*cosac_topics[group_order[0]]
        topic_counts = [[0]*global_topics.shape[0] for _ in range(J)]
        topic_counts[groups_ids[group_order[0]]] = [1]*global_topics.shape[0]
        start_ind = 1
        assignment[group_order[0]] = range(global_topics.shape[0])
    
    J_count = start_ind
    for j in group_order[start_ind:]:
        J_count += 1
        global_topics, topic_counts, assignment_j = matching_upd_j(cosac_topics[j], global_topics, topic_counts, 
                                                                   tau1=tau1, gamma=gamma, Jt=J_count,
                                                                   T=T_groups[groups_ids[j]],
                                                                   n_groups_seen=n_groups_seen, j_idx=groups_ids[j])
        assignment[j] = assignment_j
        
    return global_topics, topic_counts, assignment          


def matching_upd(cosac_topics, cent, groups_ids, J, T_groups, n_groups_seen, global_topics=None, topic_counts=None,
                 tau0=3., tau1=3., gamma=1., it=10):
    """
    Input:  list of cosac topic matrices for groups;
            global topic matrix;
            list of counts for topic popularity for groups;
            Bessel ratio
    Output: updated global topics and updated list of topic counts
    """
    cosac_topics = [normalize(cosac_topic_j - cent) for cosac_topic_j in cosac_topics]
    
    # All topics should be centered at same point and normalized to unit ball times tau0
    if global_topics is not None:
        global_topics = tau0*normalize(global_topics - cent)
        
    global_topics, topic_counts, assignment = \
        init_matching(cosac_topics, groups_ids, tau1, gamma, topic_counts, global_topics, J, n_groups_seen, T_groups)
    print 'Initialization objective is %f; number of topics is %d\n' % \
          (norm(global_topics, axis=1).sum(), global_topics.shape[0])
    
    Jt = len(cosac_topics)
    
    # Iterate over groups
    for iteration in range(it):
        random_order = np.random.permutation(Jt)
        for j in random_order:
            to_delete = []
            # Remove j
            Kj = len(assignment[j])
            for k, i in sorted(zip(range(Kj),assignment[j]), key=lambda x: -x[1]):
                topic_counts[groups_ids[j]][i] -= 1
                if sum([topic_counts[j_c][i] == 0 for j_c in range(J)]) == J:
                    for j_del in range(J):
                        del topic_counts[j_del][i]
                    to_delete.append(i)
                    for j_clean in range(Jt):
                        for idx, k_ind in enumerate(assignment[j_clean]):
                            if i < k_ind and j_clean != j:
                                assignment[j_clean][idx] -= 1
                            elif i == k_ind and j_clean != j:
                                print('Warning - weird unmatching')
                                
                else:
                    global_topics[i] -= tau1*cosac_topics[j][k]
                    
            global_topics = np.delete(global_topics,to_delete,axis=0)

            global_topics, topic_counts, assignment[j] = matching_upd_j(cosac_topics[j], global_topics, topic_counts,
                                                                        tau1=tau1, gamma=gamma, Jt=Jt,
                                                                        T=T_groups[groups_ids[j]],
                                                                        n_groups_seen=n_groups_seen,
                                                                        j_idx=groups_ids[j])
    
        print 'Iteration %d objective is %f; number of topics is %d\n' % \
              (iteration, norm(global_topics, axis=1).sum(), global_topics.shape[0])
    
    # Puncture and Uncenter
    for k in range(global_topics.shape[0]):
        global_topics[k] = puncture(global_topics[k], cent)
    
    return global_topics, topic_counts

