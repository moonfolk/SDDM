import numpy as np
from numpy.linalg import norm
from scipy.optimize import linear_sum_assignment
from sklearn.preprocessing import normalize


def puncture(new_t, cent):
    mui = cent/(-new_t)
    c = mui[mui>0]
    
    if len(c) == 0:
        print 'Warning: can\'t puncture'
        new_t = new_t + cent
        new_t[new_t < 0] = 0
        new_t = normalize(new_t, 'l1')
        return new_t
    
    c = c.min()

    return new_t*c + cent


def cosine(a, b):
    return np.dot(a, b)/(norm(a)*norm(b))


def matching_upd_j(cosac_topics_j, global_topics, topic_counts, J, gamma, tau1):
        
    K = global_topics.shape[0]
    
    param_cost = np.zeros((cosac_topics_j.shape[0], K))
    
    for i in range(global_topics.shape[0]):
        param_cost[:,i] = norm(global_topics[i] + tau1 * cosac_topics_j, axis=1) - norm(global_topics[i]) + \
                          np.log(min(topic_counts[i],10)) - np.log(J-min(topic_counts[i], 10))
    
    # Nonparametric Cost
    K = global_topics.shape[0]
    Kj = cosac_topics_j.shape[0]
    max_new = min(Kj, max(250-K, 1))
    nonparam_cost = np.ones((Kj, max_new))*tau1
    nonparam_cost -= np.log(np.arange(1, max_new+1))
    nonparam_cost += np.log(gamma/J)
    
    cost = np.hstack((param_cost, nonparam_cost))
    
    row_ind, col_ind = linear_sum_assignment(-cost)
    
    assignment_j = []
    new_K = K
    
    for k, i in zip(row_ind, col_ind):
        if i < K:
            global_topics[i] += tau1*cosac_topics_j[k]
            topic_counts[i] += 1
            assignment_j.append(i)
        # New Topic
        else:
            global_topics = np.vstack((global_topics, tau1*cosac_topics_j[k]))
            topic_counts += [1]
            assignment_j.append(new_K)
            new_K += 1
            
    return global_topics, topic_counts, assignment_j


def init_matching(cosac_topics, tau1, gamma):
    
    J = len(cosac_topics)
    group_order = sorted(range(J), key=lambda x: -cosac_topics[x].shape[0])
    
    global_topics = tau1*cosac_topics[group_order[0]]
    topic_counts = [1]*global_topics.shape[0]
    
    assignment = [[] for _ in range(J)]
    
    assignment[group_order[0]] = range(global_topics.shape[0])
    
    J_count = 1
    for j in group_order[1:]:
        J_count += 1
        global_topics, topic_counts, assignment_j = matching_upd_j(cosac_topics[j], global_topics, topic_counts, 
                                                                   J=J_count, gamma=gamma, tau1=tau1)
        assignment[j] = assignment_j
        
    return global_topics, topic_counts, assignment 


def matching_upd(cosac_topics, cent, tau1=3., gamma=1., it=3):
    """
    Input: list of cosac topic matrices for groups;
           global topic matrix;
           list of counts for topic popularity for groups;
           Bessel ratio
    Output: updated global topics and updated list of topic counts
    """
    
    # All topics should be centered at same point and normalized to unit ball times tau0
    cosac_topics = [normalize(cosac_topic_j - cent) for cosac_topic_j in cosac_topics]
    
    # Initialize
    global_topics, topic_counts, assignment = init_matching(cosac_topics, tau1, gamma)
    print 'Initialization objective is %f; number of topics is %d\n' % \
          (norm(global_topics, axis=1).sum(), global_topics.shape[0])

    J = len(cosac_topics)
    
    # Iterate over groups
    for iteration in range(it):
        random_order = np.random.permutation(J)
        for j in random_order:
            to_delete = []
            # Remove j
            Kj = len(assignment[j])
            for k, i in sorted(zip(range(Kj), assignment[j]), key=lambda x: -x[1]):
                topic_counts[i] -= 1
                if topic_counts[i] == 0:
                    del topic_counts[i]
                    to_delete.append(i)
                    for j_clean in range(J):
                        for idx, k_ind in enumerate(assignment[j_clean]):
                            if i < k_ind and j_clean != j:
                                assignment[j_clean][idx] -= 1
                            elif i == k_ind and j_clean != j:
                                print('Warning - weird unmatching')
                                
                else:
                    global_topics[i] -= tau1*cosac_topics[j][k]
                    
            global_topics = np.delete(global_topics, to_delete, axis=0)

            global_topics, topic_counts, assignment[j] = \
                matching_upd_j(cosac_topics[j], global_topics, topic_counts, J, gamma, tau1)
    
        print 'Iteration %d objective is %f; number of topics is %d\n' % \
              (iteration, norm(global_topics, axis=1).sum(), global_topics.shape[0])

    # Puncture & Uncenter
    for i in range(global_topics.shape[0]):
        global_topics[i] = puncture(global_topics[i], cent)
    
    return global_topics, topic_counts
