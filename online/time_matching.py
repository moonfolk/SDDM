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
    c = (mui[mui > 0]).min()
    return new_t*c + cent


def cosine(a, b):
    return np.dot(a, b)/(norm(a)*norm(b))


def matching_upd_t(cosac_topics, global_topics, topic_counts, T, tau0, tau1, gamma):

    if T < 2:
        print 'Warning'
        
    K = global_topics.shape[0]

    param_cost = np.zeros((cosac_topics.shape[0], K))
    
    # Parametric Cost
    for i in range(global_topics.shape[0]):
        param_cost[:,i] = norm(global_topics[i] + tau1 * cosac_topics, axis=1) - tau0 + \
                          np.log(min(topic_counts[i],10.)) - np.log(T-min(topic_counts[i], 10.))

    # Nonparametric Cost
    K = global_topics.shape[0]
    Kt = cosac_topics.shape[0]
    max_new = min(Kt, max(250-K, 1))
    nonparam_cost = np.ones((Kt, max_new))*tau1
    nonparam_cost -= np.log(np.arange(1, max_new+1))
    nonparam_cost += np.log(gamma/T)
    
    cost = np.hstack((param_cost, nonparam_cost))
    
    row_ind, col_ind = linear_sum_assignment(-cost)
    
    for k, i in zip(row_ind, col_ind):
        if i < K:
            global_topics[i] += tau1*cosac_topics[k]
            topic_counts[i] += 1
        # New Topic
        else:
            global_topics = np.vstack((global_topics, tau1*cosac_topics[k]))
            topic_counts += [1]
            
    return global_topics, topic_counts
                

def matching_upd(cosac_topics, cent, T, global_topics=None, topic_counts=None, tau0=3., tau1=3., gamma=1.):
    """
    Input: list of cosac topic matrices for groups;
           global topic matrix;
           list of counts for topic popularity for groups;
           Bessel ratio
    Output: updated global topics and updated list of topic counts
    """
    if T == 1:
        global_topics = cosac_topics
        topic_counts = global_topics.shape[0]*[1]
        return global_topics, topic_counts
    
    # All topics should be centered at same point and normalized to unit ball times tau0
    cosac_topics = normalize(cosac_topics - cent)

    global_topics = tau0*normalize(global_topics - cent)
    
    global_topics, topic_counts = matching_upd_t(cosac_topics, global_topics, topic_counts, 
                                                 T=T, tau0=tau0, tau1 = tau1, gamma=gamma)
    # Puncture & Uncenter
    for i in range(global_topics.shape[0]):
        global_topics[i] = puncture(global_topics[i], cent)
    
    return global_topics, topic_counts
