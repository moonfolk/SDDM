import numpy as np
from numpy.linalg import lstsq, norm
from sklearn.preprocessing import normalize

## Geometric Theta
def proj_on_s(beta, doc, K, ind_remain=[], first=True, distance=False):
    if first:
        ind_remain = np.arange(K)
    s_0 = beta[0,:]
    if beta.shape[0]==1:
        if distance:
            return norm(doc-s_0)
        else:
            theta = np.zeros(K)
            theta[ind_remain] = 1.
            return theta
    beta_0 = beta[1:,:]
    alpha = lstsq((beta_0-s_0).T, doc-s_0)[0]
    if np.all(alpha>=0) and alpha.sum()<=1:
        if distance:
            p_prime = (alpha*(beta_0-s_0).T).sum(axis=1)
            return norm(doc-s_0-p_prime)
        else:
            theta = np.zeros(K)
            theta[ind_remain] = np.append(1-alpha.sum(), alpha)
            return theta
    elif np.any(alpha<0):
        ind_remain = np.append(ind_remain[0], ind_remain[1:][alpha>0])
        return proj_on_s(np.vstack([s_0, beta_0[alpha>0,:]]), doc, K, ind_remain, False, distance)
    else:
        return proj_on_s(beta_0, doc, K, ind_remain[1:], False, distance)

def perplexity(docs, beta, theta='geom', scale=True, topic_weights = None, smooth=False):
  if type(theta)==str:
      theta = np.apply_along_axis(lambda x: proj_on_s(beta, x, beta.shape[0]), 1, normalize(docs, 'l1'))  
#      scale = True
      
  if topic_weights is not None:
      theta = theta*np.array(topic_weights)
      theta = normalize(theta, 'l1')
  if smooth:
      K, V = beta.shape
      theta += 1./K**2
      theta = normalize(theta, 'l1')
      beta += 1./V**2
      beta = normalize(beta, 'l1')
      est = np.log(np.dot(theta, beta))
      mtx = docs * est

  elif scale:
      est = np.dot(theta, beta)
      est = np.log(normalize(np.apply_along_axis(lambda x: x + x[x>0].min(), 1, est), 'l1'))
      mtx = docs * est
  else:
      est[est<=0] = 1.
      est = np.log(est)
      mtx = docs * est
      mtx[np.isnan(mtx)] = 0.
  return np.exp(-mtx.sum()/docs.sum())
