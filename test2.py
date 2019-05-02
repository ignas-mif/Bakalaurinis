import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gamma

shape = 7.5
iscale = 2
N = 200
dimmension = [N]

realizacijos_gamma = tf.random_gamma(
    dimmension,
    shape,
    beta=iscale
)

with tf.Session() as sess:
  
    ## Plot gamma.
    rel_gamma = sess.run(realizacijos_gamma)
    print(rel_gamma.shape)
    ## vizualizacija
    fig, ax = plt.subplots(1, 1)
    ##ax.hist(rel, bins = 100, density = True)
    x = np.linspace(gamma.ppf(0.01, a=shape, loc = 0, scale = 1.0/iscale), 
                    gamma.ppf(0.99, a=shape, loc = 0, scale = 1.0/iscale), 100)
    ax.plot(x, gamma.pdf(x, a=shape, loc = 0, scale = 1.0/iscale), 'r-', lw=5, alpha=0.6, label='gamma pdf')
    
    ## Plot Weibull.