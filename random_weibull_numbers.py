import collections
import tensorflow as tf
import tensorflow_probability as tfp
import matplotlib.pyplot as plt
import numpy as np
import scipy as sc
import scipy.stats as st
import statsmodels as sm
fig, axes = plt.subplots(2, 3)


plt.style.use('ggplot')
tf.enable_eager_execution()

# Random weibull variables 1.
random_uniform_variable = tf.random.uniform([1, 1000], minval=0, maxval=1)
bijector_collection = tfp.bijectors
weibull_bijector = bijector_collection.Weibull(10.0, 1.0)
distributed_variables = weibull_bijector._inverse(random_uniform_variable)
axes[0][0].hist(distributed_variables, bins=100, density=True, label="λ=10; k=1")
shape, loc, scale = st.weibull_min.fit(distributed_variables, floc=0)
mean = st.weibull_min.mean(shape)
dist = st.weibull_min(shape, loc, scale)
x = np.linspace(0, 100, 1000)
axes[0][0].plot(x, dist.pdf(x), 'r-', alpha=0.8, lw=2, label='Weibull PDF')
print("λ=", scale, "; k=", shape, "; mean=", mean)

# Random weibull variables 2.
weibull_bijector = bijector_collection.Weibull(2.0, 10.0)
distributed_variables = weibull_bijector._inverse(random_uniform_variable)
axes[0][1].hist(distributed_variables, bins=100, density=True, label="λ=2, k=10")
shape, loc, scale = st.weibull_min.fit(distributed_variables, floc=0)
mean = st.weibull_min.mean(shape)
dist = st.weibull_min(shape, loc, scale)
x = np.linspace(0, 10, 1000)
axes[0][1].plot(x, dist.pdf(x), 'r-', alpha=0.8, lw=2, label='Weibull PDF')
print("λ=", scale, "; k=", shape, "; mean=", mean)

# Random weibull variables 3.
weibull_bijector = bijector_collection.Weibull(5.0, 5.0)
distributed_variables = weibull_bijector._inverse(random_uniform_variable)
axes[0][2].hist(distributed_variables, bins=100, density=True, label="λ=5, k=5")
shape, loc, scale = st.weibull_min.fit(distributed_variables, floc=0)
mean = st.weibull_min.mean(shape)
dist = st.weibull_min(shape, loc, scale)
x = np.linspace(0, 10, 1000)
axes[0][2].plot(x, dist.pdf(x), 'r-', alpha=0.8, lw=2, label='Weibull PDF')
print("λ=", scale, "; k=", shape, "; mean=", mean)

# Random Gamma variables 1.
shape = 10
iscale = 1
N = 200
dimmension = [N]
## Plot gamma.
rel_gamma = tf.random_gamma(dimmension, shape, beta=iscale)
## vizualizacija
axes[1][0].hist(rel_gamma, bins = 100, density = True, label="k=10, θ=1")
x = np.linspace(st.gamma.ppf(0.01, a=shape, loc = 0, scale = 1.0/iscale), 
                st.gamma.ppf(0.99, a=shape, loc = 0, scale = 1.0/iscale), 100)
axes[1][0].plot(x, st.gamma.pdf(x, a=shape, loc = 0, scale = 1.0/iscale), 'r-', lw=2, alpha=0.8, label='Gamma PDF')
    
# Random Gamma variables 2.
shape = 2
iscale = 10
N = 200
dimmension = [N]
## Plot gamma.
rel_gamma = tf.random_gamma(dimmension, shape, beta=iscale)
print(rel_gamma.shape)
## vizualizacija
axes[1][1].hist(rel_gamma, bins = 100, density = True, label="k=2, θ=10")
x = np.linspace(st.gamma.ppf(0.01, a=shape, loc = 0, scale = 1.0/iscale), 
                st.gamma.ppf(0.99, a=shape, loc = 0, scale = 1.0/iscale), 100)
axes[1][1].plot(x, st.gamma.pdf(x, a=shape, loc = 0, scale = 1.0/iscale), 'r-', lw=2, alpha=0.8, label='Gamma PDF')
    

# Random Gamma variables 3.
shape = 5
iscale = 5
N = 200
dimmension = [N]
## Plot gamma.
rel_gamma = tf.random_gamma(dimmension, shape, beta=iscale)
## vizualizacija
axes[1][2].hist(rel_gamma, bins = 100, density = True, label="k=5, θ=5")
x = np.linspace(st.gamma.ppf(0.01, a=shape, loc = 0, scale = 1.0/iscale), 
                st.gamma.ppf(0.99, a=shape, loc = 0, scale = 1.0/iscale), 100)
axes[1][2].plot(x, st.gamma.pdf(x, a=shape, loc = 0, scale = 1.0/iscale), 'r-', lw=2, alpha=0.8, label='Gamma PDF')

axes[0][0].legend(loc='upper right')
axes[0][1].legend(loc='upper right')
axes[0][2].legend(loc='upper right')
axes[1][0].legend(loc='upper right')
axes[1][1].legend(loc='upper right')
axes[1][2].legend(loc='upper right')





plt.show()
