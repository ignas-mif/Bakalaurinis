import collections

import tensorflow as tf
import tensorflow_probability as tfp
import matplotlib.pyplot as plt
import numpy as np
import scipy as sc
import scipy.stats as st
import statsmodels as sm

plt.style.use('ggplot')
tf.enable_eager_execution()

# 1st task w/ Tensorflow.
random_uniform_variable = tf.random.uniform([1, 1000])

bijector_collection = tfp.bijectors
weibull_bijector = bijector_collection.Weibull(10.0, 2.0)
distributed_variables = weibull_bijector._inverse(random_uniform_variable)
plt.hist(distributed_variables, label="λ=10; k=2")
shape, loc, scale = st.weibull_min.fit(distributed_variables, floc=0)
mean = st.weibull_min.mean(shape)
print("λ=", scale, "; k=", shape, "; mean=", mean)

weibull_bijector = bijector_collection.Weibull(0.5, 10.0)
distributed_variables = weibull_bijector._inverse(random_uniform_variable)
plt.hist(distributed_variables, label="λ=0.5, k=10")
shape, loc, scale = st.weibull_min.fit(distributed_variables, floc=0)
mean = st.weibull_min.mean(shape)
print("λ=", scale, "; k=", shape, "; mean=", mean)

plt.legend(loc='upper right')
plt.show()
