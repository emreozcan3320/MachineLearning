"""
Source : 
    https://youtu.be/Xiab2JhwzYY
    https://www.geeksforgeeks.org/linear-regression-using-tensorflow/
"""
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

learning_rate = 0.01
epochs = 200

#number of training samples
n_samples = 30
train_x = np.linspace(0,20,n_samples)
train_y = 3 * train_x + 4 * np.random.rand(n_samples)


# Step 1 create variables for X,Y,W( weight = slope ),B( bias = y offset )
X = tf.placeholder('float')
Y = tf.placeholder('float')

W = tf.Variable(np.random.randn(),name='weights')
B = tf.Variable(np.random.randn(),name='weights')


# Step 2 define Graph or equation
#we can write like this or use tf build-in function
#pred  = X * W + B 
pred = tf.add(tf.multiply(X,W),B)

# Step 3 define the cost function
cost = tf.reduce_sum((pred - Y)**2 / (2 * n_samples))

# step 4 define optimizer
# we use (Gradient Descent)
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

#step 5 train
init = tf.global_variables_initializer()

#all computation is need to be done in a tf session  
with tf.Session() as sesh:
    sesh.run(init)
    
    for epoch in range(epochs):
        for x, y in zip(train_x,train_y):
            sesh.run(optimizer, feed_dict = {X:x,Y:y})
            
        if not epoch % 20:
            c = sesh.run(cost, feed_dict = {X:train_x,Y:y})
            w = sesh.run(W)
            b = sesh.run(B)
            print(f'epoch:{epoch:04d} c={c:.4f} w={w: .4f} b={b:.4f}')

    weight = sesh.run(W)
    bias = sesh.run(B)
    plt.plot(train_x,train_y,'o')
    plt.plot(train_x,weight * train_x + bias)
    plt.show() 













