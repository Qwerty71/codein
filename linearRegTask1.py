import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import pandas as pd

plt.style.use('ggplot')

PATH = "Salary_Data.csv"

alpha = 0.01
epochs = 200

def load_data():
  data = pd.read_csv(PATH)
  m = len(data)
  x = np.array(data[data.columns[0]]).reshape((m, 1))
  y = np.array(data[data.columns[1]]).reshape((m, 1))
  return [x, y, m]

def normalize(dataset):
  mu = np.mean(dataset)
  sigma = np.std(dataset)
  return (dataset-mu)/sigma

def split(x, y, trainSize):
  length = len(x)
  return [x[0:int(length*trainSize)], x[int(length*trainSize):length], y[0:int(length*trainSize)], y[int(length*trainSize):length]]

x, y, m = load_data()

x_train, x_test, y_train, y_test = split(x, y, 0.8)

theta = tf.Variable(tf.zeros([1, 1]))
X = tf.placeholder(tf.float32, shape=[None, 1], name = "x_input")
Y = tf.placeholder(tf.float32, shape=[None, 1], name = "y_input")
b = tf.Variable(0.0)

model = tf.add(tf.matmul(X, theta), b)

cost = tf.reduce_sum(tf.square(Y-model))/(2*m)

optimizer = tf.train.GradientDescentOptimizer(alpha).minimize(cost)
init = tf.global_variables_initializer()

with tf.Session() as sess:
  sess.run(init)
  for i in range(epochs):
    sess.run(optimizer, feed_dict={X:x_train, Y:y_train})
    loss = sess.run(cost, feed_dict={X:x_train, Y:y_train})
  theta, b = sess.run(theta), sess.run(b)

plt.plot(x, y, 'r+', label="True Values")
plt.plot(x, theta*x+b, 'blue', label="Regression Line")
plt.title("Salary vs Years of Experience")
plt.legend()
plt.show()

print("Predicted Salary for 6.5 Years of Experience:", (theta*np.array([[6.5]])+b)[0][0])
