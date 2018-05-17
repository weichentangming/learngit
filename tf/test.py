import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
#神经层函数
def add_layer(inputs,in_size,out_size,activation_function=None):
	with tf.name_scope('layer'):
		with tf.name_scope('Weights'):
			Weights = tf.Variable(tf.random_normal([in_size,out_size]),name='w')
		with tf.name_scope('biases'):
			biases = tf.Variable(tf.zeros([1,out_size])+0.1,name='biases')
		with tf.name_scope('Wx_plus_b'):
			Wx_plus_b = tf.add(tf.matmul(inputs,Weights),biases)
		if activation_function == None:
			outputs = Wx_plus_b
		else:
			outputs = activation_function(Wx_plus_b)
		return outputs

#导入数据
x_data = np.linspace(-1,1,300,dtype=np.float32)[:,np.newaxis]
noise = np.random.normal(0,0.05,x_data.shape).astype(np.float32)
y_data = np.square(x_data)-0.5+noise

with tf.name_scope('inputs'):

	xs = tf.placeholder(tf.float32,[None,1],name='x_in')
	ys = tf.placeholder(tf.float32,[None,1],name='y_in')

#搭建神经网络
l1 = add_layer(xs,1,10,activation_function = tf.nn.relu)
prediction = add_layer(l1,10,1,activation_function = None)
with tf.name_scope('loss'):
        loss = tf.reduce_mean(tf.reduce_sum(tf.square(ys-prediction),reduction_indices=[1]))
        tf.summary.scalar('loss',loss)

with tf.name_scope('train'):
	train_step = tf.train.GradientDescentOptimizer(0.1).minimize(loss)

#初始化变量	
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

#可视化
#fig = plt.figure()
#ax = fig.add_subplot(1,1,1)
#ax.scatter(x_data,y_data)
#plt.ion()
#plt.show()
#plt.pause(2)
merged = tf.summary.merge_all()
writer = tf.summary.FileWriter("learngit/tf/graph/",sess.graph)
#训练
for i in range(1000):
        sess.run(train_step,feed_dict={xs:x_data,ys:y_data})
        if i%50==0:
            rs = sess.run(merged,feed_dict={xs:x_data,ys:y_data})
            writer.add_summary(rs,i)
   
