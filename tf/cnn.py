import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

#导入数据及数据处�?mnist = input_data.read_data_sets('MNIST_data',one_hot = True)
xs = tf.placeholder(tf.float32,[None,784])
ys = tf.placeholder(tf.float32,[None,10])
keep_prob = tf.placeholder(tf.float32)
x_image  = tf.reshape(xs,[-1,28,28,1])

#定义权重
def weight_variable(shape):
        initial = tf.truncated_normal(shape,stddev=0.1)
        return tf.Variable(initial)
#定义偏置
def bias_variable(shape):
        initial = tf.constant(0.1,shape=shape)
        return tf.Variable(initial)
#定义卷积�?def conv2d(x,W):
        return tf.nn.conv2d(x,W,strides = [1,1,1,1],padding = 'SAME')
#定义 pooling
def max_pooling(x):
        return tf.nn.max_pool(x,ksize=[1,2,2,1],strides=[1,2,2,1],padding = 'SAME')
#第一层卷积层
W_conv1 = weight_variable([5,5,1,32])
b_conv1 = bias_variable([32])
h_conv1 = tf.nn.relu(conv2d(x_image,W_conv1)+b_conv1)    # 28x28x32
h_pool1 = max_pooling(h_conv1)                           #14x14x32
#第二层卷积层
W_conv2 = weight_variable([5,5,32,64])                   
b_conv2 = bias_variable([64])
h_conv2 = tf.nn.relu(conv2d(h_pool1,W_conv2)+b_conv2)    #14x14x64
h_pool2 = max_pooling(h_conv2)                           #7x7x64
#建立全链接层
h_pool2_flat = tf.reshape(h_pool2,[-1,7*7*64])
W_fc1 = weight_variable([7*7*64,1024])
b_fc1 = bias_variable([1024])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat,W_fc1)+b_fc1)
h_fc1 = tf.nn.dropout(h_fc1,keep_prob)

W_fc2 = weight_variable([1024,10])
b_fc2 = bias_variable([10])
prediction = tf.nn.softmax(tf.matmul(h_fc1,W_fc2)+b_fc2)

#loss
loss = tf.reduce_mean(-tf.reduce_sum(ys*tf.log(prediction)))
#train
train_step = tf.train.AdamOptimizer(1e-4).minimize(loss)

sess = tf.Session()
sess.run(tf.global_variables_initializer())
for i in range(100):
        batch_xs,batch_ys =mnist.train.next_batch(100)
        sess.run(train_step,feed_dict={xs:batch_xs,ys:batch_ys,keep_prob:1})
        if i%20==0:
            correct_prediction =tf.equal(tf.argmax(prediction,1),tf.argmax(ys,1))
            accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))
            print (sess.run(accuracy,feed_dict={xs:mnist.test.images,ys:mnist.test.labels,keep_prob:1}))
