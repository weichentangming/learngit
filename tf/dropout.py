import tensorflow as tf
from sklearn.datasets import load_digits
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import LabelBinarizer

#神经层函数
def add_layer(inputs,in_size,out_size,activation_function=None):
	with tf.name_scope('layer'):
		with tf.name_scope('Weights'):
			Weights = tf.Variable(tf.random_normal([in_size,out_size]),name='w')
		with tf.name_scope('biases'):
			biases = tf.Variable(tf.zeros([1,out_size])+0.1,name='biases')
		with tf.name_scope('Wx_plus_b'):
			Wx_plus_b = tf.add(tf.matmul(inputs,Weights),biases)
			Wx_plus_b = tf.nn.dropout(Wx_plus_b,keep_prob)
		if activation_function == None:
			outputs = Wx_plus_b
		else:
			outputs = activation_function(Wx_plus_b)
		return outputs
#载入数据
digits = load_digits()
X = digits.data
y = digits.target
y = LabelBinarizer().fit_transform(y)
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3)

with tf.name_scope('inputs'):
        xs = tf.placeholder(tf.float32,[None,64],name='x_in')
        ys = tf.placeholder(tf.float32,[None,10],name='y_in')
        keep_prob = tf.placeholder(tf.float32)
#搭建神经网络
l1 = add_layer(xs,64,50,activation_function = tf.nn.tanh)
prediction = add_layer(l1,50,10,activation_function =tf.nn.softmax )
with tf.name_scope('loss'):
        loss = tf.reduce_mean(-tf.reduce_sum(ys*tf.log(prediction),reduction_indices=[1]))
        tf.summary.scalar('loss',loss)

with tf.name_scope('train'):
	train_step = tf.train.GradientDescentOptimizer(0.4).minimize(loss)

#初始化变量	
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

merged = tf.summary.merge_all()
train_writer = tf.summary.FileWriter("/home/weifeng/learngit/tf/graph/train",sess.graph)
test_writer = tf.summary.FileWriter("/home/weifeng/learngit/tf/graph/test",sess.graph)
#训练
for i in range(1000):
        sess.run(train_step,feed_dict={xs:X_train,ys:y_train,keep_prob:0.6})
        if i%50==0:
            print(sess.run(loss,feed_dict={xs:X_train,ys:y_train,keep_prob:1}))
            train_result = sess.run(merged,feed_dict={xs:X_train,ys:y_train,keep_prob:1})
            test_result = sess.run(merged,feed_dict={xs:X_test,ys:y_test,keep_prob:1})
            train_writer.add_summary(train_result,i)
            test_writer.add_summary(test_result,i)
