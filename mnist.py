from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf

def compute_accuracy(v_x, v_y):
    global prediction
    #input v_x to nn and get the result with y_pre
    y_pre = sess.run(prediction, feed_dict={x:v_x})
    #find how many right
    correct_prediction = tf.equal(tf.argmax(y_pre,1), tf.argmax(v_y,1))
    #calculate average
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    #get input content
    result = sess.run(accuracy,feed_dict={x: v_x, y: v_y})
    return result

def add_layer(inputs, in_size, out_size, activation_function=None,):
    #init w: a matric in x*y
    Weights = tf.Variable(tf.random_normal([in_size, out_size]))
    #init b: a matric in 1*y
    biases = tf.Variable(tf.zeros([1, out_size]) + 0.1,)
    #calculate the result
    Wx_plus_b = tf.matmul(inputs, Weights) + biases
    #add the active hanshu
    if activation_function is None:
        outputs = Wx_plus_b
    else:
        outputs = activation_function(Wx_plus_b,)
    return outputs
    

#load mnist data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
#define placeholder for input
x = tf.placeholder(tf.float32, [None, 784])
y = tf.placeholder(tf.float32, [None, 10])
#add layer
prediction = add_layer(x, 784, 10, activation_function=tf.nn.softmax)
#calculate the loss
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y*tf.log(prediction), reduction_indices=[1]))
#use Gradientdescentoptimizer
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)
#init session
sess = tf.Session()
#init all variables
sess.run(tf.global_variables_initializer())
#start training
for i in range(1000):
    #get batch to learn easily
    batch_x, batch_y = mnist.train.next_batch(100)
    sess.run(train_step,feed_dict={x: batch_x, y: batch_y})
    if i % 50 == 0:
        print(compute_accuracy(mnist.test.images, mnist.test.labels))
