from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf

lr = 0.001
training_iters = 100000
batch_size = 128
n_input = 49
n_steps = 64
n_hidden_units = 128
n_classes = 10

def compute_accuracy(v_x, v_y):
    global prediction
    y_pre = sess.run(prediction, feed_dict={x:v_x, keep_prob:1})
    correct_prediction = tf.equal(tf.argmax(y_pre,1), tf.argmax(v_y,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    result = sess.run(accuracy,feed_dict={x: v_x, y: v_y, keep_prob:1})
    return result

def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)
    
def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

def conv2d(x, W):
    # strides=[1,x_movement,y_movement,1]
    return tf.nn.conv2d(x, W, strides=[1,1,1,1], padding='SAME')

def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')

def conv_pool_layer(X, img_len, img_hi, out_seq):
    W = weight_variable([img_len, img_len, img_hi, out_seq])
    b = bias_variable([out_seq])
    h_conv = tf.nn.relu(conv2d(X, W) + b)
    return max_pool_2x2(h_conv)

def lstm(X):
    lstm_cell = tf.contrib.rnn.BasicLSTMCell(n_hidden_units, forget_bias=1.0, state_is_tuple=True)
    _init_state = lstm_cell.zero_state(batch_size, dtype=tf.float32)
    outputs,states = tf.nn.dynamic_rnn(lstm_cell, X, initial_state=_init_state, time_major=False)
    W = weight_variable([n_hidden_units, n_classes])
    b = bias_variable([n_classes])
    outputs = tf.unstack(tf.transpose(outputs, [1,0,2]))
    results = tf.matmul(outputs[-1], W) + b
    return results

# load mnist data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

x = tf.placeholder(tf.float32, [None,784])
y = tf.placeholder(tf.float32, [None,10])
keep_prob = tf.placeholder(tf.float32)
# reshape(data you want to reshape, [-1, reshape_height, reshape_weight, imagine layers]) image layers=1 when the imagine is in white and black, =3 when the imagine is RGB 
x_image = tf.reshape(x, [-1,28,28,1])

# ********************** conv1 *********************************
# transfer a 5*5*1 imagine into 32 sequence
#W_conv1 = weight_variable([5,5,1,32])
#b_conv1 = bias_variable([32])
# input a imagine and make a 5*5*1 to 32 with stride=1*1, and activate with relu
#h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1) # output size 28*28*32
#h_pool1 = max_pool_2x2(h_conv1) # output size 14*14*32
h_pool1 = conv_pool_layer(x_image, 5, 1, 32)

# ********************** conv2 *********************************
# transfer a 5*5*32 imagine into 64 sequence
#W_conv2 = weight_variable([5,5,32,64])
#b_conv2 = bias_variable([64])
# input a imagine and make a 5*5*32 to 64 with stride=1*1, and activate with relu
#h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2) # output size 14*14*64
#h_pool2 = max_pool_2x2(h_conv2) # output size 7*7*64
h_pool2 = conv_pool_layer(h_pool1, 5, 32, 64)

# reshape data
X_in = tf.reshape(h_pool2, [-1,49,64])
X_in = tf.transpose(X_in, [0,2,1])

#put into a lstm layer
prediction = lstm(X_in)
# ********************* func1 layer *********************************
#W_fc1 = weight_variable([7*7*64, 1024])
#b_fc1 = bias_variable([1024])
# reshape the image from 7,7,64 into a flat (7*7*64)
#h_pool2_flat = tf.reshape(h_pool2, [-1,7*7*64])
#h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
#h_fc1_drop = tf.nn.dropout(h_fc1,keep_prob)

# ********************* func2 layer *********************************
#W_fc2 = weight_variable([1024, 10])
#b_fc2 = bias_variable([10])
#prediction = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)


# calculate the loss
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=y))
# use Gradientdescentoptimizer
train_step = tf.train.AdamOptimizer(lr).minimize(cross_entropy)

correct_pred = tf.equal(tf.argmax(prediction,1),tf.argmax(y,1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
# init session
sess = tf.Session()
sess.run(tf.global_variables_initializer())

for i in range(training_iters):
    batch_x, batch_y = mnist.train.next_batch(batch_size)
    sess.run(train_step,feed_dict={x: batch_x, y: batch_y, keep_prob: 0.5})
    if i % 50 == 0:
        print(sess.run(accuracy,feed_dict={x: batch_x, y: batch_y,}))
