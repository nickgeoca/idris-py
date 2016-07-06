import Data.Vect
import Python
import Python.Lib.TensorFlow.Matrix
import Python.Lib.TensorFlow
import Python.Lib.Numpy.Matrix

%default total

x : Tensor [3,4] Float32
x = ones

y : Tensor [4,3] Float32
y = ones


z : Tensor [3,3] Float32
z = matmul x y


fn : PIO ()
fn = do rslt <- run session z
        printLn' rslt

fn2 : PIO (MatrixN [3,3] DDouble)
fn2 = run session z

main : PIO ()
main = do printLn' "hey"
          printLn' x
          printOp x
          printOp $ reduce_mean x [0,1] True
          printOp $ reduce_mean x [0] False
          printOp $ reduce_mean' x True
          printOp $ reduce_mean' x False
          printOp $ reduce_sum x [0,1] True
          printOp $ reduce_sum x [0] False
          printOp $ reduce_sum' x False
          printOp $ reduce_sum' x True
  where printOp : Tensor xs dt -> PIO ()
        printOp op = do rslt <- run session op
                        printLn' rslt
                        


exports : FFI_Export FFI_Py "tf.py" []
exports =
    Fun greet2 "greet2" $
--    Fun fn "fn" $
    End
  where
    greet2 : String -> PIO String
    greet2 name = return $ "Hello " ++ name ++ "!"
    greet : String -> PIO ()
    greet name = putStrLn' $ "Hello " ++ name ++ "!"


--------------------------------------------------
-- while_loop
{-
import tensorflow as tf
import numpy as np
i = tf.ones([])
m = tf.ones([2,2])
c = lambda i, m: tf.less(i, 3)
b = lambda i, m: [tf.add(i, tf.ones([])), tf.add(m, tf.ones([2,2]))]
op = tf.while_loop(c, b, [i, m])
sess = tf.Session()
rslt = sess.run(op)
sess.close()
print(rslt)
[3.0, array([[ 3.,  3.],
             [ 3.,  3.]], dtype=float32)]
-}

{-
test_while_loop : PIO ()
test_while_loop = do rslt <- run' session loop
                     printLn' rslt
  where 
  dt : ElemType
  input'  : List (Shape, ElemType)
  output' : List (Shape, ElemType)
  condition : TensorList input' -> Bool
  body : TensorList input' -> TensorList output'
  vars : TensorList input'
  loop : TensorList output'

  dt = Float32
  input'  = [([], dt), ([2,2], dt)]
  output' = [([], dt), ([2,2], dt)]
  condition (i :: _) = i < 3
  body (i :: m :: Nil) = [1 + i, ones + m]
  vars = [1, ones]
  loop = while_loop condition body vars 10 False False
-- -}


-- Constant
-- 1 : String
-- 1 = "abc"

{-
op = tf.ones([3,4])
sess = tf.Session()
print sess.run(op)
-}



----------------------------------------------------------------------------------------------------
{-
train T Y = do t = tf.placeholder(tf.float32, [None, 10])
         cross_entropy = tf.reduce_mean(-tf.reduce_sum(t * tf.log(y), reduction_indices=[1]))
         train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)
         init = tf.initialize_all_variables()
         sess = tf.Session()
         sess.run(init)
         for i in range(1000):
           batch_xs, batch_ys = mnist.train.next_batch(100)
           sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})
  
         correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))

         accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
  
         print(sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))


_2d : _
_2d = do input [28,28]
         conv2d Relu [28,28]
         conv2d Relu [28,28]
         dense Softmax 10
-}
{- Multi Layer
x = tf.placeholder(tf.float32, shape=[None, 784])
y_ = tf.placeholder(tf.float32, shape=[None, 10])


def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)

def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)a

def conv2d(x, W):
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')

---------------
W_conv1 = weight_variable([5, 5, 1, 32])
b_conv1 = bias_variable([32])

x_image = tf.reshape(x, [-1,28,28,1])

h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)
---------------
W_conv2 = weight_variable([5, 5, 32, 64])
b_conv2 = bias_variable([64])

h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)

W_fc1 = weight_variable([7 * 7 * 64, 1024])
b_fc1 = bias_variable([1024])

h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

W_fc2 = weight_variable([1024, 10])
b_fc2 = bias_variable([10])

y_conv=tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)

--------
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y_conv), reduction_indices=[1]))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
sess.run(tf.initialize_all_variables())
for i in range(20000):
  batch = mnist.train.next_batch(50)
  if i%100 == 0:
    train_accuracy = accuracy.eval(feed_dict={
        x:batch[0], y_: batch[1], keep_prob: 1.0})
    print("step %d, training accuracy %g"%(i, train_accuracy))
  train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})

print("test accuracy %g"%accuracy.eval(feed_dict={
    x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0}))


-}

{- Single Layer
x = tf.placeholder(tf.float32, [None, 784])
W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))
y = tf.nn.softmax(tf.matmul(x, W) + b)
t = tf.placeholder(tf.float32, [None, 10])
cross_entropy = tf.reduce_mean(-tf.reduce_sum(t * tf.log(y), reduction_indices=[1]))
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)
init = tf.initialize_all_variables()
sess = tf.Session()
sess.run(init)
for i in range(1000):
  batch_xs, batch_ys = mnist.train.next_batch(100)
  sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})

correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))

accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

print(sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))
-}

{-
# Placeholder for the inputs in a given iteration.
words = tf.placeholder(tf.int32, [batch_size, num_steps])

lstm = rnn_cell.BasicLSTMCell(lstm_size)
# Initial state of the LSTM memory.
initial_state = state = tf.zeros([batch_size, lstm.state_size])

for i in range(num_steps):
    # The value of state is updated after processing each batch of words.
    output, state = lstm(words[:, i], state)

    # The rest of the code.
    # ...

final_state = state
w-}

{-


interface Layer a where
  transform : a
  fit : a
  loss : a
  optimizer : a

State
 output
   
input : _
input shape = placeholder Float32 shape
      output = 100

dense : _
  dense output_dim activation state = State output 
  where 
  weights = tf.constant(0.1, shape=shape)
            return tf.Variable(initial)
  input = output state
  output = activation $ weight * input

_1d : _
_1d = do input [784]
         dense 100 relu 
         dense 10 softmax
         -- y



-}
