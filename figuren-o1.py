import tensorflow as tf
x = tf.Variable(3, name="x")
y = tf.Variable(4, name="y")
f = x*x*y + y + 2

sess = tf.Session()
sess.run(x.initializer)
sess.run(y.initializer)
result = sess.run(f)
print(result)

sess.close()

with tf.Session() as sess:
    x.initializer.run()
    y.initializer.run()
    result = f.eval()

init = tf.global_variables_initializer()  # prepare an init node
with tf.Session() as sess:
    init.run()  # actually initialize all the variables
    result = f.eval()

sess = tf.InteractiveSession()
init.run()
result = f.eval()
print(result)
sess.close()

x1 = tf.Variable(1)
x1.graph is tf.get_default_graph()


graph = tf.Graph()
with graph.as_default():
    x2 = tf.Variable(2)

x2.graph is graph
x2.graph is tf.get_default_graph()

w = tf.constant(3)
x = w + 2
y = x + 5
z = x * 3
with tf.Session() as sess:
    print(y.eval())  # 10
    print(z.eval())  # 15

with tf.Session() as sess:
    y_val, z_val = sess.run([y, z])
    print(y_val)  # 10
    print(z_val)  # 15


