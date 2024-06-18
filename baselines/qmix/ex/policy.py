import tensorflow as tf


class Policy(object):
    def __init__(self, ob_space, ac_space, nbatch, nsteps, sess, n_hidden=64, reuse=False):
        self.sess = sess
        self.nbatch = nbatch
        self.nsteps = nsteps

        with tf.variable_scope('mlp_policy', reuse=reuse):
            self.obs_ph = tf.placeholder(tf.float32, [None] + list(ob_space.shape), name='obs')
            self.act_ph = tf.placeholder(tf.int32, [None], name='act')

            # Hidden layers
            h = tf.layers.dense(self.obs_ph, n_hidden, activation=tf.nn.relu, name='fc1')
            h = tf.layers.dense(h, n_hidden, activation=tf.nn.relu, name='fc2')

            # Q values output layer
            self.q_values_tensor = tf.layers.dense(h, ac_space.n, activation=None, name='q')

            # Action selection
            self.action = tf.argmax(self.q_values_tensor, axis=1, name='action')

            # Q value for the selected action
            action_one_hot = tf.one_hot(self.act_ph, ac_space.n)
            self.q_value = tf.reduce_sum(self.q_values_tensor * action_one_hot, axis=1)

    def step(self, obs):
        actions, q_values = self.sess.run([self.action, self.q_values_tensor], feed_dict={self.obs_ph: obs})
        return actions, q_values
