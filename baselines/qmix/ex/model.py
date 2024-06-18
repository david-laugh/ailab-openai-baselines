import tensorflow as tf
import numpy as np
from baselines.common.tf_util import get_session, initialize, save_variables, load_variables
try:
    from baselines.common.mpi_adam_optimizer import MpiAdamOptimizer
    from mpi4py import MPI
    from baselines.common.mpi_util import sync_from_root
except ImportError:
    MPI = None
import functools


class Model(object):
    def __init__(self, *, policy, ob_space, ac_space, nbatch_act, nbatch_train,
                 nsteps, ent_coef, vf_coef, max_grad_norm, mpi_rank_weight=1, comm=None, microbatch_size=None):
        self.sess = sess = get_session()

        if MPI is not None and comm is None:
            comm = MPI.COMM_WORLD

        with tf.variable_scope('qmix_model', reuse=tf.AUTO_REUSE):
            # Create individual Q networks for each agent
            self.q_nets = []
            for i in range(nbatch_act):
                q_net = policy(ob_space, ac_space, nbatch_act, nsteps, sess)
                self.q_nets.append(q_net)

            # Create mixer network
            self.mixer_net = self.create_mixer_network()

        # Placeholders for the inputs
        self.obs = [tf.placeholder(tf.float32, [None] + list(ob_space.shape)) for _ in range(nbatch_act)]
        self.actions = [tf.placeholder(tf.int32, [None]) for _ in range(nbatch_act)]
        self.rewards = tf.placeholder(tf.float32, [None])
        self.dones = tf.placeholder(tf.float32, [None])
        self.next_obs = [tf.placeholder(tf.float32, [None] + list(ob_space.shape)) for _ in range(nbatch_act)]

        # Compute Q values and loss
        self.q_values = [q_net.q_values_tensor for q_net in self.q_nets]
        self.next_q_values = [q_net.q_values_tensor for q_net in self.q_nets]

        # Combine Q values using the mixer network
        self.combined_q_values = self.mixer_net(self.q_values)
        self.combined_next_q_values = self.mixer_net(self.next_q_values)

        # Loss function
        self.loss = self.compute_loss()

        # Optimizer
        if comm is not None and comm.Get_size() > 1:
            self.trainer = MpiAdamOptimizer(comm, learning_rate=0.001, mpi_rank_weight=mpi_rank_weight, epsilon=1e-5)
        else:
            self.trainer = tf.train.AdamOptimizer(learning_rate=0.001, epsilon=1e-5)

        self._train_op = self.trainer.minimize(self.loss)

        self.save = functools.partial(save_variables, sess=sess)
        self.load = functools.partial(load_variables, sess=sess)

        initialize()
        global_variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="")
        if MPI is not None:
            sync_from_root(sess, global_variables, comm=comm)

        # Define initial_state
        self.initial_state = [np.zeros((nbatch_act, 64)) for _ in range(len(self.q_nets))]

    def create_mixer_network(self):
        # Define the mixer network here
        def mixer(q_values):
            # Example of simple summation mixer
            return tf.reduce_sum(q_values, axis=0)
        return mixer

    def compute_loss(self):
        # Compute the loss here
        td_targets = self.rewards + 0.99 * self.combined_next_q_values * (1.0 - self.dones)
        td_errors = td_targets - self.combined_q_values
        loss = tf.reduce_mean(tf.square(td_errors))
        return loss

    def train(self, obs, actions, rewards, next_obs, dones):
        feed_dict = {}
        for i in range(len(self.obs)):
            feed_dict[self.obs[i]] = obs[i]
            feed_dict[self.actions[i]] = actions[i]
            feed_dict[self.next_obs[i]] = next_obs[i]
        feed_dict[self.rewards] = rewards
        feed_dict[self.dones] = dones

        loss, _ = self.sess.run([self.loss, self._train_op], feed_dict)
        return loss

    def step(self, obs):
        actions, q_values = [], []
        for q_net in self.q_nets:
            action, q_value = q_net.step(obs)
            actions.append(action)
            q_values.append(q_value)
        return actions, q_values, self.initial_state, np.zeros(len(self.q_nets))

    def value(self, obs):
        q_values = []
        for q_net in self.q_nets:
            q_value = q_net.q_values(obs)
            q_values.append(q_value)
        return np.sum(q_values, axis=0)
