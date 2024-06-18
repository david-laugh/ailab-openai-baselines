# --- QMIX specific parameters ---


# Reference :
#   - https://github.com/oxwhirl/pymarl/blob/master/src/config/algs/qmix.yaml
#   - https://github.com/oxwhirl/pymarl/blob/master/src/config/default.yaml
def pymarl():
    return dict(
        batch_size=32,
        batch_size_run=8,
        test_nepisode=32,
        test_interval=5000,
        n_episodes=2000000,
        t_max=2000000,
        epsilon_start=1.0,
        epsilon_finish=0.05,
        epsilon_anneal_time=50000,
        buffer_size=5000,
        target_update_interval=200,
        agent_output_type="q",
        learner="q_learner",
        optimizer="RMSprop",
        optim_params=dict(
            lr=0.0005,
            alpha=0.99,
            eps=0.00001
        ),
        mixer="qmix",
        mixer_args=dict(
            embed_dim=32
        ),
        agent="rnn",
        agent_args=dict(
            rnn_hidden_dim=64,
            fc_input_dims=32,
            fc_hidden_dim=64
        )
    )
