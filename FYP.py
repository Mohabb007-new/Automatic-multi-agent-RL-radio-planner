import gymnasium 
import numpy as np
import os
import ray
from ray import air, tune
from ray.air.constants import TRAINING_ITERATION
from ray.rllib.algorithms.ppo.ppo import PPO, PPOConfig

from ray.tune.registry import register_env
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.utils.annotations import override
from ray.rllib.utils.metrics import (
    ENV_RUNNER_RESULTS,
    EPISODE_RETURN_MEAN,
    NUM_ENV_STEPS_SAMPLED_LIFETIME,
)
from ray.rllib.utils.numpy import convert_to_numpy
from ray.rllib.utils.torch_utils import convert_to_torch_tensor
from gymnasium import spaces
from gymnasium.spaces import Tuple,MultiDiscrete,Discrete
from Environment_logic import Planner

def mapp(agent_id, episode, worker, **kwargs):
    if agent_id=="agent0":
        return "agent0_policy"
    elif agent_id=="agent1":
        return "agent1_policy"
    elif agent_id=="agent2" or agent_id=="agent3" or agent_id=="agent4":
        return "agent2_3_4_policy"
    else :
        return "agent5_6_7_policy"

register_env("Planner", lambda config: Planner(config))

if __name__ == "__main__":
    
    ray.init(local_mode=True)

    config = (
        PPOConfig()
        .environment(
        env="Planner",
        env_config={
            "w": 808937,
            "e": 817333, 
            "s": 3808829,
            "n": 3816586,
            "location": "urban",
            "max_output": 100,
            "user_bandwidth": 2
        }
        )
        .framework("torch")
        .env_runners(batch_mode="truncate_episodes", num_env_runners=0)
        .multi_agent(
            policies={
                "agent0_policy": (None, spaces.Box(low=0, high=1000000, shape=(13,)), MultiDiscrete([3,3,3,3,3,3,3,3,3,3,3,3]), {"model": {
                "fcnet_hiddens": [65,260, 180],  # Two hidden layers with 128 units each
                "fcnet_activation": "tanh",  # Tanh activation function
                # LSTM parameters
                "use_lstm": True,  # Enable LSTM
                "max_seq_len": 100,  # Maximum sequence length for LSTM
                "lstm_cell_size": 256,  # Number of units in the LSTM cell
                "lstm_use_prev_action": True,  # Whether to use previous action as input
                "lstm_use_prev_reward": True,  # Whether to use previous reward as input
            }}),
                "agent1_policy": (None, spaces.Box(low=0.0, high=3846000, shape=(9,)), MultiDiscrete([3,3,3,3]),{"model": {
                "fcnet_hiddens": [45,180,60],  # Two hidden layers with 128 units each
                "fcnet_activation": "tanh",  # Tanh activation function
                "use_lstm": True,  # Enable LSTM
                "max_seq_len": 100,  # Maximum sequence length for LSTM
                "lstm_cell_size": 256,  # Number of units in the LSTM cell
                "lstm_use_prev_action": True,  # Whether to use previous action as input
                "lstm_use_prev_reward": True,  # Whether to use previous reward as input
            }}),
                "agent2_3_4_policy":(None, spaces.Box(low=0.0, high=3846000, shape=(9,)), MultiDiscrete([3,3,3,3]),{"model": {
                "fcnet_hiddens": [256,256],  # Two hidden layers with 128 units each
                "fcnet_activation": "tanh",  # Tanh activation function
                },"vf_clip_param": 12.0 }),
                "agent5_6_7_policy":(None, spaces.Box(low=0.0, high=3846000, shape=(9,)), MultiDiscrete([3,3,3,3]),{"model": {
                "fcnet_hiddens": [256,256],  # Two hidden layers with 128 units each
                "fcnet_activation": "tanh",  # Tanh activation function
                }}),
            },
            policy_mapping_fn=mapp,
        )
        # Use GPUs iff `RLLIB_NUM_GPUS` env var set to > 0.
        .resources(num_gpus=int(os.environ.get("RLLIB_NUM_GPUS", "0")))
        .training(
        gamma=0.99,
        lr=0.0004,
        train_batch_size=2000,
        )
        )

    '''
        stop = {
        TRAINING_ITERATION: args.stop_iters,
        NUM_ENV_STEPS_SAMPLED_LIFETIME: args.stop_timesteps,
        f"{ENV_RUNNER_RESULTS}/{EPISODE_RETURN_MEAN}": args.stop_reward,
    }
    
    def custom_stopping_criteria(trial_id, result):
    # Check the latest step rewards from the "info" dictionary
        if "hist_stats" in result:
            step_rewards = result["hist_stats"].get("step_reward", [])
            if any(reward >= 10 for reward in step_rewards):  # Example threshold
                return True
        return False

    stop_criteria = {
    "episodes_total": 1000  # Stop after 100 episodes
    }
    tuner = tune.Tuner(
    PPOTrainer,
    param_space=config.to_dict(),
    run_config=air.RunConfig(stop=stop_criteria, verbose=1),
    )
    
    results = tuner.fit()
    print(results)
    '''
    
    checkpoint_path='/home/master/algorithm_checkpoint_lstm' #change to your checkpoint path
    trainer = PPO(config=config)
    trainer.restore(checkpoint_path)
    print(f"Restored from checkpoint: {checkpoint_path}")
    
    for i in range(100):
        result = trainer.train()
        checkpoint_path = trainer.save('/home/master/algorithm_checkpoint_lstm')
        print(f"Checkpoint saved at {checkpoint_path}")
        break

    ray.shutdown()
