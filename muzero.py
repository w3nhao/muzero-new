import copy
import importlib
import json
import math
import pathlib
import pickle
import sys
import time

import nevergrad
import numpy
import ray
import torch
from torch.utils.tensorboard import SummaryWriter

import diagnose_model
import models
import replay_buffer
import mcts
import shared_storage
import trainer


class MuZero:
    """
    Main class to manage MuZero.

    Args:
        simulation_name (str): Name of the simulation module, it should match the name of a .py file
        in the "./simulations" directory.

        config (dict, MuZeroConfig, optional): Override the default config of the simulation.

        split_resources_in (int, optional): Split the GPU usage when using concurent muzero instances.

    Example:
        >>> muzero = MuZero("cartpole")
        >>> muzero.train()
        >>> muzero.test(render=True)
    """

    def __init__(self, simulation_name, config=None, split_resources_in=1):
        # Load the simulation and the config from the module with the simulation name
        try:
            simulation_module = importlib.import_module("simulations." + simulation_name)
            self.Simulation = simulation_module.Simulation
            self.config = simulation_module.MuZeroConfig()
        except ModuleNotFoundError as err:
            print(f'{simulation_name} is not a supported simulation name, try "cartpole" or refer to the documentation for adding a new simulation.')
            raise err

        # Overwrite the config
        if config:
            if type(config) is dict:
                for param, value in config.items():
                    if hasattr(self.config, param):
                        setattr(self.config, param, value)
                    else:
                        raise AttributeError(f"{simulation_name} config has no attribute '{param}'. Check the config file for the complete list of parameters.")
            else:
                self.config = config

        # Fix random generator seed
        numpy.random.seed(self.config.seed)
        torch.manual_seed(self.config.seed)

        # Manage GPUs
        if self.config.max_num_gpus == 0 and (
            self.config.self_explore_on_gpu
            or self.config.train_on_gpu
            or self.config.reanalyse_on_gpu
        ):
            raise ValueError("Inconsistent MuZeroConfig: max_num_gpus = 0 but GPU requested by self_explore_on_gpu or train_on_gpu or reanalyse_on_gpu.")
        
        if (
            self.config.self_explore_on_gpu
            or self.config.train_on_gpu
            or self.config.reanalyse_on_gpu
        ):
            total_gpus = (
                self.config.max_num_gpus
                if self.config.max_num_gpus is not None
                else torch.cuda.device_count()
            )
        else:
            total_gpus = 0
            
        self.num_gpus = total_gpus / split_resources_in
        
        if 1 < self.num_gpus:
            self.num_gpus = math.floor(self.num_gpus)

        ray.init(num_gpus=total_gpus, ignore_reinit_error=True)

        # Checkpoint and replay buffer used to initialize workers
        self.checkpoint = {
            "weights": None,
            "optimizer_state": None,
            "total_reward": 0,
            "muzero_reward": 0,
            "opponent_reward": 0,
            "episode_length": 0,
            "mean_value": 0,
            "training_step": 0,
            "lr": 0,
            "total_loss": 0,
            "value_loss": 0,
            "reward_loss": 0,
            "policy_loss": 0,
            "num_performed_simulations": 0,
            "num_explored_steps": 0,
            "num_reanalysed_simulations": 0,
            "terminate": False,
        }
        self.replay_buffer = {}

        cpu_actor = CPUActor.remote()
        cpu_weights = cpu_actor.get_initial_weights.remote(self.config)
        self.checkpoint["weights"], self.summary = copy.deepcopy(ray.get(cpu_weights))

        # Workers
        self.self_explore_workers = None
        self.test_worker = None
        self.training_worker = None
        self.reanalyse_worker = None
        self.replay_buffer_worker = None
        self.shared_storage_worker = None

    def train(self, log_in_tensorboard=True):
        """
        Spawn ray workers and launch the training.

        Args:
            log_in_tensorboard (bool): Start a testing worker and log its performance in TensorBoard.
        """
        if log_in_tensorboard or self.config.save_model:
            self.config.results_path.mkdir(parents=True, exist_ok=True)

        # Manage GPUs
        if 0 < self.num_gpus:
            num_gpus_per_worker = self.num_gpus / (
                self.config.train_on_gpu
                + self.config.num_workers * self.config.self_explore_on_gpu
                + log_in_tensorboard * self.config.self_explore_on_gpu
                + self.config.use_last_model_value * self.config.reanalyse_on_gpu
            )
            if 1 < num_gpus_per_worker:
                num_gpus_per_worker = math.floor(num_gpus_per_worker)
        else:
            num_gpus_per_worker = 0

        # Initialize workers
        self.training_worker = trainer.Trainer.options(
            num_cpus=0,
            num_gpus=num_gpus_per_worker if self.config.train_on_gpu else 0,
        ).remote(self.checkpoint, self.config)

        self.shared_storage_worker = shared_storage.SharedStorage.remote(
            self.checkpoint,
            self.config,
        )
        self.shared_storage_worker.set_info.remote("terminate", False)

        self.replay_buffer_worker = replay_buffer.ReplayBuffer.remote(
            self.checkpoint, self.replay_buffer, self.config
        )

        if self.config.use_last_model_value:
            self.reanalyse_worker = replay_buffer.Reanalyse.options(
                num_cpus=0,
                num_gpus=num_gpus_per_worker if self.config.reanalyse_on_gpu else 0,
            ).remote(self.checkpoint, self.config)

        self.self_explore_workers = [
            mcts.SelfExploration.options(
                num_cpus=0,
                num_gpus=num_gpus_per_worker if self.config.self_explore_on_gpu else 0,
            ).remote(
                self.checkpoint,
                self.Simulation,
                self.config,
                self.config.seed + seed,
            )
            for seed in range(self.config.num_workers)
        ]

        # Launch workers
        [
            self_explore_worker.continuous_self_exploration.remote(
                self.shared_storage_worker, self.replay_buffer_worker
            )
            for self_explore_worker in self.self_explore_workers
        ]
        self.training_worker.continuous_update_weights.remote(
            self.replay_buffer_worker, self.shared_storage_worker
        )
        if self.config.use_last_model_value:
            self.reanalyse_worker.reanalyse.remote(
                self.replay_buffer_worker, self.shared_storage_worker
            )

        if log_in_tensorboard:
            self.logging_loop(num_gpus_per_worker if self.config.self_explore_on_gpu else 0)

    def logging_loop(self, num_gpus):
        """
        Keep track of the training performance.
        """
        # Launch the test worker to get performance metrics
        self.test_worker = mcts.SelfExploration \
            .options(num_cpus=0, num_gpus=num_gpus) \
            .remote(
                self.checkpoint,
                self.Simulation,
                self.config,
                self.config.seed + self.config.num_workers,
            )
        self.test_worker.continuous_self_exploration. \
            remote(self.shared_storage_worker, None, True)

        # Write everything in TensorBoard
        writer = SummaryWriter(self.config.results_path)
        print("\nTraining...\nRun tensorboard --logdir ./results and go to http://localhost:6006/ to see in real time the training performance.\n")

        # Save hyperparameters to TensorBoard
        hp_table = [f"| {key} | {value} |" for key, value in self.config.__dict__.items()]
        writer.add_text("Hyperparameters", "| Parameter | Value |\n|-------|-------|\n" + "\n".join(hp_table))
        
        # Save model representation
        writer.add_text("Model summary", self.summary)
        
        # Loop for updating the training performance
        counter = 0
        keys = [
            "total_reward",
            "muzero_reward",
            "opponent_reward",
            "episode_length",
            "mean_value",
            "training_step",
            "lr",
            "total_loss",
            "value_loss",
            "reward_loss",
            "policy_loss",
            "num_performed_simulations",
            "num_explored_steps",
            "num_reanalysed_simulations",
        ]
        info = ray.get(self.shared_storage_worker.get_info.remote(keys))
        try:
            while info["training_step"] < self.config.training_steps:
                info = ray.get(self.shared_storage_worker.get_info.remote(keys))
                writer.add_scalar("1.Total_reward/1.Total_reward", info["total_reward"], counter)
                writer.add_scalar("1.Total_reward/2.Mean_value", info["mean_value"], counter)
                writer.add_scalar("1.Total_reward/3.Episode_length", info["episode_length"], counter)
                writer.add_scalar("1.Total_reward/4.MuZero_reward", info["muzero_reward"], counter)
                writer.add_scalar("1.Total_reward/5.Opponent_reward", info["opponent_reward"], counter)
                writer.add_scalar("2.Workers/1.self_explored_simulations", info["num_performed_simulations"], counter)
                writer.add_scalar("2.Workers/2.Training_steps", info["training_step"], counter)
                writer.add_scalar("2.Workers/3.self_explored_steps", info["num_explored_steps"], counter)
                writer.add_scalar("2.Workers/4.Reanalysed_simulations", info["num_reanalysed_simulations"], counter)
                writer.add_scalar("2.Workers/5.Training_steps_per_self_explored_step_ratio", info["training_step"] / max(1, info["num_explored_steps"]), counter)
                writer.add_scalar("2.Workers/6.Learning_rate", info["lr"], counter)
                writer.add_scalar("3.Loss/1.Total_weighted_loss", info["total_loss"], counter)
                writer.add_scalar("3.Loss/Value_loss", info["value_loss"], counter)
                writer.add_scalar("3.Loss/Reward_loss", info["reward_loss"], counter)
                writer.add_scalar("3.Loss/Policy_loss", info["policy_loss"], counter)
                print(f'Last test reward: {info["total_reward"]:.2f}. Training step: {info["training_step"]}/{self.config.training_steps}. Played simulations: {info["num_performed_simulations"]}. Loss: {info["total_loss"]:.2f}', end="\r")
                counter += 1
                time.sleep(0.5)
        except KeyboardInterrupt:
            pass

        self.terminate_workers()

        if self.config.save_model:
            # Persist replay buffer to disk
            path = self.config.results_path / "replay_buffer.pkl"
            print(f"\n\nPersisting replay buffer simulations to disk at {path}")
            pickle.dump(
                {
                    "buffer": self.replay_buffer,
                    "num_performed_simulations": self.checkpoint["num_performed_simulations"],
                    "num_explored_steps": self.checkpoint["num_explored_steps"],
                    "num_reanalysed_simulations": self.checkpoint["num_reanalysed_simulations"],
                },
                open(path, "wb"),
            )

    def terminate_workers(self):
        """
        Softly terminate the running tasks and garbage collect the workers.
        """
        if self.shared_storage_worker:
            self.shared_storage_worker.set_info.remote("terminate", True)
            self.checkpoint = ray.get(self.shared_storage_worker.get_checkpoint.remote())
        if self.replay_buffer_worker:
            self.replay_buffer = ray.get(self.replay_buffer_worker.get_buffer.remote())

        print("\nShutting down workers...")

        self.self_explore_workers = None
        self.test_worker = None
        self.training_worker = None
        self.reanalyse_worker = None
        self.replay_buffer_worker = None
        self.shared_storage_worker = None

    def test(self, render=True, opponent=None, muzero_agent=None, num_tests=1, num_gpus=0):
        """
        Test the model in a dedicated thread.

        Args:
            render (bool): To display or not the environment. Defaults to True.

            opponent (str): "self" for self-explore, "human" for exploreing against MuZero and "random"
            for a random agent, None will use the opponent in the config. Defaults to None.

            muzero_agent (int): Agent number of MuZero in case of multi-agent.
            simulations, None let MuZero explore all agents turn by turn, None will use muzero_agent in
            the config. Defaults to None.

            num_tests (int): Number of simulations to average. Defaults to 1.

            num_gpus (int): Number of GPUs to use, 0 forces to use the CPU. Defaults to 0.
        """
        opponent = opponent if opponent else self.config.opponent
        muzero_agent = muzero_agent if muzero_agent else self.config.muzero_agent
        self_explore_worker = mcts.SelfExploration \
            .options(num_cpus=0,num_gpus=num_gpus) \
            .remote(self.checkpoint, self.Simulation, self.config, numpy.random.randint(10000))
        
        results = []
        for i in range(num_tests):
            print(f"Testing {i+1}/{num_tests}")
            results.append(ray.get(
                self_explore_worker.explore_simulation.remote(0, 0, render, opponent, muzero_agent)))
        
        self_explore_worker.close_simulation.remote()

        if len(self.config.agents) == 1:
            result = numpy.mean([sum(history.reward_history) for history in results])
        else:
            result = numpy.mean(
                [
                    sum(
                        reward
                        for i, reward in enumerate(history.reward_history)
                        if history.to_explore_history[i - 1] == muzero_agent
                    )
                    for history in results
                ]
            )
        return result

    def load_model(self, checkpoint_path=None, replay_buffer_path=None):
        """
        Load a model and/or a saved replay buffer.

        Args:
            checkpoint_path (str): Path to model.checkpoint or model.weights.

            replay_buffer_path (str): Path to replay_buffer.pkl
        """
        # Load checkpoint
        if checkpoint_path:
            checkpoint_path = pathlib.Path(checkpoint_path)
            self.checkpoint = torch.load(checkpoint_path)
            print(f"\nUsing checkpoint from {checkpoint_path}")

        # Load replay buffer
        if replay_buffer_path:
            replay_buffer_path = pathlib.Path(replay_buffer_path)
            with open(replay_buffer_path, "rb") as f:
                replay_buffer_infos = pickle.load(f)
            self.replay_buffer = replay_buffer_infos["buffer"]
            self.checkpoint["num_explored_steps"] = replay_buffer_infos[
                "num_explored_steps"
            ]
            self.checkpoint["num_performed_simulations"] = replay_buffer_infos[
                "num_performed_simulations"
            ]
            self.checkpoint["num_reanalysed_simulations"] = replay_buffer_infos[
                "num_reanalysed_simulations"
            ]

            print(f"\nInitializing replay buffer with {replay_buffer_path}")
        else:
            print(f"Using empty buffer.")
            self.replay_buffer = {}
            self.checkpoint["training_step"] = 0
            self.checkpoint["num_explored_steps"] = 0
            self.checkpoint["num_performed_simulations"] = 0
            self.checkpoint["num_reanalysed_simulations"] = 0

    def diagnose_model(self, horizon):
        """
        Perform a simulation only with the learned model then explore the same trajectory in the real
        environment and display information.

        Args:
            horizon (int): Number of timesteps for which we collect information.
        """
        simulation = self.Simulation(self.config.seed)
        obs = simulation.reset()
        dm = diagnose_model.DiagnoseModel(self.checkpoint, self.config)
        dm.compare_virtual_with_real_trajectories(obs, simulation, horizon)
        input("Press enter to close all plots")
        dm.close_all()


@ray.remote(num_cpus=0, num_gpus=0)
class CPUActor:
    # Trick to force DataParallel to stay on CPU to get weights on CPU even if there is a GPU
    def __init__(self):
        pass

    def get_initial_weights(self, config):
        model = models.MuZeroNetwork(config)
        weigths = model.get_weights()
        summary = str(model).replace("\n", " \n\n")
        return weigths, summary


def load_model_menu(muzero, simulation_name):
    # Configure running options
    options = ["Specify paths manually"] + sorted(
        (pathlib.Path("results") / simulation_name).glob("*/")
    )
    options.reverse()
    print()
    for i in range(len(options)):
        print(f"{i}. {options[i]}")

    choice = input("Enter a number to choose a model to load: ")
    valid_inputs = [str(i) for i in range(len(options))]
    while choice not in valid_inputs:
        choice = input("Invalid input, enter a number listed above: ")
    choice = int(choice)

    if choice == (len(options) - 1):
        # manual path option
        checkpoint_path = input(
            "Enter a path to the model.checkpoint, or ENTER if none: "
        )
        while checkpoint_path and not pathlib.Path(checkpoint_path).is_file():
            checkpoint_path = input("Invalid checkpoint path. Try again: ")
        replay_buffer_path = input(
            "Enter a path to the replay_buffer.pkl, or ENTER if none: "
        )
        while replay_buffer_path and not pathlib.Path(replay_buffer_path).is_file():
            replay_buffer_path = input("Invalid replay buffer path. Try again: ")
    else:
        checkpoint_path = options[choice] / "model.checkpoint"
        replay_buffer_path = options[choice] / "replay_buffer.pkl"

    muzero.load_model(
        checkpoint_path=checkpoint_path,
        replay_buffer_path=replay_buffer_path,
    )


if __name__ == "__main__":
    if len(sys.argv) == 2:
        # Train directly with: python muzero.py ecosystem
        muzero = MuZero(sys.argv[1])
        muzero.train()
    elif len(sys.argv) == 3:
        # Train directly with: python muzero.py ecosystem '{"lr_init": 0.01}'
        config = json.loads(sys.argv[2])
        muzero = MuZero(sys.argv[1], config)
        muzero.train()
    else:
        print("\nWelcome to MuZero! Here's a list of simulations:")
        # Let user pick a simulation
        simulations = [
            filename.stem
            for filename in sorted(list((pathlib.Path.cwd() / "simulations").glob("*.py")))
            if filename.name != "utils.py" and filename.name != "__init__.py"
        ]
        for i in range(len(simulations)):
            print(f"{i}. {simulations[i]}")
        choice = input("Enter a number to choose the simulation: ")
        valid_inputs = [str(i) for i in range(len(simulations))]
        while choice not in valid_inputs:
            choice = input("Invalid input, enter a number listed above: ")

        # Initialize MuZero
        choice = int(choice)
        simulation_name = simulations[choice]
        muzero = MuZero(simulation_name)

        while True:
            # Configure running options
            options = [
                "Train",
                "Load pretrained model",
                "Diagnose model",
                "Render some self explore simulations",
                "Explore with CoPilot MuZero",
                "Test the simulation manually",
                "Exit",
            ]
            print()
            for i in range(len(options)):
                print(f"{i}. {options[i]}")

            choice = input("Enter a number to choose an action: ")
            valid_inputs = [str(i) for i in range(len(options))]
            while choice not in valid_inputs:
                choice = input("Invalid input, enter a number listed above: ")
            choice = int(choice)
            if choice == 0:
                muzero.train()
            elif choice == 1:
                load_model_menu(muzero, simulation_name)
            elif choice == 2:
                muzero.diagnose_model(30)
            elif choice == 3:
                muzero.test(render=True, opponent="self", muzero_agent=None)
            elif choice == 4:
                muzero.test(render=True, opponent="human", muzero_agent=0)
            elif choice == 5:
                env = muzero.Simulation(muzero.config)
                env.reset()
                env.render()

                done = False
                while not done:
                    action = env.human_to_action()
                    observation, reward, done = env.step(action)
                    print(f"\nAction: {env.action_to_string(action)}\nReward: {reward}")
                    env.render()
            else:
                break
            print("\nDone")

    ray.shutdown()
