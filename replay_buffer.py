import copy
import time

import numpy
import ray
import torch

import models


@ray.remote
class ReplayBuffer:
    """
    Class which run in a dedicated thread to store explored simulations and generate batch.
    """

    def __init__(self, initial_checkpoint, initial_buffer, config):
        self.config = config
        self.buffer = copy.deepcopy(initial_buffer)
        self.num_performed_simulations = initial_checkpoint["num_performed_simulations"]
        self.num_explored_steps = initial_checkpoint["num_explored_steps"]
        self.total_samples = sum(
            [len(simulation_history.root_values) for simulation_history in self.buffer.values()]
        )
        if self.total_samples != 0:
            print(
                f"Replay buffer initialized with {self.total_samples} samples ({self.num_performed_simulations} simulations).\n"
            )

        # Fix random generator seed
        numpy.random.seed(self.config.seed)

    def save_simulation(self, simulation_history, shared_storage=None):
        if self.config.PER:
            if simulation_history.priorities is not None:
                # Avoid read only array when loading replay buffer from disk
                simulation_history.priorities = numpy.copy(simulation_history.priorities)
            else:
                # Initial priorities for the prioritized replay (See paper appendix Training)
                priorities = []
                for i, root_value in enumerate(simulation_history.root_values):
                    priority = (
                        numpy.abs(
                            root_value - self.compute_target_value(simulation_history, i)
                        )
                        ** self.config.PER_alpha
                    )
                    priorities.append(priority)

                simulation_history.priorities = numpy.array(priorities, dtype="float32")
                simulation_history.simulation_priority = numpy.max(simulation_history.priorities)

        self.buffer[self.num_performed_simulations] = simulation_history
        self.num_performed_simulations += 1
        self.num_explored_steps += len(simulation_history.root_values)
        self.total_samples += len(simulation_history.root_values)

        if self.config.replay_buffer_size < len(self.buffer):
            del_id = self.num_performed_simulations - len(self.buffer)
            self.total_samples -= len(self.buffer[del_id].root_values)
            del self.buffer[del_id]

        if shared_storage:
            shared_storage.set_info.remote("num_performed_simulations", self.num_performed_simulations)
            shared_storage.set_info.remote("num_explored_steps", self.num_explored_steps)

    def get_buffer(self):
        return self.buffer

    def get_batch(self):
        (
            index_batch,
            theory_batch,
            theory_mask_batch,
            theory_dynamics_batch,
            action_batch,
            reward_batch,
            value_batch,
            policy_batch,
            gradient_scale_batch,
        ) = ([], [], [], [], [], [], [], [], [])
        weight_batch = [] if self.config.PER else None

        for simulation_id, simulation_history, simulation_prob in self.sample_n_simulations(
            self.config.batch_size
        ):
            simulation_pos, pos_prob = self.sample_position(simulation_history)

            values, rewards, policies, actions = self.make_target(
                simulation_history, simulation_pos
            )

            index_batch.append([simulation_id, simulation_pos])
            theory, theory_mask, theory_dynamics = simulation_history.\
                preprocess_observations(
                    simulation_pos,
                    self.config.stacked_observations,
                    len(self.config.action_space),
                )
            theory_batch.append(theory)
            theory_mask_batch.append(theory_mask)
            theory_dynamics_batch.append(theory_dynamics)
            action_batch.append(actions)
            value_batch.append(values)
            reward_batch.append(rewards)
            policy_batch.append(policies)
            gradient_scale_batch.append(
                [
                    min(
                        self.config.num_unroll_steps,
                        len(simulation_history.action_history) - simulation_pos,
                    )
                ]
                * len(actions)
            )
            if self.config.PER:
                weight_batch.append(1 / (self.total_samples * simulation_prob * pos_prob))

        if self.config.PER:
            weight_batch = numpy.array(weight_batch, dtype="float32") / max(
                weight_batch
            )

        # action_batch: batch, num_unroll_steps+1
        # value_batch: batch, num_unroll_steps+1
        # reward_batch: batch, num_unroll_steps+1
        # policy_batch: batch, num_unroll_steps+1, len(action_space)
        # weight_batch: batch
        # gradient_scale_batch: batch, num_unroll_steps+1
        return (
            index_batch,
            (
                theory_batch,
                theory_mask_batch,
                theory_dynamics_batch,
                action_batch,
                value_batch,
                reward_batch,
                policy_batch,
                weight_batch,
                gradient_scale_batch,
            ),
        )

    def sample_simulation(self, force_uniform=False):
        """
        Sample simulation from buffer either uniformly or according to some priority.
        See paper appendix Training.
        """
        simulation_prob = None
        if self.config.PER and not force_uniform:
            simulation_probs = numpy.array(
                [simulation_history.simulation_priority for simulation_history in self.buffer.values()],
                dtype="float32",
            )
            simulation_probs /= numpy.sum(simulation_probs)
            simulation_index = numpy.random.choice(len(self.buffer), p=simulation_probs)
            simulation_prob = simulation_probs[simulation_index]
        else:
            simulation_index = numpy.random.choice(len(self.buffer))
        simulation_id = self.num_performed_simulations - len(self.buffer) + simulation_index

        return simulation_id, self.buffer[simulation_id], simulation_prob

    def sample_n_simulations(self, n_simulations, force_uniform=False):
        if self.config.PER and not force_uniform:
            simulation_id_list = []
            simulation_probs = []
            for simulation_id, simulation_history in self.buffer.items():
                simulation_id_list.append(simulation_id)
                simulation_probs.append(simulation_history.simulation_priority)
            simulation_probs = numpy.array(simulation_probs, dtype="float32")
            simulation_probs /= numpy.sum(simulation_probs)
            simulation_prob_dict = dict(
                [(simulation_id, prob) for simulation_id, prob in zip(simulation_id_list, simulation_probs)]
            )
            selected_simulations = numpy.random.choice(simulation_id_list, n_simulations, p=simulation_probs)
        else:
            selected_simulations = numpy.random.choice(list(self.buffer.keys()), n_simulations)
            simulation_prob_dict = {}
        ret = [
            (simulation_id, self.buffer[simulation_id], simulation_prob_dict.get(simulation_id))
            for simulation_id in selected_simulations
        ]
        return ret

    def sample_position(self, simulation_history, force_uniform=False):
        """
        Sample position from simulation either uniformly or according to some priority.
        See paper appendix Training.
        """
        position_prob = None
        if self.config.PER and not force_uniform:
            position_probs = simulation_history.priorities / sum(simulation_history.priorities)
            position_index = numpy.random.choice(len(position_probs), p=position_probs)
            position_prob = position_probs[position_index]
        else:
            position_index = numpy.random.choice(len(simulation_history.root_values))

        return position_index, position_prob

    def update_simulation_history(self, simulation_id, simulation_history):
        # The element could have been removed since its selection and update
        if next(iter(self.buffer)) <= simulation_id:
            if self.config.PER:
                # Avoid read only array when loading replay buffer from disk
                simulation_history.priorities = numpy.copy(simulation_history.priorities)
            self.buffer[simulation_id] = simulation_history

    def update_priorities(self, priorities, index_info):
        """
        Update simulation and position priorities with priorities calculated during the training.
        See Distributed Prioritized Experience Replay https://arxiv.org/abs/1803.00933
        """
        for i in range(len(index_info)):
            simulation_id, simulation_pos = index_info[i]

            # The element could have been removed since its selection and training
            if next(iter(self.buffer)) <= simulation_id:
                # Update position priorities
                priority = priorities[i, :]
                start_index = simulation_pos
                end_index = min(
                    simulation_pos + len(priority), len(self.buffer[simulation_id].priorities)
                )
                self.buffer[simulation_id].priorities[start_index:end_index] = priority[
                    : end_index - start_index
                ]

                # Update simulation priorities
                self.buffer[simulation_id].simulation_priority = numpy.max(
                    self.buffer[simulation_id].priorities
                )

    def compute_target_value(self, simulation_history, index):
        # The value target is the discounted root value of the search tree td_steps into the
        # future, plus the discounted sum of all rewards until then.
        bootstrap_index = index + self.config.td_steps
        if bootstrap_index < len(simulation_history.root_values):
            root_values = (
                simulation_history.root_values
                if simulation_history.reanalysed_predicted_root_values is None
                else simulation_history.reanalysed_predicted_root_values
            )
            last_step_value = (
                root_values[bootstrap_index]
                if simulation_history.to_explore_history[bootstrap_index]
                == simulation_history.to_explore_history[index]
                else -root_values[bootstrap_index]
            )

            value = last_step_value * self.config.discount**self.config.td_steps
        else:
            value = 0

        for i, reward in enumerate(
            simulation_history.reward_history[index + 1 : bootstrap_index + 1]
        ):
            # The value is oriented from the perspective of the current agent
            value += (
                reward
                if simulation_history.to_explore_history[index]
                == simulation_history.to_explore_history[index + i]
                else -reward
            ) * self.config.discount**i

        return value

    def make_target(self, simulation_history, state_index):
        """
        Generate targets for every unroll steps.
        """
        target_values, target_rewards, target_policies, actions = [], [], [], []
        for current_index in range(
            state_index, state_index + self.config.num_unroll_steps + 1
        ):
            value = self.compute_target_value(simulation_history, current_index)

            if current_index < len(simulation_history.root_values):
                target_values.append(value)
                target_rewards.append(simulation_history.reward_history[current_index])
                target_policies.append(simulation_history.child_visits[current_index])
                actions.append(simulation_history.action_history[current_index])
            elif current_index == len(simulation_history.root_values):
                target_values.append(0)
                target_rewards.append(simulation_history.reward_history[current_index])
                # Uniform policy
                target_policies.append(
                    [
                        1 / len(simulation_history.child_visits[0])
                        for _ in range(len(simulation_history.child_visits[0]))
                    ]
                )
                actions.append(simulation_history.action_history[current_index])
            else:
                # States past the end of simulations are treated as absorbing states
                target_values.append(0)
                target_rewards.append(0)
                # Uniform policy
                target_policies.append(
                    [
                        1 / len(simulation_history.child_visits[0])
                        for _ in range(len(simulation_history.child_visits[0]))
                    ]
                )
                actions.append(numpy.random.choice(self.config.action_space))

        return target_values, target_rewards, target_policies, actions


@ray.remote
class Reanalyse:
    """
    Class which run in a dedicated thread to update the replay buffer with fresh information.
    See paper appendix Reanalyse.
    """

    def __init__(self, initial_checkpoint, config):
        self.config = config

        # Fix random generator seed
        numpy.random.seed(self.config.seed)
        torch.manual_seed(self.config.seed)

        # Initialize the network
        self.model = models.MuZeroNetwork(self.config)
        self.model.set_weights(initial_checkpoint["weights"])
        self.model.to(torch.device("cuda" if self.config.reanalyse_on_gpu else "cpu"))
        self.model.eval()

        self.num_reanalysed_simulations = initial_checkpoint["num_reanalysed_simulations"]

    def reanalyse(self, replay_buffer, shared_storage):
        while ray.get(shared_storage.get_info.remote("num_performed_simulations")) < 1:
            time.sleep(0.1)

        while ray.get(
            shared_storage.get_info.remote("training_step")
        ) < self.config.training_steps and not ray.get(
            shared_storage.get_info.remote("terminate")
        ):
            self.model.set_weights(ray.get(shared_storage.get_info.remote("weights")))

            simulation_id, simulation_history, _ = ray.get(
                replay_buffer.sample_simulation.remote(force_uniform=True)
            )

            # Use the last model to provide a fresher, stable n-step value (See paper appendix Reanalyze)
            if self.config.use_last_model_value:
                theory, theory_mask, theory_dynamics = [], [], []
                for i in range(len(simulation_history.root_values)):
                    _theory, _theory_mask, _theory_dynamics = simulation_history.preprocess_observations(
                        i,
                        self.config.stacked_observations,
                        len(self.config.action_space),
                    )
                    theory.append(_theory)
                    theory_mask.append(_theory_mask)
                    theory_dynamics.append(_theory_dynamics)

                
                theory = torch.tensor(numpy.array(theory), dtype=torch.int).to(next(self.model.parameters()).device)
                theory_mask = torch.tensor(numpy.array(theory_mask), dtype=torch.bool).to(next(self.model.parameters()).device)
                theory_dynamics = torch.tensor(numpy.array(theory_dynamics)).float().to(next(self.model.parameters()).device)
                values = models.support_to_scalar(
                    self.model.initial_inference(theory, theory_mask, theory_dynamics)[0],
                    self.config.support_size,
                )
                simulation_history.reanalysed_predicted_root_values = (
                    torch.squeeze(values).detach().cpu().numpy()
                )

            replay_buffer.update_simulation_history.remote(simulation_id, simulation_history)
            self.num_reanalysed_simulations += 1
            shared_storage.set_info.remote(
                "num_reanalysed_simulations", self.num_reanalysed_simulations
            )
