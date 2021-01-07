from typing import List, Dict, Tuple, Any

import numpy as np
import torch as th
from torch.nn import functional as F

from gym.envs.mujoco.mujoco_env import MujocoEnv
from stable_baselines3.sac import SAC
from stable_baselines3.common import logger
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.utils import polyak_update
from stable_baselines3.common.monitor import Monitor


from PETS.env.cartpole import CartpoleEnv
from PETS.defaults import get_cartpole_defaults
from PETS.MPC import MPC


class PETSSAC(SAC):
    def __init__(self, petssac_coef: float = 1.0, mbctrl_retrain_period: int = 5000, *args, **kwargs):
        self.petssac_coef = petssac_coef
        self.mbctrl_retrain_period = mbctrl_retrain_period
        self.env = self._get_from_args_kwargs(
            args, kwargs, argidx=1, argname="env", argisinstance=(MujocoEnv, DummyVecEnv, Monitor)
        )
        self.mbctrl = self._get_mbctrl()
        super().__init__(*args, **kwargs)

    def train(self, gradient_steps: int, batch_size: int) -> None:
        # train / retrain the PETS' dynamic model
        self.train_mbctrl()
        # Update optimizers learning rate
        optimizers = [self.actor.optimizer, self.critic.optimizer]
        if self.ent_coef_optimizer is not None:
            optimizers += [self.ent_coef_optimizer]

        # Update learning rate according to lr schedule
        self._update_learning_rate(optimizers)

        ent_coef_losses, ent_coefs = [], []
        actor_losses, critic_losses, petssac_losses = [], [], []

        for gradient_step in range(gradient_steps):
            # Sample replay buffer
            replay_data = self.replay_buffer.sample(batch_size, env=self._vec_normalize_env)

            # We need to sample because `log_std` may have changed between two gradient steps
            if self.use_sde:
                self.actor.reset_noise()

            # PETS' suggested actions for observations
            actions_mb = self.mbctrl.act(replay_data.observations)
            actions_mb = th.from_numpy(self.policy.scale_action(actions_mb)).float()

            # Action by the current actor for the sampled state
            actions_pi, log_prob = self.actor.action_log_prob(replay_data.observations)
            log_prob = log_prob.reshape(-1, 1)

            ent_coef_loss = None
            if self.ent_coef_optimizer is not None:
                # Important: detach the variable from the graph
                # so we don't change it with other losses
                # see https://github.com/rail-berkeley/softlearning/issues/60
                ent_coef = th.exp(self.log_ent_coef.detach())
                ent_coef_loss = -(self.log_ent_coef * (log_prob + self.target_entropy).detach()).mean()
                ent_coef_losses.append(ent_coef_loss.item())
            else:
                ent_coef = self.ent_coef_tensor

            ent_coefs.append(ent_coef.item())

            # Optimize entropy coefficient, also called
            # entropy temperature or alpha in the paper
            if ent_coef_loss is not None:
                self.ent_coef_optimizer.zero_grad()
                ent_coef_loss.backward()
                self.ent_coef_optimizer.step()

            with th.no_grad():
                # Select action according to policy
                next_actions, next_log_prob = self.actor.action_log_prob(replay_data.next_observations)
                # Compute the target Q value: min over all critics targets
                targets = th.cat(self.critic_target(replay_data.next_observations, next_actions), dim=1)
                target_q, _ = th.min(targets, dim=1, keepdim=True)
                # add entropy term
                target_q = target_q - ent_coef * next_log_prob.reshape(-1, 1)
                # td error + entropy term
                q_backup = replay_data.rewards + (1 - replay_data.dones) * self.gamma * target_q

            # Get current Q estimates for each critic network
            # using action from the replay buffer
            current_q_estimates = self.critic(replay_data.observations, replay_data.actions)

            # Compute critic loss
            critic_loss = 0.5 * sum([F.mse_loss(current_q, q_backup) for current_q in current_q_estimates])
            critic_losses.append(critic_loss.item())

            # Optimize the critic
            self.critic.optimizer.zero_grad()
            critic_loss.backward()
            self.critic.optimizer.step()

            # Compute actor loss
            # Alternative: actor_loss = th.mean(log_prob - qf1_pi)
            # Mean over all critic networks
            q_values_pi = th.cat(self.critic.forward(replay_data.observations, actions_pi), dim=1)
            min_qf_pi, _ = th.min(q_values_pi, dim=1, keepdim=True)
            petssac_loss = F.mse_loss(actions_pi, actions_mb, reduction="none").mean()
            sac_actor_loss = (ent_coef * log_prob - min_qf_pi).mean()
            actor_loss = sac_actor_loss + self.petssac_coef * petssac_loss

            actor_losses.append(actor_loss.item())
            petssac_losses.append(petssac_loss.item())

            # Optimize the actor
            self.actor.optimizer.zero_grad()
            actor_loss.backward()
            self.actor.optimizer.step()

            # Update target networks
            if gradient_step % self.target_update_interval == 0:
                polyak_update(self.critic.parameters(), self.critic_target.parameters(), self.tau)

        self._n_updates += gradient_steps

        logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        logger.record("train/ent_coef", np.mean(ent_coefs))
        logger.record("train/actor_loss", np.mean(actor_losses))
        logger.record("train/critic_loss", np.mean(critic_losses))
        logger.record("train/petssac_loss", np.mean(petssac_losses))
        if len(ent_coef_losses) > 0:
            logger.record("train/ent_coef_loss", np.mean(ent_coef_losses))

    def _get_from_args_kwargs(
        self,
        args: List,
        kwargs: Dict,
        argidx: int = None,
        argname: str = None,
        argisinstance: Any = None,
        pop: bool = False,
    ) -> Any:
        """
        Returns the desired argument from args (using argidx) and kwargs (using argname).
        Priority is with args if not found kwargs.
        """
        arg = None
        if argidx and len(args) > argidx:
            arg = args.pop(argidx) if pop else args[argidx]
        else:
            arg = kwargs.pop(argname, None) if pop else kwargs.get(argname, None)
        if argisinstance:
            assert isinstance(arg, argisinstance), f"Need an instance of {argisinstance}"
        return arg

    def _get_mbctrl(self) -> MPC:
        """
        We want to pass to the MPC the original env without the wrappers
        """
        mbctrl = None
        env = self.env
        if isinstance(self.env, Monitor):
            env = self.env.env
        elif isinstance(self.env, DummyVecEnv):
            env = self.env.envs[0].env
            if isinstance(env, Monitor):
                env = env.env
        else:
            env = self.env
        if isinstance(env, CartpoleEnv):
            mbctrl_params = get_cartpole_defaults()
            mbctrl = MPC(mbctrl_params)
        return mbctrl

    def train_mbctrl(self):
        if self.num_timesteps == self.learning_starts + 1:  # first training call
            batch_size = self.learning_starts
        elif self.num_timesteps % self.mbctrl_retrain_period == 0:  # periodic update of the dynamics model
            self.mbctrl.model_train_cfg["epochs"] = 1
            batch_size = self.batch_size
        else:
            return
        samples = self.replay_buffer.sample(batch_size)
        self.mbctrl.train(obs=samples.observations, next_obs=samples.next_observations, actions=samples.actions)

    def _excluded_save_params(self) -> List[str]:
        """
        Returns the names of the parameters that should be excluded from being
        saved by pickling. E.g. replay buffers are skipped by default
        as they take up a lot of space. PyTorch variables should be excluded
        with this so they can be stored with ``th.save``.

        :return: List of parameters that should be excluded from being saved with pickle.
        """
        return super(SAC, self)._excluded_save_params() + [
            "actor",
            "critic",
            "critic_target",
            "mbctrl",
        ]

    def _get_torch_save_params(self) -> Tuple[List[str], List[str]]:
        """
        Get the name of the torch variables that will be saved with
        PyTorch ``th.save``, ``th.load`` and ``state_dicts`` instead of the default
        pickling strategy. This is to handle device placement correctly.

        Names can point to specific variables under classes, e.g.
        "policy.optimizer" would point to ``optimizer`` object of ``self.policy``
        if this object.

        :return:
            List of Torch variables whose state dicts to save (e.g. th.nn.Modules),
            and list of other Torch variables to store with ``th.save``.
        """
        state_dicts = [
            "policy",
            "actor.optimizer",
            "critic.optimizer",
            "mbctrl.model",
            "mbctrl.model.optim",
        ]
        saved_pytorch_variables = ["log_ent_coef", "mbctrl.has_been_trained", "petssac_coef"]
        if self.ent_coef_optimizer is not None:
            state_dicts.append("ent_coef_optimizer")
        else:
            saved_pytorch_variables.append("ent_coef_tensor")
        return state_dicts, saved_pytorch_variables
