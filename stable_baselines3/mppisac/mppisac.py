from typing import Optional, Callable, List, Tuple

import numpy as np
import torch
from torch.nn import functional as F

from stable_baselines3.common import logger
from stable_baselines3.common.utils import polyak_update
from stable_baselines3.sac import SAC
from stable_baselines3.mppi import MPPICTRL


class MPPISAC(SAC):
    """
    NOTE: Default values are set in order to make the loading of the model to work properly.
    """

    def __init__(
        self,
        mppi_state_dim: int = 2,
        mppi_action_dim: int = 1,
        mppi_action_ub: float = 1.0,
        mppi_action_lb: float = -1.0,
        mppi_model_in_dim: Optional[int] = 1,
        mppi_model_out_dim: Optional[int] = 2,
        mppi_h_units: Optional[int] = 10,
        mppi_state_preproc: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
        mppi_state_postproc: Optional[Callable[[torch.Tensor, torch.Tensor], torch.Tensor]] = None,
        mppi_target_proc: Optional[Callable[[torch.Tensor, torch.Tensor], torch.Tensor]] = None,
        mppi_cost_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] = lambda x, y: 1,
        mppi_sigma: torch.Tensor = torch.tensor(1.0, dtype=torch.float),
        mppi_horizon: int = 10,
        mppi_n_samples: int = 100,
        mppi_lambda: float = 1.0,
        mppi_train_epoch: int = 1,
        mppi_true_dynamics_model: Optional[Callable[[torch.Tensor, torch.Tensor], torch.Tensor]] = None,
        mppisac_coef: float = 1.0,
        _init_setup_model: bool = True,
        *args,
        **kwargs,
    ):
        self.mppi_state_dim = mppi_state_dim
        self.mppi_action_dim = mppi_action_dim
        self.mppi_action_ub = mppi_action_ub
        self.mppi_action_lb = mppi_action_lb
        self.mppi_model_in_dim = None if mppi_true_dynamics_model else mppi_model_in_dim
        self.mppi_model_out_dim = None if mppi_true_dynamics_model else mppi_model_out_dim
        self.mppi_h_units = None if mppi_true_dynamics_model else mppi_h_units
        self.mppi_state_preproc = mppi_state_preproc
        self.mppi_state_postproc = mppi_state_postproc
        self.mppi_target_proc = mppi_target_proc
        self.mppi_cost_fn = mppi_cost_fn
        self.mppi_noise_sigma = mppi_sigma
        self.mppi_horizon = mppi_horizon
        self.mppi_n_samples = mppi_n_samples
        self.mppi_lambda = mppi_lambda
        self.mppi_train_epoch = None if mppi_true_dynamics_model else mppi_train_epoch
        self.mppi_true_dynamics_model = mppi_true_dynamics_model

        self.mppisac_coef = mppisac_coef
        self.use_mppi_true_dynamics_model = True if self.mppi_true_dynamics_model else False

        super().__init__(_init_setup_model=_init_setup_model, *args, **kwargs)

    def _setup_model(self) -> None:
        self.mbctrl = MPPICTRL(
            state_dim=self.mppi_state_dim,
            action_dim=self.mppi_action_dim,
            action_ub=self.mppi_action_ub,
            action_lb=self.mppi_action_lb,
            model_in_dim=self.mppi_model_in_dim,
            model_out_dim=self.mppi_model_out_dim,
            h_units=self.mppi_h_units,
            state_preproc=self.mppi_state_preproc,
            state_postproc=self.mppi_state_postproc,
            target_proc=self.mppi_target_proc,
            cost_fn=self.mppi_cost_fn,
            noise_sigma=self.mppi_noise_sigma,
            horizon=self.mppi_horizon,
            n_samples=self.mppi_n_samples,
            lambda_=self.mppi_lambda,
            train_epoch=self.mppi_train_epoch,
            true_dynamics_model=self.mppi_true_dynamics_model,
        )
        super()._setup_model()

    def train(self, gradient_steps: int, batch_size: int) -> None:
        # TODO: use the info below to make a cosine schedule for mppisac_coef
        # TODO: schedule can be a subclass of stable_baselines.common.schedules
        # print(f"Progress remaining: {self._current_progress_remaining}")
        # print(f"total_timesteps: {self._total_timesteps}")
        # print(f"num timesteps: {self.num_timesteps}")
        # print(f"learning starts: {self.learning_starts}")
        # Update optimizers learning rate
        optimizers = [self.actor.optimizer, self.critic.optimizer]
        if self.ent_coef_optimizer is not None:
            optimizers += [self.ent_coef_optimizer]

        # Update learning rate according to lr schedule
        self._update_learning_rate(optimizers)

        ent_coef_losses, ent_coefs = [], []
        actor_losses, critic_losses, mppisac_losses = [], [], []

        for gradient_step in range(gradient_steps):
            # Sample replay buffer
            replay_data = self.replay_buffer.sample(batch_size, env=self._vec_normalize_env)

            # train model based MPPICTRL dynamic model
            self.mbctrl.train(
                states=replay_data.observations, next_states=replay_data.next_observations, actions=replay_data.actions
            )

            # We need to sample because `log_std` may have changed between two gradient steps
            if self.use_sde:
                self.actor.reset_noise()

            # MPPICTRL's suggested actions for observations
            actions_mb = self.mbctrl.act(replay_data.observations)
            actions_mb = self.policy.scale_action(actions_mb)

            # Action by the current actor for the sampled state
            actions_pi, log_prob = self.actor.action_log_prob(replay_data.observations)
            log_prob = log_prob.reshape(-1, 1)

            ent_coef_loss = None
            if self.ent_coef_optimizer is not None:
                # Important: detach the variable from the graph
                # so we don't change it with other losses
                # see https://github.com/rail-berkeley/softlearning/issues/60
                ent_coef = torch.exp(self.log_ent_coef.detach())
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

            with torch.no_grad():
                # Select action according to policy
                next_actions, next_log_prob = self.actor.action_log_prob(replay_data.next_observations)
                # Compute the target Q value: min over all critics targets
                targets = torch.cat(self.critic_target(replay_data.next_observations, next_actions), dim=1)
                target_q, _ = torch.min(targets, dim=1, keepdim=True)
                # add entropy term
                target_q = target_q - ent_coef * next_log_prob.reshape(-1, 1)
                # td error + entropy term
                q_backup = replay_data.rewards + (1 - replay_data.dones) * self.gamma * target_q

                # q(s, MPPI_action)
                actions_mb_qs = torch.cat(self.critic_target(replay_data.observations, actions_mb), dim=1)
                actions_mb_q, _ = torch.min(actions_mb_qs, dim=1, keepdim=True)

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
            q_values_pi = torch.cat(self.critic.forward(replay_data.observations, actions_pi), dim=1)
            min_qf_pi, _ = torch.min(q_values_pi, dim=1, keepdim=True)
            mppisac_loss = F.mse_loss(min_qf_pi, actions_mb_q, reduction="none").mean()
            sac_actor_loss = (ent_coef * log_prob - min_qf_pi).mean()
            actor_loss = sac_actor_loss + self.mppisac_coef * mppisac_loss

            actor_losses.append(actor_loss.item())
            mppisac_losses.append(mppisac_loss.item())

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
        logger.record("train/mppi-sac_loss", np.mean(mppisac_losses))
        if len(ent_coef_losses) > 0:
            logger.record("train/ent_coef_loss", np.mean(ent_coef_losses))

    def _excluded_save_params(self) -> List[str]:
        """
        Returns the names of the parameters that should be excluded from being
        saved by pickling. E.g. replay buffers are skipped by default
        as they take up a lot of space. PyTorch variables should be excluded
        with this so they can be stored with ``th.save``.

        :return: List of parameters that should be excluded from being saved with pickle.
        """
        return super()._excluded_save_params() + ["mbctrl"]

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
        ]
        if not self.use_mppi_true_dynamics_model:
            state_dicts.extend(["mbctrl.model", "mbctrl.optimizer"])  # mppi model is a torch nn model

        saved_pytorch_variables = ["log_ent_coef"]
        if self.ent_coef_optimizer is not None:
            state_dicts.append("ent_coef_optimizer")
        else:
            saved_pytorch_variables.append("ent_coef_tensor")
        return state_dicts, saved_pytorch_variables
