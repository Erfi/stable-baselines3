from typing import Optional, Callable

import numpy as np
import torch
from torch.nn import functional as F

from stable_baselines3.common import logger
from stable_baselines3.common.utils import polyak_update
from stable_baselines3.sac import SAC
from stable_baselines3.mppi import MPPICTRL


class MPPISAC(SAC):
    def __init__(
        self,
        mppi_state_dim: int,
        mppi_action_dim: int,
        mppi_action_ub: float,
        mppi_action_lb: float,
        mppi_model_in_dim: Optional[int],
        mppi_model_out_dim: Optional[int],
        mppi_h_units: Optional[int],
        mppi_state_preproc: Optional[Callable[[torch.Tensor], torch.Tensor]],
        mppi_state_postproc: Optional[Callable[[torch.Tensor, torch.Tensor], torch.Tensor]],
        mppi_target_proc: Optional[Callable[[torch.Tensor, torch.Tensor], torch.Tensor]],
        mppi_cost_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        mppi_sigma: torch.Tensor,
        mppi_horizon: int = 10,
        mppi_n_samples: int = 100,
        mppi_lambda: float = 1.0,
        mppi_train_epoch: int = 1,
        mppi_true_dynamics_model: Optional[Callable[[torch.Tensor, torch.Tensor], torch.Tensor]] = None,
        mppisac_coef: float = 1.0,
        *args,
        **kwargs,
    ):
        self.mppisac_coef = mppisac_coef
        self.mbctrl = MPPICTRL(
            state_dim=mppi_state_dim,
            action_dim=mppi_action_dim,
            action_ub=mppi_action_ub,
            action_lb=mppi_action_lb,
            model_in_dim=mppi_model_in_dim,
            model_out_dim=mppi_model_out_dim,
            h_units=mppi_h_units,
            state_preproc=mppi_state_preproc,
            state_postproc=mppi_state_postproc,
            target_proc=mppi_target_proc,
            cost_fn=mppi_cost_fn,
            noise_sigma=mppi_sigma,
            horizon=mppi_horizon,
            n_samples=mppi_n_samples,
            lambda_=mppi_lambda,
            train_epoch=mppi_train_epoch,
            true_dynamics_model=mppi_true_dynamics_model,
        )
        super().__init__(*args, **kwargs)

    def train(self, gradient_steps: int, batch_size: int) -> None:
        print(f"Progress remaining: {self._current_progress_remaining}")
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

    # TODO: override the saving for mppictrl. What needs to be saved?
