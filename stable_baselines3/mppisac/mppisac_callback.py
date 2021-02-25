from stable_baselines3.common.logger import TensorBoardOutputFormat
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import VecEnv


class MPPISAC_TensorboardCallback(BaseCallback):
    def _on_training_start(self):
        output_formats = self.logger.Logger.CURRENT.output_formats
        # Save reference to tensorboard formatter object
        # note: the failure case (not formatter found) is not handled here, should be done with try/except.
        self.tb_formatter = next(
            formatter for formatter in output_formats if isinstance(formatter, TensorBoardOutputFormat)
        )
        self._write_on_training_start()

    def _write_on_training_start(self):
        info = {
            "Env": self.model.env.envs[0].unwrapped.__class__.__name__
            if isinstance(self.model.env, VecEnv)
            else self.model.env.unwrapped.__class__.__name__,
            "Batch_size": self.model.batch_size,
            "MPPI_use_true_dynamics": self.model.use_mppi_true_dynamics_model,
            "MPPI_h_units": self.model.mppi_h_units,
            "MPPI_train_epoch": self.model.mppi_train_epoch,
            "MPPI_horizon": self.model.mppi_horizon,
            "MPPI_sigma": self.model.mppi_noise_sigma,
            "MPPI_lambda": self.model.mppi_lambda,
            "MPPI_n_samples": self.model.mppi_n_samples,
            "MPPISAC_coef": self.model.mppisac_coef,
        }
        self.tb_formatter.writer.add_text("MPPI Config", str(info), self.num_timesteps)
        self.tb_formatter.writer.flush()

    def _on_step(self):
        pass
