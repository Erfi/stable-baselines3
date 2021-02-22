from stable_baselines3.common.logger import TensorBoardOutputFormat
from stable_baselines3.common.callbacks import BaseCallback


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
            "Batch_size": self.model.batch_size,
            "MPPI_use_true_dynamics": self.model.use_mppi_true_dynamics_model,
            "MPPI_h_units": self.model.mppi_h_units,
            "MPPI_train_epoch": self.model.mppi_train_epoch,
            "MPPI_horizon": self.model.mppi_horizon,
            "MPPI_sigma": self.model.mppi_noise_sigma,
            "MPPI_lambda": self.model.mppi_lambda,
        }
        self.tb_formatter.writer.add_text("MPPI Config", str(info), self.num_timesteps)
        self.tb_formatter.writer.flush()

    def _on_step(self):
        pass
