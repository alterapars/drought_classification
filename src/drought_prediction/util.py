"""Utility methods and classes."""
from ray.tune.integration.mlflow import MLflowLoggerCallback
from ray.tune.trial import Trial
from ray.tune.utils import flatten_dict


class CustomMlflowLoggerCallback(MLflowLoggerCallback):
    """A customized logger."""

    def log_trial_start(self, trial: Trial):
        # Create run if not already exists.
        if trial not in self._trial_runs:
            run = self.client.create_run(
                experiment_id=self.experiment_id,
                tags={"trial_name": str(trial)})
            self._trial_runs[trial] = run.info.run_id

        run_id = self._trial_runs[trial]

        # Log the config parameters.
        config = flatten_dict(trial.config)

        for key, value in config.items():
            self.client.log_param(run_id=run_id, key=key, value=value)