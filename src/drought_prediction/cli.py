"""
Command-line interface.

After installing package, e.g., via

pip install -e .

you can run the `drought` command in console.
"""
import logging
import pathlib
from typing import Optional

import click
import pystow
import ray
import tensorflow
from ray import tune as ray_tune

from .data import DEFAULT_VARIABLE_NAMES, ERA5_SHORT_VARIABLES_NAMES, load_ERA5_Land_data_from_cds, load_smi_from_ufz, scaler_resolver
from .hpo import KerasTrainable, SklearnTrainable, activation_resolver, qlograndint, sample_dims, sklearn_model_resolver
from .util import CustomMlflowLoggerCallback

# common options
# data
option_data_root = click.option(
    "--data-root",
    type=pathlib.Path,
    help="The root directory containing the data files.",
)
option_fold_id = click.option(
    "--fold",
    type=int,
    default=2,
    help="The fold on which to train",
)
option_threshold = click.option(
    "--threshold",
    type=float,
    default=0.2,
    help="The SMI threshold to define the drought class.",
)
option_window_size = click.option(
    "--window-size",
    type=int,
    default=12,
    help="The window size (in month).",
)
option_balance = click.option(
    "--balance",
    is_flag=True,
    help="Whether to balance the classes using sample weights.",
)
option_scaler = click.option(
    "--scaler",
    default=None,
    type=click.Choice(choices=scaler_resolver.options, case_sensitive=False),
    help="The normalization method to use, an sklearn scaler.",
)

# hpo
option_num_samples = click.option(
    "--num-samples",
    type=int,
    default=1,
    help="The total number of trials to start during the HPO.",
)
option_num_cpu = click.option(
    "--cpu",
    type=int,
    default=1,
    help="The number of CPUs to use for each trial.",
)

# other
option_log_level = click.option(
    "--logging-level",
    type=click.Choice(
        choices=["DEBUG", "INFO", "WARNING", "ERROR"], case_sensitive=False),
    default="INFO",
    help="The log-level to use for logging.",
)

option_debug = click.option(
    "--debug",
    is_flag=True,
    help="Enable debug mode.",
)


@click.group()
def main():
    """The main entry point."""


@main.group()
def preprocess():
    """Data preprocessing."""


@preprocess.command()
@click.option("--output-path", default=None)
@click.option("--no-sort", is_flag=True)
@option_log_level
def smi(
    output_path: Optional[pathlib.Path],
    no_sort: bool,
    logging_level: str,
):
    """Download raw SMI data if necessary, and convert to long dataframe format."""
    logging.basicConfig(level=logging_level)

    # download raw file if necessary
    input_path = pystow.ensure(
        "drought",
        "raw",
        url="https://www.ufz.de/export/data/2/248981_SMI_SM_Lall_Gesamtboden_monatlich_1951-2020_inv.nc",
    )

    # convert to data frame
    smi = load_smi_from_ufz(path=input_path, sort=not no_sort)

    # default output path
    if output_path is None:
        output_path = pystow.join("drought", "preprocessed", "smi.h5")
    output_path = output_path.expanduser().resolve()

    # save to HDF5 format
    smi.to_hdf(str(output_path), key="smi", format="table", index=False)
    logging.info(f"Saved to {output_path.as_uri()}")


@preprocess.command()
@click.option("--output-path", default=None)
@option_log_level
@click.option("--start-year", type=int, default=1981)
@click.option("--end-year", type=int, default=2010)
@click.option("--force", is_flag=True)
def era5(
    output_path: Optional[pathlib.Path],
    logging_level: str,
    start_year: int,
    end_year: int,
    force: bool,
):
    """
    Download climate data from ERA5 through the CDS API.

    .. note ::
        Limited to 100,000 data points per request.
    """
    logging.basicConfig(level=logging_level)

    # default output path
    if output_path is None:
        output_path = pystow.join("drought", "raw", "era5")
    output_path = output_path.expanduser().resolve()

    load_ERA5_Land_data_from_cds(
        data_output_root=output_path, start_year=start_year, end_year=end_year, force=force)


@main.group()
def train():
    """Train a single model."""


@train.command(name="sklearn")
@sklearn_model_resolver.get_option("--model")
@option_log_level
@option_debug
@option_threshold
@option_scaler
@option_window_size
@option_fold_id
@option_data_root
def train_sklearn(
    model: str,
    logging_level: str,
    debug: bool,
    scaler: Optional[str],
    threshold: float,
    window_size: int,
    fold: int,
    data_root: pathlib.Path,
):
    """Train a single Sklearn model."""
    logging.basicConfig(level=logging_level)
    trainable = SklearnTrainable(config=dict(
        model=dict(
            name=model,
        ),
        data=dict(
            scaler=scaler,
            window=window_size,
            n_folds=5,
            fold_no=fold,
            threshold=threshold,
            sklearn_dataformat=True,
            variables=DEFAULT_VARIABLE_NAMES,
            scaler_restrict_to_columns=ERA5_SHORT_VARIABLES_NAMES,
            add_month=True,
            add_pos=True,
            debug=debug,
            data_root=data_root,
        ),
    ))
    result = trainable.step()
    logging.info(f"Result: {result}")


@main.group()
def tune():
    """Run hyperparameter optimization."""


@tune.command()
@option_data_root
@option_num_samples
@sklearn_model_resolver.get_option("--model")
@option_log_level
@option_debug
@option_scaler
@click.option("--num-folds", type=int, default=5)
@option_fold_id
@option_window_size
@option_threshold
@option_num_cpu
@option_balance
def sklearn(
    data_root: pathlib.Path,
    num_samples: int,
    model: str,
    scaler: str,
    logging_level: str,
    debug: bool,
    num_folds: int,
    fold: int,
    window_size: int,
    threshold: float,
    cpu: int,
    balance: bool,
):
    if debug:
        ray.init(local_mode=True)
    logging.basicConfig(level=logging_level)
    ray_tune.run(
        SklearnTrainable,
        config=dict(
            model=dict(
                name=model,
            ),
            data=dict(
                data_root=data_root,
                window=window_size,
                n_folds=num_folds,
                fold_no=fold,
                threshold=threshold,
                sklearn_dataformat=True,
                add_month=True,
                add_pos=True,
                debug=debug,
                variables=DEFAULT_VARIABLE_NAMES,
                balance=balance,
                scaler=scaler,
                scaler_restrict_to_columns=ERA5_SHORT_VARIABLES_NAMES,
            ),
        ),
        num_samples=num_samples,
        fail_fast=debug,
        resources_per_trial=dict(
            gpu=0,
            cpu=cpu,
        ),
    )


@tune.command()
@option_num_samples
@click.option("--model", type=click.Choice(["dense", "lstm", "cnn"]), default="dense")
@option_log_level
@option_debug
@click.option("--eval-frequency", type=int, default=10)
@click.option("--max-num-epochs", type=int, default=1000)
@option_data_root
@click.option("--tracking-uri", type=str, default=None)
@click.option("--num-folds", type=int, default=5)
@option_fold_id
@option_window_size
@option_threshold
@option_scaler
def keras(
    num_samples: int,
    model: str,
    logging_level: str,
    debug: bool,
    eval_frequency: int,
    max_num_epochs: int,
    data_root: pathlib.Path,
    tracking_uri: str,
    num_folds: int,
    fold: int,
    window_size: int,
    threshold: float,
    scaler: str,
):
    if debug:
        ray.init(local_mode=True)
    logging.basicConfig(level=logging_level)
    loggers = None
    if tracking_uri:
        loggers = [
            CustomMlflowLoggerCallback(
                tracking_uri=tracking_uri,
                experiment_name="drought",
            ),
            # JsonLoggerCallback(),
        ]
    ray_tune.run(
        KerasTrainable,
        config=dict(
            model=dict(
                name=f"keras-{model}",
                dims=sample_dims(),
                # LeakyReLU does not work
                activation=activation_resolver.ray_tune_search_space(),
                use_batch_norm=ray_tune.choice([False, True]),
                dropout=ray_tune.choice([None, 0.1, 0.2]),
            ),
            data=dict(
                window=window_size,
                n_folds=num_folds,
                fold_no=fold,
                threshold=threshold,
                sklearn_dataformat=False,
                variables=DEFAULT_VARIABLE_NAMES,
                add_month=True,
                add_pos=True,
                debug=debug,
                data_root=data_root,
                scaler=scaler,
                scaler_restrict_to_columns=ERA5_SHORT_VARIABLES_NAMES,
            ),
            training=dict(
                batch_size=qlograndint(lower=32, upper=4096, q=32),
                verbose=1 if debug else 2,
                # eval_frequency=eval_frequency,
            ),
            optimizer=dict(
                learning_rate=ray_tune.loguniform(
                    lower=1.0e-05, upper=1.0e-01),
            ),
            logging=dict(
                level=logging_level,
            ),
            callbacks=dict(
                early_stopping=dict(
                    monitor="loss",
                    patience=10,
                ),
            ),
        ),
        num_samples=num_samples,
        stop=dict(
            training_iteration=2 * eval_frequency if debug else max_num_epochs,
        ),
        mode="max",
        metric="validation/accuracy",
        fail_fast=debug,
        resources_per_trial=dict(
            gpu=1 if len(
                tensorflow.config.list_physical_devices("GPU")) > 0 else 0,
            cpu=2,
        ),
        loggers=loggers,
    )
