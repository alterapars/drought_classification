"""Plotting scripts."""
import itertools
import json
import logging
import os
import pathlib
import tempfile
from typing import Any, Iterable, Mapping, Optional, Sequence, Tuple

import click
import numpy
import pandas
import scipy.constants
import scipy.stats
import seaborn
import xarray
from matplotlib import pyplot as plt
from tqdm.auto import tqdm

logger = logging.getLogger(__name__)

MODEL_TRANSLATION = {
    "dense": "Dense",
    "mlpclassifier": "MLP",
    "cnn": "CNN",
    "lstm": "LSTM",
    "linearsvc": "SVM",
}


def _lagged_correlation(
    *,
    x: numpy.ndarray,
    y: Optional[numpy.ndarray] = None,
    lag: int,
    correlation: str,
) -> Iterable[Tuple[int, float, float]]:
    """Compute lagged correlation."""
    correlation_func = getattr(scipy.stats, correlation + "r")
    if y is None:
        # auto correlation
        y = x
    x = x[..., :-lag if lag > 0 else None]
    y = y[..., lag:]
    yield from (
        (lag, *correlation_func(a, b))
        for a, b in zip(x, y)
    )


def _compute_auto_correlation(
    correlation: str,
    data_root: pathlib.Path,
    lower_threshold: float,
    max_months: int,
    threshold: Optional[float],
    upper_threshold: float,
    variables: Sequence[str],
) -> pandas.DataFrame:
    # TODO: move to lib?
    x, y = _get_data(data_root, lower_threshold,
                     threshold, upper_threshold, variables)
    if threshold:
        y = (y <= 0.2).astype(y.dtype)
        logger.info(f"After binarization: {y.mean():2.2%} positive class.")
    logger.info(f"Computing {correlation} correlation.")
    data = list(tqdm(itertools.chain.from_iterable(
        (_lagged_correlation(x=y, y=y, lag=lag, correlation=correlation))
        for lag in range(0, max_months + 1)
    ), unit="correlation", unit_scale=True, total=(max_months + 1) * y.shape[0]))
    return pandas.DataFrame(data=data, columns=["lag", "correlation", "p"])


def _get_data(
    data_root: pathlib.Path,
    lower_threshold: float,
    threshold: Optional[float],
    upper_threshold: float,
    variables: Sequence[str],
) -> Tuple[numpy.ndarray, numpy.ndarray]:
    x, y = _get_data_raw(data_root, lower_threshold,
                         threshold, upper_threshold, variables)

    # reshape to (num_rows * num_cols, num_time_steps, *)
    x = numpy.reshape(x, newshape=(-1, *x.shape[2:]))
    y = numpy.reshape(y, newshape=(-1, *y.shape[2:]))
    # drop nan values
    keep_mask = numpy.isfinite(y).all(axis=1)
    x = x[keep_mask]
    y = y[keep_mask]
    logger.info(f"After dropping: y.shape={y.shape}")
    return x, y


def _get_data_raw(
    data_root: pathlib.Path,
    lower_threshold: float,
    threshold: Optional[float],
    upper_threshold: float,
    variables: Sequence[str],
) -> Tuple[numpy.ndarray, numpy.ndarray]:
    from drought_prediction.data import load_ERA5_training_data, propagate_nans

    data_root = pathlib.Path(data_root).expanduser().resolve()
    data_path = data_root.joinpath("training_dataset.nc")
    logger.info(f"Loading data from {data_path.as_uri()}")
    logger.info(f"Using variables: {variables}")
    (x, y, n_features, n_timesteps, rows, cols,) = load_ERA5_training_data(
        path=data_path,
        threshold=threshold,
        input_variables=variables,
        add_month=False,
        add_pos=False,
        binary=False,
    )
    # data cleaning from binarization
    # input array to np for faster computation and to avoid in-place manipulation:
    y = y.values.astype(numpy.float32)
    # mask values by setting to NaN:
    # 1) too large
    mask = y > upper_threshold
    # 2) or too small
    mask |= y < lower_threshold
    # 3) or non-finite input
    mask |= ~numpy.isfinite(y)
    y[mask] = numpy.nan
    x, y = propagate_nans(x, y, nan_value=numpy.nan)
    # Adapt dims from
    #   (num_features, num_time_steps, num_rows, num_columns) to
    #   (num_rows, num_columns, num_time_steps, num_features)
    x = numpy.transpose(x, axes=(2, 3, 1, 0))
    y = numpy.transpose(y, axes=(1, 2, 0))
    logger.info(f"Loaded data of shape: {y.shape}")
    return x, y


@click.group()
def main():
    """Plotting commands."""


@main.command(name="auto-correlation")
@click.option("-d", "--data-root", type=pathlib.Path)
@click.option("-t", "--threshold", type=float, default=None)
@click.option("-u", "--upper-threshold", type=float, default=20.0)
@click.option("-l", "--lower-threshold", type=float, default=-20.0)
@click.option("-m", "--max-months", type=int, default=24)
@click.option("-c", "--correlation", type=click.Choice(["spearman", "pearson"], case_sensitive=False), default="spearman")
@click.option("-o", "--output-root", type=pathlib.Path, default=pathlib.Path(tempfile.gettempdir()))
@click.option("-ll", "--log-level", type=click.Choice(["DEBUG", "INFO", "WARNING", "ERROR"], case_sensitive=False), default="INFO")
@click.option("-f", "--force", is_flag=True)
def auto_correlation(
    data_root: pathlib.Path,
    threshold: Optional[float],
    upper_threshold: float,
    lower_threshold: float,
    max_months: int,
    correlation: str,
    output_root: pathlib.Path,
    log_level: str,
    force: bool,
):
    """Auto-correlation plot for SMI labels."""
    logging.basicConfig(level=log_level)
    output_root.mkdir(exist_ok=True, parents=True)
    output_stem = output_root.joinpath(f"{correlation}_auto_correlation")
    output_path = output_stem.with_suffix(suffix=".tsv.gz")
    if output_path.is_file() and not force:
        df = pandas.read_csv(output_path, sep="\t")
        logger.info(
            f"Loaded precomputed {correlation} correlation data of shape {df.shape} from {output_path.as_uri()}")
    else:
        df = _compute_auto_correlation(
            correlation, data_root, lower_threshold, max_months, threshold, upper_threshold, variables=[])
        df.to_csv(output_path, sep="\t", index=False)
        logger.info(
            f"Written {correlation} correlation data of shape {df.shape} to {output_path}")

    # %%
    grid = seaborn.relplot(
        data=df,
        x="lag",
        y="correlation",
        kind="line",
        ci="sd",
        height=3,
        aspect=scipy.constants.golden,  # golden ratio for visual pleasure :D
    )
    grid.set(
        xlim=(0, max_months),
        xlabel="lag [month]",
        ylabel=f"{correlation} correlation",
    )
    grid.ax.grid()
    grid.tight_layout()
    # grid.grid()
    output_path = output_stem.with_suffix(suffix=".pdf")
    grid.savefig(output_path)
    logger.info(f"Written {correlation} plot to {output_path}")


@main.command(name="correlation")
@click.option("-d", "--data-root", type=pathlib.Path)
@click.option("-t", "--threshold", type=float, default=None)
@click.option("-u", "--upper-threshold", type=float, default=20.0)
@click.option("-l", "--lower-threshold", type=float, default=-20.0)
@click.option("-m", "--max-months", type=int, default=24)
@click.option("-c", "--correlation", type=click.Choice(["spearman", "pearson"], case_sensitive=False), default="spearman")
@click.option("-o", "--output-root", type=pathlib.Path, default=pathlib.Path(tempfile.gettempdir()))
@click.option("-ll", "--log-level", type=click.Choice(["DEBUG", "INFO", "WARNING", "ERROR"], case_sensitive=False), default="INFO")
@click.option("-f", "--force", is_flag=True)
@click.option("-i", "--input-variables", type=str, multiple=True, default=[
    # 'total_precipitation',
    "tp",
    # 'skin_temperature',
    # "skt",
    # 'surface_pressure',
    "sp",
    # '10m_u_component_of_wind',
    "u10",
    # '10m_v_component_of_wind',
    "v10",
    # 'surface_net_solar_radiation',
    "ssr",
    # 'surface_net_thermal_radiation',
    "str",
    # 'leaf_area_index_low_vegetation',
    "lai_lv",
    # 'leaf_area_index_high_vegetation',
    "lai_hv",
    # 'surface_thermal_radiation_downwards',
    "strd",
])
def correlation_plot(
    data_root: pathlib.Path,
    threshold: Optional[float],
    upper_threshold: float,
    lower_threshold: float,
    max_months: int,
    correlation: str,
    output_root: pathlib.Path,
    log_level: str,
    force: bool,
    input_variables: Sequence[str],
) -> None:
    """Lagged correlation plot for all (relevant) input variables and SMI."""
    logging.basicConfig(level=log_level)
    output_root.mkdir(exist_ok=True, parents=True)
    output_stem = output_root.joinpath(f"{correlation}_correlation")
    output_path = output_stem.with_suffix(suffix=".tsv.gz")
    if output_path.is_file() and not force:
        df = pandas.read_csv(output_path, sep="\t")
        logger.info(
            f"Loaded precomputed {correlation} correlation data of shape {df.shape} from {output_path.as_uri()}")
    else:
        x, y = _get_data(data_root, lower_threshold, threshold,
                         upper_threshold, input_variables)
        if threshold:
            y = (y <= 0.2).astype(y.dtype)
            logger.info(f"After binarization: {y.mean():2.2%} positive class.")
        logger.info(f"Computing {correlation} correlation.")
        data = list(tqdm(itertools.chain.from_iterable(
            ((variable_name, *rest)
             for rest in _lagged_correlation(x=x[..., i], y=y, lag=lag, correlation=correlation))
            for lag in range(0, max_months + 1)
            for i, variable_name in enumerate(input_variables)
        ), unit="correlation", unit_scale=True, total=(max_months + 1) * y.shape[0] * x.shape[-1]))
        df = pandas.DataFrame(data=data, columns=[
                              "variable", "lag", "correlation", "p"])
        df.to_csv(output_path, sep="\t", index=False)
        logger.info(
            f"Written {correlation} correlation data of shape {df.shape} to {output_path}")

    # Use only selected variables
    df = df[df["variable"].isin(input_variables)]

    grid = seaborn.relplot(
        data=df,
        x="lag",
        y="correlation",
        hue="variable",
        kind="line",
        ci="sd",
        height=3,
        aspect=scipy.constants.golden,  # golden ratio for visual pleasure :D
    )
    grid.set(
        xlim=(0, max_months),
        xlabel="lag [month]",
        ylabel=f"{correlation} correlation",
    )
    grid.ax.grid()
    grid.tight_layout()
    output_path = output_stem.with_suffix(suffix=".pdf")
    grid.savefig(output_path)
    logger.info(f"Written {correlation} plot to {output_path}")


@main.command()
@click.option("-d", "--data-root", type=pathlib.Path)
@click.option("-c", "--correlation", type=click.Choice(["spearman", "pearson"], case_sensitive=False), default="spearman")
@click.option("-c", "--correlation", type=click.Choice(["spearman", "pearson"], case_sensitive=False), default="spearman")
@click.option("-o", "--output-root", type=pathlib.Path, default=pathlib.Path(tempfile.gettempdir()))
@click.option("-ll", "--log-level", type=click.Choice(["DEBUG", "INFO", "WARNING", "ERROR"], case_sensitive=False), default="INFO")
@click.option("-f", "--force", is_flag=True)
@click.option("-m", "--max-delta", type=int, default=10)
@click.option("-s", "--smoothing", type=float, default=10.0)
def variogram(
    data_root: pathlib.Path,
    correlation: str,
    output_root: pathlib.Path,
    log_level: str,
    force: bool,
    max_delta: int,
    smoothing: float,
):
    """The (spatial) variogram."""
    logging.basicConfig(level=log_level)
    output_stem = output_root.joinpath(
        f"{correlation}_spatial_correlation_{max_delta}")
    tsv_path = output_stem.with_suffix(suffix=".tsv.gz")
    if tsv_path.is_file() and not force:
        df = pandas.read_csv(tsv_path, sep="\t")
    else:
        # normalize input
        df = _get_spatial_correlation_df(
            data_root, correlation=correlation, max_delta=max_delta)
        output_root.mkdir(exist_ok=True, parents=True)
        df.to_csv(tsv_path, sep="\t", index=False)
    logger.info(f"Info:\n{df.describe()}")

    # round distance for binning
    df["distance"] = smoothing * (df["distance"] / smoothing).round()
    grid = seaborn.relplot(
        data=df,
        x="distance",
        y="correlation",
        kind="line",
        ci=None,
        height=3,
        aspect=scipy.constants.golden_ratio,
    )
    grid.savefig(output_stem.with_suffix(".pdf"))


def _get_spatial_correlation_df(
    data_root: pathlib.Path,
    correlation: str,
    max_delta: int = 10,
) -> pandas.DataFrame:
    data_root = pathlib.Path(data_root).expanduser().resolve()
    data_path = data_root.joinpath("training_dataset.nc")
    logger.info(f"Loading data from {data_path.as_uri()}")
    os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"
    with xarray.open_dataset(data_path) as dataset:
        xa = dataset["SMI"]  # time x lat x lon
        smi = numpy.asarray(xa.values)
        lat = numpy.asarray(xa.coords["latitude"])
        lon = numpy.asarray(xa.coords["longitude"])
        dd_lat = lat[1] - lat[0]
        dd_lon = lon[1] - lon[0]
    logger.info(f"Loaded data: {smi.shape}")
    data = []
    corr_func = getattr(scipy.stats, correlation + "r")
    for d_lat, d_lon in tqdm(itertools.product(range(-max_delta, max_delta + 1), repeat=2), total=(2 * max_delta + 1) ** 2):
        a = smi
        b = smi
        if d_lat > 0:
            a = a[:, d_lat:, :]
        elif d_lat < 0:
            b = b[:, -d_lat:, :]
        if d_lon > 0:
            a = a[:, :, d_lon:]
        elif d_lon < 0:
            b = b[:, :, -d_lon:]
        i, j, k = list(map(min, zip(a.shape, b.shape)))
        a = a[:i, :j, :k]
        b = b[:i, :j, :k]
        # TODO: this is wrong :)
        d = ((d_lon * dd_lon * 110) ** 2 + (d_lat * dd_lat * 110) ** 2) ** 0.5
        mask = numpy.isfinite(a) & numpy.isfinite(b)
        corr, p = corr_func(a[mask], b[mask])
        data.append((d, corr, p))
    return pandas.DataFrame(data=data, columns=["distance", "correlation", "p"])


@main.command()
@click.option("-d", "--data-root", type=pathlib.Path)
@click.option("-u", "--upper-threshold", type=float, default=20.0)
@click.option("-l", "--lower-threshold", type=float, default=-20.0)
@click.option("-o", "--output-path", type=pathlib.Path, default=pathlib.Path(tempfile.gettempdir(), "time.pdf"))
@click.option("-t", "--threshold", type=float, default=0.2)
@click.option("-b", "--binarized", is_flag=True)
def time(
    data_root: pathlib.Path,
    upper_threshold: float,
    lower_threshold: float,
    output_path: pathlib.Path,
    threshold: float,
    binarized: bool,
):
    """Plot SMI against time."""
    data_root = pathlib.Path(data_root).expanduser().resolve()
    data_path = data_root.joinpath("training_dataset.nc")
    print(f"Loading data from {data_path.as_uri()}")
    os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"
    with xarray.open_dataset(data_path) as dataset:
        xa = dataset["SMI"]  # time x lat x lon
        y = numpy.asarray(xa.values)
        lat = numpy.asarray(xa.coords["latitude"])
        lon = numpy.asarray(xa.coords["longitude"])
        time = numpy.asarray(xa.coords["time"])
    print(f"Loaded data: {y.shape}")

    # y.shape: lon, lat, time
    mask = numpy.isfinite(y)
    i_time, i_lat, i_lon = mask.nonzero()
    values = y[mask]
    print(f"Finite values: {values.shape}")
    mask2 = ((values < upper_threshold) & (values > lower_threshold))
    i_time, i_lat, i_long, values = [a[mask2]
                                     for a in (i_time, i_lat, i_lon, values)]
    print(
        f"Values after filtering for {lower_threshold} < x < {upper_threshold}: {values.shape}")

    if binarized:
        print(f"Binarization with threshold: {threshold}")
        values = (values < threshold)

    df = pandas.DataFrame(data=dict(
        smi=values, longitude=lon[i_lon], latitude=lat[i_lat], time=time[i_time]))
    print(df.describe())

    # Plot smi against time
    grid: seaborn.FacetGrid = seaborn.relplot(
        data=df,
        x="time",
        y="smi",
        kind="line",
        ci=None if binarized else "sd",
        aspect=scipy.constants.golden_ratio,
        height=3,
    )

    # Add split borders
    ax: plt.Axes = grid.ax
    for i in range(1, 5):
        j = int(i * len(time) / 5)
        ax.axvline(x=time[j], color="red", linestyle="dotted")

    if binarized:
        # update label
        grid.set(ylabel="drought frequency")
    else:
        # Add binarization threshold
        ax.axhline(y=threshold, color="darkgreen", linestyle="dashed")

    # Tight limits
    grid.set(
        xlim=[time[0], time[-1]],
        ylim=[0, 1],
    )
    grid.tight_layout()

    output_path = output_path.expanduser().resolve()
    output_path.parent.mkdir(exist_ok=True, parents=True)
    grid.savefig(output_path)
    print(f"Saved plot to {output_path.as_uri()}")


@main.command()
@click.option("-d", "--data-root", type=pathlib.Path)
@click.option("-u", "--upper-threshold", type=float, default=20.0)
@click.option("-l", "--lower-threshold", type=float, default=-20.0)
@click.option("-o", "--output-path", type=pathlib.Path, default=pathlib.Path(tempfile.gettempdir(), "location.pdf"))
@click.option("-t", "--threshold", type=float, default=0.2)
@click.option("-b", "--binarized", is_flag=True)
def location(
    data_root: pathlib.Path,
    upper_threshold: float,
    lower_threshold: float,
    output_path: pathlib.Path,
    threshold: float,
    binarized: bool,
):
    """Plot SMI per location."""
    data_root = pathlib.Path(data_root).expanduser().resolve()
    data_path = data_root.joinpath("training_dataset.nc")
    print(f"Loading data from {data_path.as_uri()}")
    os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"
    with xarray.open_dataset(data_path) as dataset:
        xa = dataset["SMI"]  # time x lat x lon
        y = numpy.asarray(xa.values)
        lat = numpy.asarray(xa.coords["latitude"])
        lon = numpy.asarray(xa.coords["longitude"])
        time = numpy.asarray(xa.coords["time"])
    print(f"Loaded data: {y.shape}")

    y[(y > upper_threshold) | (y < lower_threshold)] = numpy.nan
    print(f"Filtered for {lower_threshold} < x < {upper_threshold}")

    m = numpy.isfinite(y).all(axis=0)
    lat_nnz, lon_nnz = m.nonzero()
    a, b = lat_nnz.min(), lat_nnz.max()
    c, d = lon_nnz.min(), lon_nnz.max()
    y = y[:, a:b + 1, c:d + 1]
    lat = lat[a:b + 1]
    lon = lon[c:d + 1]
    print(f"Cut to shape {y.shape}")

    if binarized:
        print(f"Binarization with threshold: {threshold}")
        m = ~numpy.isfinite(y).all(axis=0)
        y = (y < threshold).astype(float)
        y = numpy.nanmean(y, axis=0)
        y[m] = numpy.nan
        ax: plt.Axes = seaborn.heatmap(
            y,
            xticklabels=False,
            yticklabels=False,
            square=True,
            cmap="cividis",
        )
        # ax.set_xticks(range(6, lon.shape[0], 10))
        # ax.set_xticklabels(lon[6::10])
        # ax.set_yticks(range(8, lat.shape[0], 10))
        # ax.set_yticklabels(lat[8::10])
    else:
        mean = numpy.nanmean(y, axis=0)
        std = numpy.nanstd(y, axis=0)
        fig, axes = plt.subplots(ncols=2)
        axes: Sequence[plt.Axes]
        axes[0].imshow(mean, vmin=0, vmax=1)
        axes[1].imshow(std, vmin=0, vmax=1)

    plt.tight_layout()

    output_path = output_path.expanduser().resolve()
    output_path.parent.mkdir(exist_ok=True, parents=True)
    plt.savefig(output_path)
    print(f"Saved plot to {output_path.as_uri()}")


def _collate_metrics(
    root: pathlib.Path,
) -> Iterable[Tuple[pathlib.Path, Mapping[str, Any]]]:
    """Collate metrics from result root."""
    for path in root.rglob("test_metrics.json"):
        with path.open() as f:
            j = json.load(f)
        if not isinstance(j, dict):
            j = j.replace("'", '"')
            j = json.loads(j)
        yield path, j


@main.command(name="drought-frequency")
@click.option("-i", "--input-path", type=pathlib.Path, default=pathlib.Path(tempfile.gettempdir(), "confusion_matrix_collate.tsv"))
@click.option("-o", "--output-path", type=pathlib.Path, default=pathlib.Path(tempfile.gettempdir(), "drought_frequency.pdf"))
def plot_drought_frequency(
    input_path: pathlib.Path,
    output_path: pathlib.Path,
):
    """Plot actual and predicted drought frequency."""
    # read data, TODO: error message
    input_path = input_path.expanduser().resolve()
    logger.info(f"Reading from {input_path.as_uri()}")
    df = pandas.read_csv(input_path, sep="\t")

    # create df
    # model | fold | pred class 1 freq.
    df["c**"] = df["c00"] + df["c01"] + df["c10"] + df["c11"]
    df["pred_drought_freq"] = (df["c01"] + df["c11"]) / df["c**"]

    # add GT
    df["gt_drought_freq"] = (df["c10"] + df["c11"]) / df["c**"]
    a = df.groupby(by="fold").agg({"gt_drought_freq": "first"}).reset_index()
    a["model"] = "gt"

    # merge data frames
    df2 = pandas.concat([df.loc[:, ["model", "fold", "pred_drought_freq"]].rename(columns={
                        "pred_drought_freq": "drought_freq"}), a.rename(columns={"gt_drought_freq": "drought_freq"})], ignore_index=True)

    # correct test fold
    df2["fold"] += 3

    # plot
    grid = seaborn.catplot(data=df2, x="model",
                           y="drought_freq", col="fold", kind="bar", ci="sd")
    grid.set_xticklabels(rotation=90)
    grid.set(ylim=[0, 1])
    for ax in grid.axes.flat:
        ax.grid()

    output_path = output_path.expanduser().resolve()
    grid.savefig(output_path)
    logger.info(f"Written figure to {output_path.as_uri()}")


@main.command(name="overall-point")
@click.option("-i", "--input-root", type=pathlib.Path, default=pathlib.Path(tempfile.gettempdir()))
@click.option("-o", "--output-root", type=pathlib.Path, default=pathlib.Path(tempfile.gettempdir()))
@click.option("--height", default=None, type=float)
def plot_overall_point(
    input_root: pathlib.Path,
    output_root: pathlib.Path,
    height: Optional[float],
):
    """Plot overall performance using point plot."""
    input_root = input_root.expanduser().resolve()
    output_root = output_root.expanduser().resolve()

    def _parse(tup):
        path, js = tup
        # roc_auc = js["results"]["roc_auc"]
        pr_auc = js["results"]["pr_auc"]
        f1 = js["report"]["macro avg"]["f1-score"]
        replicate, model, fold = [pp.name for pp in path.parents][:3]
        fold = int(fold[len("fold_"):]) + 3
        return fold, model, pr_auc, f1  # , roc_auc

    df = pandas.DataFrame(
        data=list(map(_parse, _collate_metrics(root=input_root))),
        columns=["fold", "model", "PR-AUC", "Macro F1"],
    )
    df = pandas.melt(df, id_vars=["fold", "model"], var_name="metric")
    grid: seaborn.FacetGrid = seaborn.relplot(
        data=df,
        x="model",
        col="fold",
        row="metric",
        y="value",
        hue="model",
        legend=False,
        facet_kws=dict(
            margin_titles=True,
            sharey="row",
        ),
        height=height,
    )
    grid.savefig(output_root.joinpath("plot.pdf"))


@main.command(name="overall")
@click.option("-i", "--input-root", type=pathlib.Path, default=pathlib.Path(tempfile.gettempdir()))
@click.option("-o", "--output-path", type=pathlib.Path, default=pathlib.Path(tempfile.gettempdir(), "results.pdf"))
@click.option("--split", is_flag=True)
@click.option("--height", default=None, type=float)
def plot_overall(
    input_root: pathlib.Path,
    output_path: pathlib.Path,
    split: bool,
    height: Optional[float],
):
    input_root = input_root.expanduser().resolve()

    if input_root.is_dir():
        dfs = [
            pandas.read_csv(input_path, sep="\t")
            for input_path in input_root.glob("*.tsv")
        ]
        if not dfs:
            raise RuntimeError(
                f"Could not find results at {input_root.as_uri()}")
        df = pandas.concat(dfs, ignore_index=True)
    elif input_root.is_file():
        df = pandas.read_csv(input_root, sep="\t")
    else:
        raise RuntimeError(f"{input_root.as_uri()} does not exist!")

    # wide to long
    df = pandas.melt(df, id_vars=["path", "model",
                     "fold", "replicate"], var_name="metric")

    # normalization
    df["metric"] = df["metric"].apply({
        "report/macro avg/f1-score": "Macro F1",
        "results/pr_auc": "PR-AUC",
        "results/roc_auc": "ROC-AUC",
    }.get)
    df["model"] = df["model"].apply(MODEL_TRANSLATION.get)
    df["fold"] += 3

    # reproducible hue order
    sorted_models = sorted(MODEL_TRANSLATION.values())

    # plot
    if split:
        for metric, group in df.groupby(by="metric"):
            grid = seaborn.catplot(
                data=group,
                x="fold",
                hue="model",
                y="value",
                kind="bar",
                ci="sd",
                height=height,
                hue_order=sorted_models,
            ).set(
                ylabel=metric,
            ).savefig(
                output_path.with_name(
                    f"{output_path.name}_{metric.lower()}.pdf"),
            )
    else:
        grid = seaborn.catplot(
            data=df,
            x="fold",
            hue="model",
            y="value",
            kind="bar",
            ci="sd",
            col="metric",
            col_order=["Macro F1", "PR-AUC"],
            hue_order=sorted_models,
            height=height,
        )
        # save figure
        grid.savefig(output_path)


if __name__ == '__main__':
    main()
