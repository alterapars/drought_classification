# noqa E226, C812

"""Utility methods for preprocessing and loading data."""

import logging
import math
import os
import pathlib
from collections import Counter
from typing import (
    Any,
    Collection,
    Mapping,
    MutableMapping,
    Optional,
    Sequence,
    Tuple,
    Union,
)

import cdsapi
import numpy
import pandas
import requests
import tensorflow
import xarray
from class_resolver import Hint, Resolver
from sklearn.base import TransformerMixin
from sklearn.preprocessing import (
    MaxAbsScaler,
    MinMaxScaler,
    PowerTransformer,
    QuantileTransformer,
    RobustScaler,
    StandardScaler,
)

logger = logging.getLogger(__name__)


ERA5_SHORT_VARIABLES_NAMES = (
    # ERA5 shortened:
    "str",
    "tp",
    "skt",
    "sp",
    "u10",
    "v10",
    "ssr",
    # "str",
    "lai_lv",
    "lai_hv",
    "strd",
    # additional:
    "ro",
    "e",
    "evatc",
    "t2m",
    "ssrd",
    "d2m",
    "evabs",
)
SOILTYPE_VARIABLE_NAMES = (
    "water",
    "evergreen_needleleaf_forest",
    "Evergreen_Broadleaf_forest",
    "Deciduous_Needleleaf_forest",
    "Deciduous_Broadleaf_forest",
    "Mixed_forest",
    "Closed_shrublands",
    "Open_shrublands",
    "Woody_savannas",
    "Savannas",
    "Grasslands",
    "Permanent_wetlands",
    "C3_Croplands",
    "Urban_and_built_up",
    "C3_Cropland_Natural_vegetation_mosaic",
    "Snow_and_ice",
    "Barren_or_sparsely_vegetated",
    "C4_fratcion_Croplands",
    "C4_fraction_Cropland_Natural_vegetation_mosaic",
)
DEFAULT_VARIABLE_NAMES = ERA5_SHORT_VARIABLES_NAMES + SOILTYPE_VARIABLE_NAMES


class WindowedDataset(tensorflow.keras.utils.Sequence):
    """
    A windowed dataset.

    Sample Format:
        (x_{r+c,t}, ..., x_{r+c,t+w}) -> y_{r+c,t+w}
    """

    def __init__(
        self,
        raw_x: numpy.ndarray,
        raw_y: numpy.ndarray,
        window_size: int,
        batch_size: int = 32,
        shuffle: bool = False,
        keep_only_last: bool = False,
    ):
        """
        Initialize the sequence.

        :param raw_x: shape: (num_row + num_col, num_time_steps, num_input_features)
        :param raw_y: shape: (num_row + num_col, num_time_steps, num_output_features)
        :param window_size:
            The window size.
        :param keep_only_last:
            Whether to keep only the last entry of a window. This can be used for ablation studies.
        """
        self.raw_x = raw_x
        if raw_y.ndim < raw_x.ndim:
            logger.warning(
                "Expanding last dimension of target values - assuming a single target value."
            )
            raw_y = raw_y[..., None]
        self.raw_y = raw_y
        self.window_size = window_size
        self.batch_size = batch_size
        (
            self.num_non_nan_pixels,
            self.num_time_steps,
            self.num_input_features,
        ) = self.raw_x.shape
        self.max_start_time = self.num_time_steps - self.window_size
        real_window_size = 1 if keep_only_last else self.window_size
        self.input_shape = (real_window_size, raw_x.shape[-1])
        self.output_shape = (raw_y.shape[-1],)
        self.num_samples = self.num_non_nan_pixels * self.max_start_time
        self.indices = numpy.arange(self.num_samples)
        self.shuffle = shuffle
        self.keep_only_last = keep_only_last
        self.on_epoch_end()

    def on_epoch_end(self):
        if self.shuffle:
            numpy.random.shuffle(self.indices)

    def __len__(self) -> int:
        return math.ceil(self.num_samples / self.batch_size)

    def __getitem__(self, idx: int) -> Tuple[numpy.ndarray, numpy.ndarray]:
        batch_indices = self.indices[self.batch_size *
                                     idx: self.batch_size * (idx + 1)]
        js, ts = numpy.unravel_index(
            batch_indices,
            shape=(self.num_non_nan_pixels, self.max_start_time),
        )
        y = self.raw_y[js, ts + self.window_size, :]
        # shape: batch_size, window_size, input_dim
        x = numpy.stack(
            [self.raw_x[j, t: t + self.window_size, :]
                for j, t in zip(js, ts)],
            axis=0,
        )
        if self.keep_only_last:
            x = x[:, -1:, ...]
        return x, y

    def __str__(self):
        return (
            f"{self.__class__.__name__}("
            f"num_samples={self.num_samples}, "
            f"batch_size={self.batch_size}, "
            f"shuffle={self.shuffle}, "
            f"input_shape={self.input_shape}, "
            f"output_shape={self.output_shape}"
            f")"
        )


def return_nan_percentage_and_dtype(input_data):
    """ print nan percentage and dtype of input array"""
    print(input_data.dtype)
    total_size = input_data.size
    nan_sum = numpy.isnan(input_data).sum()
    perc = float(nan_sum / total_size)
    print("percentage of nan values inside dataset is: %.2f" %
          float(perc) + " %")


def encode_time_dim(input_dataset: xarray.Dataset) -> Tuple[numpy.ndarray, numpy.ndarray]:
    """
    Return time as extra encoded feature in 2 arrays.

    :param input_dataset:
        The xarray dataset. Has to have dimensions "time", "latitude" (=rows), "longitude" (=columns).

    :return:
        t_cos: shape: (num_timesteps, num_rows, num_columns)
        t_sin: shape: (num_timesteps, num_rows, num_columns)
    """
    n_timesteps, n_rows, n_columns = [
        input_dataset.dims[key] for key in ("time", "latitude", "longitude")
    ]
    doy = numpy.asarray(input_dataset.time.dt.dayofyear,
                        dtype=numpy.float32) / 365.0 * numpy.pi
    t_cos, t_sin = numpy.cos(doy), numpy.sin(doy)
    t_cos, t_sin = [
        numpy.tile(numpy.reshape(t, newshape=(-1, 1, 1)),
                   reps=(1, n_rows, n_columns))
        for t in (t_cos, t_sin)
    ]
    return t_cos, t_sin


def lon_lat_dim_as_features(
    input_dataset: xarray.Dataset,
) -> Tuple[numpy.ndarray, numpy.ndarray]:
    """
    Extract lon lat from dataset and return as feature array.

    :param input_dataset:
        The xarray dataset. Has to have dimensions "time", "latitude" (=rows), "longitude" (=columns).

    :return:
        x,y, each an array with shape (n_timesteps, rows, cols)
    """
    n_timesteps, n_rows, n_columns = [
        input_dataset.dims[key] for key in ("time", "latitude", "longitude")
    ]
    lon, lat = [
        numpy.tile(
            numpy.reshape(
                input_dataset.coords[key].values.astype(numpy.float32), newshape=shape
            ),
            reps=reps,
        )
        for key, shape, reps in (
            ("longitude", (1, 1, -1), (n_timesteps, n_rows, 1)),
            ("latitude", (1, -1, 1), (n_timesteps, 1, n_columns)),
        )
    ]
    return lat, lon


def ffill(input_array):
    df = pandas.DataFrame(input_array)
    df.fillna(method="ffill", axis=1, inplace=True)
    output_array = df.to_numpy()
    return output_array


def binarize_y(
    input_xarray: xarray.DataArray,
    threshold: float,
    upper_threshold: float = 20,
    lower_threshold: float = -20,
):
    """
    Returns labeled binary mask for 3D input vector (0 for values bigger than T, 1 for labels smaller than T)

    Parameters
    ----------
    input_xarray : array
        3-D input array in the format [num_timesteps, height, width]
    threshold : int
        Binarization Threshold T to define the 2 classes

    Returns
    -------
    list of arrays
        np array [num_timesteps, height, width] with classified values

    """
    # SMI 0,20 - 0,30 = ungewöhnliche Trockenheit  \
    # SMI 0,10 - 0,20 = moderate Dürre \
    # SMI 0,05 - 0,10 = schwere Dürre \
    # SMI 0,02 - 0,05 = extreme Dürre \
    # SMI 0,00 - 0,02 = außergewöhnliche Dürre

    # print('indices for this array are: ' + str(input_xarray.indexes))
    logger.info(
        f"The threshold for the soil moisture classification is: {threshold}")

    # input array to np for faster computation and to avoid in-place manipulation:
    input_array = input_xarray.values

    # threshold
    out = (input_array <= threshold).astype(numpy.float32)

    # mask values by setting to NaN:
    # 1) too large
    mask = input_array > upper_threshold
    # 2) or too small
    mask |= input_array < lower_threshold
    # 3) or non-finite input
    mask |= ~numpy.isfinite(input_array)
    out[mask] = numpy.nan

    logger.info(
        f"Binarization Result: "
        f"0: {(out == 0).mean():2.2%}, "
        f"1: {(out == 1).mean():2.2%}, "
        f"NaN: {(out != out).mean():2.2%}",
    )
    return out


def load_ERA5_training_data(
    path: Union[str, pathlib.Path],
    threshold: float,
    input_variables: Collection[str],
    target_variable: str = "SMI",
    binary: bool = True,
    add_month: bool = True,
    add_pos: bool = True,
) -> Tuple[numpy.ndarray, numpy.ndarray, int, int, int, int, Sequence[str]]:
    """
    Parameters
    ----------
    path : string
       absolute path of data location
    threshold:
        The threshold for target binarization.
    input_variables : tuple
        variables the dataset should consist of
    target_variable : string
        the name of the target variable. must not be in input_variables.
    binary:
        Whether to binarize the target values.
    add_month:
        Whether to add an encoding of time to the input data.
    add_pos:
        Whether to add an encoding of position to the input data.

    Returns
    -------
    X
        Training Data [n_features, n_timesteps, rows, cols)]
    Y
        Labels, shape: [n_timesteps, rows, cols]
    n_features
        number of features / variables in the Training set
    """
    if target_variable is input_variables:
        raise ValueError(
            "The target variable must not be in the input variables.")

    # check for duplicate input variable names
    duplicates = {
        v: count
        for v, count in Counter(input_variables).items()
        if count > 1
    }
    if duplicates:
        raise ValueError(f"Duplicates in input variable names: {duplicates}")

    # normalize input
    path = pathlib.Path(path).expanduser().resolve()
    logger.info(f"Loading data from {path.as_uri()}")

    os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"
    input_variables = list(input_variables)
    columns = []
    with xarray.open_dataset(path) as dataset:
        available_variables = set(dataset.keys())
        missing = set(input_variables).difference(available_variables)
        if missing:
            raise ValueError(
                f"Missing variables: {sorted(missing)}. Available are {sorted(available_variables)}")
        # load variables into memory
        dataset[input_variables]
        var_list = []
        # temporal encoding: circular encoding of month
        if add_month:
            var_list.extend(encode_time_dim(dataset))
            columns.extend(("time_cos", "time_sin"))
        # adding lon and lat as extra features
        if add_pos:
            var_list.extend(lon_lat_dim_as_features(dataset))
            columns.extend(("latitude", "longitude"))
        for i in input_variables:
            data_var = numpy.array(dataset[i])
            # fill nans if present:
            if numpy.isnan(data_var).sum() > 0:
                data_var_2D = numpy.reshape(
                    data_var, (data_var.shape[0],
                               data_var.shape[1] * data_var.shape[2])
                )
                data_var_without_nans = ffill(data_var_2D)
                data_var = numpy.reshape(
                    data_var_2D,
                    (
                        data_var_without_nans.shape[0],
                        data_var.shape[1],
                        data_var.shape[2],
                    ),
                )
            var_list.append(data_var)
            columns.append(i)

        X = numpy.stack(var_list, axis=0)
        n_features = int(X.shape[0])
        n_timesteps = int(X.shape[1])
        rows = int(X.shape[2])
        cols = int(X.shape[3])

        # labels:
        if binary:
            Y = binarize_y(dataset[target_variable], threshold=threshold)
        else:
            Y = dataset[target_variable]

    return X, Y, n_features, n_timesteps, rows, cols, columns


def propagate_nans(
    x: numpy.ndarray,
    y: numpy.ndarray,
    nan_value=numpy.nan,
) -> Tuple[numpy.ndarray, numpy.ndarray]:
    """
    Fill input / output with NaN where the other is NaN.

    :param x: shape: (num_features, n_timesteps, n_rows, n_cols)
        The input data.
    :param y: shape: (n_timesteps, n_rows, n_cols)
        The target data.
    :param nan_value:
        The fill value.

    :return:
        A tuple (x, y) with
        - x: shape: (num_features, n_timesteps, n_rows, n_cols)
        - y: shape: (n_timesteps, n_rows, n_cols)
    """
    # set input to NaN where output is Nan
    fill_mask = numpy.repeat(~numpy.isfinite(
        y)[None, ...], repeats=x.shape[0], axis=0)
    x[fill_mask] = nan_value

    # propagate input nans to target
    fill_mask = ~numpy.isfinite(x).all(axis=0)
    y[fill_mask] = nan_value

    return x, y


def non_nan_values_per_image(input_data):
    """Create boolean mask to index nans in original data,
    assuming all samples in the dataset have nans at the same position"""
    boolean_nan_idx = numpy.isnan(input_data[:, 0, 0])
    count = numpy.count_nonzero(numpy.isnan(input_data[:, 0, 0]))
    non_nan_values = int(len(input_data)) - int(count)
    return non_nan_values, boolean_nan_idx


def labels_nan_removal(raw_y, non_nan_values, n_timesteps):
    """ removing nans in y"""
    print(raw_y.shape)
    y_without_nans_1D = raw_y[~numpy.isnan(raw_y)]

    y_without_nans_3D = numpy.reshape(
        y_without_nans_1D,
        (non_nan_values, n_timesteps, raw_y.shape[-1]),
    )

    # To multiclass:
    class_0 = y_without_nans_3D[:, :, 0]
    # invert for class no_drought:
    class_1 = numpy.logical_not(class_0)
    labels_2D = numpy.stack((class_0, class_1), axis=2)
    return labels_2D


def adapt_dims_to_WindowedDataset_class(X, Y):
    """"adapt dims to format [rows, cols, n_timesteps, n_features]"""
    X_transposed = numpy.stack(X, axis=0).T
    X_adapted = numpy.swapaxes(X_transposed, 0, 1)
    Y_swapaxis = numpy.swapaxes(Y.T, 0, 1)
    Y_adapted = numpy.expand_dims(Y_swapaxis, axis=3)
    return X_adapted, Y_adapted


def combine_rows_and_col(X, Y):
    """"adapt dims to format [rows * cols, n_timesteps, n_features]"""
    print(Y.shape[-1])
    rows = X.shape[0]
    cols = X.shape[1]
    n_timesteps = X.shape[2]
    n_features = X.shape[3]
    X_reshaped = numpy.reshape(X, (rows * cols, n_timesteps, n_features))
    Y_reshaped = numpy.reshape(Y, (rows * cols, n_timesteps, Y.shape[-1]))
    return X_reshaped, Y_reshaped


def load_ERA5_Land_data_from_cds(
    data_output_root: pathlib.Path,
    start_year: int,
    end_year: int,
    force: bool = False,
) -> None:
    """
    Download climate data from ERA5 through the CDS API.

    :param data_output_root:
        The output root directory.
    :param start_year:
        The start year.
    :param end_year:
        The end year.
    :param force:
        Whether to enforce re-downloading existing files.

    .. note ::
        Limited to 100,000 data points per request.

    """
    # ensure pathlib.Path
    data_output_root = pathlib.Path(data_output_root).expanduser().resolve()
    logger.info(f"Resolved output root: {data_output_root.as_uri()}")

    # ERA5-Land monthly :
    # https://cds.climate.copernicus.eu/cdsapp#!/dataset/reanalysis-era5-land-monthly-means?tab=overview
    c = cdsapi.Client()
    for i in range(start_year, end_year + 1):
        logger.info(
            f"[{i - start_year + 1:3}/{end_year - start_year:3}] Retrieving year {i}"
        )

        output_path = data_output_root.joinpath(
            f"ERA5_monthly-mean_germany_{i}.nc")

        if output_path.is_file() and not force:
            logger.info(f"Skipping existing file: {output_path.as_uri()}")
            continue

        c.retrieve(
            "reanalysis-era5-land-monthly-means",
            {
                "format": "netcdf",
                "product_type": "monthly_averaged_reanalysis",
                "variable": [
                    "10m_u_component_of_wind",
                    "10m_v_component_of_wind",
                    "2m_dewpoint_temperature",
                    "2m_temperature",
                    "evaporation_from_bare_soil",
                    "evaporation_from_open_water_surfaces_excluding_oceans",
                    "evaporation_from_the_top_of_canopy",
                    "evaporation_from_vegetation_transpiration",
                    "forecast_albedo",
                    "lake_bottom_temperature",
                    "lake_ice_depth",
                    "lake_ice_temperature",
                    "lake_mix_layer_depth",
                    "lake_mix_layer_temperature",
                    "lake_shape_factor",
                    "lake_total_layer_temperature",
                    "leaf_area_index_high_vegetation",
                    "leaf_area_index_low_vegetation",
                    "potential_evaporation",
                    "runoff",
                    "skin_reservoir_content",
                    "skin_temperature",
                    "snow_albedo",
                    "snow_cover",
                    "snow_density",
                    "snow_depth",
                    "snow_depth_water_equivalent",
                    "snow_evaporation",
                    "snowfall",
                    "snowmelt",
                    "soil_temperature_level_1",
                    "soil_temperature_level_2",
                    "soil_temperature_level_3",
                    "soil_temperature_level_4",
                    "sub_surface_runoff",
                    "surface_latent_heat_flux",
                    "surface_net_solar_radiation",
                    "surface_net_thermal_radiation",
                    "surface_pressure",
                    "surface_runoff",
                    "surface_sensible_heat_flux",
                    "surface_solar_radiation_downwards",
                    "surface_thermal_radiation_downwards",
                    "temperature_of_snow_layer",
                    "total_evaporation",
                    "total_precipitation",
                    "volumetric_soil_water_layer_1",
                    "volumetric_soil_water_layer_2",
                    "volumetric_soil_water_layer_3",
                    "volumetric_soil_water_layer_4",
                ],
                "year": str(i),
                "month": [f"{i:2}" for i in range(1, 12 + 1)],
                "area": [
                    56,
                    6,
                    44,
                    15,
                ],
                "time": "00:00",
            },
            output_path,
        )
        logger.info(f"Saved to {output_path.as_uri()}")


def load_SMI_data_from_Helmholtz():
    """
    Downloading SMI topsoil data from helmholtz website
    """
    url = "https://www.ufz.de/export/data/2/237851_SMI_L02_Oberboden_monatlich_1951_2018_inv.zip"  # noqa: E501
    response = requests.get(url)
    return response


def save_SMI_data(DATA_OUTPUT_PATH):
    r = load_SMI_data_from_Helmholtz()
    with open(DATA_OUTPUT_PATH + "topsoil_helmholtz.zip", "wb") as f:
        f.write(r.content)


def normalize_data(training_data, scaler_type):
    """
    Normalizes every feature of the input data seperatly
    according to the specified scaler

    Parameters
    ----------
    training_data: array
       input array, format: (n_features, n_samples, width*height)
    scaler_type: string
        Robust, Standard or MinMax

    Returns
    -------
    array
        scaled Training Data

    """
    # TODO: We need access to the training scaler to normalize the evaluation data
    # Standard scaler = includes outliers, robust scaler = doesnt inculde outliers
    if scaler_type == "Robust":
        scaler = RobustScaler()
    elif scaler_type == "Standard":
        scaler = StandardScaler()
    elif scaler_type == "MinMax":
        scaler = MinMaxScaler()
    else:
        raise ValueError(f"Invalid scalar_type: {scaler_type}")

    result_arr = []
    for idx, i in enumerate(training_data):
        scaled_data = scaler.fit_transform(i)
        logger.info(scaled_data.shape)
        result_arr.append(scaled_data)
        logger.info("feature " + str(idx) + " normalized")
    logger.info("Creating scaled Training data...")
    Training_data_scaled = numpy.stack(result_arr, axis=0)

    return Training_data_scaled


def load_smi_from_ufz(
    path: pathlib.Path,
    sort: bool = True,
) -> pandas.DataFrame:
    """
    Load SMI data from file.

    Drop all NaN values, and convert to dataframe.

    :param path:
        The path.
    :param sort:
        Whether to sort the dataframe for unique order.

    :return:
        A dataframe with columns "t" | "lon" | "lat" | "smi".
    """
    path = path.expanduser().resolve()
    logger.info(f"Loading data from {path.as_uri()}")

    with xarray.open_dataset(path) as ufz_data:
        # SMI: shape: (time, lon, lat)
        smi = numpy.asarray(ufz_data["SMI"])
        logger.info(f"Loaded SMI data of shape: {smi.shape}")

        # time: shape: (time,)
        time = numpy.asarray(ufz_data["time"])

        # coord: shape: (lon, lat)
        lon = numpy.asarray(ufz_data["lon"])
        lat = numpy.asarray(ufz_data["lon"])

    # remove nan values
    mask = numpy.isfinite(smi)
    t, x, y = mask.nonzero()
    smi = smi[t, x, y]
    t = time[t]
    lon = lon[x, y]
    lat = lat[x, y]
    df = pandas.DataFrame(data=dict(t=t, lon=lon, lat=lat, smi=smi))
    n = df.shape[0]
    logger.info(
        f"After dropping nan values, {n :,} ({n / mask.size:2.2%}) data points remain."
    )

    if sort:
        df = df.sort_values(by=["t", "lon", "lat"])
        logger.info("Sorted for unique order.")

    return df


def _to_sklearn(ds: WindowedDataset) -> Tuple[numpy.ndarray, numpy.ndarray]:
    xs, ys = [], []
    for x, y in ds:
        x = numpy.reshape(x, (x.shape[0], -1))
        xs.append(x)
        ys.append(y)
    # sklearn models expect single dimension targets as a 1D array.
    y = numpy.concatenate(ys, axis=0)
    y = numpy.reshape(y, newshape=(y.shape[0],))
    return numpy.concatenate(xs, axis=0), y


def adapted_time_series_kfold_split(
    x: numpy.ndarray,
    y: numpy.ndarray,
    fold_no: int = 0,
    n_folds: int = 5,
) -> MutableMapping[str, Tuple[numpy.ndarray, numpy.ndarray]]:
    """
    K-fold time series split.

    Parameters
    ----------
    x: shape: (nnz, num_time_steps, num_features)
        The input data.
    y: shape: (nnz, num_time_steps)
        The target data.
    fold_no:
        The fold number.
    n_folds
        The total number of folds.

    Returns
    -------
        A mapping with keys "train", "validation", "test", and pairs of input / output numpy arrays as values.
    """
    allowed_folds = set(range(n_folds - 2))
    if fold_no not in allowed_folds:
        raise ValueError(
            f"{fold_no} is invalid. Allowed are fold_no={sorted(allowed_folds)}"
        )

    # split along time dimension
    split_idx = numpy.linspace(start=0, stop=x.shape[1], num=n_folds + 1, dtype=int)[
        fold_no + 1: fold_no + 4
    ]
    return dict(
        zip(
            ("train", "validation", "test"),
            map(
                tuple,
                zip(
                    *(
                        numpy.split(a, indices_or_sections=split_idx, axis=1)
                        for a in (x, y)
                    )
                ),
            ),
        )
    )


def _class_stat(y):
    return {
        "0": (y == 0).mean(),
        "1": (y == 1).mean(),
        "NaN": (y != y).mean(),
    }


scaler_resolver = Resolver(
    classes={
        StandardScaler,
        RobustScaler,
        MinMaxScaler,
        MaxAbsScaler,
        QuantileTransformer,
        PowerTransformer,
    },
    base=TransformerMixin,  # type: ignore
    suffix="Scaler",
)


def load_data(
    data_root: Union[pathlib.Path, os.PathLike],
    variables: Sequence[str] = ("ro", "e", "tp", "ssro", "skt"),
    add_month: bool = True,
    add_pos: bool = True,
    threshold: float = 0.2,
    batch_size: int = 32,
    debug: bool = False,
    fold_no: int = 0,
    n_folds: int = 5,
    window: int = 12,
    sklearn_dataformat: bool = False,
    scaler: Hint[TransformerMixin] = None,
    scaler_restrict_to_columns: Optional[Sequence[str]] = None,
    scaler_kwargs: Optional[Mapping[str, Any]] = None,
    keep_only_last: bool = False,
) -> Mapping[str, Union[WindowedDataset, numpy.ndarray]]:
    data_root = pathlib.Path(data_root).expanduser().resolve()
    data_path = data_root.joinpath("training_dataset.nc")
    logger.info(f"Loading data from {data_path}")
    logger.info(f"Using variables: {variables}")

    (x, y, n_features, n_timesteps, rows, cols, column_names) = load_ERA5_training_data(
        path=data_path,
        threshold=threshold,
        input_variables=variables,
        add_month=add_month,
        add_pos=add_pos,
        binary=True,
    )
    x, y = propagate_nans(x, y, nan_value=numpy.nan)

    # Adapt dims from
    #   (num_features, num_time_steps, num_rows, num_columns) to
    #   (num_rows, num_columns, num_time_steps, num_features)
    x, y = numpy.transpose(x, axes=(2, 3, 1, 0)
                           ), numpy.transpose(y, axes=(1, 2, 0))
    logger.info(f"Loaded data of shape: {x.shape}")

    # reshape to (num_rows * num_cols, num_time_steps, *)
    x = numpy.reshape(x, newshape=(-1, *x.shape[2:]))
    y = numpy.reshape(y, newshape=(-1, *y.shape[2:]))

    # drop nan values
    keep_mask = numpy.isfinite(y).all(axis=1)
    x = x[keep_mask]
    y = y[keep_mask]

    logger.info(
        f"After dropping NaN values: " f"x.shape: {x.shape}, " f"y.shape: {y.shape}",
    )

    # TODO: debug case
    if debug:
        num_debug_samples = 32
        logger.info(f"DEBUG: using only first {num_debug_samples} samples.")
        x = x[:num_debug_samples]
        y = y[:num_debug_samples]

    # adapted k-fold time series split
    xy = adapted_time_series_kfold_split(
        x=x,
        y=y,
        fold_no=fold_no,
        n_folds=n_folds,
    )
    logger.info(
        "\n".join(
            (
                "Split Statistics:",
                *(
                    f"{key:20}: class counts:"
                    + ", ".join(
                        f"{cls}: {freq:2.2%}"
                        for cls, freq in sorted(_class_stat(y).items())
                    )
                    + f" x.shape={x.shape}, y.shape: {y.shape}"
                    for key, (x, y) in xy.items()
                ),
            )
        )
    )

    if scaler_restrict_to_columns is None:
        column_indices = None
    else:
        scaler_restrict_to_columns = sorted(scaler_restrict_to_columns)
        column_indices = [column_names.index(
            col) for col in scaler_restrict_to_columns]
        logger.info(
            f"Restricting scaling to columns: {sorted(zip(column_indices, scaler_restrict_to_columns))}")

    # Merge with normalize_data?
    if scaler is not None:
        input_scaler = scaler_resolver.make(
            query=scaler, pos_kwargs=scaler_kwargs)
        logger.info(f"Normalizing input with {input_scaler}")
        x, y = xy["train"]
        if column_indices is not None:
            x = x[..., column_indices]
        input_scaler = input_scaler.fit(
            numpy.reshape(x, newshape=(-1, x.shape[-1])))
        for key in xy.keys():
            x, y = xy[key]
            x_tmp = x
            if column_indices is not None:
                x = x[..., column_indices]
            shape = x.shape
            x = numpy.reshape(x, newshape=(-1, shape[-1]))
            x = input_scaler.transform(x)
            x = numpy.reshape(x, newshape=shape)
            if column_indices is not None:
                x_tmp = numpy.copy(x_tmp)
                x_tmp[..., column_indices] = x
                x = x_tmp
            xy[key] = (x, y)

    # create windowed dataset
    xy_ds = {
        key: WindowedDataset(
            raw_x=raw_x,
            raw_y=raw_y,
            window_size=window,
            batch_size=batch_size,
            shuffle=key == "train",
            keep_only_last=keep_only_last,
        )
        for key, (raw_x, raw_y) in xy.items()
    }
    logging.info("Created windowed datasets.")
    for key, value in xy_ds.items():
        logging.info(f"{key:20}: {value}")

    # convert to long array for sklearn models
    if not sklearn_dataformat:
        return xy_ds

    logging.info("Converting to sklearn data format")
    xy_sk = {key: _to_sklearn(value) for key, value in xy_ds.items()}
    for key, (_x, _y) in xy_sk.items():
        logging.info(f"{key:20}: {_x.shape} -> {_y.shape}")

    return xy_sk
