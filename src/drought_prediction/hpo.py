"""Hyperparameter optimization."""
import datetime
import inspect
import logging
import math
import pathlib
import pprint
import random
from typing import (Any, Iterable, Mapping, MutableMapping, Optional, Sequence,
                    Tuple, Type, Union)

import joblib
import numpy
from class_resolver import Resolver
from ray import tune
from ray.tune.sample import Function
from ray.tune.utils import flatten_dict
from sklearn import metrics
from sklearn.base import ClassifierMixin, RegressorMixin
from sklearn.metrics import classification_report
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import LinearSVC, NuSVC, SVC
from sklearn.utils import compute_sample_weight
from tensorflow import keras

from .data import WindowedDataset, load_data

logger = logging.getLogger(__name__)

classes = {
    # mlp
    MLPClassifier,
    # svm
    SVC,
    NuSVC,
    LinearSVC,
    # knn
    KNeighborsClassifier,
}

# try to replace gradient boosting by histogram gradient boosting
try:
    # explicitly require this experimental feature
    # now you can import normally from ensemble
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.experimental import enable_hist_gradient_boosting  # noqa

    gradient_boosting_cls = HistGradientBoostingClassifier
except ImportError:
    # gradient boosting
    from sklearn.ensemble import GradientBoostingClassifier

    gradient_boosting_cls = GradientBoostingClassifier
logger.info(f"Using {gradient_boosting_cls.__name__} for gradient boosting.")
classes.add(gradient_boosting_cls)

sklearn_model_resolver = Resolver(
    classes=classes,
    base=ClassifierMixin,  # type: ignore
    default=gradient_boosting_cls,
)

# Default configurations for sklearn models
DEFAULT_SKLEARN_MODEL_CONFIGS: Mapping[Type[ClassifierMixin], Mapping[str, Any]] = {
    gradient_boosting_cls: dict(max_leaf_nodes=64, min_samples_leaf=32),
    MLPClassifier: dict(),
    SVC: dict(),
    NuSVC: dict(),
    LinearSVC: dict(),
    KNeighborsClassifier: dict(leaf_size=256, n_jobs=-1),
}


def qlograndint(lower, upper, q, base=10):
    """Work-around for older ray.tune versions."""

    lower = max(lower, q)

    def _sample(spec):
        # log space
        log_lower = math.log(lower, base)
        log_upper = math.log(upper, base)
        # uniform sample
        x = random.uniform(log_lower, log_upper)
        # back from log space
        x = math.pow(base, x)
        # round
        x = int(round(x / q) * q)
        return x

    return tune.sample_from(_sample)


def _load_data(
    data_config: MutableMapping[str, Any],
    batch_size: int,
) -> Mapping[str, Union[WindowedDataset, numpy.ndarray]]:
    data_config.setdefault("data_root", r"C:\Users\jsgot\dev\PhD\anomaly-detection\DATA\data-modified")
    return load_data(batch_size=batch_size, **data_config)


class SklearnTrainable(tune.Trainable):
    """A trainable for sklearn models for drought prediction."""

    model: Union[ClassifierMixin, RegressorMixin]

    def setup(self, config: MutableMapping[str, Any]):
        """Initialize model from config."""
        logging.basicConfig(level=config.get(
            "logging", {}).get("level", logging.INFO))
        # Initialize model
        model_config = config.pop("model")
        model = model_config.pop("name")
        # lookup model class
        model = sklearn_model_resolver.lookup(query=model)
        # get default configuration
        this_model_config = DEFAULT_SKLEARN_MODEL_CONFIGS.get(model, {})
        logger.info(
            f"Loaded default config for class {model}: \n\t{pprint.pformat(this_model_config, indent=2)}"
        )
        # override values
        this_model_config.update(model_config)
        # initialize model
        self.model = sklearn_model_resolver.make(
            model, pos_kwargs=this_model_config)
        logger.info(f"Instantiated model: {self.model}")

        # Load data
        self.is_classification = isinstance(self.model, ClassifierMixin)
        data_config = config.pop("data", {})
        # set sklearn_dataformat if not given
        data_config.setdefault("sklearn_dataformat", True)
        self.balance = data_config.pop("balance", False)
        self.data = _load_data(data_config=data_config, batch_size=1)

    def step(self):
        """Perform one training iteration."""
        logger.info(f"Fitting model: \n\t{self.model}")
        x, y = self.data["train"]
        kwargs = dict()
        if self.balance:
            # inverse class weights
            if "sample_weight" not in inspect.signature(self.model.fit).parameters.keys():
                raise ValueError(
                    f"{self.model.__class__.__name__}.fit does not support sample weights."
                )
            kwargs["sample_weight"] = weights = compute_sample_weight(
                class_weight="balanced", y=y
            )
            logger.info(f"Computed class weights: {numpy.unique(weights)}")
        self.model.fit(x, y, **kwargs)
        logger.info("Evaluating model")
        assert self.is_classification
        result = dict()
        for key, (x, y_true) in self.data.items():
            y_pred = self.model.predict(x)
            result[key] = dict(
                report=classification_report(
                    y_true=y_true, y_pred=y_pred, output_dict=True
                ),
                pr_auc=metrics.average_precision_score(
                    y_true=y_true, y_score=y_pred),
                roc_auc=metrics.roc_auc_score(y_true=y_true, y_score=y_pred),
            )

        # stop trial
        result[tune.result.DONE] = True
        # replace spaces by _
        return {
            key.replace(" ", "_"): value for key, value in flatten_dict(result).items()
        }

    def save_checkpoint(self, tmp_checkpoint_dir: str):
        root = pathlib.Path(tmp_checkpoint_dir).expanduser().resolve()
        logger.info(f"Checkpointing to {root.as_uri()}")
        joblib.dump(self.model, root.joinpath("model.joblib"))

    def load_checkpoint(self, checkpoint: str):
        root = pathlib.Path(checkpoint).expanduser().resolve()
        logger.info(f"Loading checkpoint from {root.as_uri()}")
        self.model = joblib.load(root.joinpath("model.joblib"))

    def _train(self):
        raise AssertionError("Do not call _train.")

    def _save(self, tmp_checkpoint_dir):
        raise AssertionError("Do not call _save.")

    def _restore(self, checkpoint):
        raise AssertionError("Do not call _restore.")


pooling_resolver = Resolver(
    classes={
        keras.layers.GlobalAveragePooling1D,
        keras.layers.GlobalMaxPooling1D,
    },
    base=keras.layers.Layer,  # type: ignore
    synonyms=dict(
        avg=keras.layers.GlobalAveragePooling1D,
        max=keras.layers.GlobalMaxPooling1D,
    ),
)


class Tanh(keras.layers.Activation):
    def __init__(self, **kwargs):
        super().__init__(activation="tanh", **kwargs)


class Softplus(keras.layers.Activation):
    def __init__(self, **kwargs):
        super().__init__(activation="softplus", **kwargs)


class Sigmoid(keras.layers.Activation):
    def __init__(self, **kwargs):
        super().__init__(activation="sigmoid", **kwargs)


activation_resolver = Resolver(
    classes={
        keras.layers.ReLU,
        keras.layers.PReLU,
        keras.layers.LeakyReLU,
        Softplus,
        Tanh,
        Sigmoid,
    },
    base=keras.layers.Layer,  # type: ignore
)


def _get_dense_layers(
    dims: Sequence[int],
    activation: str = "relu",
    use_batch_norm: bool = False,
    dropout: Optional[float] = None,
) -> Iterable[keras.layers.Layer]:
    """
    Create a sequence of dense layers according to the configuration.

    :param dims:
        The *hidden* dimensions.
    :param activation:
        The activation function to use for the hidden layers.
    :param use_batch_norm:
        Whether to use batch normalization.
    :param dropout:
        The dropout rate.

    :return:
        A sequence of dense layers with specified activation / batch normalization / dropout modules in between.
    """
    # flatten input to vector; no need to specify input shape of next layer
    yield keras.layers.Flatten()
    for dim in dims:
        yield keras.layers.Dense(
            units=dim,
            activation=None,
            use_bias=not use_batch_norm,
        )
        if use_batch_norm:
            yield keras.layers.BatchNormalization()
            yield activation_resolver.make(query=activation)
        else:
            yield activation_resolver.make(query=activation)
        if dropout:
            yield keras.layers.Dropout(rate=dropout)


def _get_LSTM_layers(
    dims: Sequence[int],
    activation: str = "relu",
    use_batch_norm: bool = False,
    dropout: Optional[float] = None,
    bi_directional: bool = False,
    pooling: str = "avg",
) -> Iterable[keras.layers.Layer]:
    """
    Create a sequence of LSTM layers according to the configuration.

    :param dims:
        The *hidden* dimensions.
    :param activation:
        The activation function to use for the hidden layers.
    :param use_batch_norm:
        Whether to use batch normalization.
    :param dropout:
        The dropout rate.

    :return:
        A sequence of dense layers with specified activation / batch normalization / dropout modules in between.
    """
    for dim in dims:
        layer = keras.layers.LSTM(
            units=dim,
            return_sequences=True,
            activation=None,
            use_bias=not use_batch_norm,
        )
        if bi_directional:
            layer = keras.layers.Bidirectional(layer=layer)
        yield layer
        if use_batch_norm:
            yield keras.layers.BatchNormalization()
            yield activation_resolver.make(query=activation)
        else:
            yield activation_resolver.make(query=activation)
        if dropout:
            yield keras.layers.Dropout(rate=dropout)
    # TODO: other pooling options, e.g., only last state?
    yield pooling_resolver.make(query=pooling)


def iter_cnn_layers(
    dims: Sequence[int],
    activation: str = "relu",
    kernel_size: int = 3,
    use_batch_norm: bool = False,
    dropout: Optional[float] = None,
    pooling: str = "avg",
) -> Iterable[keras.layers.Layer]:
    """Yield CNN layers."""
    for dim in dims:
        yield keras.layers.Conv1D(
            filters=dim,
            kernel_size=kernel_size,
            padding="same",
            use_bias=not use_batch_norm,
            activation=None,
        )
        if use_batch_norm:
            yield keras.layers.BatchNormalization()
            yield activation_resolver.make(query=activation)
        else:
            yield activation_resolver.make(query=activation)
        if dropout:
            yield keras.layers.Dropout(0.5)
    yield pooling_resolver.make(query=pooling)


def _create_keras_model(
    input_shape: Tuple[int, ...],
    dims: Sequence[int] = tuple(),
    model_type: str = "dense",
    activation: str = "relu",
    use_batch_norm: bool = False,
    output_dim: int = 1,
    is_classification: bool = True,
    dropout: Optional[float] = None,
) -> keras.Sequential:
    """
    Create a Keras model with the given configuration.

    :param dims:
        The *hidden* dimensions.
    :param model_type:
        The model type. Supported types: "dense". Case insensitive.
    :param activation:
        The activation function to use for the hidden layers.
    :param use_batch_norm:
        Whether to use batch normalization.
    :param output_dim:
        The dimension of the output.
    :param is_classification:
        Whether to use sigmoid on the output.
    :param dropout:
        The dropout rate.

    :return:
        A Keras sequential model according to the configuration.
    """
    # normalize model type
    model_type = model_type.lower()

    layers = [keras.layers.InputLayer(input_shape=input_shape)]

    # sequence encoder
    if model_type == "dense":
        layers.extend(
            _get_dense_layers(
                dims=dims,
                activation=activation,
                use_batch_norm=use_batch_norm,
                dropout=dropout,
            )
        )
    elif model_type == "lstm":
        layers.extend(
            _get_LSTM_layers(
                dims=dims,
                activation=activation,
                use_batch_norm=use_batch_norm,
                dropout=dropout,
            )
        )
    elif model_type == "cnn":
        layers.extend(
            iter_cnn_layers(
                dims=dims,
                activation=activation,
                use_batch_norm=use_batch_norm,
                dropout=dropout,
            )
        )
    else:
        raise NotImplementedError(f"Unknown model type: {model_type}")
    # decoder
    layers.append(keras.layers.Dense(output_dim))
    if is_classification:
        layers.append(activation_resolver.make("sigmoid"))
    return keras.Sequential(layers=layers)


optimizer_resolver = Resolver.from_subclasses(
    base=keras.optimizers.Optimizer, default=keras.optimizers.Adam
)


def sample_dims(
    min_num_layers: int = 1,
    max_num_layers: int = 5,
    upper: int = 256,
    q: int = 16,
) -> Function:
    """
    Create a sampler for hidden dimensions.

    :param min_num_layers:
        The minimum number of hidden layers.
    :param max_num_layers:
        The maximum number of hidden layers.
    :param upper:
        The maximum number of units per layer.
    :param q:
        An integer number. The units per layer are chosen to be multiple of `q`.

    :return:
        A sampler for hidden dims.
    """

    def _sample_dims(conf):
        """Sample hidden dimensions."""
        # first sample number of layers
        num_layers = random.randrange(min_num_layers, max_num_layers)
        # then randomly sample number of units per layer; make sure the units are increasing
        return sorted((random.randrange(q, upper, q) for _ in range(num_layers)))

    return tune.sample_from(_sample_dims)


class KerasTrainable(tune.Trainable):
    """A trainable for Keras models."""

    def setup(self, config):
        logging.basicConfig(level=config.get(
            "logging", {}).get("level", logging.INFO))
        logging.info(f"Full config: {pprint.pformat(config)}")

        training_config = config.get("training", {})
        self.eval_frequency = training_config.pop("eval_frequency", 1)
        batch_size = training_config["batch_size"]
        self.fit_kwargs = training_config

        self.all_data_loader = _load_data(
            config.get("data"), batch_size=batch_size)

        # Infer n_features, window_size from data
        input_shape = self.all_data_loader["train"].input_shape

        # Initialize model
        model_config = config.pop("model")
        model = model_config.pop("name").lower()
        prefix = "keras-"
        assert model.startswith(prefix)
        model = model[len(prefix):]
        self.model = _create_keras_model(
            model_type=model,
            input_shape=input_shape,
            **model_config,
        )
        logger.info(f"Instantiated model: {self.model}")

        # create optimizer
        opt_config = config.get("optimizer", {})
        opt_name = opt_config.pop("name", None)
        optimizer = optimizer_resolver.make(opt_name, pos_kwargs=opt_config)

        # compile
        self.model.compile(
            loss="binary_crossentropy",
            optimizer=optimizer,
            metrics=[
                keras.metrics.BinaryAccuracy(name="accuracy"),
                keras.metrics.Precision(name="precision"),
                keras.metrics.Recall(name="recall"),
                keras.metrics.AUC(curve="ROC", name="roc_auc"),
                keras.metrics.AUC(curve="PR", name="pr_auc"),
            ],
        )
        self.model.summary()

        # callbacks
        callback_config = config.get("callbacks", {})
        self.callbacks = []
        if "early_stopping" in callback_config:
            self.callbacks.append(
                keras.callbacks.EarlyStopping(
                    **callback_config["early_stopping"]),
            )
        if "tensorboard" in callback_config:
            tensorboard_config: MutableMapping = callback_config["tensorboard"]
            tensorboard_config.setdefault(
                "log_dir",
                pathlib.Path(
                    "~", "logs", datetime.datetime.now().isoformat().replace(":", "__")
                ),
            )
            self.callbacks.append(
                keras.callbacks.TensorBoard(**tensorboard_config),
            )

        self.epoch = 0

    def step(self):  # noqa: D102

        class_weight = {0: 1.0, 1: 4}

        self.model.fit(
            x=self.all_data_loader["train"],
            epochs=self.eval_frequency,
            shuffle=True,
            callbacks=self.callbacks,
            **self.fit_kwargs,
            class_weight=class_weight,
        )
        self.epoch += self.eval_frequency
        result = dict(
            epoch=self.epoch,
        )
        for key, data_loader in self.all_data_loader.items():
            result[key] = self.model.evaluate(
                x=data_loader,
                return_dict=True,
                verbose=self.fit_kwargs.get("verbose", 2),
            )
        logger.info(f"Result:\n{pprint.pformat(result)}")
        # return flatten_dict(result)
