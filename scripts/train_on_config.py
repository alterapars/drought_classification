"""Hyperparameter optimization."""
import datetime
import os
import inspect
import logging
import math
import pathlib
import pprint
import random
from typing import (
    Any,
    Iterable,
    Mapping,
    MutableMapping,
    Optional,
    Sequence,
    Tuple,
    Type,
    Union,
)

import joblib
import numpy
import pandas as pd

# from class_resolver import Resolver
from sklearn import metrics
from sklearn.base import ClassifierMixin, RegressorMixin
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import LinearSVC, NuSVC, SVC
from sklearn.utils import compute_sample_weight
from tensorflow import keras
from pathlib import Path
from class_resolver import Hint, Resolver
from tensorflow import keras
from datetime import datetime
import matplotlib.pyplot as plt

from drought_prediction.data import (
    WindowedDataset,
    load_data,
)

# from drought_prediction.hpo import (
#     _create_keras_model,
#     _get_LSTM_layers,
#     _get_dense_layers,
#     iter_cnn_layers,
# )


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
    input_shape,
    dims,
    model_type,
    activation,
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


def load_data_with_config(data_config):
    return load_data(
        data_root=data_config.pop("data_root"),
        variables=data_config.pop("vars"),
        add_month=data_config.pop("add_month"),
        add_pos=data_config.pop("add_pos"),
        threshold=data_config.pop("Threshold"),
        debug=data_config.get("debug"),
        fold_no=data_config.pop("fold_no"),
        n_folds=data_config.pop("n_folds"),
        window=data_config.pop("window"),
        sklearn_dataformat=data_config.get("sklearn_dataformat"),
        scaler=data_config.pop("scaler"),
    )


def setup_model(training_config, input_shape):
    return _create_keras_model(
        input_shape=input_shape,
        dims=training_config.pop("dims"),
        model_type=training_config.pop("model_type"),
        activation=training_config.pop("activation"),
        use_batch_norm=training_config.pop("use_batch_norm"),
        output_dim=1,
        is_classification=True,
        dropout=training_config.pop("dropout"),
    )


def plot_confusion_matrix(cm, viz_path, plot_name, fold_no, model_name):
    plt.figure()
    plt.tight_layout()
    plt.imshow(cm, interpolation="nearest", cmap=plt.cm.Pastel2)
    classNames = ["Negative", "Positive"]
    plt.title("Confusion Matrix - Test Data on fold " + str(fold_no) + str(model_name))
    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    tick_marks = numpy.arange(len(classNames))
    plt.xticks(tick_marks, classNames, rotation=45)
    plt.yticks(tick_marks, classNames)
    s = [["TN", "FP"], ["FN", "TP"]]
    for i in range(2):
        for j in range(2):
            plt.text(j, i, str(s[i][j]) + " = " + str(cm[i][j]))
    plt.savefig(viz_path + "/" + plot_name)


def train_model(
    model, model_name, training_config, data_config, all_data_loader, fold_no
):  # noqa: D102
    # config
    eval_frequency = training_config.pop("eval_frequency")
    batch_size = training_config.pop("batch_size")
    class_weight = training_config.pop("class_weight")
    learning_rate = training_config.pop("learning_rate")
    viz_path = data_config.pop("viz_path")
    model_output_path = data_config.pop("model_output_path")
    fit_kwargs = training_config

    # compile
    model.compile(
        loss="binary_crossentropy",
        optimizer=keras.optimizers.Adam(learning_rate),
        metrics=[
            keras.metrics.BinaryAccuracy(name="accuracy"),
            keras.metrics.Precision(name="precision"),
            keras.metrics.Recall(name="recall"),
            keras.metrics.AUC(curve="ROC", name="roc_auc"),
            keras.metrics.AUC(curve="PR", name="pr_auc"),
        ],
    )
    print(model.summary())

    # callbacks
    callbacks = [
        keras.callbacks.TensorBoard(
            log_dir=os.path.join(
                model_output_path,
                "logs",
                "scalars",
                datetime.now().isoformat().replace(":", "__"),
            ),
            update_freq="batch",
            histogram_freq=1,
        ),
        keras.callbacks.ModelCheckpoint(
            filepath=model_output_path,
            monitor="roc_auc",
            mode="max",
            save_best_only=True,
        ),
    ]
    # step
    history = model.fit(
        x=all_data_loader["train"],
        epochs=eval_frequency,
        shuffle=True,
        # callbacks=callbacks,
        **fit_kwargs,
        class_weight=class_weight,
        verbose=2,
    )

    model.save(
        model_output_path
        + str(model_name)
        + str(fold_no)
        + "_"
        + str(datetime.now().strftime("%Y%m%d-%H%M"))
        + ".h5"
    )

    test_dataset = all_data_loader["test"]

    # Evaluate for test data
    results = model.evaluate(x=test_dataset)
    print("\ntest loss, test acc:", results)
    # test_accuracy = results[1]

    # convert list of pairs to pair of lists
    x_batches, y_batches = list(zip(*test_dataset))

    # stack into one large array
    x_test = numpy.concatenate(x_batches, axis=0)
    y_test = numpy.concatenate(y_batches, axis=0)
    # Calculate and display the error metrics
    y_pred = (model.predict(x_test) > 0.5).astype("int32")
    y_pred_int = y_pred.astype(int)

    cMatrix = confusion_matrix(y_test, y_pred_int)
    pScore = precision_score(y_test, y_pred_int)
    rScore = recall_score(y_test, y_pred_int)
    f1Score = f1_score(y_test, y_pred_int, average="macro")

    # plot_name = (
    #     str(model_name)
    #     + "cm_fold"
    #     + str(fold_no)
    #     + "_"
    #     + str(datetime.now().strftime("%Y%m%d-%H%M"))
    #     + ".png"
    # )
    # plot_confusion_matrix(cMatrix, viz_path, plot_name, fold_no, model_name)

    print("Confusion matrix: \n", cMatrix)
    print("\nP-Score: %.3f, R-Score: %.3f" % (pScore, rScore))
    print("f1Score: ", f1Score)

    return cMatrix, pScore, rScore, f1Score


if __name__ == "__main__":
    cMatrix_list = []
    pScore_list = []
    rScore_list = []
    f1score_list = []

    ############SPECIFY#############
    model_name = "lstm"

    # iterate over folds in config:
    for i in range(0, 3):
        fold_no = i

        print("------------------------------------------------------------------------")
        print(f"Training for fold {fold_no} ...")

        config = dict(
            data=dict(
                classification=True,
                scaler="MinMax",
                window=6,
                n_folds=5,
                fold_no=fold_no,
                Threshold=0.2,
                # sklearn_dataformat=True,
                sklearn_dataformat=False,
                vars=[
                    # ERA5 shortened:
                    "str",
                    "tp",
                    "skt",
                    "sp",
                    "u10",
                    "v10",
                    "ssr",
                    "str",
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
                    # soiltype:
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
                    # target var
                    "SMI",
                ],
                add_month=True,
                add_pos=True,
                debug=False,
                # WINDOWS
                # data_root=r"C:/Users/gott_ju/dev/anomaly-detection/DATA/data-modified/final-input-data-all-vars",
                # model_output_path=r"C:/Users/gott_ju/dev/anomaly-detection/DATA/",
                # viz_path=r"C:/Users/gott_ju/dev/anomaly-detection/DATA/",
                # MISTRAL
                viz_path="/work/bd1083/b309184/trained_models/viz",
                data_root="/work/bd1083/b309184/DATA/final_input_data_1981-2018_on_ERA5grid/",
                model_output_path="/work/bd1083/b309184/trained_models",
            ),
            training=dict(
                # # # lstm
                batch_size=201,
                activation="softplus",
                dims=[80],
                use_batch_norm=True,
                learning_rate=0.000015245,
                class_weight={0: 1, 1: 4},
                eval_frequency=100,
                model_type="lstm",
                dropout=0.2,
                # # # # cnn
                # batch_size=7,  # 7
                # activation="softplus",
                # dims=[128],
                # use_batch_norm=False,
                # learning_rate=0.00036695,
                # class_weight={0: 1, 1: 4},
                # eval_frequency=150,
                # model_type="cnn",
                # dropout=0.1,
                # # # # dense
                # batch_size=1952,
                # activation="softplus",
                # dims=[96],
                # use_batch_norm=True,
                # learning_rate=0.0013663,
                # class_weight={0: 1, 1: 4},
                # eval_frequency=150,
                # model_type="dense",
                # dropout=0.2,
            ),
        )

    data_config = config.get("data")
    training_config = config.get("training")

    xy_sk = load_data_with_config(data_config)
    input_shape = xy_sk["train"].input_shape
    model = setup_model(training_config, input_shape)
    # train model
    cMatrix, pScore, rScore, f1Score = train_model(
        model, model_name, training_config, data_config, xy_sk, fold_no
    )
    cMatrix_list.append(cMatrix)
    pScore_list.append(pScore)
    rScore_list.append(rScore)
    f1score_list.append(f1Score)

    metrics_filename = (
        "/pf/b/b309184/anomaly-detection/scripts/evaluation_txtfiles/"
        + str(model_name)
        + "metrics_on_test_data_lstm2"
        + str(datetime.now().strftime("%Y%m%d"))
    )

    print(metrics_filename)

    with open(metrics_filename + ".txt", "w") as f:
        f.write("confusion matrix: \n")
        for item in cMatrix_list:
            f.write("%s\n" % item)
        f.write("p-score: \n")
        for item in pScore_list:
            f.write("%s\n" % item)
        f.write("r-score: \n")
        for item in rScore_list:
            f.write("%s\n" % item)
        f.write("F1-score: \n")
        for item in f1score_list:
            f.write("%s\n" % item)

    # Create a DataFrame object
    # dictionary of lists
    dict = {"f1Score": f1score_list, "rScore": rScore_list, "pScore": pScore_list}
    df = pd.DataFrame(dict)
    # saving the dataframe
    df.to_csv(metrics_filename + ".csv")
