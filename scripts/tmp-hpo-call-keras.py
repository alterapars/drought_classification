import logging

from ray import tune as ray_tune

from drought_prediction.hpo import KerasTrainable, qlograndint, sample_dims

config = dict(
    model=dict(
        name="dense-lstm",
        dims=sample_dims(),
        # LeakyReLU does not work
        activation=ray_tune.choice(["relu", "softplus"]),
        use_batch_norm=ray_tune.choice([False, True]),
        dropout=ray_tune.choice([None, 0.1, 0.2]),
    ),
    data=dict(
        classification=True,
        window=6,
        n_folds=5,
        fold_no=2,
        Threshold=0.2,
        # sklearn_dataformat=True,
        sklearn_dataformat=False,
        vars=[
            # ERA5:
            'surface_net_thermal_radiation',
            'total_precipitation',
            'skin_temperature',
            'surface_pressure',
            '10m_u_component_of_wind',
            '10m_v_component_of_wind',
            'surface_net_solar_radiation',
            'surface_net_thermal_radiation',
            'leaf_area_index_low_vegetation',
            'leaf_area_index_high_vegetation',
            'surface_thermal_radiation_downwards',
            # soiltype:
            'water',
            'evergreen_needleleaf_forest',
            'Evergreen_Broadleaf_forest',
            'Deciduous_Needleleaf_forest',
            'Deciduous_Broadleaf_forest',
            'Mixed_forest',
            'Closed_shrublands',
            'Open_shrublands',
            'Woody_savannas',
            'Savannas',
            'Grasslands',
            'Permanent_wetlands',
            'C3_Croplands',
            'Urban_and_built_up',
            'C3_Cropland_Natural_vegetation_mosaic',
            'Snow_and_ice',
            'Barren_or_sparsely_vegetated',
            'C4_fratcion_Croplands',
            'C4_fraction_Cropland_Natural_vegetation_mosaic',
            # target var
            'SMI',
        ],
        add_month=True,
        add_pos=True,
        debug=False,
        data_root="/final_input_data_1981-2018_on_ERA5grid/",
    ),
    training=dict(
        batch_size=qlograndint(lower=32, upper=4096, q=32),
        # eval_frequency=eval_frequency,
    ),
    optimizer=dict(
        learning_rate=ray_tune.loguniform(lower=1.0e-05, upper=1.0e-01),
    ),
    logging=dict(
        level=logging.DEBUG,
    ),
    callbacks=dict(
        early_stopping=dict(
            monitor="loss",
            patience=5,
            min_delta=0.01,
            mode="min",
        ),
    ),
)

if __name__ == "__main__":
    # fixme: for debugging
    # ray.init(
    #     local_mode=True,
    #     logging_level=logging.DEBUG,
    # )
    ray_tune.run(
        KerasTrainable,
        resources_per_trial={'gpu': 2},
        config=config,
        mode="max",
        metric="validation/accuracy",
        fail_fast=True,
    )
