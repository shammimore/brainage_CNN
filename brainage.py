"""Brain age prediction."""

# %% Internal package import

from brainage import BrainAgePredictor

# %% Class definition

bap = BrainAgePredictor(
    data_path='./brainage/data/datasets/ixi_fsl_bet_flirt_antsBC_train.csv',
    age_filter=[42, 82],
    image_dimensions=(160, 192, 160),
    steps=('normalize_image', 'crop_center'),
    learning_rate=0.0001,
    number_of_epochs=10,
    batch_size=3,
    train_all_layers=False,
    architecture='sfcn',
    optimizer='adam',
    pretrained_weights='./brainage/models/exports/pretrained_weights/'
    'run_20190719_00_epoch_best_mae.p',
    metrics=('CORR', 'MSE', 'MAE'))

# %% Example prediction

prediction = bap.predict(
            ('brainage/data/datasets/images/sub-IXI332/final/'
             'highres001_BrainExtractionBrain_flirt.nii.gz',
             'brainage/data/datasets/images/sub-IXI597/final/'
             'highres001_BrainExtractionBrain_flirt.nii.gz'))

metrics = bap.evaluate([bap.data_loader.sets['test'].loc[0, 'age'],
                        bap.data_loader.sets['test'].loc[1, 'age']],
                       prediction)
