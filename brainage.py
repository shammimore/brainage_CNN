"""Brain age prediction."""

# %% Internal package import

from brainage import BrainAgePredictor

# %% Class definition

bap = BrainAgePredictor(
    data_path='./brainage/data/datasets/'
    'ixi_fsl_bet_flirt_antsBC_train.csv',
    age_filter=[42, 82],
    image_dimensions=(160, 192, 160),
    learning_rate=0.0001,
    number_of_epochs=10,
    batch_size=3,
    train_all_layers=False,
    architecture='sfcn',
    optimizer='adam',
    pretrained_weights='./brainage/models/exports/pretrained_weights/'
    'run_20190719_00_epoch_best_mae.p')
