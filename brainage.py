"""Brain age prediction."""

# %% External package import

import time

# %% Internal package import

from brainage import BrainAgePredictor

# %% EXAMPLE WORKFLOW

# %% Class initialization

bap = BrainAgePredictor(
    data_path='/data/project/brainage_deeplearning/Hida_AgePrediction/dl_brainage/data/T1/ixi_fsl_bet_flirt_antsBC.csv',
    age_filter=[42, 82],
    image_dimensions=(160, 192, 160),
    steps=('normalize_image', 'crop_center'),
    learning_rate=0.0001,
    number_of_epochs=240,
    batch_size=3,
    train_all_layers=False,
    architecture='sfcn',
    optimizer='adam',
    pretrained_weights='./brainage/models/exports/pretrained_weights/'
    'run_20190719_00_epoch_best_mae.p',
    metrics=('CORR', 'MSE', 'MAE'),
    save_label='ixi_adam_0.0001_240')

# %% Model fitting

start_time = time.time()
bap.fit()
end_seconds = time.time()
print("--- %s minutes ---" % ((time.time() - start_time)/60))

# %% Model prediction

prediction = bap.predict(tuple(bap.data_loader.sets['test']['file_path']))

# %% Model evaluation

metrics = bap.evaluate(tuple(bap.data_loader.sets['test']['age']), prediction)

# %% Results visualization
bap.plot("Training_loss")
bap.plot("Validation_loss")
