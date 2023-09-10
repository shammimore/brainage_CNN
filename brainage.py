"""Brain age prediction."""
import time

# %% Internal package import

from brainage import BrainAgePredictor

# %% Class initialization

bap = BrainAgePredictor(
    data_path='./brainage/data/datasets/1000brains_subject_list_new.csv',
    age_filter=[42, 82],
    image_dimensions=(160, 192, 160),
    steps=('normalize_image', 'crop_center'),
    learning_rate=0.01,
    number_of_epochs=3,
    batch_size=4,
    early_stopping_rounds=20,
    reduce_lr_on_plateau={'rounds': 10, 'factor': 0.5},
    train_all_layers=False,
    architecture='rank_sfcn',
    optimizer='adam',
    pretrained_weights='./brainage/models/exports/pretrained_weights/'
    'run_20190719_00_epoch_best_mae.p',
    metrics=('CORR', 'MSE', 'MAE'),
    save_label='ixi_adam_0.01_trail1')

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
