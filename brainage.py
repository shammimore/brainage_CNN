"""Brain age prediction."""

# %% Internal package import

from brainage import BrainAgePredictor
import numpy as np

# %% Class definition

bap = BrainAgePredictor(
    data_path='./brainage/data/datasets/'
    'ixi_fsl_bet_flirt_antsBC_train.csv',
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
    'run_20190719_00_epoch_best_mae.p')

images = bap.data_loader.get_images(which='test')
age_values = bap.data_loader.get_age_values(which='test')
image_label = bap.data_preprocessor.preprocess(images, age_values)
image_label_data = next(image_label)
data = image_label_data[0]
label = image_label_data[1]
bap.data_model_predictor.run_prediction_model(data, label)