#%%
import pandas as pd
from pathlib import Path
from os.path import exists
from pathlib import Path
from seaborn import regplot, lineplot
from matplotlib.pyplot import savefig, subplots

from brainage import BrainAgePredictor

pretrained_weights='./brainage/models/exports/pretrained_weights/run_20190719_00_epoch_best_mae.p'

# # Testing original model on whole/test IXI
# trained_weights = './brainage/models/exports/pretrained_weights/run_20190719_00_epoch_best_mae.p'
# test_csv = './brainage/data/datasets/ixi_fsl_bet_flirt_antsBC_test.csv'
# save_label = 'ixi/ixi_test_pretrained_sfcn'
# architecture = 'sfcn'

# Testing original model on whole/test 1000brains
trained_weights = './brainage/models/exports/pretrained_weights/run_20190719_00_epoch_best_mae.p'
test_csv = './brainage/data/datasets/1000brains_subject_list_new_test.csv'
save_label = '1000brains/1000brains_test_pretrained_sfcn'
architecture = 'sfcn'

# # Testing our trained models on test 1000brains
# trained_weights = './brainage/models/exports/model_results/1000brains/1000brains_adam_0.001_1000_sfcn/state_dict.pt'
# test_csv = './brainage/data/datasets/1000brains_subject_list_new_test.csv'
# save_label = '1000brains/xx/1000brains_adam_0.001_240_sfcn'
# architecture = 'sfcn'

# trained_weights = './brainage/models/exports/model_results/1000brains/1000brains_adam_0.01_1000_sfcn/state_dict.pt'
# test_csv = './brainage/data/datasets/1000brains_subject_list_new_test.csv'
# save_label = '1000brains/1000brains_adam_0.01_1000_sfcn'
# architecture = 'sfcn'

# Test Rank-SFCN model
# trained_weights = './brainage/models/exports/model_results/1000brains/1000brains_adam_0.001_1000_earlystop_rank_sfcn/state_dict.pt'
# test_csv = './brainage/data/datasets/1000brains_subject_list_new_test.csv'
# save_label = '1000brains/1000brains_adam_0.001_1000_earlystop_rank_sfcn'
# architecture = 'rank_sfcn'

# # # Test Rank-Resnet model: set data_path=test_csv, train_all_layers=True, pretrained_weights=None
# trained_weights = './brainage/models/exports/model_results/1000brains/before_earlystopping/1000brains_adam_0.001_240_rank_resnet3d/state_dict.pt'
# test_csv = './brainage/data/datasets/1000brains_subject_list_new_test.csv'
# save_label = '1000brains/before_earlystopping1000brains_adam_0.001_240_rank_resnet3d'
# architecture = 'rank_resnet3d'


# %% Class initialization without weights
bap = BrainAgePredictor(
    data_path=None,
    age_filter=[42, 82],
    image_dimensions=(160, 192, 160),
    steps=('normalize_image', 'crop_center'),
    learning_rate=0.0001,
    number_of_epochs=240,
    batch_size=3,
    train_all_layers=False,
    architecture=architecture,
    optimizer='adam',
    pretrained_weights=pretrained_weights,
    metrics=('CORR', 'MSE', 'MAE'),
    save_label=save_label)

# %%
print('trainable parameters: ', sum(p.numel() for p in bap.data_model_predictor.model.parameters() if p.requires_grad))

#%%
# Load the weights now after class initialization
bap.update_parameters(trained_weights)

#%% load predictions if they exists else read test data and get predictions
prediction_file = Path(bap.data_model_predictor.save_path, 'predictions.csv')
if exists(prediction_file):
    results_df = pd.read_csv(prediction_file)
    metrics = bap.evaluate(results_df['age'], results_df['prediction'])
    print(metrics)
else:
    #load the test data and predict using filepaths
    test_df = pd.read_csv(test_csv)
    test_df = test_df[test_df['age'].between(42, 82)]
    test_df = test_df.reset_index(drop=True)
    print('test_df.shape', test_df.shape)

    # predict using filepaths
    file_path = tuple(test_df['file_path'])
    prediction = bap.predict(file_path)

    # evaluate the model perforamance
    true_label = tuple(test_df['age'])
    metrics = bap.evaluate(true_label, prediction)
    print(metrics)

    # save the results
    results_df = pd.DataFrame([true_label, prediction], index=['age', 'prediction']).T
    results_df.to_csv(Path(bap.data_model_predictor.save_path, 'predictions.csv'), index=False)


# # %% plot true vs predicted scatter plot
# Create a new figure and axis
figure, axis = subplots(nrows=1, figsize=(5, 5))

lineplot(x=results_df['age'], y=results_df['age'], color="darkgreen", 
         linestyle='dashed', label="Equality line")
                        
# Add the scatter/regression plot
label =  save_label.split('/')[1]
regplot(x=results_df['age'],
        y=results_df['prediction'], label=label,
        scatter_kws={"color":"darkgreen"},
        line_kws={"color": "orange"})

# Set the plot title
axis.set_title('MAE:' + str(int(metrics['MAE'] * 100) / 100) + '   CORR:' 
               + str(int(metrics['CORR'] * 100) / 100), fontweight='bold')

# Set the both axis to be equal
axis.set_box_aspect(1)

# Set the x- and y-label
axis.set_xlabel('True age', fontsize=12)
axis.set_ylabel('Predicted age', fontsize=12)

# Set the x-limits and y-limits
import numpy as np
x_lower =  np.min([results_df['age'].min(), results_df['prediction'].min()])
x_upper =  np.max([results_df['age'].max(), results_df['prediction'].max()])
axis.set_xlim(x_lower-1, x_upper+1)
axis.set_ylim(x_lower-1, x_upper+1)

# Set the facecolor for the axis
axis.set_facecolor("whitesmoke")

# Specify the grid with a subgrid
axis.grid(which='major', color='lightgray', linewidth=0.8)
axis.grid(which='minor', color='lightgray', linestyle=':',
            linewidth=0.5)
axis.minorticks_on()
axis.legend()

# Apply a tight layout to the figure
figure.tight_layout()

# Save the plot
file_nm_out = Path(bap.data_model_predictor.save_path, 'test_true_vs_pred.png')
savefig(file_nm_out)
