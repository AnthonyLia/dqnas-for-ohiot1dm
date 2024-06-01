import tensorflow
import tensorflow.keras
import numpy as np
from time_series2 import *
import warnings
warnings.filterwarnings('ignore')
import pandas as pd
import csv

unnorm_patient_test_df = np.array([])
unnorm_preds = np.array([])
patient_numbers = [540,544,552,559,563,567,570,575,584,588,591,596]
def ground_truth_seg_regions(val, level, risk_levels):
    if 0 <= val <= 120:
        if level == "A":
            risk_levels["None"] += 1
        elif level == "B":
            risk_levels['Slight'] += 1
        elif level == "C":
            risk_levels['Moderate'] += 1
        elif level == "D":
            risk_levels['Extreme'] += 1
        elif level == "E":
            risk_levels['Extreme'] += 1
    elif 120 < val <= 240:
        if level == "A":
            risk_levels["Slight"] += 1
        elif level == "B":
            risk_levels["None"] += 1
        elif level == "C":
            risk_levels['Slight'] += 1
        elif level == "D":
            risk_levels['Great'] += 1
        elif level == "E":
            risk_levels['Extreme'] += 1
    elif 240 < val <= 360:
        if level == "A":
            risk_levels["Moderate"] += 1
        elif level == "B":
            risk_levels["Slight"] += 1
        elif level == "C":
            risk_levels["None"] += 1
        elif level == "D":
            risk_levels['Moderate'] += 1
        elif level == "E":
            risk_levels['Great'] += 1
    elif 360 < val <= 480:
        if level == "A":
            risk_levels["Great"] += 1
        elif level == "B":
            risk_levels["Moderate"] += 1
        elif level == "C":
            risk_levels["Slight"] += 1
        elif level == "D":
            risk_levels["None"] += 1
        elif level == "E":
            risk_levels['Moderate'] += 1
    elif 480 < val <= 600:
        if level == "A":
            risk_levels["Extreme"] += 1
        elif level == "B":
            risk_levels["Great"] += 1
        elif level == "C":
            risk_levels["Moderate"] += 1
        elif level == "D":
            risk_levels["Slight"] += 1
        elif level == "E":
            risk_levels["None"] += 1
user_choice = input('For predictions vs. ground truth plots, enter "1" for normalized and "2" for unnormalized: ')

for patient in patient_numbers:
    model = tensorflow.keras.models.load_model(f'{patient}_weights.keras')
    window = WindowGenerator(patient, input_width=6, label_width=1, shift=1,
                          label_columns=['Glucose Level'])
    patient_test_df = window.test_df
    unnorm_patient_test_df = patient_test_df * window.train_std.iloc[0]
    unnorm_patient_test_df += window.train_mean.iloc[0]
    
    plot_df = patient_test_df if user_choice == '1' else unnorm_patient_test_df
    
    test_set = tensorflow.data.Dataset.from_tensor_slices(patient_test_df.values).window(7, shift=1, drop_remainder=True).flat_map(lambda x: x.batch(7))
    test_set = test_set.map(lambda x: (x[:-1], x[-1])).batch(32).prefetch(1)
    preds = model.predict(test_set, verbose=0)
    unnorm_preds = preds * window.train_std.iloc[0]
    unnorm_preds += window.train_mean.iloc[0]
    plot_preds = preds if user_choice == '1' else unnorm_preds

    normalized_rmse = np.sqrt( np.average( (unnorm_preds - unnorm_patient_test_df.values[6:])**2  ))
    #print(normalized_rmse)
    model.compile(optimizer=model.optimizer, loss=model.loss, metrics=[tensorflow.keras.metrics.MeanAbsoluteError(),tensorflow.keras.metrics.RootMeanSquaredError()])
    eval = model.evaluate(test_set, verbose=0)
    #print(eval[2]*window.train_std.iloc[0])

    normalized_mae = np.abs( patient_test_df.values[6:] - np.array(preds) ).sum() / preds.shape[0]
    print(f'Normalized MAE for patient {patient} is : {normalized_mae}')
    print(f'Unnormalized MAE for patient {patient} is : {normalized_mae * window.train_std[0]}\n')

#------------------- Ground Truth vs. Predictions -------------------------#
    '''plt.plot(plot_df.values[6:], color='black', linestyle='--', label='Ground Truth')
    plt.plot(plot_preds.squeeze(), color='blue', label='Predictions')
    plt.title(f'Patient {patient}')
    plt.xlabel('Sample Number (5 Minutes/Sample)')
    plt.ylabel('Blood Glucose Value' if user_choice == '2' else 'Normalized Glucose Level')
    plt.legend()
    plt.show()'''

#------------------------------ SEG Plot ----------------------------------#
    '''fig, ax = plt.subplots()
    img = plt.imread('seg_canvas.jpg')
    ax.imshow(img, extent=[0,600,0,600])
    
    
    for i in range(min([len(unnorm_preds),len(unnorm_patient_test_df)])):
        plt.scatter(unnorm_patient_test_df.values[6+i], unnorm_preds[i], color='blue',s=1)
    plt.title(f'Surveillance Error Grid for Patient {patient}')
    plt.xlabel('Measured Blood Glucose Values (ml/dl)')
    plt.ylabel('Predicted Blood Glucose Values (ml/dl)')
    plt.show()'''

#--------------------------- SEG Calculations ------------------------------#
    '''data_points = len(unnorm_preds)
    risk_levels = {
    'None': 0,
    'Slight': 0,
    'Moderate': 0,
    'Great': 0,
    'Extreme': 0,
    }
    #seg_df = pd.DataFrame(np.concatenate(unnorm_preds,unnorm_patient_test_df[6:,:]),axis=1)
    for p, g in zip(unnorm_preds, np.array(unnorm_patient_test_df[6:])):
        if 0 <= p <= 120:
            ground_truth_seg_regions(g, "A", risk_levels)
        elif 120 < p <= 240:
            ground_truth_seg_regions(g, "B", risk_levels)
        elif 240 < p <= 360:
            ground_truth_seg_regions(g, "C", risk_levels)
        elif 360 < p <= 480:
            ground_truth_seg_regions(g, "D", risk_levels)
        elif 480 < p <= 600:
            ground_truth_seg_regions(g, "E", risk_levels)
    risk_levels_percentage = {k: (v / data_points) * 100 for k, v in risk_levels.items()}
    with open("seg_risk_levels_{}.csv".format(patient), 'w') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=risk_levels_percentage.keys())
        writer.writeheader()
        writer.writerow(risk_levels_percentage)'''

