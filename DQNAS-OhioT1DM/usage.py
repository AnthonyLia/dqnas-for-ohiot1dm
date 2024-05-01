import tensorflow
import tensorflow.keras
import numpy as np
from time_series2 import *
import warnings
warnings.filterwarnings('ignore')

unnorm_patient_test_df = np.array([])
unnorm_preds = np.array([])
patient_numbers = [540,544,552,559,563,567,570,575,584,588,591,596]

for patient in patient_numbers:
    model = tensorflow.keras.models.load_model(f'{patient}_weights.keras')

    window = WindowGenerator(patient, input_width=6, label_width=1, shift=1,
                          label_columns=['Glucose Level'])
    patient_test_df = window.test_df
    test_set = tensorflow.data.Dataset.from_tensor_slices(patient_test_df.values).window(7, shift=1, drop_remainder=True).flat_map(lambda x: x.batch(7))
    test_set = test_set.map(lambda x: (x[:-1], x[-1])).batch(32).prefetch(1)
    preds = model.predict(test_set, verbose=0)

    normalized_mae = np.abs( patient_test_df.values[6:] - np.array(preds) ).sum() / preds.shape[0]
    print(f'Normalized MAE for patient {patient} is : {normalized_mae}')
    print(f'Unnormalized MAE for patient {patient} is : {normalized_mae * window.train_std[0]}\n')

    plt.plot(patient_test_df.values[6:], color='black', linestyle='--', label='Ground Truth')
    plt.plot(preds.squeeze(), color='blue', label='Predictions')
    plt.title(f'Patient {patient}')
    plt.legend()
    plt.show()


    fig, ax = plt.subplots()
    img = plt.imread('seg_canvas.jpg')
    ax.imshow(img, extent=[0,600,0,600])
    unnorm_patient_test_df = patient_test_df * window.train_std.iloc[0]
    unnorm_patient_test_df += window.train_mean.iloc[0]
    unnorm_preds = preds * window.train_std.iloc[0]
    unnorm_preds += window.train_mean.iloc[0]
    
    for i in range(min([len(unnorm_preds),len(unnorm_patient_test_df)])):
        plt.scatter(unnorm_patient_test_df.values[i], unnorm_preds[i], color='blue',s=1)
    plt.title(f'Surveillance Error Grid for Patient {patient}')
    plt.xlabel('Measured Blood Glucose Values (ml/dl)')
    plt.ylabel('Predicted Blood Glucose Values (ml/dl)')
    plt.show()
