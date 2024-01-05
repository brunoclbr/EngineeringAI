import tensorflow as tf
import executePlan as ep
import pandas as pd
import numpy as np


directory = "Models/LBV_Model/"
model_name = directory + "model"  # trained model
expPlanFileStr = directory + 'parameters_actual' # parameters for the trained model
dataFileStr = 'InputData/Input_actual_no_HelpIdx' # dataset (required for the normalization)
predictDataStr = 'InputData/Input_predict' # data which should be predicted

def main():
    # load data
    expPlan, dataWhole = ep.load_data(dataFileStr, expPlanFileStr) # load training dataset for normalization
    predictionDataWhole = pd.read_csv(predictDataStr + '.csv', header=0)  # load from csv
    predictionDataWhole = predictionDataWhole.set_index('Name')

    # read input columns names for the plot
    rows = expPlan.itertuples(index=False, name=None)
    rows=list(rows)
    row=rows[0]
    fuelClasses = row[1]
    inputColumns = row[2]

    # cut the relevant data feature and the label column (RON/MON/ON)
    data, fuels_by_classes = ep.cut_data(dataWholeLocal=dataWhole, inputColumns=inputColumns, fuelClasses=fuelClasses)
    predictionData, fuels_by_classes = ep.cut_data(dataWholeLocal=predictionDataWhole, inputColumns=inputColumns, fuelClasses=fuelClasses)

    # cut out the y-values (RON/MON/OS)
    yName = list(data.columns.values)[-1]
    # read out fuel names
    fuel_names = predictionData.index.values

    is_DCN = False
    if yName == 'ln(t_ign)':
        is_DCN = True

    train_labels = data.pop(yName)
    measurements = predictionData.pop(yName)

    # normalize input data, as the trained model expects normalized input
    x_std = data.std(axis=0)
    x_mean = data.mean(axis=0)

    predictionData_normed = ( predictionData - x_mean ) / x_std

    # labels dont have to be normalized but their std and mean is necessary to denorm the model output
    label_std = train_labels.std(axis=0)
    label_mean = train_labels.mean(axis =0)

    # load trained model
    loaded_model = tf.keras.models.load_model(model_name, custom_objects=None, compile=True)

    prediction_normed = loaded_model.predict(predictionData_normed)
    prediction = (prediction_normed * label_std + label_mean).flatten()


    if is_DCN:
        prediction = 4.460 + 186.6/np.exp(prediction)
        measurements = 4.460 + 186.6/np.exp(measurements)
        prediction_error = prediction - measurements
        metric = 'DCN'
    else:
        prediction_error = prediction - measurements
        metric = yName

    mae = np.mean(abs(prediction_error))
    rmse = np.sqrt((np.square(prediction_error)).mean(axis=0))

    print('MAE in {} space: {}'.format(metric, mae))
    print('RMSE in {} space: {}'.format(metric, rmse))

    prediction_df = pd.DataFrame(prediction, columns= ['Prediction'], index = fuel_names)

    prediction_error_df = pd.DataFrame(np.asarray(prediction_error), columns=['Error'], index = fuel_names)
    #prediction_error_df = prediction_error.rename(columns = ['Error'])

    save_frame = pd.concat([predictionDataWhole, prediction_df, prediction_error_df], sort = False, axis = 1)
    print(save_frame)

    save_frame.to_csv(directory + 'test_set_prediction.csv')
    print('Results saved.')
if __name__ == "__main__":
    main()