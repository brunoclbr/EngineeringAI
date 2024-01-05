
import tensorflow as tf
import executePlan as ep
import pandas as pd
import numpy as np


directory = 'Models/LBV_Model/'
model_name = directory + "model"  # trained model
dataFileStr = 'InputData/Input_actual_no_HelpIdx'  # dataset used in SaveModel (also used for normalization)
expPlanFileStr = directory + 'parameters_actual'  # parameters for the trained model
sensitivityDataStr = 'InputData/Input_sensitivities' # New, Same data as Input_predict
save_sens = True  # save the outputs as csv


def main():
    # load data
    expPlan, dataWhole = ep.load_data(dataFileStr, expPlanFileStr)
    # NEW, read Input_predict, not Input used in SaveModel
    sensitivityDataWhole = pd.read_csv(sensitivityDataStr + '.csv', header=0)  # load from csv
    sensitivityDataWhole = sensitivityDataWhole.set_index('Name')

    # read input columns names for the plot
    rows = expPlan.itertuples(index=False, name=None)
    rows=list(rows)
    row=rows[0]
    fuelClasses = row[1]
    inputFeatures = row[2] #This is called InputColumns in predict.py

    # cut the relevant data feature and the label column (RON/MON/ON)
    data, fuels_by_classes = ep.cut_data(dataWholeLocal=dataWhole, inputColumns=inputFeatures, fuelClasses=fuelClasses)

    # NEW, cut relevant data
    sensitivityData, fuels_by_classes = ep.cut_data(dataWholeLocal=sensitivityDataWhole, inputColumns=inputFeatures, fuelClasses=fuelClasses)



    # cut out the y-values (RON/MON/OS)
    yName = list(data.columns.values)[-1] # y-Data from Original Input
    train_labels = data.pop(yName) # y-Data from Original Input, i.e., 6500 LBVs
    # read out fuel names
    fuel_names = sensitivityData.index.values # Fuels in new Input
    sensitivityData.pop(yName) # y-Data must be droped out from sensitivityData, just like in predict.py


    # normalize input data, as the trained model expects normalized input
    # NORMALIZE WITH ORIGINAL TRAIN DATA, i.e., Input used in SaveModel
    x_mean = data.mean(axis=0)
    x_std = data.std(axis=0)
 
    sensitivity = (sensitivityData - x_mean) / x_std # Data for Sensitivity Analysis (Input_predict) must be normalized with ORIGINAL TRAIN DATA
    # in Original local_sensitivities the variable was called "train_data", important for row 84


    # labels dont have to be normalized but their std and mean is necessary to denorm the model output
    # Normalized with original data
    label_std = train_labels.std(axis=0)
    label_mean = train_labels.mean(axis =0)


    # load trained model
    loaded_model = tf.keras.models.load_model(model_name, custom_objects=None, compile=True)

    # features_str is a list of strings naming the features
    features_str = list(sensitivity.columns.values)


    """
    Local Sensitivity: 
    Fuel by Fuel, Feature by Feature. [Fuel number, Feature Index, sensitivity distance ]
    A single feature value gets altered (+/-) by a multiple of the standard deviation.
    Effect on the model is tested and from that the normed sensitivity is calculated.
    
    """
    # sensitivities are calculated from 2 points +- 0.2 std away from the real fuel
    sensitivity_distance_list = [-1,1]

    # storage arrays for the model predictions and the calculated sensitivities
    # here also instead of train_data --> sensitivity
    prediction_results = np.zeros((sensitivity.shape[0] ,sensitivity.shape[1], len(sensitivity_distance_list)),dtype=float)
    sensitivity_results = np.zeros((sensitivity.shape[0] ,sensitivity.shape[1]),dtype=float)

    # Iteration over every fuel
    #Here every "sensitivity" variable was train_data in original local_sensitivities
    for fuel_idx in range(sensitivity.shape[0]): 

        feature_row = sensitivity.iloc[[fuel_idx]]

        # Iteration over every feature
        for ft_idx in range(sensitivity.shape[1]):

            # Iteration over the (2) points used to calculated sensitivity
            for dist_idx in range(len(sensitivity_distance_list)):

                input = feature_row.as_matrix().copy()

                input[0,ft_idx] = input[0,ft_idx] + sensitivity_distance_list[dist_idx]
                prediction_results[fuel_idx,ft_idx,dist_idx] = loaded_model.predict(input)


    # denorm model output (RON/MON..)
    prediction_results = prediction_results * label_std +label_mean
    

    # calculate sensitivities
    # iterating over fuels
    for fuel_idx in range(sensitivity.shape[0]):
        #iteration over features
        for ft_idx in range(len(x_mean)): 
            sensitivity_results[fuel_idx,ft_idx] = np.diff(prediction_results[fuel_idx,ft_idx,:])/(np.diff(sensitivity_distance_list)*x_std[ft_idx])

#            sensitivity_results[fuel_idx,ft_idx] = np.diff(prediction_results[fuel_idx,ft_idx,:])/(np.diff(sensitivity_distance_list))


    sensitivity_results_df = pd.DataFrame(sensitivity_results,
                                   columns=[fn + "_Sens" for fn in features_str],
                                   index=fuel_names)

    save_frame = pd.concat([sensitivityDataWhole,sensitivity_results_df], axis=1, sort=False)

    save_frame = save_frame.reindex(index=sensitivityDataWhole.index)
    save_frame.to_csv(directory + 'Sensitivity_1std.csv')



    print('Saved Sensitivity in {}Sensitivity.csv'.format(directory))

if __name__ == "__main__":
    main()