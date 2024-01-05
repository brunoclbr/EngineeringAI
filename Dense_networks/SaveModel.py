import tensorflow as tf
import executePlan as ep

path = 'Models/LBV_Model' # path for model folder, contains the parameters.csv file like in expParameters.csv
#only 1 row in parameters.csv allowed
#optimal Epochs has to be determined with cross-validation (executePlan.py) before and typed in "maxEpoch"-column
expPlanFileStr = path + '/parameters_actual'  # define inputs and hyperparameters
dataFileStr = 'InputData/Input_actual_no_HelpIdx' # define Input File

#cut# seedTrainTest = 1859  # seed to split between train and test set
#cut# train_fraction = 1  # =1 when using k folds
savePredictionsBool = False # True to save the detailed predictions

expPlan = []
dataWhole = []
yName= 'LBV'

def calculate_model(row, dataWhole):
    global yName

    expStr = row[0]
    fuelClasses = row[1]
    inputFeatures = row[2]
    testSeedList = row[3]
    activationStr = row[4]
    numberKF = row[5]
    n1 = row[6]
    n2 = row[7]
    n3 = row[8]
    d1 = row[9]
    d2 = row[10]
    d3 = row[11]
    alpha = row[12]
    beta1 = row[13]
    beta2 = row[14]
    maxEpoch = row[15]
    adjustment = row[16]  # indicates the feature adjustment made for this row
    iterationStep = row[17]

    numberOfFeatures = len(inputFeatures.split())
    optEpoch = maxEpoch

    # sets neuron activation according to the exp plan, defaults to sigmoid
    if activationStr == 'relu':
        activation = tf.nn.relu
    elif activationStr == 'tanh':
        activation = tf.nn.tanh
    elif activationStr == 'leaky':
        activation = tf.nn.leaky_relu
    else:
        activation = tf.nn.sigmoid

    modelHyperparameters = (n1, n2, n3, d1, d2, d3, activation, alpha, beta1, beta2)

    # cut relevant data from raw data, save the used input Columns to name the output file
    data, fuels_by_classes = ep.cut_data(dataWholeLocal=dataWhole, inputColumns=inputFeatures, fuelClasses=fuelClasses)
    yName = list(data.columns.values)[-1]

    # arrange data and split into train and test set
    (train_data, train_labels), (test_data, test_labels) = ep.arrange_data(seed=1, data=data,
                                                                        train_fraction_local=1,
                                                                        yName=yName)

    # normalize data
    train_data, train_labels, test_data, test_labels, label_mean, label_std = ep.normalizeDataStd(train_data, train_labels,
                                                                                               test_data, test_labels)



    model = ep.buildModel(modelHyperparameters, train_data.shape[1])
    model.fit(train_data, train_labels, epochs=optEpoch, verbose=0)

    return (model, expStr)

def start_main(argv):
    assert len(argv) == 1

    main(expPlanFileStr, dataFileStr)

def save_model(model, model_name):

    filepath = path + '/' + 'model'
    tf.keras.models.save_model(model, filepath, overwrite=True, include_optimizer=False)



def main(expPlanFileStr, dataFileStr):
    global yName
    model_list = []
    model_name_list=[]

    # Load dataset and exp Plan from csv
    expPlan, dataWhole = ep.load_data(dataFileStr, expPlanFileStr)

    # read rows from csv data
    rows = expPlan.itertuples(index=False, name=None)
    rows=list(rows)

    for row in rows:
        (model, model_name) = calculate_model(row, dataWhole)
        model_list.append(model)
        model_name_list.append(model_name)

    i = 0
    for model in model_list:
        model_name = model_name_list[i]
        save_model(model, model_name)
        i+=1

if __name__ == "__main__":
    # The Estimator periodically generates "INFO" logs; make these logs visible.
    tf.logging.set_verbosity(tf.logging.ERROR)
    tf.app.run(main=start_main)