import matplotlib
import os
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from sklearn.preprocessing import StandardScaler
from src.model_training2 import Hybrid, GraphEncoder
from src.data_processing import EDA, PreProcess
from src.merge_batch import get_preprocessed_dataset
from src.graph_utils.tfrecords.serialize import decode_fn
import pickle
import keras_tuner as kt
import joblib
import datetime

# Load the data
if True:
    cat_data = pd.read_excel('../small_data.xlsx')
else:
    cat_data = pd.read_excel('../cat_data.xlsx')  # REMEMBER TO CHOOSE FULL_PROCESSED_DATA.PKL AND FULL.TFRECORDS

# set booleans
eda_bool = False
serialize_data = False  # saves molecules2graphs as tfrecords in cwd, move to src.
# Change output_file_names accordingly in graph_preprocess.py line 223
train_bool = True
save_bool = False

if __name__ == "__main__":

    # see inside the data and bring to readable format
    if eda_bool:
        #  EDA
        eda = EDA(cat_data)
        alloys = eda.alloy_elements
        products = eda.products_df
        miller = eda.miller
        energy = eda.energy

        data = {
            'alloys': alloys,
            'miller': miller,
            'products': products,
            'energy': energy
        }
        with open('processed_data.pkl', 'wb') as f:
            pickle.dump(data, f)

    # load EDA data, preprocess it and train model
    if train_bool:
        with open('processed_data.pkl', 'rb') as f:
            loaded_data = pickle.load(f)
        alloys, miller, products, energy = (loaded_data['alloys'], loaded_data['miller'], loaded_data['products'],
                                            loaded_data['energy'])

        prepsd_data = PreProcess(alloys, miller, products, energy, serialize_data)

        print('TENSOR MANAGEMENT BEGINS')

        valid_mask = prepsd_data.mask_tensor
        prepsd_data.alloy_tensor = prepsd_data.alloy_tensor[valid_mask]
        prepsd_data.miller_tensor = prepsd_data.miller_tensor[valid_mask]
        prepsd_data.energy_tensor = prepsd_data.energy_tensor[valid_mask]
        """
        Number of graphs = Len of Mask Tensor, but because all True. NOW THIS MIGHT BE A PROBLEM WITH THIS HARDCORE
        SPLITTING OF MY TRAINING DATA --> at least it trained first run
        """
        # ADD relative path to config to load pickles and tfrecords
        train_path = os.path.join(os.getcwd(),  "..", "tfrecords", "graph_molecules_train_2F.tfrecord")
        val_path = os.path.join(os.getcwd(), "..", "tfrecords", "graph_molecules_val_2F.tfrecord")
        test_path = os.path.join(os.getcwd(), "..", "tfrecords", "graph_molecules_test_2F.tfrecord")

        train_graph = tf.data.TFRecordDataset([train_path]).map(decode_fn)
        batch_size=12
        train_graph_batched = train_graph.batch(batch_size)
        val_graph_batched = tf.data.TFRecordDataset([val_path]).map(decode_fn).batch(batch_size)
        test_graph_batched = tf.data.TFRecordDataset([test_path]).map(decode_fn).batch(batch_size)
        model_input_graph_spec = train_graph.element_spec

        def split_list(slicing_data):
            n = len(slicing_data)
            split1 = int(n * 0.70)  # First 70%
            split2 = int(n * 0.90)  # Next 20% (up to 90%)

            list1 = slicing_data[:split1]  # First 70%
            list2 = slicing_data[split1:split2]  # Next 20%
            list3 = slicing_data[split2:]  # Last 10%

            return list1, list2, list3


        # Only normalize target variable as other ones are in ranges from 0-1 (review graph)
        scaler = StandardScaler()  # this should be done after separating train/test
        #scaler = joblib.load('scaler.pkl')
        Y1_tr, Y1_val, Y1_test = split_list(prepsd_data.energy_tensor)

        Y1_tr = Y1_tr.reshape(-1, 1)  # Converts Series to 2D array
        Y1_tr = scaler.fit_transform(Y1_tr)
        #joblib.dump(scaler, 'scaler.pkl')
        #y_test = y_test.reshape(-1, 1)  # Converts Series to 2D array
        Y1_val = scaler.transform(Y1_val.reshape(-1, 1))
        Y1_test = scaler.transform(Y1_test.reshape(-1, 1))

        Y1_tr = tf.data.Dataset.from_tensor_slices(tf.convert_to_tensor(Y1_tr, dtype=tf.float32)).batch(9)
        Y1_val = tf.data.Dataset.from_tensor_slices(tf.convert_to_tensor(Y1_val, dtype=tf.float32)).batch(9)
        Y1_test = tf.data.Dataset.from_tensor_slices(tf.convert_to_tensor(Y1_test, dtype=tf.float32)).batch(9)

        X1_tr, X1_val, X1_test = split_list(prepsd_data.alloy_tensor)
        X2_tr, X2_val, X2_test = split_list(prepsd_data.miller_tensor)
        X3_tr = train_graph_batched
        X3_val = val_graph_batched
        X3_test = test_graph_batched
        #for i, element in enumerate(X3_tr):
        #    print(i, element)
        X1_tr = tf.data.Dataset.from_tensor_slices(tf.convert_to_tensor(X1_tr, dtype=tf.float32)).batch(batch_size)
        X1_val = tf.data.Dataset.from_tensor_slices(tf.convert_to_tensor(X1_val, dtype=tf.float32)).batch(batch_size)
        X1_test = tf.data.Dataset.from_tensor_slices(tf.convert_to_tensor(X1_test, dtype=tf.float32)).batch(batch_size)

        X2_tr = tf.data.Dataset.from_tensor_slices(tf.convert_to_tensor(X2_tr, dtype=tf.float32)).batch(batch_size)
        X2_val = tf.data.Dataset.from_tensor_slices(tf.convert_to_tensor(X2_val, dtype=tf.float32)).batch(batch_size)
        X2_test = tf.data.Dataset.from_tensor_slices(tf.convert_to_tensor(X2_test, dtype=tf.float32)).batch(batch_size)
        #  I want the Dataset format for supervised training
        #  then I need to keep (features, target) but unpack the 3 features, hence lambda
        #  finally batch the data for training
        dataset = tf.data.Dataset.zip(((X1_tr, X2_tr, X3_tr), Y1_tr)).map(lambda x, y: ((x[0], x[1], x[2]), y))

        # Ensure dataset is structured correctly
        #dataset = tf.data.Dataset.from_tensor_slices(
        #    ({"cat": X1_tr, "miller_indices": X2_tr, "graph_tensor": X3_tr}, Y1_tr)
        #)

        # Ensure model is built before training

        #this works but throws same error when training
        #dataset = tf.data.Dataset.zip(
        #    (({"cat": X1_tr}, {"miller_indices": X2_tr}, {"graph_tensor": X3_tr}), Y1_tr)).map(
        #    lambda x, y: ((x[0], x[1], x[2]), y))

        #dataset = tf.data.Dataset.zip(
        #    ({"cat": X1_tr, "miller_indices": X2_tr, "graph_tensor": X3_tr}, Y1_tr)
        #)

        g1 = dataset.take(1).get_single_element() #--> batched data
        g1 = dataset.take(1).get_single_element()[0] #--> first element of tuple, i.e., input features
        g1 = dataset.take(1).get_single_element()[0][2] # --> 3d feature, i.e., all graphs
        #for i, element in enumerate(g1):
        #    print(i, element)
        # g1.node_sets['atoms'
        #g1.edge_sets['bonds'].adjacency.source
        dataset_val = tf.data.Dataset.zip(
            ((X1_val, X2_val, X3_val), Y1_val)).map(lambda x, y: ((x[0], x[1], x[2]), y))

        dataset_test = tf.data.Dataset.zip(
            ((X1_test, X2_test, X3_test), Y1_test)).map(lambda x, y: ((x[0], x[1], x[2]), y))

        #  AS NUMPY ITERATOR TO INSPECT THIS
        # mas entendimiento sobre HP tunning, pq usar relu linear etc.
        # mejorar eficiencia gpu, tratar de usar cluster eventualmente
        tune = False
        old = False
        if tune:
            tuner = kt.Hyperband(
                                lambda hp: Hybrid.build_hp(hp, graph_spec=model_input_graph_spec),
                                 objective='val_mae',
                                 max_epochs=10,
                                 factor=3,
                                 directory='my_dir',
                                 project_name='intro_to_kt'
                                 )
            stop_early = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)

            tuner.search(dataset, validation_data=dataset_val, epochs=10,  callbacks=[stop_early])

            # Get the optimal hyperparameters
            best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]

            print(f"""
            The hyperparameter search is complete. The optimal number of units in the first densely-connected
            layer is {best_hps.get('units')} and the optimal learning rate for the optimizer
            is {best_hps.get('learning_rate')}.
            """)

            joblib.dump(best_hps, 'best_hps.pkl')

        elif old:
            gnn_model = GraphEncoder()  # graph_spec=model_input_graph_spec
            model = Hybrid.build(gnn_model=gnn_model,
                                 graph_spec=model_input_graph_spec,
                                 input_shape_x1=(prepsd_data.alloy_tensor.shape[1],),
                                 input_shape_x2=(prepsd_data.miller_tensor.shape[1],))
            model.compile(
                optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
                loss="mean_squared_error",
                metrics=["mae"]
            )

            model.summary()
            log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
            tensorboard = keras.callbacks.TensorBoard(log_dir=log_dir)
            history = model.fit(dataset, validation_data=dataset_val, epochs=15, callbacks=[tensorboard])
        else:
            # Instantiate the GNN model
            gnn_model = GraphEncoder()  # graph_spec=model_input_graph_spec
            #nn_model = NN(gnn_model)

            # Instantiate the hybrid model properly (no need for .build()) hybrid part instantiated within
            # Hybrid Class definition in model_training2.py
            model = Hybrid(
                gnn_model=gnn_model)

            # Compile with optimizer, loss, and metrics
            model.compile(
                optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
                loss_fn=tf.keras.losses.MeanSquaredError(),
                metrics=[tf.keras.metrics.MeanAbsoluteError()]
            )
            #model.build([(None, X1_tr.shape[1]), (None, X2_tr.shape[1]), X3_tr.element_spec])
            #model.summary()

            # Model summary
            #model.summary()

            # TensorBoard callback
            log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
            tensorboard = tf.keras.callbacks.TensorBoard(log_dir=log_dir)

            # Train the model
            history = model.fit(
                dataset,
                validation_data=dataset_val,
                epochs=15,
                callbacks=[tensorboard]
            )

        matplotlib.use('Agg')  # Non-GUI backend, pycharm is somehow not showing plots interactively

        for i in history.history:
            print(i)

        loss = history.history['loss']
        val_loss = history.history['val_loss']
        epochs = range(1, len(loss) + 1)
        plt.plot(epochs, loss, 'y', label='Training loss')
        plt.plot(epochs, val_loss, 'r', label='Validation loss')
        plt.title('Training and validation loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.savefig("/presentations/results/loss_val.png")

    if save_bool:

        model.save('cats_gnn_main.keras')
        joblib.dump(scaler, 'saved_scaler.pkl')
        joblib.dump(dataset_test, 'test_data_features.pkl')
        joblib.dump(Y1_test, 'test_data_targets.pkl')
