import tensorflow as tf
import tensorflow_gnn as tfgnn
from tensorflow.keras import Model
from tensorflow.keras.layers import Input, Layer, Dense, Concatenate, Dropout
from tensorflow_gnn.models import gcn
from tensorflow_gnn.models import graph_sage
from tensorflow_gnn.keras.layers import MapFeatures
# from tfrecords.serialize import train_ds, val_ds
# from mutag import train_ds, val_ds


class EdgeInitLayer(Layer):
    def __init__(self, units):
        super().__init__()
        # here i want to define the usage of weights within the layer so that tf keeps track of them
        # set initial state
        # Define layers here so TensorFlow tracks them

        self.edge_dense = Dense(units, activation="relu", name="edge_init_dense")

    def call(self, inputs):
        # "forward pass"
        return self.edge_dense(inputs)


class NodeInitLayer(Layer):
    def __init__(self, units):
        super().__init__()
        self.node_dense = Dense(units, activation="relu", name="node_init_dense")

    def call(self, inputs):
        # "forward pass"
        return self.node_dense(inputs)


class GraphEncoder(tf.keras.Model):

    """
    Custom subclassed model that processes the GraphTensor separately and returns a latent vector. By subclassing the
    Model class, you should define your layers in __init__() and you should implement the model's forward pass in call()
    """
    def __init__(self, units=27, activation="relu", node_dim=60, edge_dim=4,
                 num_message_passing=1, message_dim=60, next_state_dim=46):
        super().__init__()
        self.units = units
        self.node_dim = node_dim
        self.edge_dim = edge_dim
        self.node_dense = NodeInitLayer(16)
        self.edge_dense = EdgeInitLayer(16)
        self.num_message_passing = num_message_passing

        def set_initial_node_state(node_set, *, node_set_name):
            features = node_set.features
            return self.node_dense(
                tf.keras.layers.concatenate([v for _, v in sorted(features.items())]))

        def set_initial_edge_state(edge_set, *, edge_set_name):
            features = edge_set.features
            return self.edge_dense(tf.keras.layers.concatenate([v for _, v in sorted(features.items())]))

        self.mapped_graph = MapFeatures(node_sets_fn=set_initial_node_state, edge_sets_fn=set_initial_edge_state)

        self.gcn = gcn.GCNHomGraphUpdate(units=message_dim, name="gcn" + "_layer_")

        self.readout_features = tfgnn.keras.layers.Pool(
            tfgnn.CONTEXT, "sum", node_set_name="atoms")

        self.graphencoder_final_layer = tf.keras.layers.Dense(self.units, activation="linear", name="entropy")

    # CALL METHOD
    def call(self, graph_tensor, training=False, mask=None):  # WHY DOES KWARGS FIX THE WARNING?
        #batched_graph = graph_tensor.merge_batch_to_components()

        graph = self.mapped_graph(graph_tensor)

        # NodeSetUpdate updates hidden state of the atoms
        for message_i in range(self.num_message_passing):
            graph = self.gcn(graph)

        readout_features = self.readout_features(graph)
        return self.graphencoder_final_layer(readout_features)


class GnnHybrid:
    @staticmethod
    def build(gnn_model, graph_spec, input_shape_x1, input_shape_x2):

        input_cat = Input(shape=input_shape_x1, name="cat")
        input_mill = Input(shape=input_shape_x2, name="miller_indices")
        #graph_input = tfgnn.keras.layers.GraphTensorInput(name="graph_input")
        graph_input = Input(type_spec=graph_spec, name="graph_tensor")
        batched_graph = graph_input.merge_batch_to_components()
        # Define GNN submodel
        print("does this even work x2?")
        gnn_output = gnn_model(batched_graph)
        print("does this even work x3?")
        # Fully connected layers for external featuresmodel_training.py
        alloys_dense = Dense(183, activation="relu", name="cat_dense")(input_cat)
        miller_dense = Dense(78, activation="relu", name="miller_indices_dense")(input_mill)

        # Concatenate and pass through dense layers
        concatenated_features = Concatenate()([alloys_dense, miller_dense, gnn_output])
        dense_1 = Dense(1064, activation="relu", name="after_concat_dense1")(concatenated_features)
        dense_1dp = Dropout(0.5, name="dp1")(dense_1)
        dense_2 = Dense(563, activation="relu", name="dense_2")(dense_1dp)
        dense_2dp = Dropout(0.5, name="dp2")(dense_2)
        dense_3 = Dense(274, activation="relu", name="dense_3")(dense_2dp)
        dense_3dp = Dropout(0.5, name="dp3")(dense_3)
        output = Dense(1, activation="linear", name="output")(dense_3dp)

        # Create and compile the model
        # Think if your activation layers are the best
        hyb_model = Model(
            inputs=[input_cat, input_mill, graph_input],
            outputs=output,
        )

        return hyb_model


if __name__ == "__main__":

    pass
