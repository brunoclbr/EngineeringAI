import tensorflow as tf
import tensorflow_gnn as tfgnn
from tensorflow.keras import Model
from tensorflow.keras.layers import Input, Layer, Dense, Concatenate, Dropout
from tensorflow_gnn.models.gcn import gcn_conv
from tensorflow_gnn.models import graph_sage
from tensorflow_gnn.keras.layers import MapFeatures

# --- Custom Layer Definitions ---

class EdgeInitLayer(Layer):
    def __init__(self, units):
        super().__init__()
        self.units = units
        self.edge_dense = Dense(units, activation="relu", name="edge_init_dense")


    def call(self, inputs):
        return self.edge_dense(inputs)


class NodeInitLayer(Layer):
    def __init__(self, units):
        super().__init__()
        self.node_dense = Dense(units, activation="relu", name="node_init_dense")

    def call(self, inputs):
        return self.node_dense(inputs)


# --- Graph Encoder (GNN) ---

class GraphEncoder(Model):
    def __init__(self, units=27, activation="relu", node_dim=16, edge_dim=4,
                 num_message_passing=2, message_dim=16, next_state_dim=16):
        super().__init__()
        self.units = units
        self.node_dense = NodeInitLayer(16)
        self.edge_dense = EdgeInitLayer(16)
        self.num_message_passing = num_message_passing

        self.mapped_graph = MapFeatures(
            node_sets_fn=self.set_initial_node_state,
            edge_sets_fn=self.set_initial_edge_state
        )

        #self.gcn = gcn_conv.GCNConv(units=message_dim, name="gcn")
        self.gcn = graph_sage.GraphSAGEGraphUpdate(
        node_set_names=["atoms"],
        receiver_tag=tfgnn.TARGET,
        reduce_type="sum",
        use_pooling=True,
        use_bias=False,
        dropout_rate=0.01,
        units=message_dim,
        hidden_units=message_dim * 2,
        l2_normalize=True,
        combine_type="sum",
        activation="relu")
        self.readout_features = tfgnn.keras.layers.Pool(tfgnn.CONTEXT, "sum", node_set_name="atoms")
        self.graphencoder_final_layer = Dense(self.units, activation="linear", name="entropy")

    def set_initial_node_state(self, node_set, *, node_set_name):
        return self.node_dense(tf.keras.layers.concatenate([v for _, v in sorted(node_set.features.items())]))

    def set_initial_edge_state(self, edge_set, *, edge_set_name):
        #print(edge_set['features']['adjacency'])
        return {} #self.edge_dense(edge_set[tfgnn.HIDDEN_STATE])
                #(tf.keras.layers.concatenate([v for _, v in sorted(edge_set.features.items())])))

    def call(self, graph_tensor, training=False, mask=None):

        print("Before GNN model processing:")
        print("GraphTensor Type:", type(graph_tensor))
        print("GraphTensor Spec:", graph_tensor.spec)

        # Debug individual edge sets
        print("Adjacency source indices:", graph_tensor.edge_sets["bonds"].adjacency.source)
        print("Adjacency target indices:", graph_tensor.edge_sets["bonds"].adjacency.target)

        # Debug individual node sets
        for node_set_name, node_set in graph_tensor.node_sets.items():
            print(f"Node set {node_set_name}: Features shape: {node_set.features}")

        print("Merging batch to components...")

        graph = graph_tensor.merge_batch_to_components()
       # graph, mask = tfgnn.keras.layers.PadToTotalSizes(10)(graph)
        graph = self.mapped_graph(graph)

        for _ in range(self.num_message_passing):
            graph = self.gcn(graph)

        readout_features = self.readout_features(graph)

        output = self.graphencoder_final_layer(readout_features)

        if output is None:  # Debugging step
            raise ValueError("GraphEncoder call() is returning None!")

        return output


# --- NN Model ---
class NN(Model):
    def __init__(self, gnn_model):
        super().__init__()

        self.gnn_model = gnn_model
        # Fully connected layers for external features
        self.alloys_dense = Dense(183, activation="relu", name="cat_dense")
        self.miller_dense = Dense(78, activation="relu", name="miller_indices_dense")

        # Fully connected layers after concatenation
        self.concat_layer = Concatenate()
        self.dense_1 = Dense(1064, activation="relu", name="after_concat_dense1")
        self.dropout_1 = Dropout(0.5, name="dp1")
        self.dense_2 = Dense(563, activation="relu", name="dense_2")
        self.dropout_2 = Dropout(0.5, name="dp2")
        self.dense_3 = Dense(274, activation="relu", name="dense_3")
        self.dropout_3 = Dropout(0.5, name="dp3")
        self.output_layer = Dense(1, activation="linear", name="output")

    def call(self, inputs, training=False, mask=None):
        input_cat, input_mill, graph_input = inputs
        # Input layers
        print("SHAPE OF 1st INPUT WHEN CALL NN", input_cat.shape)
        #input_cat = Input(shape=input_cat.shape[1], name="cat")
        #input_mill = Input(shape=input_mill.shape[1], name="miller_indices")
        #graph_input = Input(type_spec=graph_input.spec, name="graph_tensor")

        # Pass through respective layers
        cat_features = self.alloys_dense(input_cat)
        miller_features = self.miller_dense(input_mill)
        print("now going into gnn model")
        gnn_output = self.gnn_model(graph_input)
        print("going into concatenated network")

        # Concatenation and fully connected layers
        concatenated_features = self.concat_layer([cat_features, miller_features, gnn_output])
        x = self.dense_1(concatenated_features)
        x = self.dropout_1(x, training=training)
        x = self.dense_2(x)
        x = self.dropout_2(x, training=training)
        x = self.dense_3(x)
        x = self.dropout_3(x, training=training)

        return self.output_layer(x)


# --- Hybrid Model ---

class Hybrid(Model):
    def __init__(self, gnn_model):
        super().__init__()
        self.gnn_model = gnn_model
        # create instance of class NN which requires one positional argument "gnn_model"
        self.nn_model = NN(self.gnn_model)

        #self.nn_model = nn_model
        #self.graph_spec = graph_spec

        # GNN processing
        #self.batched_graph = self.graph_input.merge_batch_to_components()
        #self.gnn_output = self.gnn_model(self.graph_spec)

    def compile(self, optimizer, loss_fn, metrics):
        super().compile()
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.metrics_list = metrics

    def train_step(self, data):
        inputs, labels = data

        with tf.GradientTape() as tape:  # tape is grad(loss, W) which is a tensor (derivative aka gradient)
            """Run the fp under GradientTape scope. A Python scope will record the tensor operations that run
            inside it, in the form of a computation graph"""
            # gnn model
            # nn model
            # hybrid model
            # here im calling nn_model object
            predictions = self.nn_model(inputs, training=True)  # object call requires inputs
            loss = self.loss_fn(labels, predictions)

        # Compute gradients only for trainable variables
        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)  # use tape to retrieve the gradient of
        # loss w.r.t trainable vars

        # Ensure gradients are not None (debugging step)
        print("Gradients Computed:")
        for var, grad in zip(self.trainable_variables, gradients):
            if grad is None:
                print(f"No gradient for: {var.name}")
            else:
                print(f"Gradient found for: {var.name}, Shape: {grad.shape}")

        # Apply gradients. This is w1 = w0 -alpha*gradient
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))

        # Update metrics
        for metric in self.metrics_list:
            metric.update_state(labels, predictions)

        return {m.name: m.result() for m in self.metrics_list}

