import tensorflow as tf
import tensorflow_gnn as tfgnn
from tensorflow_gnn.models import gcn
from tensorflow_gnn.models import graph_sage
from tensorflow.keras.layers import Input, Dense, Lambda
from tensorflow_gnn.keras.layers import MapFeatures, Pool
from matplotlib import pyplot as plt
from tfrecords.serialize import train_ds, val_ds
#from mutag import train_ds, val_ds
from tensorflow_gnn.models.gcn.gcn_conv import GCNConv


print("TensorFlow version:", tf.__version__)
print("TensorFlow GNN version:", tfgnn.__version__)


class GCNLayer(tf.keras.layers.Layer):
    """Custom GCN Layer wrapping around TensorFlow GNN's GCNConv."""

    def __init__(self, out_dim):
        super().__init__()
        self.gcn = GCNConv(out_dim)

    def call(self, graph):
        updated_node_features = self.gcn(graph, edge_set_name="bonds")
        #tf.print("Updated node features shape:", tf.shape(updated_node_features))
        #tf.print("Graph node set sizes:", graph.node_sets["atoms"].sizes)
        return graph.replace_features(
            node_sets={"atoms": {"hidden_state": updated_node_features}}
        )


# Create the GNN model
def create_gnn_model(graph_spec: tfgnn.GraphTensorSpec,
                     node_dim=10,
                     edge_dim=16,
                     message_dim=32,
                     next_state_dim=46,
                     num_message_passing=2,
                     l2_regularization=5e-4,
                     dropout_rate=0.5, ) -> tf.keras.Model:
    graph_convolution_type = "graph_sage"  # read docu of other gcn models
    input_graph = Input(type_spec=graph_spec)

    # IMPORTANT: All TF-GNN modeling code assumes a GraphTensor of shape [] (i.e. scalar graph)

    """
    For example, consider reading three GraphTensors from disk with 4, 5 and 6 nodes, resp., in the NodeSet "docs". 
    Parsing them as a batch of size 3 creates a GraphTensor of shape [3] with graph.node_sets["docs"].sizes equal to 
    [[4], [5], [6]]. The edges in each graph refer to node indices 0,...,3; 0,...,4; and 0,...,5, respectively. 
    Likewise, node features have a shape [3, (node_indices), ...] where (node_indices) is a RAGGED DIMENSION. 
    The result of graph.merge_batch_to_components() is a new GraphTensor with shape [], node set sizes [4, 5, 6],
     node indices 0,...,14, and feature shape [15, ...], with nodes concatenated in order.
    """
    batched_graph = input_graph.merge_batch_to_components()
    tf.print("NodeFeature shape after merge_batch:", batched_graph.node_sets["atoms"][tfgnn.HIDDEN_STATE].shape)

    # Nodes and edges (features) 2 trainable embedding table.
    # Apply linear projection of hidden state tensors
    def set_initial_node_state(node_set, dense_dim=node_dim, *, node_set_name):
        # Since we only have one node set, we can ignore node_set_name.
        return tf.keras.layers.Dense(dense_dim)(node_set[tfgnn.HIDDEN_STATE])  # units should be variable,
        # "atomic_number" before tfgnn.HIDDEN_STATE, same below

    def set_initial_edge_state(edge_set, dense_dim=edge_dim, *, edge_set_name):
        if graph_convolution_type != "default":
            return {}
        return tf.keras.layers.Dense(dense_dim)(edge_set[tfgnn.HIDDEN_STATE])

    # Map input features (atomic numbers as node features)
    current_graph = MapFeatures(node_sets_fn=set_initial_node_state, edge_sets_fn=set_initial_edge_state)(batched_graph)

    # Add a context feature which will be the target property, and we want to add a context
    # to the original (batched) graph (i.e. molecule[s])

    # MY GUESS IS THAT THE CONSTANT 2 DIMENSIONS OF THE CONTEXT ARE FIXED HERE INSTEAD OF VARY WITH BATCH SIZE
    num_atoms = tf.expand_dims(tf.cast(current_graph.node_sets["atoms"].sizes, dtype=tf.float32), axis=-1)
    num_bonds = tf.expand_dims(tf.cast(current_graph.edge_sets["bonds"].sizes, dtype=tf.float32), axis=-1)
    tf.print("Num atoms shape:", num_atoms.shape)
    tf.print("Num bonds shape:", num_bonds.shape)

    current_graph = current_graph.replace_features(
        context={
            tfgnn.HIDDEN_STATE: tf.concat([num_atoms, num_bonds], axis=1)
        })
    tf.print("Context Shape after MapFeatures & context addition:",
             current_graph.context[tfgnn.HIDDEN_STATE].shape)

    # Reshape the node features for all node sets in the graph

    def dense(units, activation="relu"):
        """A Dense layer with regularization (L2 and Dropout)."""
        regularizer = tf.keras.regularizers.l2(l2_regularization)
        return tf.keras.Sequential([
            tf.keras.layers.Dense(
                units,
                activation=activation,
                kernel_regularizer=regularizer,
                bias_regularizer=regularizer),
            tf.keras.layers.Dropout(dropout_rate)
        ])

    # min 17.55 something about hyperparameter tuning: https://www.youtube.com/watch?v=b8cA4Vjh9D0
    # NodeSetUpdate updates hidden state of the atoms
    for message_i in range(num_message_passing):
        if graph_convolution_type == "default":
            current_graph = tfgnn.keras.layers.GraphUpdate(
                node_sets={
                    "atoms": tfgnn.keras.layers.NodeSetUpdate(
                        edge_set_inputs={"bonds": tfgnn.keras.layers.SimpleConv(  # these are edge_set convolutions
                            message_fn=dense(message_dim),  #sender_edge_feature=tfgnn.HIDDEN_STATE,
                            reduce_type="sum",
                            receiver_tag=tfgnn.TARGET)},
                        next_state=tfgnn.keras.layers.NextStateFromConcat(dense(next_state_dim)))},
            )(current_graph)
        elif graph_convolution_type == "gcn":
            current_graph = gcn.GCNHomGraphUpdate(
                units=message_dim,
                name=graph_convolution_type + "_layer_" + str(message_i))(current_graph)
        elif graph_convolution_type == "graph_sage":
            print(message_i, tfgnn.TARGET)
            current_graph = graph_sage.GraphSAGEGraphUpdate(
                node_set_names=["atoms"],
                receiver_tag=tfgnn.TARGET,
                reduce_type="sum",
                use_pooling=True,
                use_bias=False,
                dropout_rate=dropout_rate,
                units=message_dim,
                hidden_units=message_dim * 2,
                l2_normalize=True,
                combine_type="sum",
                activation="relu",
                name="gsage_layer_" + str(message_i))(current_graph)
        elif graph_convolution_type == "custom":
            print(message_i, current_graph)
            tf.print(f"Iteration {message_i}: node features shape:",
                     current_graph.node_sets["atoms"][tfgnn.HIDDEN_STATE].shape)
            current_graph = GCNLayer(32)(current_graph)
            #print('Applying GCN Layer 1')
            #current_graph = GCNLayer(64)(current_graph)

            # Dense layers to predict entropy
            #current_graph = Dense(16, activation="relu")(graph_embedding)
            #entropy_output = Dense(1, activation="linear", name="entropy")(dense_output)
            print("OUT OF CONVOLUTIONS")

    """
    After the GNN has computed a context-aware representation of the "atoms", the model reads out a representation 
    for the graph as a whole by averaging (pooling) node states into the graph context (Pool2Context). 
    The context is global to each input graph of the batch, so the first dimension of the result corresponds to 
    the batch dimension of the inputs (same as the labels).
    """
    # Context  has a hidden-state feature, concatenate the aggregated node vectors with the hidden-state to get
    # the final vector
    #tf.print("Nodes Shape after GCN:",
    #         current_graph.node_sets["atoms"][tfgnn.HIDDEN_STATE].shape)

    readout_features = tfgnn.keras.layers.Pool(
        tfgnn.CONTEXT, "mean", node_set_name="atoms")(current_graph)

    #tf.print("Readout_features Shape after Pool2Context:",
    #         readout_features.shape)

    feat = tf.concat([readout_features, current_graph.context[tfgnn.HIDDEN_STATE]], axis=1)

    # I BELIEVE EITHER HERE OR BEFORE IS THE OUTPUT I WANT TO CONCATENATE TO MY OTHER NNs
    # ACTUALLY GASGE_LAYER_1 SEEMS TO BE LAST LAYER WITH TRAINABLE PARAMETERS AND HENCE THATS THE CONCATANATION LAYER
    # FOR MY MODEL. OR MAYBE TRY WITH READ OUT FEATURES THAT SHOULD BE POOLING AFTER DENSE LAYER.
    # STILL REMEMBER TO THINK IN BATCHES, SO IF 2 GRAPHS --> TO CONTEXTS. INTUITION SHOULD BE EXPORT FEAT FOR CONCAT
    # SO FUCK WHAT I SAID BEFORE.

    #tf.print("Feat concat current_graph:", feat.shape)

    output = tf.keras.layers.Dense(1, activation="relu", name="entropy")(feat)

    return tf.keras.Model(inputs=[input_graph], outputs=[output])


if __name__ == "__main__":

    # Print the preprocessed data
    #g, y = train_ds.take(1).get_single_element()
    #print(g.node_sets['atoms'])
    #print(g.node_sets['atoms'].features[tfgnn.HIDDEN_STATE])

    #look at one example
    #g, y = train_ds.take(1).get_single_element()

    # BUILD THE MODEL
    print('Building GNN Model')

    batch_size = 1
    train_ds_batched = train_ds.batch(batch_size=batch_size).repeat()
    for i, batch in enumerate(train_ds_batched):
        print(f"Batch {i + 1}:")
        graph_tensor, labels = batch
        print("GraphTensor:")
        print(graph_tensor)
        print("Node sets:", graph_tensor.node_sets)
        print("Edge sets:", graph_tensor.edge_sets)
        print("Labels:", labels)
        break

    val_ds_batched = train_ds.batch(batch_size=batch_size)
    model_input_graph_spec, label_spec = train_ds.element_spec
    gnn_model = create_gnn_model(model_input_graph_spec)
    gnn_model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss=tf.keras.losses.MeanSquaredError(),
        metrics=["mae"]
    )
    # Summary of the model
    gnn_model.summary()
    """
     batch_size, number of nodes, and node_dim. the first shape [15,1] is [batch_size*number of nodes, 1] 
     and [5,3] is [batch_size, node_dim]. 
    """
    steps_per_epoch = 15
    epochs = 30

    history = gnn_model.fit(x=train_ds_batched,
                            steps_per_epoch=steps_per_epoch,
                            epochs=epochs,
                            validation_data=None)

    for k, hist in history.history.items():
        plt.plot(hist)
        plt.title(k)
        plt.show()
