import os
import requests
import zipfile
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import tensorflow_gnn as tfgnn
from tensorflow_gnn.models import gcn
from tensorflow_gnn.models import graph_sage
from google.protobuf import text_format
import tensorflow_gnn.proto.graph_schema_pb2 as schema_pb2
from tfrecords.serialize import train_ds, val_ds

t1 = train_ds
t2 = val_ds
#print(f'Running TF-GNN {tfgnn.__version__} with TensorFlow {tf.__version__}.')

# Step 1: Download the file
url = "https://storage.googleapis.com/download.tensorflow.org/data/mutag.zip"
output_file = "mutag.zip"

response = requests.get(url)
with open(output_file, "wb") as f:
    f.write(response.content)

# Step 2: Unzip the file
with zipfile.ZipFile(output_file, "r") as zip_ref:
    zip_ref.extractall(".")  # Extracts files to the current directory

# Optional: Clean up by removing the zip file
os.remove(output_file)

train_path = os.path.join(os.getcwd(), 'mutag', 'train.tfrecords')
val_path = os.path.join(os.getcwd(), 'mutag', 'val.tfrecords')

# @title Declare GraphSchema { vertical-output: true }
schema_pbtx = """
# proto-file: //third_party/py/tensorflow_gnn/proto/graph_schema.proto
# proto-message: tensorflow_gnn.GraphSchema
context {
  features {
    key: "label"
    value: {
      description: "compound mutagenicity."
      dtype: DT_INT32
    }
  }
}
node_sets {
  key: "atoms"
  value {
    features {
      key: "hidden_state"
      value {
        description: "atom type."
        dtype: DT_FLOAT
        shape { dim { size: 7 } }
      }
    }
  }
}
edge_sets {
  key: "bonds"
  value {
    source: "atoms"
    target: "atoms"
    features {
      key: "hidden_state"
      value {
        description: "bond type."
        dtype: DT_FLOAT
        shape { dim { size: 4 } }
      }
    }
  }
}
"""
graph_schema = text_format.Merge(schema_pbtx, schema_pb2.GraphSchema())
#print(f"graph_schema: {graph_schema}")
graph_tensor_spec = tfgnn.create_graph_spec_from_schema_pb(graph_schema)


def decode_fn(record_bytes):
    graph = tfgnn.parse_single_example(graph_tensor_spec, record_bytes, validate=True)

    # extract label from context and remove from input graph
    context_features = graph.context.get_features_dict()
    label = context_features.pop('label')
    new_graph = graph.replace_features(context=context_features)

    return new_graph, label


train_ds = tf.data.TFRecordDataset([train_path]).map(decode_fn)
val_ds = tf.data.TFRecordDataset([val_path]).map(decode_fn)
#from declare_graph_schema import train_ds, val_ds
batch_size = 2 #@param {type:"integer"}
train_ds_batched =train_ds.batch(batch_size=batch_size).repeat()
val_ds_batched = val_ds.batch(batch_size=batch_size)

# @title GNN Model { vertical-output: true }

if __name__ == "__main__":
    graph_convolution_type = "gcn"  # @param ["default", "gcn", "graph_sage"]


    def _build_model(
            graph_tensor_spec,
            # Dimensions of initial states.
            node_dim=16,
            edge_dim=16,
            # Dimensions for message passing.
            message_dim=64,
            next_state_dim=64,
            # Dimension for the logits.
            num_classes=2,
            # Number of message passing steps.
            num_message_passing=3,
            # Other hyperparameters.
            l2_regularization=5e-4,
            dropout_rate=0.5,
    ):
        # Model building with Keras's Functional API starts with an input object
        # (a placeholder for the eventual inputs). Here is how it works for
        # GraphTensors:
        input_graph = tf.keras.layers.Input(type_spec=graph_tensor_spec)

        # IMPORTANT: All TF-GNN modeling code assumes a GraphTensor of shape []
        # in which the graphs of the input batch have been merged to components of
        # one contiguously indexed graph. (There are no edges between components,
        # so no information flows between them.)
        graph = input_graph.merge_batch_to_components()

        # Nodes and edges have one-hot encoded input features. Sending them through
        # a Dense layer effectively does a lookup in a trainable embedding table.
        def set_initial_node_state(node_set, *, node_set_name):
            # Since we only have one node set, we can ignore node_set_name.
            return tf.keras.layers.Dense(node_dim)(node_set[tfgnn.HIDDEN_STATE])

        def set_initial_edge_state(edge_set, *, edge_set_name):
            if graph_convolution_type != "default":
                return {}
            return tf.keras.layers.Dense(edge_dim)(edge_set[tfgnn.HIDDEN_STATE])

        # MapFeatures layer receives callbacks as input for each graph piece: node_set
        # edge_set and context. Each callback applies a transformation over the
        # existing features of the respective graph piece while using a Keras
        # Functional API to call new Keras Layers. For more information and examples
        # about the MapFeatures layer please check out its docstring. This call here
        # initializes the hidden states of the edge and node sets.
        graph = tfgnn.keras.layers.MapFeatures(
            node_sets_fn=set_initial_node_state, edge_sets_fn=set_initial_edge_state)(
            graph)

        num_atoms = tf.expand_dims(tf.cast(graph.node_sets["atoms"].sizes, dtype=tf.float32), axis=-1)
        num_bonds = tf.expand_dims(tf.cast(graph.edge_sets["bonds"].sizes, dtype=tf.float32), axis=-1)
        graph = graph.replace_features(
            context={
                tfgnn.HIDDEN_STATE: tf.concat([num_atoms, num_bonds], axis=1)
            })
        tf.print("Context Shape after MapFeatures & context addition:", graph.context[tfgnn.HIDDEN_STATE].shape)

        # This helper function is just a short-hand for the code below.
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

        # The GNN core of the model does `num_message_passing` many updates of node
        # states conditioned on their neighbors and the edges connecting to them.
        # More precisely:
        #  - Each edge computes a message by applying a dense layer `message_fn`
        #    to the concatenation of node states of both endpoints (by default)
        #    and the edge's own unchanging feature embedding.
        #  - Messages are summed up at the common TARGET nodes of edges.
        #  - At each node, a dense layer is applied to the concatenation of the old
        #    node state with the summed edge inputs to compute the new node state.
        # Each iteration of the for-loop creates new Keras Layer objects, so each
        # round of updates gets its own trainable variables.
        for i in range(num_message_passing):
            if graph_convolution_type == "default":
                graph = tfgnn.keras.layers.GraphUpdate(
                    node_sets={
                        "atoms": tfgnn.keras.layers.NodeSetUpdate(
                            {"bonds": tfgnn.keras.layers.SimpleConv(
                                sender_edge_feature=tfgnn.HIDDEN_STATE,
                                message_fn=dense(message_dim),
                                reduce_type="sum",
                                receiver_tag=tfgnn.TARGET)},
                            tfgnn.keras.layers.NextStateFromConcat(dense(next_state_dim)))},
                )(graph)
            elif graph_convolution_type == "gcn":
                graph = gcn.GCNHomGraphUpdate(
                    units=message_dim,
                    name=graph_convolution_type + "_layer_" + str(i))(graph)
            elif graph_convolution_type == "graph_sage":
                graph = graph_sage.GraphSAGEGraphUpdate(
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
                    name="gsage_layer_" + str(i))(graph)

        # After the GNN has computed a context-aware representation of the "atoms",
        # the model reads out a representation for the graph as a whole by averaging
        # (pooling) node states into the graph context. The context is global to each
        # input graph of the batch, so the first dimension of the result corresponds
        # to the batch dimension of the inputs (same as the labels).
        readout_features = tfgnn.keras.layers.Pool(
            tfgnn.CONTEXT, "mean", node_set_name="atoms")(graph)
        # Context  has a hidden-state feature, concatenate the aggregated node vectors
        # with the hidden-state to get the final vector,
        feat = tf.concat([readout_features, graph.context[tfgnn.HIDDEN_STATE]], axis=1)
        tf.print("Readout_features Shape after Pool2Context:", readout_features.shape)
        # Put a linear classifier on top (not followed by dropout).
        logits = tf.keras.layers.Dense(1)(feat)

        # Build a Keras Model for the transformation from input_graph to logits.
        return tf.keras.Model(inputs=[input_graph], outputs=[logits])


    node_dim = 16  # @param {type:"integer"}
    edge_dim = 16  # @param {type:"integer"}
    message_dim = 64  # @param {type:"integer"}
    next_state_dim = 64  # @param {type:"integer"}
    num_classes = 2  # @param {type:"integer"}
    num_message_passing = 3  # @param {type:"integer"}
    l2_regularization = 5e-4  # @param {type:"number"}
    dropout_rate = 0.5  # @param {type:"number"}

    model_input_graph_spec, label_spec = train_ds.element_spec
    del label_spec  # Unused.
    model = _build_model(model_input_graph_spec,
                         node_dim=node_dim,
                         edge_dim=edge_dim,
                         message_dim=message_dim,
                         next_state_dim=next_state_dim,
                         num_classes=num_classes,
                         num_message_passing=num_message_passing,
                         l2_regularization=l2_regularization,
                         dropout_rate=dropout_rate)
    loss = tf.keras.losses.BinaryCrossentropy(from_logits=True)
    metrics = [tf.keras.metrics.BinaryAccuracy(threshold=0.), tf.keras.metrics.BinaryCrossentropy(from_logits=True)]

    # @title Optimizer
    learning_rate = 1e-3  # @param {type:"number"}
    learning_rate_decay = True  # @param {type:"boolean"}
    steps_per_epoch = 10  # @param {type:"integer"}
    epochs = 43  # @param {type:"integer"}

    if learning_rate_decay:
        learning_rate = tf.keras.optimizers.schedules.CosineDecay(learning_rate, steps_per_epoch * epochs)

    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

    model.compile(optimizer, loss=loss, metrics=metrics)
    model.summary()

    steps_per_epoch = 10  # @param {type:"integer"}
    epochs = 43  # @param {type:"integer"}

    history = model.fit(train_ds_batched,
                        steps_per_epoch=steps_per_epoch,
                        epochs=epochs,
                        validation_data=val_ds_batched)

    for k, hist in history.history.items():
        plt.plot(hist)
        plt.title(k)
        plt.show()