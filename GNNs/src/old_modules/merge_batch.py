import tensorflow as tf
import tensorflow_gnn as tfgnn


def _make_preprocessing_model(graph_tensor_spec):
    """Returns Keras model to preprocess a batched and parsed GraphTensor."""
    graph = input_graph = tf.keras.layers.Input(type_spec=graph_tensor_spec)
    def set_initial_node_state(node_set, dense_dim=20, *, node_set_name):
        #features = node_set.features
        # applies linear transform to all features in one node_set
        return {
            # Retain the original atomic properties
            "atomic_number": node_set["atomic_number"],
            "electro_negativity": node_set["electro_negativity"]
        }

    def set_initial_edge_state(edge_set, *, edge_set_name):

        return {
            # Retain the original atomic properties
            tfgnn.HIDDEN_STATE: edge_set[tfgnn.HIDDEN_STATE]
        }

  # Convert input features to suitable representations for use on GPU/TPU.
  # Drop unused features (like id strings for tracking the source of examples).
    graph = tfgnn.keras.layers.MapFeatures(node_sets_fn=set_initial_node_state,edge_sets_fn=set_initial_edge_state )(graph)

  ### IMPORTANT: All TF-GNN modeling code assumes a GraphTensor of shape []
  ### in which the graphs of the input batch have been merged to components of
  ### one contiguously indexed graph. There are no edges between components,
  ### so no information flows between them.
    graph = graph.merge_batch_to_components()
    node_sizes = graph.node_sets["atoms"].sizes  # Shape: [num_graphs]

    # Compute cumulative sum of node sizes to adjust edge indices
    node_offsets = tf.concat([[0], tf.cumsum(node_sizes[:-1])], axis=0)

    # Adjust edge indices
    edge_src = graph.edge_sets["bonds"].adjacency.source + tf.gather(node_offsets, graph.edge_sets["bonds"]["graph_index"])
    edge_tgt = graph.edge_sets["bonds"].adjacency.targets + tf.gather(node_offsets, graph.edge_sets["bonds"]["graph_index"])

    # Update edge set with new indices
    graph = graph.replace_features(
        edge_sets={"bonds": {"source": edge_src, "target": edge_tgt}}
    )

    return tf.keras.Model(input_graph, graph)

def get_preprocessed_dataset(ds, example_input_spec, *, batch_size=4):
    """
    The dataset returned by _get_preprocessed_dataset() is not symbolic; it contains actual GraphTensor objects and
    labels ready for training.The dataset is fully preprocessed before being passed into model.fit(), meaning:
    Graphs are already merged into components (fixing your previous issue with incorrect node indexing). Labels are
    extracted and separated from the graph structure. Size constraints are enforced before training. This preprocessing
    model ensures that when model.fit(train_ds) is called, the model receives clean, correctly indexed,
    and batched graph data. 🚀
    :param ds: serialized to graphtensor data
    :param example_input_spec:
    :param batch_size:
    :return:
    """
    #ds = ds.repeat()
    ds = ds.batch(batch_size)  # in docu this is done to raw data before serializing to ensure that each replica GPU/TPU
    # gets an appropriate batch size
    # Apply preprocessing Model
    # Why use map() instead of applying transformations directly?
    # This allows the dataset to be processed efficiently in parallel (num_parallel_calls=tf.data.AUTOTUNE).
    ds = ds.map(_make_preprocessing_model(example_input_spec))

    return ds
