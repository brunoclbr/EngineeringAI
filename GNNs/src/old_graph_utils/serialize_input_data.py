import tensorflow as tf
import tensorflow_gnn as tfgnn
import numpy as np
from google.protobuf import text_format
from declare_graph_schema import schema_ptbx
from google.protobuf import text_format
import tensorflow_gnn.proto.graph_schema_pb2 as schema_pb2
from mol2graph import graph1


# Assuming you already have your `GraphTensor` and its schema
def save_graph_tensors_to_tfrecord(graph_tensors, schema_str, output_file):
    """
    Save a list of GraphTensor objects to a TFRecord file.
    """
    # Parse the schema string into a GraphSchema object
    schema = schema_pb2.GraphSchema()
    text_format.Merge(schema_str, schema)

    # Write the schema to a .pbtxt file
    schema_filename = output_file.replace(".tfrecord", "_schema.pbtxt")
    with open(schema_filename, "w") as schema_file:
        schema_file.write(text_format.MessageToString(schema))

    # Create the GraphSpec from the parsed schema
    graph_spec = tfgnn.create_graph_spec_from_schema_pb(schema)

    # Write the GraphTensor objects to a TFRecord file
    with tf.io.TFRecordWriter(output_file) as writer:
        for graph_tensor in graph_tensors:
            serialized_example = tfgnn.write_example(graph_tensor)
            writer.write(serialized_example.SerializeToString())


# Save your graph data to a TFRecord
graph_tensors = np.array([graph1, graph1, graph1, graph1, graph1, graph1])  # List of GraphTensor objects
output_file = "C:/Users/BCOPS7AJ/Aerostack_Projects/CatDevel/catAds/graph_utils/graphs.tfrecord"
save_graph_tensors_to_tfrecord(graph_tensors, schema_ptbx, output_file)


def load_batched_graph_dataset(tfrecord_file, schema_file, batch_size):
    """
    Load a TFRecord file of serialized GraphTensor objects into a batched Dataset.
    """
    # Read the schema
    graph_schema = tfgnn.read_schema(schema_file)
    graph_spec = tfgnn.create_graph_spec_from_schema_pb(graph_schema)

    # Load the dataset
    dataset = tf.data.TFRecordDataset(filenames=[tfrecord_file])
    dataset = dataset.map(
        lambda serialized: tfgnn.parse_single_example(graph_spec, serialized)
    )

    # Batch the dataset
    dataset = dataset.batch(batch_size)
    return dataset


# Load your dataset
#tfrecord_file = "C:/Users/BCOPS7AJ/Aerostack_Projects/CatDevel/catAds/graph_utils/graphs.tfrecord"
#schema_file = "C:/Users/BCOPS7AJ/Aerostack_Projects/CatDevel/catAds/graph_utils/graphs_schema.pbtxt"
#batch_size = 2
#batched_dataset = load_batched_graph_dataset(tfrecord_file, schema_file, batch_size)

