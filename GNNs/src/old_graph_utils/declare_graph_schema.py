import tensorflow as tf
import tensorflow_gnn as tfgnn
from google.protobuf import text_format
import tensorflow_gnn.proto.graph_schema_pb2 as schema_pb2
import os
print(f'Running TF-GNN {tfgnn.__version__} with TensorFlow {tf.__version__}.')

# the dim of each feature must be the CONST, MAX dim of my molecule representation (so far its 61) and its features
schema_ptbx = """
context {
  features {
    key: "label"
    value: {
      description: "compound entropy"
      dtype: DT_FLOAT
    }
  }
}
node_sets {
  key: "atoms"
  value {
    features {
      key: "hidden_state"
      value {
        description: "atomic_number representing atom type"
        dtype: DT_FLOAT
        shape {dim { size: -1 }}  # <-- Allow variable-sized tensors
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

graph_schema = text_format.Merge(schema_ptbx, schema_pb2.GraphSchema())

graph_tensor_spec = tfgnn.create_graph_spec_from_schema_pb(graph_schema)


def decode_fn(record_bytes):
    """
    Calling tfgnn.create_graph_spec_from_schema_pb with the GraphSchema will provide us the GraphTensorSpec.
    GraphTensorSpec can be used to parse serialized tf.Example protos into the GraphTensor using
    tfgnn.parse_single_example api with the tf.data.dataset's map function.
    :param record_bytes:
    :return: new_graph, label:
    """

    print(record_bytes)
    graph = tfgnn.parse_single_example(graph_tensor_spec, record_bytes, validate=True)

    # extract label from context and remove from input graph
    context_features = graph.context.get_features_dict()
    label = context_features.pop('label')
    new_graph = graph.replace_features(context=context_features)
    print(new_graph)

    return new_graph, label


train_path = os.path.join(os.getcwd(), 'graphs.tfrecord')
val_path = os.path.join(os.getcwd(), 'graphs_val.tfrecord')
train_ds = tf.data.TFRecordDataset([train_path]).map(decode_fn)
val_ds = tf.data.TFRecordDataset([val_path]).map(decode_fn)


