import tensorflow as tf
import tensorflow_gnn as tfgnn
from google.protobuf import text_format
import tensorflow_gnn.proto.graph_schema_pb2 as schema_pb2
import os
print(f'Running TF-GNN {tfgnn.__version__} with TensorFlow {tf.__version__}.')


schema = tfgnn.parse_schema("""
node_sets {
  key: "atoms"
  value {
    features {
      key: "atomic_number"
      value {
        description: "atomic_number representing atom type"
        dtype: DT_INT64
        shape { dim { size: 3 } }
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
      key: "bond_type"
      value {
        description: "bond type."
        dtype: DT_INT64
        shape { dim { size: 4 } }
      }
    }
  }
}
""")

graph_tensor_spec = tfgnn.create_graph_spec_from_schema_pb(schema)
graph_spec = tfgnn.create_graph_spec_from_schema_pb(schema)


graph_schema = text_format.Merge(schema, schema_pb2.GraphSchema())
print(f"graph_schema: {graph_schema}")


def decode_fn(record_bytes):
    print(record_bytes)
    graph = tfgnn.parse_single_example(graph_tensor_spec, record_bytes, validate=True)

    # extract label from context and remove from input graph
    context_features = graph.context.get_features_dict()
    #label = context_features.pop('label')
    new_graph = graph.replace_features(context=context_features)
    print(new_graph)

    return new_graph  # label


train_path = os.path.join(os.getcwd(), 'graphs.tfrecord')
train_ds = tf.data.TFRecordDataset([train_path]).map(decode_fn)
