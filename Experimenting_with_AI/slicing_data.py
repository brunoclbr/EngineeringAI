import tensorflow as tf
data = [100, 200, 300, 400, 500]
tf_data = tf.data.Dataset.from_tensor_slices(data)
print(len(tf_data))
for i in tf_data:
    print(i.numpy())

# Create a dataset where features and labels are paired
features = [[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]]
print(features)
labels = ["A", "B", "C"]

# Print each feature-label pair
sup_data = tf.data.Dataset.from_tensor_slices((features, labels))
for feature, label in sup_data:
    print("Feature:", feature.numpy(), "Label:", label.numpy())

# Create a dataset where inputs are stored as a dictionary
data = {
    "feature1": [1, 2, 3],
    "feature2": [4.0, 5.0, 6.0]
}

# Print each dictionary element

dict_data = tf.data.Dataset.from_tensor_slices(data)
"""
This automatically slices each key-value pair along the first axis.
The dataset will have 3 elements, each being a dictionary with two values.
The dictionary itself is not treated as a single tensor.
Instead, each value inside the dictionary is treated as an independent tensor.
from_tensor_slices() slices the dictionary by taking the corresponding index from each key.
"""

for element in dict_data:
    print({key: value.numpy() for key, value in element.items()})

    
# Slicing a tuple of 1D tensors produces tuple elements containing
# scalar tensors. So far I had given one tensor, but giving multiple tensors
# as inputs confuses me
"""
The input is a tuple of three 1D tensors. The shape of each tensor is (2,),
meaning each has 2 elements. It slices along the first dimension (axis 0) 
of each tensor independently. Since each tensor has shape (2,), 
slicing along axis 0 results in 2 slices.
"""
datasett = tf.data.Dataset.from_tensor_slices(([1, 2], [3, 4], [5, 6]))
print(list(datasett.as_numpy_iterator()))


# Rank 0 tensor (scalar)
tensor_rank_0 = tf.constant(42)
#dataset_rank_0 = tf.data.Dataset.from_tensor_slices(tensor_rank_0)
#print("Rank 0 tensor dataset:")
#for element in dataset_rank_0:
#    print(element.numpy())

# Rank 1 tensor (vector)
tensor_rank_1 = tf.constant([1, 2, 3, 4, 5])
dataset_rank_1 = tf.data.Dataset.from_tensor_slices(tensor_rank_1)
print("\nRank 1 tensor dataset:", tensor_rank_1.shape)
for element in dataset_rank_1:
    print(element.numpy())

# Rank 2 tensor (matrix)
tensor_rank_2 = tf.constant([[1, 2], [3, 4], [5, 6]])
dataset_rank_2 = tf.data.Dataset.from_tensor_slices(tensor_rank_2)
print(f"\nRank 2 tensor dataset: {dataset_rank_2}")
for element in dataset_rank_2:
    print(element.numpy())

# Rank 3 tensor
tensor_rank_3 = tf.constant([
    [
        [1, 2], [3, 4]
        ], [
            [5, 6], [7, 8]
            ]
    ])
dataset_rank_3 = tf.data.Dataset.from_tensor_slices(tensor_rank_3)
print(f"\nRank 3 tensor dataset: {dataset_rank_3}")
for element in dataset_rank_3:
    print(element.numpy())

# Rank 4 tensor
tensor_rank_4 = tf.constant(
    [
        [
            [[1], [2]], 
            [[3], [4]]
                ], 
    [
        [[5], [6]], 
        [[7], [8]]
        ]
    ])
dataset_rank_4 = tf.data.Dataset.from_tensor_slices(tensor_rank_4)
print(f"\nRank 4 tensor dataset: {dataset_rank_4}")
i=1
for element in dataset_rank_4:
    print(i)
    print(element.numpy())
    i+=1

# Rank 5 tensor
tensor_rank_5 = tf.constant([[[[[1], [2]], [[3], [4]]], [[[5], [6]], [[7], [8]]]]])
dataset_rank_5 = tf.data.Dataset.from_tensor_slices(tensor_rank_5)
print(f"\nRank 5 tensor dataset: {dataset_rank_5}")
for element in dataset_rank_5:
    print(element.numpy())
