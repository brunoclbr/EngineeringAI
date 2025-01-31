import tensorflow as tf

# Example 1: Squaring each element in the dataset
dataset = tf.data.Dataset.from_tensor_slices([1, 2, 3, 4, 5])
squared_dataset = dataset.map(lambda x: x ** 2)
for element in squared_dataset:
    print(element.numpy())

# Example 2: Adding a constant value to each element
dataset = tf.data.Dataset.from_tensor_slices([1, 2, 3, 4, 5])
added_dataset = dataset.map(lambda x: x + 10)
for element in added_dataset:
    print(element.numpy())

# Example 3: Normalizing each element in the dataset
dataset = tf.data.Dataset.from_tensor_slices([1.0, 2.0, 3.0, 4.0, 5.0])
mean = 3.0
std = 1.0
normalized_dataset = dataset.map(lambda x: (x - mean) / std)
for element in normalized_dataset:
    print(element.numpy())

# Example 4: Converting each element to a string
dataset = tf.data.Dataset.from_tensor_slices([1, 2, 3, 4, 5])
string_dataset = dataset.map(lambda x: tf.strings.as_string(x))
for element in string_dataset:
    print(element.numpy())

# Example 5: Applying a custom function to each element
def custom_function(x):
    return x * 2 + 1

dataset = tf.data.Dataset.from_tensor_slices([1, 2, 3, 4, 5])
custom_dataset = dataset.map(lambda x: custom_function(x))
for element in custom_dataset:
    print(element.numpy())