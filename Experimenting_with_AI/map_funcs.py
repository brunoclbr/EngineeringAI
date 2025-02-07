import tensorflow as tf
"""
map(
    map_func, num_parallel_calls=None, deterministic=None, name=None
)
This transformation applies map_func to each element of this dataset, and returns 
a new dataset containing the transformed elements, in the same order as they appeared in the input.

anonymous_function = lambda arguments : expression 
--> print(anonymous_function(variable)) 

When you iterate over a dataset, each element isa tf.Tensor object. Calling .numpy() method
on a tf.Tensor object returns the value of the tensor as a NumPy array.

The zip() function returns a zip object, which is an iterator of tuples where the first item in each passed
iterator is paired together, and then the second item in each passed iterator are paired together etc.

"""
dataset = tf.data.Dataset.from_tensor_slices([1, 2, 3, 4, 5])
# Example 1: Squaring each element in the dataset

squared_dataset = dataset.map(lambda x : x**2)
for element in squared_dataset:
    print(element.numpy())

# Example 2: Adding a constant value to each element
added_dataset = dataset.map(lambda x : x +10)
for element in added_dataset:
    print(element.numpy())

# Example 3: Normalizing each element in the dataset with mean and std
dataset2 = tf.data.Dataset.from_tensor_slices([1.0, 2.0, 3.0, 4.0, 5.0])
mean = 3.0
std = 1.0

normalized_dataset = dataset2.map(lambda x : (x - mean)/std)
for element in normalized_dataset:
    print(element.numpy())

# Example 4: Converting each element to a string

string_dataset = dataset.map(lambda x: tf.strings.as_string(x))
for element in string_dataset:
    print(element.numpy())

# Example 5: Applying a custom function to each element
def custom_function(x):
    return x * 2 + 1

custom_dataset = dataset.map(lambda x : custom_function(x))
for element in custom_dataset:
    print(element.numpy())

# Example 6: Filter dataset to keep elements greater than 50

dataset3 = tf.data.Dataset.from_tensor_slices([10, 20, 30, 40, 50, 60, 70, 80, 90, 100])

filtered_data = dataset3.map(lambda x :  x if x > 50 else 0)
for element in filtered_data:
    print(element.numpy())

dataset4 = tf.data.Dataset.from_tensor_slices(([1, 2, 3], [10, 20, 30]))
summed_data = dataset4.map(lambda x, y: x + y)
for element in summed_data:
    print(element.numpy())
for i in dataset4:
    print(i)

dts = tf.data.Dataset.range(100)
def dataset_fn(ds):
    return ds.filter(lambda x: x < 5)
dts2 = dataset_fn(dts)
for i in dts2:
    print(i)