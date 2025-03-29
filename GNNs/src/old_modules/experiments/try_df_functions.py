import pandas as pd
import re
data = {'A': [1, 2, 3, 4, 5], 'B': ['x', 'y', 'z', 'w', 'v']}
df = pd.DataFrame(data)


# Condition for dropping rows
df1 = df[df['A'] >= 5]  # Keeps only rows where column 'A' is >= 5
boolean_mask = [True, False, True, False, False]
df2 = df[boolean_mask]


# Example data
data = {'facet': ['1100', '11-00', '-1100', '1100.0', '-1-10']}
df = pd.DataFrame(data)


# Function to standardize and convert facet values
def process_facet_value(facet_str):
    # Remove decimal points for float-like strings
    if isinstance(facet_str, str) and '.' in facet_str:
        facet_str = facet_str.replace('.', '')

    # Replace '-' with 'n' temporarily
    standardized_str = facet_str.replace('-', 'n')
    return standardized_str


# Apply processing function
df['processed_facet'] = df['facet'].apply(process_facet_value)

# Function to split the processed facet string into exactly four integer components
def split_facet_components(facet_str):
    # Replace 'n' back to '-' for handling negative components, then split by characters
    facet_str = facet_str.replace('n', '-')

    # Regex to capture each number group, considering '-' signs
    components = re.findall(r'-?\d', facet_str)
    components = [int(c) for c in components]

    # Adjust list to ensure exactly 4 components
    if len(components) > 4:
        # Truncate to 4 elements if more than 4
        components = components[:4]
    elif len(components) < 4:
        # Pad with zeros if less than 4
        components.extend([0] * (4 - len(components)))

    # Debugging output to check lengths of each component list
    print(f"Processed {facet_str}: {components} (length: {len(components)})")

    return components


# Apply function to get individual components and handle lengths
component_lists = df['processed_facet'].apply(split_facet_components).to_list()

# Assign columns to the DataFrame
df[['facet_1', 'facet_2', 'facet_3', 'facet_4']] = pd.DataFrame(component_lists, index=df.index)

print("DataFrame with Split Facet Components:")
print(df)

# Sample data (replace with your DataFrame)
data = {
    'Column1': [1, 2, None, 4],
    'Column2': [5, None, 7, 8],
    'Column3': [9, 10, 11, None]
}
df = pd.DataFrame(data)

# Drop rows where 'Column2' has NaN values
df_cleaned = df.dropna(subset=['Column2'])

print(df_cleaned)

x = lambda a: a + 10
print(x(5))

def myfunc(n):
    """ n is used to define the multiplier within the function f(a)
        and a is the actual argument you pass to the function once 'instantiated'
        lamda functions apparently are used in this dynamic form"""
    return lambda a: a * n


mydoubler = myfunc(2)  #here 2=n

print(mydoubler(11))

