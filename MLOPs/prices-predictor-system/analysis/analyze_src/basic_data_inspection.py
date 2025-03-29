from abc import ABC, abstractmethod

import pandas as pd


# Abstract Base Class for Data Inspection Strategies
# --------------------------------------------------
# This class defines a common interface for data inspection strategies.
# Subclasses must implement the inspect method.
class DataInspectionStrategy(ABC):
    @abstractmethod
    def inspect(self, df: pd.DataFrame):
        """
        Perform a specific type of data inspection.

        Parameters:
        df (pd.DataFrame): The dataframe on which the inspection is to be performed.

        Returns:
        None: This method prints the inspection results directly.
        """
        pass


# Concrete Strategy for Data Types Inspection
# --------------------------------------------
# This strategy inspects the data types of each column and counts non-null values.
class DataTypesInspectionStrategy(DataInspectionStrategy):
    def inspect(self, df: pd.DataFrame):
        """
        Inspects and prints the data types and non-null counts of the dataframe columns.

        Parameters:
        df (pd.DataFrame): The dataframe to be inspected.

        Returns:
        None: Prints the data types and non-null counts to the console.
        """
        print("\nData Types and Non-null Counts:")
        print(df.info())
        print("\nData Description:")
        print(df.describe())


# Concrete Strategy for Summary Statistics Inspection
# -----------------------------------------------------
# This strategy provides summary statistics for both numerical and categorical features.
class SummaryStatisticsInspectionStrategy(DataInspectionStrategy):
    def inspect(self, df: pd.DataFrame):
        """
        Prints summary statistics for numerical and categorical features.

        Parameters:
        df (pd.DataFrame): The dataframe to be inspected.

        Returns:
        None: Prints summary statistics to the console.
        """
        print("\nSummary Statistics (Numerical Features):")
        print(df.describe())
        print("\nSummary Statistics (Categorical Features):")
        print(df.describe(include=["O"]))


# Context Class that uses a DataInspectionStrategy
# ------------------------------------------------
# This class allows you to switch between different data inspection strategies.
class DataInspector:
    def __init__(self, strategy: DataInspectionStrategy):
        """
        Initializes the DataInspector with a specific inspection strategy.

        Parameters:
        strategy (DataInspectionStrategy): The strategy to be used for data inspection.

        Returns:
        None
        """
        self._strategy = strategy

    def set_strategy(self, strategy: DataInspectionStrategy):
        """
        Sets a new strategy for the DataInspector.

        Parameters:
        strategy (DataInspectionStrategy): The new strategy to be used for data inspection.

        Returns:
        None
        """
        self._strategy = strategy

    def execute_inspection(self, df: pd.DataFrame):
        """
        Executes the inspection using the current strategy.

        Parameters:
        df (pd.DataFrame): The dataframe to be inspected.

        Returns:
        None: Executes the strategy's inspection method.
        """
        self._strategy.inspect(df)


# Example usage
if __name__ == "__main__":
    # Example usage of the DataInspector with different strategies.

    # Load the data
    df = pd.read_csv('C:/Users/bruno/OneDrive_main/Folders/GitHub/Deep_Learning_Projects/MLOPs/prices-predictor-system/extracted_data/AmesHousing.csv')
    # 1. Data Inspection
    print(type(df))
    print(df.head())
    print(df.info())
    print(df.describe()) # DataFrame(df, path) car = Car("VW"), Car(car, "VW"), car.describe() <=> Car.describe(car)

    #2 Missing Data Handling
    # df.isnull().sum()
    # df.dropna()
    # df.fillna(value)

    #3 Data Cleaning & Transformation
    #df.drop_duplicates()
    #df.rename(columns={'old': 'new'})
    #df.astype({'col': 'type'})
    #df.replace(['old'], ['new'])
    #df.reset_index()
    #df.drop(['col'], axis=1)

    #4 Data Selection & Filtering
    #df.loc['label', 'col'] --> selects data by labels/conditions
    #df.iloc[] --> acces data using integer positions
    #df[df['col'] > value]

    #5 Data Aggregation & Analysis
    #df.groupby('col').agg(['mean']) --> groups and applies aggregation function
    #df.sort_values('col', ascending=False)
    #df.value_counts()-->count unique values in a column
    #df.apply() --> apply function to rows/columns
    #df.pivot_table(values, index, columns) --> creates pivot ? from data

    #6 Data Combining/Merging
    #pd.concat([df1, df2])
    #pd.merge(df1, df2, on='key')
    #df1.join(df2) 
    #df1.append(df2)

    # Initialize the Data Inspector with a specific strategy
    # inspector = DataInspector(DataTypesInspectionStrategy())
    # inspector.execute_inspection(df)
    # I also could create an object of a DataInspectionStrategy and pass it to my DataInspector object

    # Change strategy to Summary Statistics and execute
    # inspector.set_strategy(SummaryStatisticsInspectionStrategy())
    # inspector.execute_inspection(df)
   
