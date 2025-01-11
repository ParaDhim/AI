#############
## Imports ##
#############

import pickle
import pandas as pd
import numpy as np
import bnlearn as bn
from test_model import test_model

######################
## Boilerplate Code ##
######################

def load_data():
    """Load train and validation datasets from CSV files."""
    # Implement code to load CSV files into DataFrames
    # Example: train_data = pd.read_csv("train_data.csv")
    train_data = pd.read_csv("/Users/parasdhiman/Desktop/assmt/AI/AI Assmt/Assmt3/code_zip/train_data.csv")
    val_data = pd.read_csv("/Users/parasdhiman/Desktop/assmt/AI/AI Assmt/Assmt3/code_zip/validation_data.csv")
    
    # Convert categorical variables to discrete values
    categorical_columns = ['Distance', 'Route_Type', 'Fare_Category']
    
    for col in categorical_columns:
        train_data[col] = pd.Categorical(train_data[col]).codes
        val_data[col] = pd.Categorical(val_data[col]).codes
    
    # Convert numerical columns to integer bins
    numerical_columns = ['Start_Stop_ID', 'End_Stop_ID', 'Zones_Crossed']
    n_bins = 10
    
    for col in numerical_columns:
        train_data[col] = pd.qcut(train_data[col], q=n_bins, labels=False, duplicates='drop')
        val_data[col] = pd.qcut(val_data[col], q=n_bins, labels=False, duplicates='drop')
    
    return train_data, val_data
    pass

def make_network(df):
    """Define and fit the initial Bayesian Network."""
    # Code to define the DAG, create and fit Bayesian Network, and return the model
    # Define edges for the initial network
    edges = [
        ['Start_Stop_ID', 'Distance'],
        ['Start_Stop_ID', 'Zones_Crossed'],
        ['Start_Stop_ID', 'Route_Type'],
        ['Start_Stop_ID', 'Fare_Category'],
        ['End_Stop_ID', 'Distance'],
        ['End_Stop_ID', 'Zones_Crossed'],
        ['End_Stop_ID', 'Route_Type'],
        ['End_Stop_ID', 'Fare_Category'],
        ['Distance', 'Zones_Crossed'],
        ['Distance', 'Fare_Category'],
        ['Zones_Crossed', 'Fare_Category'],
        ['Route_Type', 'Fare_Category']
    ]
    
    # Create adjacency matrix
    nodes = list(df.columns)
    n = len(nodes)
    adj_matrix = np.zeros((n, n))
    
    # Fill adjacency matrix based on edges
    node_to_idx = {node: idx for idx, node in enumerate(nodes)}
    for edge in edges:
        i, j = node_to_idx[edge[0]], node_to_idx[edge[1]]
        adj_matrix[i, j] = 1
    
    # Create model dictionary
    model = {
        'adjmat': pd.DataFrame(adj_matrix, index=nodes, columns=nodes),
        'model': None,
        'values': {col: sorted(df[col].unique()) for col in df.columns}
    }
    
    # Fit the parameters
    model = bn.parameter_learning.fit(model, df, methodtype='maximumlikelihood')
    
    return model
    pass

def make_pruned_network(df):
    """Define and fit a pruned Bayesian Network."""
    # Code to create a pruned network, fit it, and return the pruned model
    # Define edges for the pruned network
    edges = [
        ['Distance', 'Fare_Category'],
        ['Zones_Crossed', 'Fare_Category'],
        ['Route_Type', 'Fare_Category'],
        ['Distance', 'Zones_Crossed']
    ]
    
    # Create adjacency matrix
    nodes = list(df.columns)
    n = len(nodes)
    adj_matrix = np.zeros((n, n))
    
    # Fill adjacency matrix based on edges
    node_to_idx = {node: idx for idx, node in enumerate(nodes)}
    for edge in edges:
        i, j = node_to_idx[edge[0]], node_to_idx[edge[1]]
        adj_matrix[i, j] = 1
    
    # Create model dictionary
    model = {
        'adjmat': pd.DataFrame(adj_matrix, index=nodes, columns=nodes),
        'model': None,
        'values': {col: sorted(df[col].unique()) for col in df.columns}
    }
    
    # Fit the parameters
    model = bn.parameter_learning.fit(model, df, methodtype='maximumlikelihood')
    
    return model
    pass

def make_optimized_network(df):
    """Perform structure optimization and fit the optimized Bayesian Network."""
    # Code to optimize the structure, fit it, and return the optimized model
    # Use hill climbing to learn structure
    model = bn.structure_learning.fit(df, methodtype='hc')
    
    # Fit parameters
    model = bn.parameter_learning.fit(model, df, methodtype='maximumlikelihood')
    
    return model
    pass

def save_model(fname, model):
    """Save the model to a file using pickle."""
    with open(fname, 'wb') as f:
        pickle.dump(model, f)
    pass

def evaluate(model_name, val_df):
    """Load and evaluate the specified model."""
    with open(f"{model_name}.pkl", 'rb') as f:
        model = pickle.load(f)
        correct_predictions, total_cases, accuracy = test_model(model, val_df)
        print(f"Total Test Cases: {total_cases}")
        print(f"Total Correct Predictions: {correct_predictions} out of {total_cases}")
        print(f"Model accuracy on filtered test cases: {accuracy:.2f}%")

############
## Driver ##
############

def main():
    # Load data
    train_df, val_df = load_data()

    # Create and save base model
    base_model = make_network(train_df.copy())
    save_model("base_model.pkl", base_model)

    # Create and save pruned model
    pruned_network = make_pruned_network(train_df.copy())
    save_model("pruned_model.pkl", pruned_network)

    # Create and save optimized model
    optimized_network = make_optimized_network(train_df.copy())
    save_model("optimized_model.pkl", optimized_network)

    # Evaluate all models on the validation set
    evaluate("base_model", val_df)
    evaluate("pruned_model", val_df)
    evaluate("optimized_model", val_df)

    print("[+] Done")

if __name__ == "__main__":
    main()

