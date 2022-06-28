import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

#######################################################################################
# Processing helpers                                                                  #
#######################################################################################


def get_total_missing_count(df):
    """
    Count the number of missing (nan) values in a dataframe

    Args:
        df (dataframe): A pandas dataframe

    Returns:
        int: Total number of missing values
    """
    col_counts = []
    for col in df.columns:
        col_missing = df[col].isna().sum()
        col_counts.append(col_missing)
    total_counts = sum(col_counts)
    return total_counts

def convert_to_nan(df, summary):
    """
    Convert coded missing values to nan

    Args:
        df (dataframe): The dataframe with the missing values
        summary (dataframe): The dataframe with the codes of missing values

    Returns:
        dataframe: A new dataframe with converted missing values
    """
    df = df.copy()
    for i in range(summary.shape[0]):
        codes = summary["missing_or_unknown"].iloc[i]
        codes = codes.strip("[]").split(",")
        codes = [int(code) if code not in ["","X", "XX"] else code for code in codes]
        attribute = summary["attribute"].iloc[i]
        df[attribute] = df[attribute].replace(codes, np.nan)
    return df

def get_nan_summaries(df, sort=True):
    """
    Summarize nan counts and percentages per column

    Args:
        df (dataframe): A dataframe containing the missing values
        sort (bool, optional): Sort by nan percentage in ascending order. Defaults to True.

    Returns:
        dataframe: A summary dataframe containing column name, nan count, and nan percentage
    """
    col_names, counts, percentages = [], [], []
    for col in df.columns:
        count = df[col].isna().sum()
        percentage = (count / df.shape[0]) * 100
        col_names.append(col)
        counts.append(count)
        percentages.append(percentage)
    nan_summaries = pd.DataFrame({"attribute": col_names, "missing_count": counts, "missing_percentage": percentages})
    if sort:
        nan_summaries.sort_values(by=["missing_percentage"], inplace=True)
    return nan_summaries

def plot_nan_dist(df, num_bins=20):
    """
    Plot the distribution of the counts missing values

    Args:
        df (dataframe): A summarized dataframe of missing value counts per column
        num_bins (int, optional): The number of bins in the histogram. Defaults to 20.
    """
    nan_summaries = get_nan_summaries(df)
    fig, ax = plt.subplots(figsize=(12, 10))
    ax.hist(nan_summaries["missing_count"], bins= num_bins)  
    ax.set(xlabel="Number of missin values", ylabel="Number of columns", title="Distribution of missing value counts");
    
def plot_col_nan(df, summary_type="percentage"):
    """
    Plot a bar chart of each column and its corresponding nan count/percentage

    Args:
        df (dataframe): A summarized dataframe of missing value counts and percentages per column
        summary_type (str, optional): ["count", "percentage"]. Defaults to "percentage".
    """
    nan_summaries = get_nan_summaries(df).set_index("attribute")
    if summary_type == "count":
        nan_summaries["missing_count"].plot(kind="barh", color="C0", figsize=(12, 18));
        return
    nan_summaries["missing_percentage"].plot(kind="barh", color="C0", figsize=(12, 18));

def get_top_missing(df, top_n=20):
    """
    Find columns with missing value percentage equal to or above a certain threshold

    Args:
        df (dataframe): A summarized dataframe of missing value percentages per column
        top_n (int, optional): The percentage of missing value. Defaults to 20.

    Returns:
        list: A list of column names
    """
    nan_summaries = get_nan_summaries(df)
    mask = nan_summaries["missing_percentage"] >= top_n
    top_missing = nan_summaries[mask]["attribute"]
    return top_missing.tolist()

def add_missing_summarie(df):
    """
    Add two columns containg nan count and nan percentage in each row

    Args:
        df (dataframe): A pandas dataframe

    Returns:
        dataframe: A new dataframe with two new columns for nan count and nan percentage in each row
    """
    df = df.copy()
    df["missing_counts"] = df.isna().sum(axis=1)
    df["missing_percentages"] = (df["missing_counts"] / df.shape[1]) * 100
    return df

def get_cols_naperc(df, perc=0):
    """
    Get columns with nan percentage equal to a certain value

    Args:
        df (dataframe): A pandas dataframe
        perc (int, optional): The percentage of nan values in a column. Defaults to 0.

    Returns:
        list: A list of column names
    """
    nan_summaries = get_nan_summaries(df)
    mask = nan_summaries["missing_percentage"] == perc
    cols = nan_summaries[mask]["attribute"]
    return cols.tolist()

def compare_cols(df1, df2, cols):
    """
    Plot two bar charts comparing the value counts of the same column in different dataframe

    Args:
        df1 (dataframe): The first dataframe to compare
        df2 (dataframe): The second dataframe to compare
        cols (list): A list of column names to compare their value counts. Must be present in the the two dataframe
    """
    for i , col in enumerate(cols):
        fig = plt.figure(figsize=(14, 2))
        counts_1 = df1[col].value_counts()
        counts_2 = df2[col].value_counts()
        ax1 = fig.add_subplot(1, 2, 1)
        ax1.bar(counts_1.index, height=counts_1)
        ax1.set_xlabel(col)
        ax2 = fig.add_subplot(1, 2, 2)
        ax2.bar(counts_2.index, height=counts_2)
        ax2.set_xlabel(col)
        if i == 0:
            ax1.set_title("Subset with lower or no missing values")
            ax2.set_title("Subset with higher missing value")

def get_numeric_cols(df):
    """
    Get all columns of numeric type in a dataframe

    Args:
        df (dataframe): A pandas dataframe

    Returns:
        list: A list of column names
    """
    numeric_cols = []
    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            numeric_cols.append(col)
    return numeric_cols

def count_col_types(df, summary=None):
    """
    Count number of columns per data type in a dataframe

    Args:
        df (dataframe): A pandas dataframe
        summary (dataframe, optional): A summary dataframe with column name and corresponding data type. Defaults to None.

    Returns:
        dictionary: A dictionary of data types and columns count
    """
    if summary is not None:
        attribute_types = summary.groupby("type").count()["attribute"].to_dict()
        return attribute_types
    attribute_types = []
    for col in df.columns:
        col_dtype = df[col].dtype
        attribute_types.append(col_dtype)
    attribute_types = pd.Series(attribute_types).value_counts()
    return attribute_types.to_dict()    

def get_cat_features(df, summary):
    """
    Get the columns of categorical type in a dataframe provided a summary dataframe

    Args:
        df (dataframe): A pandas dataframe
        summary (dataframe): A summary of features and data types in the provided dataframe

    Returns:
        list: A list of column names
    """
    cat_features = []
    for attribute, attribute_type in zip(summary["attribute"], summary["type"]):
        if attribute in df.columns:
            if attribute_type == "categorical":
                cat_features.append(attribute)
    return cat_features

def get_binary_feats(df, cols):
    """
    Get columns with binary values from a list of categorical columns in a dataframe

    Args:
        df (dataframe): A pandas dataframe
        cols (list): A list of categorical columns in the provided dataframe

    Returns:
        list: A list of column names
    """
    binary_feats = []
    for col in cols:
        if col in df.columns:
            unique_vals = df[col].unique()
            if len(unique_vals) == 2:
                binary_feats.append(col)
    return binary_feats           

def get_mixed_feats(df, summary):
    """
    Get columns of mixed type in a dataframe provided a summary dataframe

    Args:
        df (dataframe): A pandas dataframe
        summary (dataframe): A summary of features and data types in the provided dataframe

    Returns:
        list: A list of column names
    """
    mixed_feats = []
    for attribute, attribute_type in zip(summary["attribute"], summary["type"]):
        if attribute in df.columns:
            if attribute_type == "mixed":
                mixed_feats.append(attribute)
    return mixed_feats


#######################################################################################
# Modeling helpers                                                                    #
#######################################################################################


def do_pca(n_components, scaled_data):
    '''
    Transforms data using PCA to create n_components, and provides back the results of the
    transformation.

    Args: 
        n_components  (int): the number of principal components to create
        scaled_data (dataframe): the data you would like to transform in scaled form

    Returns: 
        pca: the pca object created after fitting the data
        X_pca: the transformed X matrix with new number of components
    '''
    pca = PCA(n_components)
    X_pca = pca.fit_transform(scaled_data)
    return pca, X_pca

def scree_plot(pca):
    '''
    Creates a scree plot associated with the principal components 
    
    Args: 
        pca: the result of instantian of PCA in scikit learn
            
    '''
    num_components = len(pca.explained_variance_ratio_)
    ind = np.arange(num_components)
    vals = pca.explained_variance_ratio_
 
    plt.figure(figsize=(16, 8))
    ax = plt.subplot(111)
    cumvals = np.cumsum(vals)
    ax.bar(ind, vals)
    ax.plot(ind, cumvals)
    for i in range(num_components):
        ax.annotate(r"%s%%" % ((str(vals[i]*100)[:4])), (ind[i]+0.2, vals[i]), va="bottom", ha="center", fontsize=12)
 
    ax.xaxis.set_tick_params(width=0)
    ax.yaxis.set_tick_params(width=2, length=12)
 
    ax.set_xlabel("Principal Component")
    ax.set_ylabel("Variance Explained (%)")
    plt.title('Explained Variance Per Principal Component')
    
def get_feature_weights_by_component(df, pca, component):
    """
    Get the weight for each feature in your data according to a chosen principal component

    Args:
        df (dataframe): The dataset on which PCA was fitted
        pca : The PCA model
        component (int): A number specifying which component you want

    Returns:
        weights (series): A series of sorted weights 
    """
    weights = pca.components_[component - 1]
    weights = pd.Series(weights)
    weights.index = df.keys()
    weights = weights.sort_values(ascending=False) 
    return weights

def plot_feature_weights_by_component(weights, component):
    """
    Produces a bar plot of features and corresponding weights.

    Args:
        weights (series): The feature weights for a certain principal component
        component (int): The chosen principal component
    """
    fig, ax = plt.subplots(figsize = (16,8))
    weights.plot(ax=ax, kind = 'bar', color="C0")
    ax.set_title(f"Feature Weights for component number {component}")
    ax.set_ylabel("Feature Weights");
    
def get_interesting_features_by_components(df, pca, component, n):
    """
    Get 2n features with the highest weights both positive and negative sign and plot the result.

    Args:
        df (dataframe): The dataset on which PCA was fitted
        pca : The PCA model
        component (int): The chosen principal component
        n (int): The number of top features
    """
    weights = get_feature_weights_by_component(df, pca, component)
    high_n = weights.iloc[:n]
    print(f"Highest {n} weights are:\n")
    print(high_n, "\n")
    low_n = weights.iloc[-n:]
    print(f"lowest {n} weights are:\n")
    print(low_n.sort_values())
    top_feats = pd.concat([high_n, low_n])
    plot_feature_weights_by_component(top_feats, component)
    
def get_kmeans_score(data, center):
    """
    Get the computed score results of fitting KMeans models with specified K value.

    Args:
        data (dataframe): The dataset to fit the model to
        center (int): The specified K value

    Returns:
        score (int): The computed score
    """
    kmeans = KMeans(n_clusters=center)
    model = kmeans.fit(data)
    score = np.abs(model.score(data))
    return score

def get_clusters_summary(clusters_1, clusters_2):
    """
    Compare the result of tranforming two dataset using the produced cluster labels by generating the count and percentage of each cluster in each dataset.

    Args:
        clusters_1 (array): Cluster labels for the first dataset
        clusters_2 (array): Cluster labels for the second dataset

    Returns:
        cluster_data (dataframe): A pandas dataframe containing the result of the comparison
    """
    
    clusters_counts_1 = pd.Series(clusters_1).value_counts().sort_index()
    clusters_counts_2 = pd.Series(clusters_2).value_counts().sort_index()
    clusters_proportions_1 = clusters_counts_1 / clusters_1.shape[0]
    clusters_proportions_2 = clusters_counts_2 / clusters_2.shape[0]
    
    
    population_data = pd.DataFrame({
        "clusters": clusters_counts_1.index + 1,
        "counts": clusters_counts_1,
        "proportions": clusters_proportions_1,
        "category":"population",
    })
    
    customers_data = pd.DataFrame({
        "clusters": clusters_counts_2.index + 1,
        "counts": clusters_counts_2,
        "proportions": clusters_proportions_2,
        "category":"customers",
    })
    
    clusters_data = pd.concat([population_data, customers_data])
    clusters_data = clusters_data.sort_values("clusters")
    clusters_data.index = list(range(clusters_data.shape[0]))
    
    return clusters_data       