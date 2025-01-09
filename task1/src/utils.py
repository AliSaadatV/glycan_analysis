# Load YAML Configuration
def load_config(config_file):
    import yaml

    with open(config_file, 'r') as file:
        return yaml.safe_load(file)
    

# For reproducibility
def seed_everything(seed: int):
    import random, os
    import numpy as np

    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)


def calculate_cv_per_class(data_mat, sample_meta):
    """
    Calculate the Coefficient of Variation (CV) for each feature across all classes.
    
    Parameters:
        data_mat (pd.DataFrame): Data matrix containing feature intensities with samples as rows and features as columns.
        sample_meta (pd.DataFrame): Metadata containing sample information, including 'sample' and 'class' columns.
        
    Returns:
        pd.DataFrame: DataFrame with features as rows and classes as columns, containing CV values.
    """

    import pandas as pd
    # Merge data_mat with sample_meta to ensure classes are aligned
    data_with_meta = data_mat.merge(sample_meta[['sample', 'class']], on='sample', how='left')
    
    # Initialize an empty dictionary to store CVs
    cv_dict = {}
    
    # Iterate through each class
    for sample_class in sample_meta['class'].unique():
        # Subset data for the current class
        class_data = data_with_meta[data_with_meta['class'] == sample_class].iloc[:, 1:-1]
        
        # Calculate CV for each feature (std/mean)
        class_cv = class_data.std(axis=0) / class_data.mean(axis=0)
        
        # Add to dictionary
        cv_dict[sample_class] = class_cv
    
    # Create a DataFrame from the CV dictionary
    cv_df = pd.DataFrame(cv_dict)
    
    return cv_df.T.reset_index().rename(columns={'index': 'class'})

    
def calculate_d_ratio(data_mat, qc_samples, bio_samples):
    """
    Calculate D-Ratio for each feature, comparing technical variability (QC samples)
    to biological variability (biological samples).
    
    Parameters:
        data_mat (pd.DataFrame): Data matrix with samples as rows and features as columns.
        qc_samples (list): List of QC sample identifiers.
        bio_samples (list): List of biological sample identifiers.
    
    Returns:
        pd.DataFrame: DataFrame with features as rows and columns for QC CV, Bio CV, and D-Ratio.
    """
    import pandas as pd
    import numpy as np

    # Subset data for QC and biological samples
    qc_data = data_mat[data_mat['sample'].isin(qc_samples)].iloc[:, 1:]
    bio_data = data_mat[data_mat['sample'].isin(bio_samples)].iloc[:, 1:]
    
    # Calculate Coefficient of Variation (CV) for QC and biological samples
    qc_cv = qc_data.std(axis=0) / qc_data.mean(axis=0)
    bio_cv = bio_data.std(axis=0) / bio_data.mean(axis=0)
    
    # Calculate D-Ratio for each feature
    d_ratio = qc_data.std(axis=0) / np.sqrt(bio_data.std(axis=0)**2 + qc_data.std(axis=0)**2)
    
    # Combine results into a DataFrame
    variability_metrics = pd.DataFrame({
        'Feature': qc_data.columns,
        'QC_CV': qc_cv.values,
        'Bio_CV': bio_cv.values,
        'D_Ratio': d_ratio.values
    })
    
    return variability_metrics


def calculate_detection_rate(data_mat, sample_meta, groupby="class", threshold=0):
    """
    Computes the fraction of samples where a feature was detected.

    Parameters
    ----------
    groupby : str, List[str] or None
        If groupby is a column or a list of columns of sample metadata, the
        values of detection rate are computed on a per-group basis. If None, the detection
        rate is computed for all samples in the data.
    threshold : float
        Minimum value to consider a feature detected.

    Returns
    -------
    result : pd.DataFrame or pd.Series
        Detection rates computed per group or overall.
    """

    import pandas as pd

    def dr_per_feature(column, threshold):
        if isinstance(column, pd.DataFrame):
            n = column.shape[0]
        else:
            n = column.size

        dr = (column > threshold).sum().astype(int) / n
        return dr

    
    data_mat = data_mat.merge(sample_meta[['sample', 'class']], how='left')
    data_mat = data_mat.drop('sample', axis=1)

    if groupby is not None:

        # Compute detection rate per group
        result = data_mat.groupby(groupby).apply(
            lambda group: dr_per_feature(group, threshold=threshold)
        )
    else:
        # Compute detection rate for all samples
        result = data_mat.apply(
            lambda column: dr_per_feature(column, threshold=threshold)
        )

    return result.reset_index().rename(columns={'index': 'class'})




def correct_blank(
    data_matrix, samples_meta, sample_classes, blank_classes
):
    """
    Corrects the values of a data matrix based on specified classes and a correction factor.

    Parameters
    ----------
    data_matrix : pd.DataFrame
        DataFrame containing the data matrix (samples x features).
    samples_meta : pd.DataFrame
        DataFrame containing metadata for samples.
    sample_classes : list
        List of classes in `samples_meta` to be processed.
    blank_classes : list
        List of classes in `samples_meta` to be used for correction.

    Returns
    -------
    pd.DataFrame
        Data matrix with corrected values.
    """

    import pandas as pd

    # Identify samples and blanks using samples_meta
    samples = data_matrix[samples_meta['class'].isin(sample_classes)]
    blanks = data_matrix[samples_meta['class'].isin(blank_classes)]

    # Compute correction values (mean by default)
    correction = blanks.mean()

    # Correct samples
    corrected = samples - correction
    corrected[corrected < 0] = 0
    data_matrix.loc[samples_meta['class'].isin(sample_classes)] = corrected

    return data_matrix
