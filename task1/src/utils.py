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


def get_low_cv_features(data_mat, sample_meta, ft_columns, threshold=0.3):
    cv_df = calculate_cv_per_class(data_mat, sample_meta)

    # Reshape the DataFrame to long format
    cv_df_long = cv_df.melt(id_vars='class', value_vars=ft_columns, 
                    var_name='Feature', value_name='Value')

    cv_df_long = cv_df_long[(cv_df_long['class'].isin(['QC'])) & (cv_df_long['Value']<=threshold)]

    low_cv_features = list(set(cv_df_long['Feature'].values.tolist()))

    return low_cv_features


def get_high_detection_rate_features(data_mat, sample_meta, ft_columns, bio_samples, prevelance_threshold=0.7, detection_threshold=0):
    detection_rate = calculate_detection_rate(data_mat, sample_meta, threshold=detection_threshold)
    detection_rate_filtered = detection_rate.loc[detection_rate['class'].isin(bio_samples), ft_columns]  >= prevelance_threshold
    high_detection_rate_features = detection_rate_filtered.all()
    high_detection_rate_features = high_detection_rate_features.loc[high_detection_rate_features].index.tolist()
    return high_detection_rate_features


def get_high_mz_features(feat_meta, mz_threshold=500):
    return feat_meta[feat_meta['mz'] > mz_threshold]['feature'].values.tolist()




def pairwise_comparison(data_matrix, class1, class2):
    """
    Analyze features in the data matrix to compute Wilcoxon test p-values, fold-change, and effect size 
    between two specified classes.
    
    Parameters:
        data_matrix (pd.DataFrame): Data matrix (samples x features) with a 'class' column indicating class labels.
        class1 (str): The first class label.
        class2 (str): The second class label.
        
    Returns:
        pd.DataFrame: DataFrame with feature-level statistics: p-value, fold-change, and effect size.
    """

    from scipy.stats import ttest_ind
    import numpy as np
    import pandas as pd

    # Check if the specified classes exist in the 'class' column
    if class1 not in data_matrix['class'].unique() or class2 not in data_matrix['class'].unique():
        raise ValueError("Specified classes not found in the 'class' column.")
    
    results = []

    for feature in data_matrix.columns.drop('class'):
        # Split feature values by class
        group1 = data_matrix[data_matrix['class'] == class1][feature]
        group2 = data_matrix[data_matrix['class'] == class2][feature]
        
        # Wilcoxon rank-sum test (Mann-Whitney U test)
        #stat, p_value = mannwhitneyu(group1, group2, alternative='two-sided')
        stat, p_value = ttest_ind(group1, group2)

        #stat, p_value  = permutation_test((group1, group2), statistic='mean')

        # Fold-change (log2 of ratio of medians)
        median1, median2 = np.median(group1), np.median(group2)
        fold_change = np.log2(median2 / (median1 + 1e-6) ) #if median1 > 0 else np.nan
        
        # Append results
        results.append({
            'feature': feature,
            'p_value': p_value,
            'log2FC': fold_change,
            'Group': f'{class2}_vs_{class1}'
        })
    
    # Convert results to DataFrame
    results_df = pd.DataFrame(results)
    
    return results_df.sort_values(by='log2FC', ascending=False)


def volcano_plot(
    df, log2fc_col, padj_col, feature_col, group_col, significance_threshold=0.05, fc_threshold=1.0, save_path=None
):
    """
    Create a volcano plot to visualize log2FC vs -log10(adjusted p-value) with shapes based on groups and labels for significant features.
    
    Parameters:
        df (pd.DataFrame): DataFrame containing the data.
        log2fc_col (str): Column name for log2 fold change.
        padj_col (str): Column name for adjusted p-values.
        feature_col (str): Column name for feature labels.
        group_col (str): Column name for group labels (used for shapes).
        significance_threshold (float): Threshold for adjusted p-value to consider significance.
        fc_threshold (float): Threshold for log2FC to highlight features with high fold change.
        save_path (str, optional): File path to save the plot. If None, the plot is only displayed.
        
    Returns:
        None: Displays and/or saves the volcano plot.
    """
    import matplotlib.pyplot as plt
    import seaborn as sns
    import numpy as np

    # Calculate -log10(padj)
    df['-log10(padj)'] = -np.log10(df[padj_col])

    # Define colors and conditions
    conditions = [
        (df[padj_col] < significance_threshold) & (df[log2fc_col] > fc_threshold),
        (df[padj_col] < significance_threshold) & (df[log2fc_col] < -fc_threshold),
    ]
    labels = ['Upregulated', 'Downregulated']
    df['Significance'] = np.select(conditions, labels, default='Not Significant')

    # Create the plot
    plt.figure()
    sns.scatterplot(
        data=df,
        x=log2fc_col,
        y='-log10(padj)',
        hue='Significance',
        style=group_col,
        palette={'Upregulated': 'red', 'Downregulated': 'blue', 'Not Significant': 'grey'},
        alpha=0.7
    )

    # Add threshold lines
    plt.axhline(-np.log10(significance_threshold), color='black', linestyle='--', linewidth=1)
    plt.axvline(fc_threshold, color='black', linestyle='--', linewidth=1)
    plt.axvline(-fc_threshold, color='black', linestyle='--', linewidth=1)

    # Label significant features
    for _, row in df.iterrows():
        if row[padj_col] < significance_threshold and abs(row[log2fc_col]) > fc_threshold:
            plt.text(
                row[log2fc_col], row['-log10(padj)'], row[feature_col],
                fontsize=8, color='black', alpha=0.8
            )

    # Add labels and title
    plt.xlabel('log2 Fold Change')
    plt.ylabel('-log10(P adjusted)')
    plt.title('Volcano Plot')
    plt.grid(visible=True, linestyle='--', alpha=0.5)
    plt.tight_layout()

    # Save or display the plot
    if save_path:
        plt.savefig(save_path)

    plt.show()


def get_feature_importance(data_mat, selected_features, group1, group2, save=True):
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
    from sklearn.metrics import balanced_accuracy_score, classification_report
    from sklearn.inspection import permutation_importance
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns

    X = data_mat.loc[data_mat['class'].isin([group1, group2]), selected_features]
    y = data_mat.loc[data_mat['class'].isin([group1, group2]), 'class']

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, stratify=y, train_size=0.8, random_state=42, shuffle=True
    )

    # Hyperparameter tuning
    param_grid = {
        "n_estimators": [3, 4, 5],
        "max_depth": [None, 1, 2, 3, 4, 5],
        "min_samples_leaf": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        "max_features": ["sqrt", "log2", None],
        "bootstrap": [True, False],
        "criterion": ["gini", "entropy"]
    }
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    grid_search = GridSearchCV(
        RandomForestClassifier(random_state=42), 
        param_grid, 
        cv=cv, 
        n_jobs=-1, 
        scoring='balanced_accuracy'
    )
    grid_search.fit(X_train, y_train)
    best_params = grid_search.best_params_
    print(f'best_params: {best_params}')
    print("*"*20)

    # Train the best model
    best_model = RandomForestClassifier(**best_params, random_state=42)
    best_model.fit(X_train, y_train)

    # Evaluate the model
    y_pred = best_model.predict(X_test)
    balanced_acc = balanced_accuracy_score(y_test, y_pred)
    print(f"Balanced Accuracy: {balanced_acc:.3f}")
    print("*"*20)
    print('Classification report :\n', classification_report(y_test, y_pred,\
                                                            target_names=[group1, group2]))

    importance_result = best_model.feature_importances_
    importance_df = pd.DataFrame({
        "feature": selected_features,
        "importance_mean": importance_result
    }).sort_values(by="importance_mean", ascending=False)
    

    plt.figure()
    sns.barplot(importance_df, x='feature', y='importance_mean')
    plt.xticks(rotation=90)
    plt.xlabel(None)
    plt.title(f'feature importance for comparison {group1} vs {group2}')
    if save:
        plt.savefig(f"../results/figures/feature_importance_{group1}_vs_{group2}.png")
    plt.show()

    return importance_df




def make_boxplot_for_features(data_mat, importance_df, group1, group2, log_scale=True, save=True):
    import matplotlib.pyplot as plt
    import seaborn as sns

    selected_features = importance_df[importance_df['importance_mean']>0]['feature'].unique().tolist()
    data_mat2 = data_mat.loc[data_mat['class'].isin([group1, group2]), selected_features + ['class']]

    plt.figure()
    sns.boxplot(data_mat2.melt(id_vars='class', var_name='feature', value_name='intensity'),\
                hue='class', x='feature', y='intensity')
    if log_scale:
        plt.yscale('log')
        plt.ylabel('log (scaled intensity)')
    else:
        plt.ylabel('scaled intensity')
    
    if save:
        plt.savefig(f"../results/figures/boxplot_important_features_{group1}_vs_{group2}.png")
    
    plt.show()
    


    

