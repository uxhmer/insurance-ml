import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
import phik
from scipy.stats import sem, t, norm
from statsmodels.stats.outliers_influence import (
    variance_inflation_factor as vif
    )
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import GridSearchCV, BaseCrossValidator
from sklearn.base import ClassifierMixin
from sklearn.compose import ColumnTransformer


def encode(data: pd.DataFrame) -> pd.DataFrame:
    """Encodes variables into [0, 1] values.
    Specific binary variables are expected to be passed as data.

    :param data: A dataframe containing binary features 
    with specific values.
    :return: A dataframe of uniformly encoded binary features."""
    data = data.copy()
    for feature in data.columns:
        data[feature] = data[feature].map(
            lambda x: 1 if x in [1, "Yes", "Government Sector"] else 0
            )
    return data


def stats_overview(
        data: pd.DataFrame, 
        feature: str, 
        global_palette: str
        ) -> pd.DataFrame:
    """Outputs statistical description and three distribution 
    plots of the provided feature in the data.
    
    :param data: The analysed dataframe.
    :param feature: The variable of interest in the data.
    :param global_palette: Color palette to use in the plots.
    :return: A statistical description of the feature of interest."""
    bin_size = min(len(data[feature].unique()),15)
    plt.figure(figsize=(12,3))
    plt.subplot(1,3,1)
    sns.histplot(data, x=feature, bins=bin_size)
    plt.subplot(1,3,2)
    sns.kdeplot(data, x=feature, hue="TravelInsurance", palette=[
        sns.color_palette(global_palette)[0], 
        sns.color_palette(global_palette)[-1]
        ]);
    plt.subplot(1,3,3)
    sns.histplot(data, 
                 x=feature, 
                 hue="TravelInsurance", 
                 bins=bin_size, 
                 multiple="fill", 
                 palette=[
                     sns.color_palette(global_palette)[0], 
                     sns.color_palette(global_palette)[-1]
                     ]
                     )
    plt.suptitle(f"{feature} distributions by insurance class");
    plt.tight_layout();
    return data[
        [feature,"TravelInsurance"]
        ].groupby("TravelInsurance").describe()


def get_ci_mean(data: pd.Series) -> tuple:
    """Calculates mean confidence interval based on t-statistic 
    and 95% confidence.
    
    :param data: A series of values to calculate the mean and CI from.
    return: A tuple containing the margin of error, 
    lower and upper bound of the confidence interval."""
    n = len(data)
    mean = np.mean(data)
    se = sem(data)
    confidence = 0.95
    t_val = t.ppf((1 + confidence) / 2, df=n-1)
    moe = t_val * se
    lower = mean - moe
    upper = mean + moe
    return moe, lower, upper


def get_ci_proportion(data: pd.Series) -> tuple:
    """Calculates proportion confidence interval based on 
    z-score and 95% confidence for a one-tailed z-test.
    
    :param data: A series of binary values to 
    calculate the proportion of CI from.
    :return: A tuple containing the margin of error and 
    estimated proportion."""
    n = len(data)
    yes = len(data[data == 1])
    p_hat = yes/n
    z = norm.ppf(0.975)
    se = np.sqrt(p_hat * (1 - p_hat) / n)
    margin = z * se
    return margin, p_hat


def binary_var_proportions(data: pd.DataFrame) -> None:
    """Creates a plot for each binary variable in the provided 
    data to visualise proportions and proportion confidence
    intervals of customers who bought travel insurance per class 
    per feature.
    
    :param data: A dataframe containing only binary features.
    :return: None"""
    _, axes = plt.subplots(nrows=2, ncols=3, figsize=(12, 3), sharex=True)
    plt.subplots_adjust(top=0.8)
    axes = axes.flatten()
    
    for i, (feature, ax) in enumerate(
        zip(data.columns[:-1].sort_values(), axes)
        ):
        counts = pd.crosstab(data[feature], data["TravelInsurance"])
        total = counts.sum(axis=1)
        success = counts[1]
        proportion = success / total
        margin = (
            norm.ppf(0.975) * 
            np.sqrt(proportion * 
                    (1 - proportion) / total)
                    )
    
        proportion.plot(kind="barh", xerr=margin, ax=ax, alpha=0.7, 
                        legend=False, capsize=5)
    
        for rect, val, err in zip(ax.patches, proportion, margin):
            width = rect.get_width() + err
            y = rect.get_y() + rect.get_height() / 2
            ax.text(width + 0.02, y, f"{val*100:.1f} ± {err*100:.1f}%", 
                    va="center", fontsize=11)
    
            ax.set_title(feature)
            ax.set_ylabel("")
            ax.set_xticks([])
            ax.grid(False)
        plt.grid(False)
        axes[-1].axis("off")
        plt.tight_layout()
        plt.subplots_adjust(top=0.83);


def outlier_mask(data: pd.DataFrame, feature: str) -> pd.DataFrame:
    """Calculates outliers using IQR in a particular feature 
    and returns an outlier mask as dataframe.
    
    :param data: The dataframe containing analysed data
    :param feature: A string referring to the variable in
    the data for which outliers will be checked
    :return: An outlier mask"""
    q1 = data[feature].quantile(0.25)
    q3 = data[feature].quantile(0.75)
    iqr = q3 - q1
    mask = (
        (data[feature] > (q3 + iqr * 1.5)) 
        | (data[feature] < (q1 - iqr * 1.5))
    )
    return data[mask]


def check_outliers(data: pd.DataFrame, feature: str) -> pd.DataFrame:
    """Returns outliers, or prints out a message if there are no
    outliers in a feature.

    :param data: The dataframe containing analysed data
    :param feature: A string referring to the variable in
    the data for which outliers will be checked
    :return: A dataframe containing outliers in the specified feature.
    """
    outliers = (outlier_mask(data, feature).sort_values(by=feature))
    if outliers.size == 0:
        return None
    else:
        return outliers
    

def corr_heatmap(data, method):
    if method == "phik":
        corr_matrix = data.phik_matrix(interval_cols="").round(2)
    else:
        corr_matrix = data.corr(method=method).round(2)
    triu_mask = np.zeros(corr_matrix.shape)
    triu_mask[np.triu_indices(corr_matrix.shape[0])] = np.nan
    corr_matrix += triu_mask
    return corr_matrix


def check_vif(data: pd.DataFrame, palette: str, sort: bool=True) -> None:
    """Calculates variance inflation factors and plots them.
    
    :param data: The dataframe containing variables 
    whose VIF scores will be calculated
    :param palette: Str representation of a color palette 
    to be used in the plot
    :param sort: A boolean indicating if the variables 
    in the plot should be sorted
    by VIF score or not"""
    X = sm.add_constant(data)
    vif_df = pd.DataFrame()
    vif_df["Feature"] = X.columns
    vif_df["VIF"] = [vif(X.values, i) for i in range(X.shape[1])]
    if sort == True:
        vifs = vif_df[1:].sort_values(
            by="VIF", 
            ascending=False
            ).set_index("Feature").T
    else:
        vifs = vif_df[1:].set_index("Feature").T    
    ax = sns.barplot(vifs, orient="h", palette=palette)
    [ax.bar_label(container, fmt=" %g") for container in ax.containers]
    plt.title("Variance inflation factors")
    ax.grid(False)
    ax.tick_params(axis="x", which="both", labelbottom=False);


def grid_cv(
        model: ClassifierMixin, 
        param_grid: dict, 
        data: pd.DataFrame, 
        cv: BaseCrossValidator, 
        preprocessor: ColumnTransformer, 
        palette: str="viridis", 
        plotting: bool=True
        ) -> None:
    """Performs hyperparameter tuning using `GridSearchCV`. 
    Plots a heatmap with recall scores, and prints out the 
    best combination of hyperparameters.
    
    :param model: Classifier to be used in the pipeline.
    :param param_grid: A disctionary of hyperparameters 
    and their values to be tuned.
    :param data: The analysed dataframe.
    :param cv: A cross-validation algorithm to be used in 
    `GridSearchCV`.
    :param preprocessor: A transformer to be used in the 
    pipeline.
    :param palette: The color palette to be used in the 
    grid search heatmap.
    :param plotting: Defines whether a heatmap should be 
    plotted or not, `True` by default.
    :return: None
    """
    pipe = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("smote", SMOTE(random_state=54)),
        ("classifier", model),
    ])
    
    cv = GridSearchCV(pipe, 
                  param_grid=param_grid, 
                  cv=cv, 
                  scoring="recall",
                  n_jobs=-1,
                  return_train_score=True)
    with warnings.catch_warnings(): 
        warnings.simplefilter("ignore", category=FutureWarning)
        cv.fit(data.drop(columns=["TravelInsurance"]), data["TravelInsurance"])
    
    print("Best Recall:", round(cv.best_score_, 5))
    print("Parameters:")
    for param, value in cv.best_params_.items():
        print(f"   {param}: {value}")
    print("")

    if plotting == True:
        results = pd.DataFrame(cv.cv_results_)
        pivot = results.pivot(
            index=f"param_{list(param_grid.keys())[0]}", 
            columns=iterate_params(param_grid), 
            values="mean_test_score")
        
        sns.heatmap(pivot, annot=True, fmt=".4f", cmap=palette)
        plt.title("Grid search recall results");


def iterate_params(param_grid: dict) -> list:
    """A helper function that provides a list of parameter names.
    
    :param param_grid: A disctionary of hyperparameters and their values.
    :return: A list of hyperparameter names (keys in the provided 
    dictionary)."""
    keys = list(param_grid.keys())
    return [f"param_{keys[i]}" for i in list(range(1, len(param_grid)))]