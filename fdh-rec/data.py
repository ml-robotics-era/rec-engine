"""
Contains functions that preprocess data loaded from main.
"""

import pandas as pd
import numpy as np
from numpy.random import default_rng
from typing import Dict, Tuple
from utils import (
    PRICE,
    SHRINK_SIMILAR_OWNERSHIP,
    SEX,
    OVERCAT,
    GOAL_PRIORITY,
    N_RECOMMEND,
    N_REQUEST,
    N_PURCHASE,
)


def preprocess(
    prod_features: pd.DataFrame,
    prod_goals: pd.DataFrame,
    prod_issues: pd.DataFrame,
    prod_subcats: pd.DataFrame,
    user_goals: pd.DataFrame,
    user_issues: pd.DataFrame,
    user_subcats: pd.DataFrame,
    global_params: Dict,
    n_goals: int,
    n_issues: int,
    goal_ranking: Dict,
    issues_ranking: Dict,
) -> Tuple[pd.DataFrame]:
    """
    Cleans data and weighs product goals and issues as well as subcategoies of
    products users already own according to predefined importance metrics, in
    order to generate informative inputs for cosine similarity matching, which
    computes a similarity score between the user vector and the product vector.

    Parameters:
      n_goals: Total number of goals.
      n_issues: Total number of issues.
      goal_ranking: Dictionary of relative importance of goals.
      issues_ranking: Dictionary of relative importance of issues.
      global_params: Dictionary of miscellaneous weights.
      Relevant DataFrames.

    Returns:
      Tuple of:
        user_cos_frame: Processed goals, issues, and subcats of users joined together.
        prod_cos_frame: Processed goals, issues, and subcats of products joined together.
        Other processed DataFrames.
    """
    # Fill products that have null prices with the mean price of all products.
    prod_features[PRICE] = prod_features.price.fillna(prod_features.price.mean())

    # Discourage similar ownership by turning it negative.
    user_subcats = user_subcats.applymap(
        lambda x: x * -global_params[SHRINK_SIMILAR_OWNERSHIP]
    )

    # Weigh goals and issues according to predefined rankings.
    for goal in prod_goals:
        prod_goals[goal] *= global_params[GOAL_PRIORITY] * goal_ranking[goal] / n_goals
    for issue in prod_issues:
        prod_issues[issue] *= issues_ranking[issue] / n_issues

    # Convert comma-separated strs to list.
    prod_features[SEX] = prod_features.sex.str.split(",")
    prod_features[OVERCAT] = prod_features.overcat.str.split(",")

    # Join goals, issues, and subcats to form the arguments passed into cosine similarity.
    prod_cos_frame = prod_goals.join(prod_issues).join(prod_subcats)
    user_cos_frame = user_goals.join(user_issues).join(user_subcats)
    return (
        user_subcats,
        prod_features,
        prod_goals,
        prod_issues,
        user_cos_frame,
        prod_cos_frame,
    )


def fill_indices(
    prod_features: pd.DataFrame,
    prod_goals: pd.DataFrame,
    prod_issues: pd.DataFrame,
    prod_subcats: pd.DataFrame,
    user_features: pd.DataFrame,
    user_goals: pd.DataFrame,
    user_issues: pd.DataFrame,
    user_subcats: pd.DataFrame,
) -> Tuple[pd.DataFrame]:
    """
    TEMPORARY: Generates random id's for products and users. Remove during deployment.

    Parameters: Dataframes that are lacking ids.
    Returns: Tuple of DataFrames with id's filled in.
    """
    # Generate random numbers for pids.
    rng = default_rng()
    pids = rng.choice(
        np.arange(10000, 100000), size=len(prod_features), replace=False
    )  # For each product, choose one id from 10000 to 99999 without replacement.
    prod_features.index = pids
    prod_goals.index = pids
    prod_issues.index = pids
    prod_subcats.index = pids
    prod_features[N_RECOMMEND] = np.random.randint(5, size=len(prod_features))
    prod_features[N_REQUEST] = np.random.randint(5, size=len(prod_features))
    prod_features[N_PURCHASE] = np.random.randint(5, size=len(prod_features))
    # Generate random numbers for uids.
    uids = rng.choice(
        np.arange(10000, 100000), size=len(user_features), replace=False
    )  # For each user, choose one id from 10000 to 99999 without replacement.
    user_features.index = uids
    user_goals.index = uids
    user_issues.index = uids
    user_subcats.index = uids
    return (
        prod_features,
        prod_goals,
        prod_issues,
        prod_subcats,
        user_features,
        user_goals,
        user_issues,
        user_subcats,
    )
