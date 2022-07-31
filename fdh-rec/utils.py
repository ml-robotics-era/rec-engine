"""
Asserts that base conditions hold; prints matching checks for debugging.
"""

import numpy as np
import pandas as pd

RECOMMENDED = "Recommended"
PURCHASED = "Purchased"
ZERO = 0
OTHER = ""
TEST_PRINT_K = 5
PRICE = "price"
SHRINK_SIMILAR_OWNERSHIP = "shrink_similar_ownership"
SEX = "sex"
OVERCAT = "overcat"
GOAL_PRIORITY = "goal_priority"
N_RECOMMEND = "n_recommend"
N_REQUEST = "n_request"
N_PURCHASE = "n_purchase"

NATURAL = "natural"
SIMILARITY_BOOST = "similarity_boost"
DEVICE = "device"
VIRTUAL = "virtual"
SUBTRACT_VIRTUAL = "subtract_virtual"
ADD_DEVICE = "add_device"
POPULARITY = "popularity"
SHRINK_CAC = "shrink_cac"
DEMO_PRIORITY = "demo_priority"


def test(
    uid: int,
    prod_ids: pd.Series,
    wellness_flag: bool,
    prod_goals: pd.DataFrame,
    prod_issues: pd.DataFrame,
    prod_subcats: pd.DataFrame,
    status: pd.DataFrame,
    user_goals: pd.DataFrame,
    user_issues: pd.DataFrame,
    user_subcats: pd.DataFrame,
    user_features: pd.DataFrame,
    prod_features: pd.DataFrame,
) -> None:
    """
    Checks the correctness and quality of recommendations.

    Parameters:
      uid: User id.
      prod_ids: Series of product id's that the matching function outputs for the user.
      wellness_flag: Boolean indicating whether the user is only being recommended
        wellness products.
      Relevant DataFrames.

    Prints: Vectors where index k is boolean of whether recommended product k satisfies a
    demographic condition.
    Returns nothing.
    """
    goal_match_vec = (
        prod_goals.loc[prod_ids].to_numpy() @ user_goals.loc[uid].to_numpy()
    )
    issue_match_vec = (
        prod_issues.loc[prod_ids].to_numpy() @ user_issues.loc[uid].to_numpy()
    )
    subcat_overlap_vec = (
        prod_subcats.loc[prod_ids].to_numpy() @ user_subcats.loc[uid].to_numpy()
    )

    past_prods = status[status.uid == uid]
    used_pids = set(
        past_prods[
            (past_prods.status == RECOMMENDED) | (past_prods.status == PURCHASED)
        ].pid
    )

    # Constraint 1 is user cannot have already been recommended or bought this product.
    assert (
        not set(prod_ids).intersection(used_pids),
        "User cannot have already been recommended or bought this product.",
    )

    if (
        (user_goals.loc[uid] != ZERO).any() or (user_issues.loc[uid] != ZERO).any()
    ) and not wellness_flag:  # has goals/issues
        # Constraint 3 is if user has goals or issues, must match at least one goal or issue.
        assert (
            np.any((goal_match_vec + issue_match_vec) > ZERO),
            "If user has goals or issues, must match at least one goal or issue.",
        )

    u = user_features.loc[uid].to_numpy()
    sex, age, weight, income_l, income_r, price_ceil, tier = u

    # Constraint 2 is sex must match.
    if sex == OTHER:
        sex_match_vec = np.ones(len(prod_ids))
    else:
        sex_match_vec = (
            prod_features.loc[prod_ids]
            .sex.apply(lambda x: sex in x)
            .astype("int")
            .to_numpy()
        )
    assert (
        np.all((sex_match_vec > ZERO)),
        "Targeted sex of the product did not match user's listed sex.",
    )

    # Logical_and checks if the user's demographic attributes are within range
    # of products' attributes.
    age_match_vec = np.logical_and(
        prod_features.loc[prod_ids].age_l.apply(
            lambda x: x <= age
        ),  # Vector of whether eligible products' age lower bounds are less than user age.
        prod_features.loc[prod_ids].age_r.apply(
            lambda x: x >= age
        ),  # Vector of whether eligible products' age upper bounds are greater than user age.
    ).astype("int")
    weight_match_vec = np.logical_and(
        prod_features.loc[prod_ids].weight_l.apply(lambda x: x <= weight),
        prod_features.loc[prod_ids].weight_r.apply(lambda x: x >= weight),
    ).astype("int")
    income_match_vec = np.logical_and(
        prod_features.loc[prod_ids].income_l.apply(lambda x: x <= income_l),
        prod_features.loc[prod_ids].income_r.apply(lambda x: x >= income_r),
    ).astype("int")
    net_pay = prod_features.loc[prod_ids].price - tier * prod_features.loc[prod_ids].cac
    price_vec = (net_pay <= price_ceil).astype("int")

    # TODO: Change from print to log statements.
    print("SUBCATS OVERLAP: ", subcat_overlap_vec)
    print("AGE MATCH: ", age_match_vec.to_numpy())
    print("WEIGHT MATCH: ", weight_match_vec.to_numpy())
    print("INCOME MATCH: ", income_match_vec.to_numpy())
    print("NOT TOO EXPENSIVE: ", price_vec.to_numpy())
