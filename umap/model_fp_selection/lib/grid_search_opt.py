from itertools import product


def grid_search_opt(df, logger, param_grid, objective, model_class=None):
    """
    Deterministic grid search over hyperparameters.
    """

    keys = list(param_grid.keys())
    values = list(param_grid.values())

    best_rmse = float("inf")
    best_params = None

    for combination in product(*values):
        params = dict(zip(keys, combination))

        rmse = objective(
            args_dict=params,
            #logger=logger,
            data=df,
            model_class=model_class
        )

        if logger:
            logger.info(f"Tested params: {params} → RMSE: {rmse:.4f}")

        if rmse < best_rmse:
            best_rmse = rmse
            best_params = params

    if logger:
        logger.info(f"Best RMSE: {best_rmse:.4f}")

    return best_params
