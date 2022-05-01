import itertools
from collections import defaultdict
from warnings import warn
import numpy as np
import pandas as pd
import lightgbm as lgb
import shap
import matplotlib.pyplot as plt
import seaborn as sns
from . import globalvar


class LGBMVisualizer:
    def __init__(
        self,
        params,
        data,
        target_column,
        weight_column=None,
        validation_column=None,
        validation_percent=0.3,
        n_trees=None,
        early_stopping_rounds=100,
        pc_data_for_interaction_detection=0.0,
        twofold_shap=True,
        model=None,
        main_class_for_multiclassification=None,
        seed=101,
    ):
        """
        Visualizer helps to find and visualize two-way interactions through SHAP values. Interactions
        are obtained by first creating an instance of this Class where the relevant model(s) are fit
        to the data and then calling `interactions()` method from this class. In the arguments below
        please pay close attention to the `double_shap` description. This will greatly impact performance
        and may impact inference of important interactions as they are different methods.

        Arguments
        ----------
        params : dict
            LightGBM parameters that to fit `df` on `y_col`
        data : pandas.DataFrame
            Data that has your predictor variables along with target & weight(optional) columns. Categorical variables are encoded so feel free to leave them as is.
        target_column : str
            Target variable column name 
        weight_column : str
            Observation weight column name. Only used to learn relationships between predictors & target variables. Not used when averaging the interactions values.
        validation_column : str
            Validation indicator column name which is optional. Should only have 2 unique values. Note that the resulting validation set is only used to find the optimal 
            number of trees , so if the `n_trees` value is specified this argument is redundant.
        validation_percent : float
            In the event that the user doesnt specify `validation_column` and `n_trees` , this argument is the percentage of data used for validation set.
        n_trees : int
            Total trees fit in LGBM to determine interactions between predictors and target variable. If set to None,
            a validation data (based on `validation_column` or `validation_percent`) will be used to find optimal number of trees.
        early_stopping_rounds : int
            Early stopping for determining the right number of number of trees to fit. Ignored if `n_trees` is provided
        pc_data_for_interaction_detection : float
            After a model is fit, SHAP values are calculated to find interactions between variables
            in the data. If you have large `data`, then the user can use this argument to limit the amount of data to be used for interaction detection since computing 
            SHAP values is the most time consuming process of this package.
            A value of 0.0 means no reduction in the resulting dataset size & a value of 0.9 means only using 10% of the data for interaction detection
        twofold_shap : bool
            Whether to use a modified SHAP logic (True) for interaction detection or use the default provided
            by the SHAP package (False). 
            If your data is large and `pc_data_for_interaction_detection` is low, setting this to False
            could take a very long time, where the modified custom SHAP method will be much more faster. 
            However, if your data is small or if use `pc_data_for_interaction_detection` to limit the no. of observations for interaction 
            detection, then your runtime will faster if you set this to False. 
            Another scenario when setting this to False is when initializing the `LGBMVisualizer` instance will take longer to run but plotting the interactions 
            will be much faster.
            If set to True,  initalizing of the `LGBMVisualizer` instance will be much faster but an additional model will be fit for every interaction variable.
            Most of the time, the runtime results are similar and only varies when the dataset it too large.
            Recommendation is to use both the methods to compare results if you have time & stick with one if the runtime difference is too large
        model : lightgbm.Booster
            Pre-trained lightgbm model can be passed here if you have large dataset & an already trained model at hand to reduce runtime.
        main_class_for_multiclassification : int or None
            If "objective" is "multiclass" in `params` then set `twofold_shap` to True since shap package doesnt support interaction
            values for "multiclass" LightGBM models. When `twofold_shap` is True then shap values are returned for each class. 
            This argument is the value of the main class that is used to determine the interaction analysis. 
            If None then the majority class is used.
        seed : int
            random.seed for reproducibility

        """
        np.random.seed(seed)
        if "multiclass" in params.get("objective", "undefined") and not twofold_shap:
            raise ValueError("SHAP package doesnt support interactions for multiclass problems. Instead set twofold_shap to True and run again for multiclass data"
                             )
        self.twofold_shap = twofold_shap

        N_SAMPLES = len(data)
        if validation_column:
            if data[validation_column].nunique() != 2:
                raise RuntimeError("More than 1 unique value in 'validation_column' detected. `validation_column` must only have 2 unique values..")
            else:
                validation_idxs = data[validation_column] == data[validation_column].max().values
                data = data.drop(validation_column, axis=1)
        else:
            validation_idxs = np.random.choice([True, False], size=N_SAMPLES, p=[validation_percent, 1 - validation_percent])

        self.target = data[target_column].values
        self.target_train = self.target[~validation_idxs]
        self.target_valid = self.target[validation_idxs]

        drop_list = [target_column]
        if weight_column:
            self.weight = data[weight_column].values
            self.weight_train = self.weight[~validation_idxs]
            self.weight_valid = self.weight[validation_idxs]
            drop_list.append(weight_column)
        else:
            self.weight = np.ones(N_SAMPLES)
            self.weight_train = np.ones(len(self.target_train))
            self.weight_valid = np.ones(len(self.target_valid))

        data = data.drop(drop_list, axis=1)

        categorical_columns = list(data.select_dtypes("object"))
        for col in categorical_columns:
            data[col] = data[col].astype("category")

        # saving data before categorical columns are encoded
        self.data = data.copy()

        if not twofold_shap:
            for col in categorical_columns:
                data[col] = data[col].cat.codes

        self.x_train = data.loc[~validation_idxs]
        self.x_valid = data.loc[validation_idxs]

        # Checking if pre-trained model specified by the user can be used to make predictions.
        if model: 
            self.model = model
            try:
                _ = self.model.predict(self.x_valid.head())
            except ValueError:
                raise ValueError("The model supplied is not supported by the data provided, could be because data has additional or missing columns, or even that format of some columns are different")
        else:
            self.model = self.training(
                params=params,
                early_stopping_rounds=early_stopping_rounds,
                n_trees=n_trees,
            )
        explainer = shap.TreeExplainer(self.model)


        if pc_data_for_interaction_detection > 0.0:
            sample_percent = 1 - pc_data_for_interaction_detection
            permutation_indices = np.random.permutation(N_SAMPLES)[: int(sample_percent * N_SAMPLES)]
            data = data.iloc[permutation_indices]
            self.data = self.data.iloc[permutation_indices]
        if twofold_shap:
            #The modified custom shap logic is only implemented when calculating interactions & visualising them
            shaps = explainer.shap_values(data)
            """
            If binary or multiclass objectives is specified, then the result is a list of length num_classes where each
            item is 2d shap values. 
            Binary objective is straightforward, we just grab one of the array's since the other is negative of that. 
            For multiclass, we need to get the shap values of the class of interest(ie. the one specified in the `main_class_for_multiclassification` argument) 
            """
            if isinstance(shaps, list):
                dominant_class = np.unique(self.target, return_counts=True)[1].argmax()
                if main_class_for_multiclassification:
                    # If no main class is specified then set it to the dominant class
                    try:
                        shaps = shaps[main_class_for_multiclassification]
                    except IndexError:    
                        warn("`main_class_for_multiclassification` does not match the dimension of the shap values. `main_class_for_multiclassification` has been set to the dominant class"
                        )
                        shaps = shaps[dominant_class]
                else:
                    shaps = shaps[dominant_class]
            self.shap_data = pd.DataFrame(shaps, columns=list(data), index=data.index)
        else:
            
            shaps = explainer.shap_interaction_values(data)
            data_columns = list(data)
            column_tuples = [item for item in itertools.product(data_columns, data_columns)]
            deduplicated_columns = sorted(set(tuple(sorted(t)) for t in column_tuples))

            complete_col_names = []
            for c1, c2 in column_tuples:
                if c1 == c2:
                    complete_col_names.append(c1)
                else:
                    complete_col_names.append(f"{c1}{globalvar.CONCAT_STRING}{c2}")
            retain_col_names = []
            main_effect_columns = []
            for c1, c2 in deduplicated_columns:
                if c1 == c2:
                    retain_col_names.append(c1)
                    main_effect_columns.append(c1)
                else:
                    retain_col_names.append(f"{c1}{globalvar.CONCAT_STRING}{c2}")

            self.shap_data = pd.DataFrame(
                shaps.reshape(shaps.shape[0], -1),
                columns=complete_col_names,
                index=data.index,
            )
            self.shap_data = self.shap_data[retain_col_names]

            # interaction effect is split in the matrix, so multiplying by 2
            for col in retain_col_names:
                if globalvar.CONCAT_STRING in col:
                    self.shap_data[col] = self.shap_data[col] * 2

        # using `gain` from feature importance to make sure its consistent between twofold_shap = True or False
        self.feature_importance = self.feature_imp_gain(self.model)

    def feature_imp_shap(self, shap_data, return_dict=True):
        feat_imp = shap_data.abs().sum(axis=0).sort_values(ascending=False)
        if return_dict:
            return feat_imp.to_dict()
        else:
            return feat_imp

    def feature_imp_gain(self, model, return_dict=True):
        feat_imp = {
            column_name: gain_importance
            for column_name, gain_importance in zip(
                model.feature_name(), model.feature_importance("gain")
            )
        }
        result = sorted(feat_imp.items(), key=lambda kv: kv[1], reverse=True)
        if return_dict:
            return {column_name: gain_importance for column_name, gain_importance in result}
        else:
            return result

    def training(self, params, early_stopping_rounds, n_trees):

        params["verbosity"] = -1
        target = np.concatenate([self.target_train, self.target_valid])
        weight = np.concatenate([self.weight_train, self.weight_valid])
        lgbm_data = lgb.Dataset(pd.concat([self.x_train, self.x_valid], axis=0), target, weight=weight)

        if n_trees:
            mod = lgb.train(
                params,
                lgbm_data,
                n_trees,
            )
        else:
            lgb_data_train = lgb.Dataset(self.x_train, self.target_train, weight=self.weight_train)
            lgb_data_valid = lgb.Dataset(self.x_valid, self.target_valid, weight=self.weight_valid)
            mod = lgb.train(
                params,
                lgb_data_train,
                5000,
                valid_sets=[lgb_data_valid],
                valid_names=["val"],
                early_stopping_rounds=early_stopping_rounds,
                verbose_eval=False,
            )
            mod = lgb.train(
                params,
                lgbm_data,
                mod.best_iteration,
            )
        return mod

    def custom_shap_target(self, var):
        variable_bin = f"{var}_bin"
        variable_average_shap = f"{var}_avg_shap"
        variable_df = self.shap_data[[var]].copy()

        if globalvar.BIN_OVER_ROLLING:
            total_unique = self.data[var].nunique()
            if total_unique <= globalvar.MAXIMUM_BINS:
                variable_df[variable_bin] = self.data[var]
            else:
                try:
                    variable_df[variable_bin] = pd.qcut(
                        self.data[var],
                        q=globalvar.MAXIMUM_BINS,
                        labels=False,
                        duplicates="drop",
                    )
                except (ValueError, TypeError):
                    variable_df[variable_bin] = self.data[var]
            average_shap = variable_df.groupby(variable_bin)[var].mean()
            variable_df[variable_average_shap] = variable_df[variable_bin].map(average_shap)
            
            # In cases where variable_bin is categorical, the mapping sets 
            # variable_average_shap to categorical as well, so converting it to float
            variable_df[variable_average_shap] = variable_df[variable_average_shap].astype(np.float32)

        else:
            variable_df[variable_bin] = self.data[var]
            total_unique = variable_df[variable_bin].nunique()
            if (
                total_unique > globalvar.ROLLING_THRESHOLD
                and variable_df[variable_bin].dtype.name != "category"
            ):
                variable_df = variable_df.sort_values(by=variable_bin)
                variable_df[variable_average_shap] = (
                    variable_df[var]
                    .rolling(
                        window=globalvar.ROLLING_THRESHOLD,
                        min_periods=1,
                        center=True,
                    )
                    .mean()
                )
            else:
                average_shap = variable_df.groupby(variable_bin)[var].mean()
                variable_df[variable_average_shap] = variable_df[variable_bin].map(average_shap)
                variable_df[variable_average_shap] = variable_df[variable_average_shap].astype(np.float32)

        return (variable_df[var] - variable_df[variable_average_shap]).values

    def shap_calculations(self, col_name, lgbm_params=None, n_trees_second_model=500, twofold_shap=True):
        if twofold_shap:
            if lgbm_params is None:
                lgbm_params = globalvar.DEFAULT_PARAMS
            target = self.custom_shap_target(col_name)
            lgb_train = lgb.Dataset(self.data, target)
            secondary_model = lgb.train(lgbm_params, lgb_train, n_trees_second_model)
            explainer = shap.TreeExplainer(secondary_model)
            shaps = explainer.shap_values(self.data)
            final_shap_data = pd.DataFrame(
                shaps, columns=list(self.data), index=self.data.index
            )
            # Setting the final shap value of the variable as  Oiginal shap - vertical dispersion + 
            # dispersion explained by the variable itself
            final_shap_data[col_name] = self.shap_data[col_name] - target + final_shap_data[col_name]
        else:  
            variable_list = [col for col in list(self.shap_data) if col_name in col]
            final_shap_data = self.shap_data[variable_list].copy()
            shap_col_names = []
            for col in list(final_shap_data):
                splits = col.split(globalvar.CONCAT_STRING)
                if len(splits) == 1:
                    shap_col_names.append(splits[0])
                else:
                    if splits[0] == col_name:
                        shap_col_names.append(splits[1])
                    else:
                        shap_col_names.append(splits[0])
            final_shap_data.columns = shap_col_names
        return final_shap_data

    def lineplot(
        self,
        data,
        title,
        xlabel,
        ylabel,
        figsize,
        style=globalvar.STYLE,
        lw=globalvar.LINEWIDTH,
        markersize=globalvar.MARKER_SIZE / 4,
        xtick_rotation=globalvar.XTICK_ROTATION,
    ):
        data_type = data.index.dtype.name
        ax = data.plot(
            style=style,
            title=title,
            xlabel=xlabel,
            ylabel=ylabel,
            figsize=figsize,
            lw=lw,
            markersize=markersize,
        )
        if data_type == "object" or data_type == "category":
            plt.xticks(np.arange(len(data)), data.index, rotation=xtick_rotation)
        plt.show()
        plt.close()
        return ax

    def scatterplot(
        self, x, y, title, xlabel, ylabel, hue=None, size=globalvar.MARKER_SIZE
    ):
        data_type = x.dtype.name
        sns.scatterplot(
            x=x,
            y=y,
            hue=hue,
            s=size,
        )
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)

        if data_type == "object" or data_type == "category":
            plt.xticks(np.arange(len(x)), x.index, rotation=globalvar.XTICK_ROTATION)
        plt.show()
        plt.close()

    def find_interactions(
        self,
        variables=None,
        n_variables_explore=5,
        total_interactions_per_var=2,
        total_bins_for_num_cols=3,
        params_for_second_model=None,
        n_trees_second_model=500,
        scatter_plot=False,
        moving_avg_window_size=5,
        moving_avg_min_periods=1,
        custom_bin_for_interactionplot=None,
        remove_var_morethan_n_levels=10,
        figsize=(8.5, 6),
        impute_missing_with=-1,
    ):
        """
        Flexible interface for detecting and plotting potential interaction candidates

        Arguments
        ----------
        variables : None, str, list of str's, tuple of str's, or list of tuples or str's
            Predictor variables in `df` for interaction detection.
            > If None: Interactions with the `n_variables_explore` most important features are plotted
            > If a str: The `total_interactions_per_var` most important interactions involving the variable represented
            by the str are plotted
            > If a list of str's': The `total_interactions_per_var` most important interactions involving all the variables
            in the list are plotted
            > If a tuple of str's: A single interaction between the first item in the tuple and the second item
            of the tuple will be plotted. Note the tuple must be of length two and the first variable in the tuple
            will be on the x-axis
            > If a list of tuples of str's: Multiple specific interactions between the first item and the second
            item in each of the tuples will be plotted
        n_variables_explore : int
            If `variables` is None, this determines how many of the most important features in the model
            to return interactions for
        total_interactions_per_var : int
            Unless tuple(s) are provided to `variables` this is how many interactions to return for each
            variable
        total_bins_for_num_cols : int
            If the secodary variable that determines the color of the markers or lines is continuous, this determines
            how many equally-sized bins so that the plot can be more easily interpreted. Note that the actual number of
            bins might vary from this if the secondary variable has missing values in which case that always gets its own color for
            plotting since missings may be very important
        params_for_second_model : dict
            If `double_shap` is True, this is the parameters for the model that attempts to explain the vertical
            dispersion in SHAP values for the interaction target of interest. If None, defaults are provided which
            tend to work pretty well
        n_trees_second_model : int
            If `double_shap` is True, this is the number of trees for the model that attempts to explain the vertical
            dispersion in SHAP values for the interaction target of interest.
        scatter_plot : bool
            True will produce a scatter plot and False will produce a line plot where the individual data points are
            averaged to produce the line
        moving_avg_window_size : int
            If the variable plotted along the x-axis is continuous, computing a moving average with a sufficient
            `window size` will aid in interpretation
        moving_avg_min_periods : int
            When computing the moving average for a continuous variable plotted along the x-axis, this is the minimum
            number of periods in the window required to produce a value. Otherwise it will produce a NaN
        custom_bin_for_interactionplot : sequence of scalars
            Defines the bin edges allowing for non-uniform width of the variable whose effect is going to be plotted
            across the levels of the main variable. This option is only valid if a single interaction is requested
            by passing a tuple of two variables to `variables` and the variable is non-categorical
        remove_var_morethan_n_levels : int
            If a high-cardinality categorical variable's effects are plotted as the lines, the plot can be too messy
            to interpret (e.g. lots of lines and huge legend). This argument prevents interaction recommendations
            from being displayed if a categorical variable's effects are to be plotted as lines and that categorical
            variable as more than this number of unique levels
        figsize : tuple
            The size of the figures to be plotted: (width, height)
        impute_missing_with : float
            If the variable to be plotted on the x-axis has missing values and is numeric, then this determines what
            value on the x-axis is represented by missingness

        """
        sns.set(rc={"figure.figsize": figsize})
        n_bins = total_bins_for_num_cols
        y_label = "Linear predictor effect"

        is_tuple = isinstance(variables, tuple)
        if custom_bin_for_interactionplot and not is_tuple:
            raise RuntimeError("Since `custom_bin_for_interactionplot` argument is used, `variables` should be tuple of 2 column names")

        def _tuple_length_check(t):
            if len(t) != 2:
                raise RuntimeError(
                    "If you intend to pass tuple to `variables` argument, it must contain 2 column names"
                )

        variable_dict = defaultdict(list)
        if isinstance(variables, str):
            variable_dict[variables]
        elif isinstance(variables, list):
            if isinstance(variables[0], tuple):
                _ = [_tuple_length_check(tup) for tup in variables]
                for t in variables:
                    v1, v2 = t
                    variable_dict[v1].append(v2)
            elif isinstance(variables[0], str):
                # Initialising empty list
                [variable_dict[var] for var in variables]
            else:
                raise RuntimeError("`variables` provided with unsupported format")
        elif isinstance(variables, tuple):
            _tuple_length_check(variables)
            v1, v2 = variables[0], variables[1]
            variable_dict[v1].append(v2)
            if custom_bin_for_interactionplot:
                if self.data[v2].dtype.name == "category":
                    raise TypeError(
                        "The interaction variable specified is categorical. `custom_bin_for_interactionplot` doesnt apply for categorical"
                    )
                n_bins = custom_bin_for_interactionplot
        elif variables is None:
            variables = [
                key
                for i, key in enumerate(self.feature_importance.keys())
                if i < n_variables_explore
            ]
            [variable_dict[var] for var in variables]
        for var_interact1, var_interact2 in variable_dict.items():
            original_variable = f"o_{var_interact1}"
            final_shap_df = self.shap_calculations(
                var_interact1, params_for_second_model, n_trees_second_model, self.twofold_shap
            )
            # if var_interact is an empty list, fill it
            if not var_interact2:
                main_feature_importance = self.feature_imp_shap(
                    final_shap_df.drop(var_interact1, axis=1), return_dict=False
                )

                # Interactions with high cardinality categorical variables are not calculated & plotted.
                # Hence any categorical variable must have less than `remove_var_morethan_n_levels` unique levels
                importance_order = main_feature_importance.index
                datatypes = self.data.dtypes
                counter = n = 0
                while counter < total_interactions_per_var:
                    try:
                        var_interact_temp = importance_order[n]
                    except IndexError:
                        break
                    var_interact_type = datatypes[var_interact_temp].name
                    if var_interact_type == "category":
                        if self.data[var_interact_temp].nunique() <= remove_var_morethan_n_levels:
                            var_interact2.append(var_interact_temp)
                            counter += 1
                    else:
                        var_interact2.append(var_interact_temp)
                        counter +=1
                    n+= 1

            final_shap_df[original_variable] = self.data[var_interact1]

            if final_shap_df[original_variable].dtype.name == "category":
                # Calling astype(str) twice converts NaN's to string format
                final_shap_df[original_variable] == final_shap_df[original_variable].astype(str).astype(str)
            else:
                final_shap_df[original_variable] = final_shap_df[original_variable].fillna(impute_missing_with)

            if var_interact1 == var_interact2[0]: # main interaction effect
                if scatter_plot:
                    self.scatterplot(
                        x=final_shap_df[original_variable],
                        y=final_shap_df[var_interact1],
                        title=f"Main effect for {var_interact1}",
                        xlabel=var_interact1,
                        ylabel=y_label,
                    )
                else:
                    variable_movAvg = final_shap_df.groupby(original_variable)[var_interact1].mean()
                    length_var_avg = len(variable_movAvg)
                    if length_var_avg < (moving_avg_window_size * globalvar.MIN_ROLLING_MEAN_MULTIPLIER):
                        window_size_use = max(
                            length_var_avg // globalvar.MIN_ROLLING_MEAN_MULTIPLIER, 1
                        )
                    else:
                        window_size_use = moving_avg_window_size
                    moving_avg_min_periods = min(window_size_use, moving_avg_min_periods)
                    variable_movAvg = variable_movAvg.rolling(
                        window=window_size_use, min_periods=moving_avg_min_periods, center=True
                    ).mean()
                    ax = self.lineplot(
                        data=variable_movAvg,
                        title=f"Main effect for {var_interact1}",
                        xlabel=var_interact1,
                        ylabel=y_label,
                        figsize=figsize,
                        style="-" if length_var_avg > 50 else globalvar.STYLE,
                    )
            else:  
                for var2 in var_interact2:
                    n_levels = self.data[var2].nunique()
                    
                    if n_levels <= total_bins_for_num_cols:
                        bins = self.data[var2].astype("category")
                    else:
                        try:
                            bins = pd.cut(
                                self.data[var2],
                                bins=n_bins,
                                labels=None,
                                duplicates="drop",
                            )
                        except (ValueError, TypeError):
                            bins = self.data[var2].astype("category")
                            

                    # Making sure when NaNs are plotted, the categories are ordered correctly
                    if bins.isnull().sum() > 0:
                        cat_order = {0: "nan"}
                        start = 1
                    else:
                        cat_order = {}
                        start = 0
                    cat_order.update(
                        {
                            i: str(bin)
                            for i, bin in enumerate(bins.cat.categories, start=start)
                        }
                    )
                    final_shap_df = final_shap_df.rename(columns={var2: y_label})
                    final_shap_df[var2] = bins.astype(str).astype(str)
                    if globalvar.NORMALIZE:
                        variable_movAvg = final_shap_df.groupby(var2)[y_label].mean()
                        final_shap_df["avg_var_interact_by_bin"] = final_shap_df[var2].map(
                            variable_movAvg
                        )
                        final_shap_df[y_label] = (
                            final_shap_df[y_label] - final_shap_df["avg_var_interact_by_bin"]
                        )
                    if scatter_plot:
                        final_shap_df["original_order"] = np.arange(len(final_shap_df))
                        sorted_dict = {v:k for k,v in cat_order.items()}
                        final_shap_df["sort_order"] = final_shap_df[var2].map(sorted_dict)
                        final_shap_df = final_shap_df.sort_values(by="sort_order")
                        self.scatterplot(
                            x=final_shap_df[original_variable],
                            y=final_shap_df[y_label],
                            hue=final_shap_df[var2],
                            title=f"{var_interact1}_X_{var2}",
                            xlabel=var_interact1,
                            ylabel=y_label,
                        )
                        final_shap_df = final_shap_df.sort_values(by="original_order")
                        final_shap_df = final_shap_df.drop(["original_order", "sort_order"], axis=1)
                    else:
                        lineplot_bin = []
                        bin_tmp = []
                        for n in range(len(cat_order)):
                            n_bin = cat_order[n]
                            bin_tmp.append(n_bin)
                            df0 = final_shap_df.loc[final_shap_df[var2] == n_bin]
                            variable_movAvg = df0.groupby(original_variable)[y_label].mean()
                            length_var_avg = len(variable_movAvg)
                            if length_var_avg < (
                                moving_avg_window_size * globalvar.MIN_ROLLING_MEAN_MULTIPLIER
                            ):
                                window_size_use = max(
                                    length_var_avg // globalvar.MIN_ROLLING_MEAN_MULTIPLIER, 1
                                )
                            else:
                                window_size_use = moving_avg_window_size
                            moving_avg_min_periods = min(window_size_use, moving_avg_min_periods)
                            variable_movAvg = variable_movAvg.rolling(
                                window=window_size_use,
                                min_periods=moving_avg_min_periods,
                                center=True,
                            ).mean()
                            lineplot_bin.append(variable_movAvg)
                        avg_bin_df = pd.concat(lineplot_bin, axis=1)
                        avg_bin_df.sort_index(inplace=True)
                        avg_bin_df = avg_bin_df.interpolate(
                            axis=0, limit_direction="both"
                        )

                        
                        avg_bin_df.columns = [f"{var2} {a_bin}" for a_bin in bin_tmp]
                        
                        
                        ax = self.lineplot(
                            data=avg_bin_df,
                            title=f"{var_interact1}_X_{var2}",
                            xlabel=var_interact1,
                            ylabel=y_label,
                            figsize=figsize,
                            style="-" if len(avg_bin_df) > 50 else globalvar.STYLE,
                        )
                    # Removing the current interaction pair for the next loop
                    final_shap_df = final_shap_df.drop([var2, y_label], axis=1)
