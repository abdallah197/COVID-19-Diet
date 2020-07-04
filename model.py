class Model:
    def __init__(self, data):
        datasets = {
            "fat": "data/datasets_618335_1290506_Fat_Supply_Quantity_Data.csv",
            "protein": "data/datasets_618335_1290506_Protein_Supply_Quantity_Data.csv",
            "kg": "data/datasets_618335_1290506_Food_Supply_Quantity_kg_Data.csv",
            "kcal": "data/datasets_618335_1290506_Food_Supply_kcal_Data.csv",
        }
        self.df = self.clean_data(datasets[data])
        self.data = train_test_split(
            self.df.loc[:, "Alcoholic Beverages":"Vegetables"],
            self.df["Rec_Rate"],
            test_size=0.2,
            random_state=100,
        )
        self.best_ml_appraoch(self.df)

    def clean_data(self, path):
        """
        preprocess the data
        """
        data = pd.read_csv(path)
        data["Rec_Rate"] = data[["Confirmed", "Active", "Recovered"]].apply(
            lambda x: (x["Recovered"] / (x["Confirmed"] - x["Active"])) * 100, axis=1
        )
        data["Total Confirmed"] = data[["Confirmed", "Population"]].apply(
            lambda row: (row["Confirmed"] / 100) * row["Population"], axis=1
        )
        data = data.sort_values(by=["Rec_Rate"], ascending=False)

        median = data.Undernourished[data.Undernourished != "<2.5"].median()
        data["Obesity"].interpolate(
            method="linear", inplace=True, limit_direction="both"
        )
        data = data.loc[data["Total Confirmed"] >= 5000]

        data["Undernourished"].replace(to_replace="<2.5", value=median, inplace=True)
        data["Undernourished"] = data["Undernourished"].apply(lambda x: float(x))
        data["Undernourished"].interpolate(
            method="linear", inplace=True, limit_direction="both"
        )
        return data

    def plot_feature_importance(self, df, importance):

        df = self.df
        importance_formatted = ["%.5f" % elem for elem in importance]
        importance_formatted = [elem * -1 for elem in importance]

        # importance_formatted.sort(reverse=True)

        sns.set(style="ticks", color_codes=True)

        sns.set(font_scale=1)

        X = df.loc[:, "Alcoholic Beverages":"Vegetables"]

        dt_feature_names = list(X.columns)

        dt_feature_names = [w.replace("&", "and") for w in dt_feature_names]

        dt_feature_names_sorted = [
            x for _, x in sorted(zip(importance_formatted, dt_feature_names))
        ]

        importance_formatted.sort(reverse=True)

        dt_feature_names_sorted.reverse()

        # Set up the matplotlib figure
        f, ax = plt.subplots(figsize=(11, 10))

        ax = sns.barplot(
            importance_formatted, y=dt_feature_names_sorted, palette="deep"
        )

        ax.set_title("Permutation Feature Importance", pad=20, fontsize=25)

        if isinstance(ax, np.ndarray):
            for idx, ax in np.ndenumerate(ax):
                self._show_on_single_plot(ax)
        else:
            self._show_on_single_plot(ax)

    def _show_on_single_plot(self, ax):
        for p in ax.patches:
            _x = p.get_x() + p.get_width() + float(0.05)
            _y = p.get_y() + p.get_height() + float(-0.3)
            value = float("%.5f" % p.get_width())
            ax.text(_x, _y, value, ha="left")

    def decision_tree(self):

        X_train, X_test, y_train, y_test = self.data

        tree = DecisionTreeRegressor(criterion="mse", random_state=42)

        # grid_search_params
        parameters = {
            "max_depth": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            "min_samples_leaf": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        }

        gridforest = GridSearchCV(tree, parameters, cv=5, n_jobs=-1, verbose=0)
        gridforest.fit(X_train, y_train)

        # Training
        model = DecisionTreeRegressor(
            criterion="mse",  # Initialize and fit regressor
            max_depth=gridforest.best_params_["max_depth"],
            random_state=42,
            min_samples_leaf=gridforest.best_params_["min_samples_leaf"],
        )
        model.fit(X_train, y_train)

        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)

        ##########
        # getting the most seginificant features using Permuatation
        results = permutation_importance(
            model, X_test, y_test, scoring="neg_root_mean_squared_error"
        )
        # get importance
        importance = results.importances_mean

        ########## printing  results
        print("********* " + "Decision Tree" + " ********* ")
        print(
            "********* " + "RMSE on Training data:",
            round(np.sqrt(mean_squared_error(y_train, y_train_pred)), 2),
            " *********",
        )
        print(
            "********* " + "RMSE on Test Set:",
            round(np.sqrt(mean_squared_error(y_test, y_test_pred)), 2),
            " *********",
        )

        dt_feature_names = list(
            self.df.loc[:, "Alcoholic Beverages":"Vegetables"].columns
        )

        dt_feature_names = [w.replace("&", "and") for w in dt_feature_names]

        # print the tree
        dot_data = StringIO()
        export_graphviz(
            model,
            out_file=dot_data,
            filled=True,
            rounded=True,
            feature_names=dt_feature_names,
            special_characters=True,
        )
        graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
        # plot the most significant features
        return (
            model,
            Image(graph.create_png()),
            round(np.sqrt(mean_squared_error(y_test, y_test_pred)), 2),
            importance,
        )

    def adaboost(self, tree):

        X_train, X_test, y_train, y_test = self.data

        ##Random Search for best HP
        ada_regressor = AdaBoostRegressor(tree)

        random_search = {
            "n_estimators": [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000],
            "learning_rate": [0.0001, 0.001, 0.01, 0.1, 1],
            "loss": ["linear", "sqaure", "exponential"],
        }
        ada = AdaBoostRegressor()
        rf_random = RandomizedSearchCV(
            estimator=ada,
            param_distributions=random_search,
            n_iter=100,
            cv=3,
            verbose=0,
            random_state=42,
            n_jobs=-1,
        )
        rf_random.fit(X_train, y_train)

        best_params = rf_random.best_params_

        # Grid Search in order to find the best parameters for the adaboost model
        parameters = {
            "n_estimators": [
                *range(
                    int(best_params["n_estimators"] * 0.5),
                    best_params["n_estimators"] * 2,
                    50,
                )
            ],
            "learning_rate": np.linspace(
                best_params["learning_rate"] / 10, best_params["learning_rate"] * 10, 5
            ),
            "loss": ["linear", "sqaure", "exponential"],
        }

        gridforest = GridSearchCV(ada_regressor, parameters, cv=5, n_jobs=-1, verbose=0)
        gridforest.fit(X_train, y_train)

        ##Adaboost Training
        ada_regressor = AdaBoostRegressor(
            tree,
            n_estimators=gridforest.best_params_["n_estimators"],
            learning_rate=gridforest.best_params_["learning_rate"],
            loss=gridforest.best_params_["loss"],
        )

        ada_regressor.fit(X_train, y_train)

        y_train_pred = ada_regressor.predict(X_train)
        y_test_pred = ada_regressor.predict(X_test)

        ##########
        # Permuatation
        results = permutation_importance(
            ada_regressor, X_test, y_test, scoring="neg_root_mean_squared_error"
        )
        # get importance
        importance = results.importances_mean

        ##########
        print("********* " + "Adaboost" + " ********* ")
        print(
            "********* " + "RMSE on Training data:",
            round(np.sqrt(mean_squared_error(y_train, y_train_pred)), 2),
            " *********",
        )
        print(
            "********* " + "RMSE on Test Set:",
            round(np.sqrt(mean_squared_error(y_test, y_test_pred)), 2),
            " *********",
        )

        return round(np.sqrt(mean_squared_error(y_test, y_test_pred)), 2), importance

    def gradboost(self):

        X_train, X_test, y_train, y_test = self.data

        # random_search
        random_grid = {
            "max_depth": [5, 6, 7, 8, 9, 10],
            "learning_rate": [0.001, 0.01, 0.1, 1],
            "min_samples_leaf": [1, 2, 3, 4, 5],
            "loss": ["ls", "lad", "huber", "quantile"],
            "n_estimators": [100, 200, 300, 400, 500],
            "min_samples_split": [2, 4, 5, 6, 7],
        }

        gb = GradientBoostingRegressor()
        gb_random = RandomizedSearchCV(
            estimator=gb,
            param_distributions=random_grid,
            n_iter=100,
            cv=3,
            verbose=0,
            random_state=42,
            n_jobs=-1,
        )
        gb_random.fit(X_train, y_train)

        best_params = gb_random.best_params_
        parameters = {
            "max_depth": [
                *range(best_params["max_depth"] - 3, best_params["max_depth"] * 2, 2)
            ],
            "learning_rate": np.linspace(
                best_params["learning_rate"] / 10, best_params["learning_rate"] * 10, 5
            ),
            "n_estimators": [
                *range(
                    int(best_params["n_estimators"] * 0.5),
                    best_params["n_estimators"] * 2,
                    50,
                )
            ],
            "min_samples_leaf": [
                *range(
                    max(best_params["min_samples_leaf"] - 3, 0),
                    best_params["min_samples_leaf"] * 2,
                    2,
                )
            ],
            "min_samples_split": [
                *range(
                    max(best_params["min_samples_split"] - 3, 0),
                    best_params["min_samples_split"] * 2,
                    2,
                )
            ],
            "loss": ["ls", "lad", "huber", "quantile"],
        }

        grad_regressor = GradientBoostingRegressor()
        gridforest = GridSearchCV(
            grad_regressor, parameters, cv=5, n_jobs=-1, verbose=0
        )
        gridforest.fit(X_train, y_train)

        ##Adaboost
        grad_regressor = GradientBoostingRegressor(
            learning_rate=gridforest.best_params_["learning_rate"],
            loss=gridforest.best_params_["loss"],
            max_depth=gridforest.best_params_["max_depth"],
            min_samples_leaf=gridforest.best_params_["min_samples_leaf"],
            n_estimators=gridforest.best_params_["n_estimators"],
            min_samples_split=gridforest.best_params_["min_samples_split"],
        )

        grad_regressor.fit(X_train, y_train)

        y_train_pred = grad_regressor.predict(X_train)
        y_test_pred = grad_regressor.predict(X_test)

        ##########
        # Permuatation
        results = permutation_importance(
            grad_regressor, X_test, y_test, scoring="neg_root_mean_squared_error"
        )
        # get importance
        importance = results.importances_mean

        ##########
        print("********* " + "GradBoost" + " ********* ")
        print(
            "********* " + "RMSE on Training data:",
            round(np.sqrt(mean_squared_error(y_train, y_train_pred)), 2),
            " *********",
        )
        print(
            "********* " + "RMSE on Test Set:",
            round(np.sqrt(mean_squared_error(y_test, y_test_pred)), 2),
            " *********",
        )

        return round(np.sqrt(mean_squared_error(y_test, y_test_pred)), 2), importance

    def random_forest(self):

        X_train, X_test, y_train, y_test = self.data

        model = RandomForestRegressor(criterion="mse", random_state=42)

        # Random search to find the best parameters for the model
        random_grid = {
            "max_depth": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            "n_estimators": [200, 300, 400, 500, 600, 700, 800, 900, 1000],
            "min_samples_leaf": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            "max_features": ["auto", "sqrt", "log2"],
            "bootstrap": [True, False],
            "min_samples_split": [2, 4, 5, 6, 7],
        }
        # Use the random grid to search for best hyperparameters
        rf = RandomForestRegressor()
        rf_random = RandomizedSearchCV(
            estimator=rf,
            param_distributions=random_grid,
            n_iter=100,
            cv=3,
            verbose=0,
            random_state=42,
            n_jobs=-1,
        )
        rf_random.fit(X_train, y_train)

        best_params = rf_random.best_params_

        # Grid Search over Random Search best values
        parameters = {
            "max_depth": [
                *range(best_params["max_depth"] - 3, best_params["max_depth"] * 2, 2)
            ],
            "n_estimators": [
                *range(
                    int(best_params["n_estimators"] * 0.5),
                    best_params["n_estimators"] * 2,
                    50,
                )
            ],
            "min_samples_leaf": [
                *range(
                    max(best_params["min_samples_leaf"] - 2, 0),
                    best_params["min_samples_leaf"] * 2,
                    2,
                )
            ],
            "max_features": [best_params["max_features"]],
            "bootstrap": [best_params["bootstrap"]],
            "min_samples_split": [
                *range(
                    max(best_params["min_samples_split"] - 2, 0),
                    best_params["min_samples_split"] * 2,
                    2,
                )
            ],
        }

        gridforest = GridSearchCV(model, parameters, cv=5, n_jobs=-1, verbose=0)
        gridforest.fit(X_train, y_train)

        model = RandomForestRegressor(
            criterion="mse",  # Initialize and fit regressor
            max_depth=gridforest.best_params_["max_depth"],
            random_state=42,
            min_samples_leaf=gridforest.best_params_["min_samples_leaf"],
            n_estimators=gridforest.best_params_["n_estimators"],
            max_features=gridforest.best_params_["max_features"],
        )
        model.fit(X_train, y_train)

        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)

        ##########
        # Permuatation
        results = permutation_importance(
            model, X_test, y_test, scoring="neg_root_mean_squared_error"
        )
        # get importance
        importance = results.importances_mean
        print("********* " + "Random Forest" + " ********* ")
        print(
            "********* " + "RMSE on Training data:",
            round(np.sqrt(mean_squared_error(y_train, y_train_pred)), 2),
            " *********",
        )
        print(
            "********* " + "RMSE on Test Set:",
            round(np.sqrt(mean_squared_error(y_test, y_test_pred)), 2),
            " *********",
        )

        ##########
        return round(np.sqrt(mean_squared_error(y_test, y_test_pred)), 2), importance

    def best_ml_appraoch(self, df):
        """
        get the output of:
        * Decision Tree
        * Adaboost
        * Random Forest
        * Gradient Boost
        """
        d_tree, graph, mse_dt, importance_dt = self.decision_tree()
        mse_ada, importance_ada = self.adaboost(d_tree)
        mse_rf, importance_rf = self.random_forest()
        mse_gb, importance_gb = self.gradboost()
        trees = {
            mse_dt: (graph, importance_dt),
            mse_ada: (importance_ada, "Adaboost"),
            mse_rf: (importance_rf, "Random Forest"),
            mse_gb: (importance_gb, "Gradient Boosting"),
        }
        min_mse = min(trees.keys())
        if min_mse is mse_dt:

            print(
                "*********************** Decision Tree is the best Algorithm ***********************"
            )
            graph, importance = trees[min_mse]
            self.plot_feature_importance(df, importance)

        else:
            importance, algorithm = trees[min_mse]
            print(
                f"*********************** {algorithm} is the best Algorithm ***********************"
            )
            self.plot_feature_importance(df, importance)
