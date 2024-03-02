from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, BaggingClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier


def get_models():
    models = {}

    nb = GaussianNB()
    nb_params = {}
    models['Naive Bayes'] = [nb, nb_params]

    svc = SVC()
    svc_params = {
        'C': [0.1, 1, 10],
        'kernel': ['linear', 'rbf'],
        'gamma': ['scale', 'auto', 0.1, 1],
    }
    models['SVC'] = [svc, svc_params]

    df = DecisionTreeClassifier()
    df_params = {
        'criterion': ['gini', 'entropy'],
        'max_depth': [None, 10, 20, 30],
    }
    models['Decision Tree'] = [df, df_params]

    boosting = AdaBoostClassifier()
    boosting_params = {
        'n_estimators': [50, 100, 200],
        'learning_rate': [0.01, 0.1, 1.0],
    }
    models['Boost'] = [boosting, boosting_params]

    bagging_model = BaggingClassifier(df)
    bagging_params = {
        'n_estimators': [50, 100, 200],
    }
    models['Bagging'] = [bagging_model, bagging_params]
    
    return models


def select_the_best(model, param_grid, X_train, y_train):
    grid_search = GridSearchCV(model, param_grid, cv=5, scoring='accuracy')
    grid_search.fit(X_train, y_train)

    best_rf_model = grid_search.best_estimator_
    print(best_rf_model.get_params())

    return best_rf_model
