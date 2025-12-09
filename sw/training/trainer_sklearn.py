def train_sklearn(model, X_train, y_train):
    model.fit(X_train, y_train)
    return model
