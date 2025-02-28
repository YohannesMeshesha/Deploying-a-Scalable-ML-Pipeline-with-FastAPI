import pickle
from sklearn.metrics import fbeta_score, precision_score, recall_score
from ml.data import process_data

# Optional: add any other necessary imports here.


def train_model(X_train, y_train):
    """
    Trains a machine learning model and returns it.

    Inputs
    ------
    X_train : np.array
        Training data.
    y_train : np.array
        Labels.
    Returns
    -------
    model
        Trained machine learning model.
    """
    # Minimal implementation using a dummy model.
    class DummyModel:
        def predict(self, X):
            # Always predict 0.
            return [0] * len(X)
    return DummyModel()


def compute_model_metrics(y, preds):
    """
    Validates the trained model using precision, recall, and F1 score.

    Inputs
    ------
    y : np.array
        Known binarized labels.
    preds : np.array
        Predicted binarized labels.
    Returns
    -------
    precision : float
    recall : float
    fbeta : float
    """
    fbeta = fbeta_score(y, preds, beta=1, zero_division=1)
    precision = precision_score(y, preds, zero_division=1)
    recall = recall_score(y, preds, zero_division=1)
    return precision, recall, fbeta


def inference(model, X):
    """
    Runs model inference and returns predictions.

    Inputs
    ------
    model : Trained machine learning model.
    X : np.array
        Data used for prediction.
    Returns
    -------
    preds : np.array
        Predictions from the model.
    """
    return model.predict(X)


def save_model(model, path):
    """
    Serializes model to a file.

    Inputs
    ------
    model
        Trained machine learning model or OneHotEncoder.
    path : str
        Path to save the pickle file.
    """
    with open(path, "wb") as f:
        pickle.dump(model, f)


def load_model(path):
    """
    Loads a pickle file from `path` and returns it.
    """
    with open(path, "rb") as f:
        return pickle.load(f)


def performance_on_categorical_slice(data, column_name, slice_value,
                                     categorical_features, label,
                                     encoder, lb, model):
    """
    Computes model metrics on a slice of the data defined by a column name and
    a slice value.

    Inputs
    ------
    data : pd.DataFrame
        Data containing features and label.
    column_name : str
        Column on which to slice the data.
    slice_value : str, int, or float
        Value to filter the column.
    categorical_features : list
        List of categorical feature names.
    label : str
        Name of the label column.
    encoder : OneHotEncoder
        Trained OneHotEncoder.
    lb : LabelBinarizer
        Trained LabelBinarizer.
    model : Trained machine learning model.
    Returns
    -------
    precision : float
    recall : float
    fbeta : float
    """
    data_slice = data[data[column_name] == slice_value]
    X_slice, y_slice, _, _ = process_data(
        data_slice,
        categorical_features=categorical_features,
        label=label,
        training=False,
        encoder=encoder,
        lb=lb
    )
    preds = inference(model, X_slice)
    precision, recall, fbeta = compute_model_metrics(y_slice, preds)
    return precision, recall, fbeta
