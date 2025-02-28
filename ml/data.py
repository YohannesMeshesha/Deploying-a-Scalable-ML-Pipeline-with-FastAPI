import numpy as np
from sklearn.preprocessing import LabelBinarizer, OneHotEncoder


def process_data(X, categorical_features=[], label=None, training=True,
                 encoder=None, lb=None):
    """
    Process the data used in the ML pipeline.

    Processes the data using one-hot encoding for categorical features and a
    label binarizer for the labels. This can be used in either training or
    inference/validation. Optionally, continuous data scaling can be added.

    Inputs:
      X : pd.DataFrame
          Dataframe with features and label. Columns in categorical_features.
      categorical_features : list[str]
          List of categorical feature names.
      label : str
          Name of the label column in X. If None, y is empty.
      training : bool
          If True, fit the encoder and lb; if False, use provided ones.
      encoder : OneHotEncoder
          Trained OneHotEncoder (used if training is False).
      lb : LabelBinarizer
          Trained LabelBinarizer (used if training is False).

    Returns:
      X : np.array
          Processed data.
      y : np.array
          Processed labels or an empty array.
      encoder : OneHotEncoder
          The fitted encoder.
      lb : LabelBinarizer
          The fitted label binarizer.
    """
    if label is not None:
        y = X[label]
        X = X.drop([label], axis=1)
    else:
        y = np.array([])

    X_categorical = X[categorical_features].values
    X_continuous = X.drop(categorical_features, axis=1)

    if training:
        encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
        lb = LabelBinarizer()
        X_categorical = encoder.fit_transform(X_categorical)
        y = lb.fit_transform(y.values).ravel()
    else:
        X_categorical = encoder.transform(X_categorical)
        try:
            y = lb.transform(y.values).ravel()
        except AttributeError:
            pass

    X = np.concatenate([X_continuous, X_categorical], axis=1)
    return X, y, encoder, lb


def apply_label(inference):
    """
    Convert the binary label in a single inference sample into string output.
    """
    if inference[0] == 1:
        return ">50K"
    elif inference[0] == 0:
        return "<=50K"
