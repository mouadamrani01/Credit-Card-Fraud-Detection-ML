import numpy as np
import pandas as pd
import pytest
from src.train import FEATURES, split_and_prepare, validate_dataset


def sample_data():
    rng = np.random.default_rng(7)
    data = pd.DataFrame(rng.normal(size=(120, len(FEATURES))), columns=FEATURES)
    data['Class'] = np.array([0]*100 + [1]*20)
    return data


def test_validation_missing_column():
    with pytest.raises(ValueError, match='missing columns'):
        validate_dataset(sample_data().drop(columns=['V1']))


def test_resample_training_only():
    X_train, X_test, y_train, y_test = split_and_prepare(sample_data())
    assert len(X_test) == 24
    assert len(y_test) == 24
    assert y_test.sum() == 4
    assert int((y_train == 0).sum()) == int((y_train == 1).sum())
    assert list(X_train.columns) == FEATURES
