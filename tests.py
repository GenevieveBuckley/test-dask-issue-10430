import os

import dask.dataframe as dd
import pandas as pd
import pytest


@pytest.fixture
def df():
    data = {
        "calories": [420, 380, 390],
        "duration": [50, 40, 45]
    }
    return pd.DataFrame(data)


def test_pandas(tmp_path, df):
    save_filename = tmp_path / "df.csv"
    df.to_csv(save_filename)
    assert os.path.exists(save_filename)
    new_df = pd.read_csv(save_filename, index_col="Unnamed: 0")
    pd.testing.assert_frame_equal(df, new_df)


def test_long_pandas(tmp_path, df):
    save_filename = os.path.join(tmp_path, ('a' * 256) + ".csv")
    df.to_csv(save_filename)
    assert os.path.exists(save_filename)
    new_df = pd.read_csv(save_filename, index_col="Unnamed: 0")
    pd.testing.assert_frame_equal(df, new_df)


def test_long_pandas_fix(tmp_path, df):
    save_filename = os.path.join(r"\\?\\" + tmp_path, ('a' * 256) + ".csv")
    df.to_csv(save_filename)
    assert os.path.exists(save_filename)
    new_df = pd.read_csv(save_filename, index_col="Unnamed: 0")
    pd.testing.assert_frame_equal(df, new_df)


def test_dask(tmp_path, df):
    save_filename = tmp_path / "df.csv"
    df.to_csv(save_filename)
    assert os.path.exists(save_filename)
    new_df = dd.read_csv(save_filename, partitions=2, index_col="Unnamed: 0")
    pd.testing.assert_frame_equal(df, new_df.compute())


def test_long_dask(tmp_path, df):
    save_filename = os.path.join(tmp_path, ('a' * 256) + ".csv")
    df.to_csv(save_filename)
    assert os.path.exists(save_filename)
    new_df = dd.read_csv(save_filename, partitions=2, index_col="Unnamed: 0")
    pd.testing.assert_frame_equal(df, new_df.compute())


def test_long_dask_fix(tmp_path, df):
    save_filename = os.path.join(r"\\?\\" + tmp_path, ('a' * 256) + ".csv")
    df.to_csv(save_filename)
    assert os.path.exists(save_filename)
    new_df = dd.read_csv(save_filename, partitions=2, index_col="Unnamed: 0")
    pd.testing.assert_frame_equal(df, new_df.compute())