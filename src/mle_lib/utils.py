"""
Common utilities function that can be used across project to maintain uniformity and enhance re-usability.
"""
import pickle

import pandas as pd


def read(path):
    """
    Function to read files of various formats

    Parameters:
    -----------
    path:
        String, Path of file to read
    """
    _type = path.split(".")[-1]

    if _type == "csv":
        return pd.read_csv(path)
    elif _type == "pkl":
        return pickle.load(open(path, "rb"))
    else:
        pass


def write(data, path):
    """
    Function to write files of various formats

    Parameters:
    -----------
    data:
        data to write
    path:
        String, Path where file should be written
    """
    _type = path.split(".")[-1]

    if _type == "csv":
        data.to_csv(path)
    elif _type == "pkl":
        pickle.dump(data, open(path, "wb"))
    else:
        pass
