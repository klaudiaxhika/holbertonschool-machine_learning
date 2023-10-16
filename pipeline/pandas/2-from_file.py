#!/usr/bin/env python3
"""
A function that loads data from a file as a pd.DataFrame
"""
import pandas as pd


def from_file(filename, delimiter):
    """
    A function that loads data from a file as a pd.DataFrame
    filename: the file to load from
    delimiter: the column separator
    returns: the loaded pd.DataFrame
    """
    df = pd.read_csv(filename, delimiter=delimiter)
    return df
