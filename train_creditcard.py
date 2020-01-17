import sys
import os
import pickle
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer

from alphagan_class import AlphaGAN
from util import dump_column_transformers, load_column_transformers, split_data, preprocess_data

os.environ["PATH"] += os.pathsep + 'C:/Program Files (x86)/Graphviz2.38/bin/'


if __name__ == '__main__':
    if len(sys.argv) > 1:
        train_df = pd.read_csv(sys.argv[1])
        preprocess_data(train_df, './data/ranges.csv')

        X_train = train_df.to_numpy()

        ag = AlphaGAN()
        ag.train(X_train=X_train, epochs=4000, batch_size=32)

