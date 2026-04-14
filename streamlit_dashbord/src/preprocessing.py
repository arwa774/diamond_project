"""
src/preprocessing.py
Fonctions utilitaires de préprocessing réutilisables.
Auteur : Hadj Mbarek Arwa
"""

import pandas as pd
import numpy as np


def charger_et_nettoyer(filepath: str) -> pd.DataFrame:
    df = pd.read_csv(filepath)
    if 'Unnamed: 0' in df.columns:
        df = df.drop(columns=['Unnamed: 0'])
    df = df[(df['x'] > 0) & (df['y'] > 0) & (df['z'] > 0)]
    df = df[(df['y'] < 20) & (df['z'] < 20)]
    df = df.drop_duplicates()
    return df.reset_index(drop=True)


def encoder_variables_ordinales(df: pd.DataFrame) -> pd.DataFrame:
    cut_mapping = {'Fair': 1, 'Good': 2, 'Very Good': 3, 'Premium': 4, 'Ideal': 5}
    color_mapping = {'J': 1, 'I': 2, 'H': 3, 'G': 4, 'F': 5, 'E': 6, 'D': 7}
    clarity_mapping = {'I1': 1, 'SI2': 2, 'SI1': 3, 'VS2': 4, 'VS1': 5, 'VVS2': 6, 'VVS1': 7, 'IF': 8}
    df = df.copy()
    df['cut_encoded'] = df['cut'].map(cut_mapping)
    df['color_encoded'] = df['color'].map(color_mapping)
    df['clarity_encoded'] = df['clarity'].map(clarity_mapping)
    df = df.drop(columns=['cut', 'color', 'clarity'])
    return df


def creer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df['volume'] = df['x'] * df['y'] * df['z']
    df['carat_per_volume'] = df['carat'] / (df['volume'] + 1e-6)
    return df


def pipeline_complet(filepath: str):
    df = charger_et_nettoyer(filepath)
    df = encoder_variables_ordinales(df)
    df = creer_features(df)
    X = df.drop(columns=['price'])
    y = df['price']
    return X, y
