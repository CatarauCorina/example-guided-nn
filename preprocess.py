import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from math import floor, ceil
from pylab import rcParams


def remove_duplicates(merged_df):
    merge_vector = ["school", "sex", "age", "address",
                    "famsize", "Pstatus", "Medu", "Fedu",
                    "Mjob", "Fjob", "reason", "nursery", "internet"]

    duplicated_mask = merged_df.duplicated(keep=False, subset=merge_vector)
    duplicated_df = merged_df[duplicated_mask]
    unique_df = merged_df[~duplicated_mask]
    both_courses_mask = duplicated_df.duplicated(subset=merge_vector)
    both_courses_df = duplicated_df[~both_courses_mask].copy()
    both_courses_df["course"] = "both"
    students_df = unique_df.append(both_courses_df)
    print(students_df.shape)
    return students_df


def merge_csv(csv_1, csv_2):
    merged_df = csv_1.append(csv_2)
    return merged_df


def encode(series):
    return pd.get_dummies(series.astype(str))


def split_data(data_x, data_y, train_size):

    train_cnt = floor(data_x.shape[0] * train_size)
    x_train = data_x.iloc[0:train_cnt].values
    y_train = data_y.iloc[0:train_cnt].values
    x_test = data_x.iloc[train_cnt:].values
    y_test = data_y.iloc[train_cnt:].values

    return x_train, y_train, x_test, y_test


def process_student_data():
    sns.set(style='ticks', palette='Spectral', font_scale=1.5)

    material_palette = ["#4CAF50", "#2196F3", "#9E9E9E", "#FF9800", "#607D8B", "#9C27B0"]
    sns.set_palette(material_palette)
    rcParams['figure.figsize'] = 16, 8

    plt.xkcd();
    random_state = 42
    np.random.seed(random_state)
    tf.set_random_seed(random_state)

    math = pd.read_csv("student-mat.csv")
    port = pd.read_csv("student-por.csv")
    math["course"] = "math"
    port["course"] = "portuguese"
    merged_df = merge_csv(math, port)
    unique = remove_duplicates(merged_df)
    students_df = unique.sample(frac=1)
    students_df['alcohol'] = (students_df.Walc * 2 + students_df.Dalc * 5) / 7
    students_df['alcohol'] = students_df.alcohol.map(lambda x: ceil(x))
    students_df['drinker'] = students_df.alcohol.map(lambda x: "yes" if x > 2 else "no")

    train_x = pd.get_dummies(students_df.school)
    train_x['age'] = students_df.age
    train_x['absences'] = students_df.absences
    train_x['g1'] = students_df.G1
    train_x['g2'] = students_df.G2
    train_x['g3'] = students_df.G3
    train_x = pd.concat([train_x, encode(students_df.sex), encode(students_df.Pstatus),
                         encode(students_df.Medu), encode(students_df.Fedu),
                         encode(students_df.guardian), encode(students_df.studytime),
                         encode(students_df.failures), encode(students_df.activities),
                         encode(students_df.higher), encode(students_df.romantic),
                         encode(students_df.reason), encode(students_df.paid),
                         encode(students_df.goout), encode(students_df.health),
                         encode(students_df.famsize), encode(students_df.course)
                         ], axis=1)

    train_y = encode(students_df.drinker)
    train_x, train_y, test_x, test_y = split_data(train_x, train_y, 0.9)

    return train_x, train_y, test_x, test_y

