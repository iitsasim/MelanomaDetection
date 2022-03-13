import os

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sys

"""Set Initial Constants"""
GCS_PATH = "../data"
OUT_PATH = '../output'
TRAIN_DIR = GCS_PATH + '/train.csv'
TEST_DIR = GCS_PATH + '/test.csv'

file_path = OUT_PATH+'/mel_data_viz.txt'
sys.stdout = open(file_path, "w")

def viz_data():


    """Data Set Counts:"""
    train = pd.read_csv(TRAIN_DIR)
    test = pd.read_csv(TEST_DIR)
    os.chdir(OUT_PATH)
    print("--------------Data Set Counts:-----------------")
    print('Train Data: ', train.shape)
    print("Test Data:", test.shape)

    """Preview Train Data"""
    print("--------------Preview Train Data:-----------------")
    print(train.head())

    """Train Data Columns"""
    print("--------------Preview Train Data Columns:-----------------")
    print(train.info())

    print(test.head())
    print(test.info())

    """Data Seggregation - Benign vs Malignant"""
    print("--------------Data Seggregation - Benign vs Malignant:-----------------")
    print(train['benign_malignant'].value_counts(normalize=True))
    sns.countplot(train['benign_malignant'])
    plt.savefig("benign_malignant.png")

    """Data Variations (Sex, Age.. etc)"""
    print("--------------Data Variations (Sex, Age.. etc):-----------------")
    print(train['sex'].value_counts(normalize=True))
    print(train['target'].groupby(train['sex']).mean())
    sns.countplot(train['sex'], hue=train['target'])
    plt.savefig("age_gender.png")

    """Data Segregation By Age"""
    print("--------------Data Segregation By Age:-----------------")
    print(train['target'].groupby(train['age_approx']).mean())
    plt.figure(figsize=(8, 5))
    sns.countplot(train['age_approx'], hue=train['target'])
    plt.savefig("age_apprx.png")

    """Data Seggregation By Body Part"""
    print("--------------Data Seggregation By Body Part:-----------------")
    train['anatom_site_general_challenge'].value_counts(normalize=True)
    train['target'].groupby(train['anatom_site_general_challenge']).mean()
    plt.figure(figsize=(10, 5))
    sns.countplot(train['anatom_site_general_challenge'], hue=train['target'])
    plt.savefig("body_part.png")

    """Data Seggregation By Diagnosis"""
    print("--------------Process finished with exit code 0:-----------------")
    train['diagnosis'].value_counts(normalize=True)
    train['target'].groupby(train['diagnosis']).mean()
    plt.figure(figsize=(15, 5))
    sns.countplot(train['diagnosis'], hue=train['target'])
    plt.savefig("diagnosis.png")

    train_df = train[['sex', 'age_approx', 'anatom_site_general_challenge', 'diagnosis', 'target']]
    train_df.head()

    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()
    train_df = train_df.apply(lambda col: le.fit_transform(col.astype(str)), axis=0, result_type='expand')
    train_df.head()

    g = sns.pairplot(train_df, hue="diagnosis")

    sns.heatmap(train_df.corr(), annot=True, linewidths=0.2)
    fig = plt.gcf()
    fig.set_size_inches(20, 12)
    plt.savefig("All_Figs.png")
    plt.show()


if __name__ == "__main__":
   viz_data()