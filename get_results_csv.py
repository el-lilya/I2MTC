import torch
import pandas as pd


def main():
    results_full = pd.read_csv('./data/hparams_table_max_k=5.csv')
    results = pd.DataFrame({'k': results_full['k'].unique()})
    dict_mean = results_full.groupby('k')['test_accuracy'].mean().to_dict()
    dict_std = results_full.groupby('k')['test_accuracy'].std().to_dict()
    results['mean'] = results['k'].map(dict_mean)
    results['std'] = results['k'].map(dict_std)
    print(results.head())
    results.to_csv('results_k=1_3_5.csv')


if __name__ == "__main__":
    main()