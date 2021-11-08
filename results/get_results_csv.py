import pandas as pd


def get_results(name: str, name_csv: str):
    results_full = pd.read_csv(name)
    results = pd.DataFrame({'k': results_full['k'].unique()})
    dict_mean = results_full.groupby('k')['test_accuracy'].mean().to_dict()
    dict_std = results_full.groupby('k')['test_accuracy'].std().to_dict()
    results['mean'] = results['k'].map(dict_mean)
    results['std'] = results['k'].map(dict_std)
    print(results)
    results.to_csv(name_csv)


def main():
    get_results('hparams_table_max_k=5.csv', 'results_k=1_3_5.csv')


if __name__ == "__main__":
    main()
