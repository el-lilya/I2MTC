from process_results_utils import get_mean_std_metric, plot_train_images, plot_false_predictions


def main():
    log_dir = 'runs_gpu_baseline'
    # get mean and std accuracy
    max_k = 7
    get_mean_std_metric(f'hparams_table_max_k={max_k}_17.csv', f'results_max_k={max_k}_17.csv', log_dir=log_dir)

    # plot false predictions
    k = 7
    exp = 4
    labels = [2, 1]
    plot_false_predictions('false_predictions_17.csv', k=k, exp=exp, labels=labels, log_dir=log_dir)

    # plot train images
    plot_train_images(k=k, exp=exp, labels=labels, log_dir=log_dir)


if __name__ == "__main__":
    main()
