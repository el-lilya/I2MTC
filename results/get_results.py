from process_results_utils import get_mean_std_acc, plot_train_images, plot_false_predictions


def main():
    log_dir = 'runs_gpu'
    # get mean and std accuracy
    max_k = 7
    get_mean_std_acc(f'hparams_table_max_k={max_k}.csv', f'results_max_k={max_k}.csv', log_dir=log_dir)

    # plot false predictions
    k = 7
    exp = 0
    labels = [1, 19]
    plot_false_predictions('false_predictions.csv', k=k, exp=exp, labels=labels, log_dir=log_dir)

    # plot train images
    plot_train_images(k=k, exp=exp, labels=labels, log_dir=log_dir)


if __name__ == "__main__":
    main()
