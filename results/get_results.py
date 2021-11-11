from process_results_utils import get_mean_std_acc, plot_train_images, plot_false_predictions


def main():
    # get_results('hparams_table_max_k=5.csv', 'results_max_k=5.csv')
    k = 5
    exp = 3
    labels = [8, 1, 13, 6]
    plot_false_predictions('false_predictions.csv', k=k, exp=exp, labels=labels)
    plot_train_images(k=k, exp=exp, labels=labels)


if __name__ == "__main__":
    main()
