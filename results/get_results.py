from process_results_utils import get_mean_std_acc, plot_train_images, plot_false_predictions


def main():
    # get_results('hparams_table_max_k=5.csv', 'results_max_k=5.csv')
    plot_false_predictions('false_predictions.csv', k=1, exp=0, label=1)
    plot_train_images(k=2, exp=1)


if __name__ == "__main__":
    main()
