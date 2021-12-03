from process_results_utils import get_mean_std_metric, plot_train_images, plot_false_predictions

colab = True


def main():
    max_k = 5
    name = f'hparams_table_max_k={max_k}_all_stages.csv'
    name_output = f'results_max_k={max_k}_all_stages.csv'
    log_dir = 'runs'
    if colab:
        name = f'/content/drive/MyDrive/I2MTC/results/{name}'
        name_output = f'/content/drive/MyDrive/I2MTC/results/{name_output}'
        log_dir = '/content/drive/MyDrive/I2MTC/runs'

    get_mean_std_metric(name=name, name_output=name_output, log_dir=log_dir)


if __name__ == "__main__":
    main()
