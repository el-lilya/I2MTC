import torchvision

from checkpoints.BBN_master.lib.net.network import Network


def main():
    root = 'data'
    torchvision.datasets.INaturalist(root=root, version='2021_train_mini')


if __name__ == "__main__":
    main()
