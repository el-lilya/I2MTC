import argparse

from checkpoints.BBN_master.lib.config import update_config, cfg
from checkpoints.BBN_master.lib.net.network import Network


def main():
    full_inat_num_classes = 8142
    model = Network(cfg, mode="train", num_classes=full_inat_num_classes)
    print(model.cfg.BACKBONE)
    model_path = "checkpoints/BBN_master/BBN.iNaturalist2018.res50.180epoch.best_model.pth"
    model.load_model(model_path)
    model.classifier = nn.Linear()


if __name__ == "__main__":
    main()
