import hydra
from kaggle import KaggleApi
from omegaconf import DictConfig

from src.utils.kaggle_utils import Deploy


@hydra.main(version_base=None, config_path="/workspace/configs/", config_name="config")
def main(cfg: DictConfig):
    client = KaggleApi()
    client.authenticate()

    deploy = Deploy(cfg=cfg, client=client)
    deploy.push_output()
    deploy.push_huguingface_model()
    deploy.push_code()


if __name__ == "__main__":
    main()
