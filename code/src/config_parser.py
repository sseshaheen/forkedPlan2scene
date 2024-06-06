import json
import os.path as osp
import logging


class Config:
    """
    Configuration item.
    """

    def __init__(self, config_dict: dict):
        """
        Initialize configuration item using a dictionary.
        :param config_dict: Configuration dictionary.
        """
        for k, v in config_dict.items():
            if isinstance(v, dict):
                v = Config(v)
            self.__dict__[k] = v

    def __getitem__(self, key):
        return self.__dict__[key]


def parse_config(config_path: str):
    """
    Parses a JSON config file into a Config object or a list.
    :param config_path: Path to the JSON config file.
    """
    if not osp.exists(config_path):
        logging.error(f"Config file not found: {config_path}")
        return None

    try:
        with open(config_path, "r") as f:
            config_dict = json.loads(f.read())
            logging.debug(f"Contents of {config_path}: {config_dict}")
            if isinstance(config_dict, dict):
                return Config(config_dict)
            elif isinstance(config_dict, list):
                return config_dict
            else:
                logging.error(f"Config file {config_path} does not contain a dictionary or a list.")
                return None
    except Exception as e:
        logging.error(f"Error reading configuration file {config_path}: {e}")
        return None
