"""
Currently, the config files are only present for BiSeNetV2.
Eventually, if found useful, we can add config files for other models as well.
"""
import importlib
import json
from config.config_schema import TestConfig

# class cfg_dict(object):

#     def __init__(self, d):
#         self.__dict__ = d
#         self.get = d.get

# def set_cfg_from_file(cfg_path):
#     spec = importlib.util.spec_from_file_location('cfg_file', cfg_path)
#     cfg_file = importlib.util.module_from_spec(spec)
#     spec_loader = spec.loader.exec_module(cfg_file)
#     cfg = cfg_file.cfg
#     return cfg_dict(cfg)


def load_config(cfg_path: str) -> TestConfig:
    with open(cfg_path, 'r') as f:
        cfg_dict = json.load(f)
    return TestConfig(**cfg_dict)