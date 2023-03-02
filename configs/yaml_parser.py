import yaml
from yaml import CLoader as Loader


class BunchDict(dict):
    def __init__(self, **kw):
        dict.__init__(self, kw)
        self.__dict__.update(kw)


def yaml_parser(path_to_yaml):
    """Converts file in path from YAML to dictionary.

    Supports bunch dictionary up to depth = 2.

    :param path_to_yaml: path to the yaml you wish to parse.
    :return:dict: BunchDict. dictionary containing the yaml data.
    """
    with open(path_to_yaml, 'r') as f:
        yaml_content = yaml.load(f.read(), Loader=Loader)
    for k in yaml_content:
        if type(yaml_content[k]) == dict:
            for inner_k in yaml_content[k]:
                if type(yaml_content[k][inner_k]) == dict:
                    yaml_content[k][inner_k] = BunchDict(**yaml_content[k][inner_k])
            yaml_content[k] = BunchDict(**yaml_content[k])
    return BunchDict(**yaml_content)
