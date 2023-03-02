import yaml
from yaml import CLoader as Loader

with open('configs/example.yaml', 'r') as f:
    yaml_content = yaml.load(f.read(), Loader=Loader)

print(yaml_content)
