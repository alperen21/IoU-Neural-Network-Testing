import yaml

with open("config.yaml", "r") as file:
    config = yaml.safe_load(file)

DATABASE_URI = config["mixed_list"]
print(DATABASE_URI)