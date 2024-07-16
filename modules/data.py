import json

from .globals import logger


class Data:
    def __init__(self) -> None:
        self.file_name = dict(dev="dev.json", test="test.json", train="train.json")
        self.data_path = "./round1_training_data/"
        self.data = self.load_data()

    def load_data(self):
        # read data
        data = dict()
        for key in self.file_name:
            with open(self.data_path + self.file_name[key], "r") as f:
                data[key] = json.load(f)
                logger.info(f"load {key} data, len: {len(data[key])}")

        return data

    def get_item(self, key, idx):
        return self.data[key][idx]

    def get_split(self, key):
        return self.data[key]
