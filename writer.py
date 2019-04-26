import os
import dill
import pandas as pd
from pathlib import Path


class Writer:

    @staticmethod
    def save_pickled_object(obj, file_path):
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, 'wb') as output:
            dill.dump(obj, output, dill.HIGHEST_PROTOCOL)
        print("WROTE " + file_path)

    @staticmethod
    def load_pickled_object(file_path):
        my_file = Path(file_path)
        if my_file.is_file():
            return dill.load(open(file_path, "rb"))
        else:
            return None

    @staticmethod
    def save_csv_object(obj, file_path):
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        obj.to_csv(file_path)

    @staticmethod
    def load_csv_object(file_path):
        my_file = Path(file_path)
        if my_file.is_file():
            return pd.DataFrame.from_csv(my_file)
        else:
            return None

    @staticmethod
    def log(*args):
        message = ' '.join(args) + '\n'
        with open("log.txt", "a") as log_file:
            log_file.write(message)