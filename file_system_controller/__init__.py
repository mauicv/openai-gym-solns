import csv
from datetime import datetime
from os import path
import os


class FileSystemConfig:
    def __init__(self, dirname, name):
        self.name = name
        self.dirname = dirname

        if not path.exists(dirname):
            os.mkdir(dirname)
        self.filename = path.join(dirname, name)

    @property
    def file_loc(self):
        return path.join(self.dirname, self.name+'.csv')


class FileWriter(FileSystemConfig):
    def __init__(self, dirname, name):
        super().__init__(dirname, name)

    def _get_writer(self, csvfile):
        return csv.writer(csvfile, delimiter=',',
                          quotechar='|',
                          quoting=csv.QUOTE_MINIMAL)

    def init_file(self, val):
        with open(self.file_loc, 'w', newline='') as csvfile:
            writer = self._get_writer(csvfile)
            writer.writerow(['timestamp', *val])

    def write_val(self, val):
        with open(self.file_loc, 'a', newline='') as csvfile:
            writer = self._get_writer(csvfile)
            writer.writerow([get_time_to_millisecond(), val])


class FileReader(FileSystemConfig):
    def __init__(self, dirname, name):
        super().__init__(dirname, name)

    def _get_reader(self, csvfile):
        return csv.reader(csvfile, delimiter=',',
                          quotechar='|')

    def __call__(self):
        data = {}
        with open(self.file_loc, 'r', newline='') as csvfile:
            reader = self._get_reader(csvfile)
            t_key, v_key = next(reader)
            data[v_key] = []
            data[t_key] = []
            for row in reader:
                data[t_key].append(float(row[0]))
                data[v_key].append(float(row[1]))
        return data


def get_time_to_millisecond():
    return float((datetime.utcnow() - datetime(1970, 1, 1))
                 .total_seconds())
