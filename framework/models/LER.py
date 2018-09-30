import tempfile
import os
import subprocess
import math

from framework.util.csv_io.csv_filewriter import CSVFileWriter


class LER:
    NLS_KEYS = ['L', 'k', 'x_mid']
    LM_KEYS = ['intercept', 'x_b']

    def __init__(self, data):
        self._coefficients = self._calc_coefficients(data)

    def _calc_coefficients(self, data):
        f = tempfile.NamedTemporaryFile()
        CSVFileWriter(f.name, data)

        cmd = [os.path.join(os.environ['R_SCRIPTS'], 'LER.R'), '-i', f.name]
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE)
        out = proc.communicate()[0]

        result = dict()
        for entry in out.strip().split("\n"):
            content = entry.split(":")
            result[content[0]] = float(content[1])

        # determine type of model
        if set(self.NLS_KEYS) == set(result.keys()):
            self._type = "NLS"
        elif set(self.LM_KEYS) == set(result.keys()):
            self._type = "LM"
        else:
            raise Exception("Unkown model type")

        return result

    def get_slope(self, x):
        if self._type == "NLS":
            L = self._coefficients['L']
            k = self._coefficients['k']
            x_mid = self._coefficients['x_mid']

            result = (L * k * math.exp(-k * (x - x_mid))) /\
                     ((1 + math.exp(-k * (x - x_mid)))**2)
        elif self._type == "LM":
            if x == 0:
                x += 0.01

            result = self._coefficients['x_b']/x

        return result
