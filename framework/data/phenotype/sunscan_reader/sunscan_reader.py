import os
import ast
import re
import StringIO
import csv
import logging

import framework.util.config_parser as config_parser
from framework.util.csv_io.csv_filereader import CSVFileReader
from framework.data import formats
SINGLE_FILE = "file"
MULTIPLE_FILES = "multiple_files"


class SunScanReader:

    default_ini = os.path.join(os.environ['ALF_CONFIG'], "sunscan.ini")

    def __init__(self, location, **kwargs):
        # read the sunscan.ini config file
        self._config = config_parser.SpaceAwareConfigParser()
        self._config.read(os.path.expanduser(self.default_ini))

        # get the format for this field/year
        format_dict = ast.literal_eval(self._config.get(location.get_name(),
                                       'format'))

        if 'all' in format_dict.keys():
            self._format = format_dict['all']
        else:
            self._format = format_dict[location.get_year()]

        self._location = location
        self._data = self.read_directory(location.get_sunscan_dir())

    def read_directory(self, path):
        if self._format == SINGLE_FILE:
            data = CSVFileReader(os.path.join(path,
                                 "transmission.csv")).get_content()

            for entry in data:
                entry[formats.UID] = entry['Plot']
                entry[formats.PSEUDO_REP] = 0
                entry[formats.MEASUREMENT] = 0
                entry[formats.DATE] = "%s %s" % (entry['Day'], entry['Year'])
                entry[formats.TIME] = None

                del entry['Plot']
                del entry['Year']
                del entry['Day']

                entry = formats.on_read(entry, self._location, "%j %Y")

            return data

        elif self._format == MULTIPLE_FILES:
            # read all txt files in path
            files = os.listdir(path)
            r = re.compile(".*(TXT|txt)$")
            files = filter(r.match, files)
            data = []
            for f in files:
                data += self.read_txt(os.path.join(path, f))

            return data

    def read_txt(self, filename):
        f = open(filename)
        # TODO remove
        content = f.readlines()
        f.close()

        # use the filename to determine measurement date
        basename = os.path.basename(filename)
        date = re.split("_|\.| ", basename)[1]

        # grab the line that will be used as the key for the csv
        i, key = next((i, line) for i, line in enumerate(content)
                      if re.match("Time\t.*tip 64", line))

        # slice off the top header as it contains NOTHING important
        content = content[i+2:]
        content = ''.join(content)

        mock_f = StringIO.StringIO(content)
        key = key.strip()  # remove the endline madness
        key = key.split("\t")
        reader = csv.DictReader(mock_f, delimiter="\t", fieldnames=key)

        data = []
        calibration = []
        for row in reader:
            # insert the measurement date
            row['Date'] = date
            row['PAR'] = float(row['PAR'])

            row['Total'] = float(row['Total'])
            if row['Total'] < 0.3:
                continue

            row = self._fix_synonyms(row)

            entry = dict()

            # process as calibration data
            if row['Plot'] == "256":
                entry['PAR'] = row['PAR']
                entry['Total'] = row['Total']
                calibration.append(entry)
                continue

            # process as normal data
            entry[formats.UID] = row['Plot']
            row['Sample'] = int(row['Sample'])
            if row['Sample'] % 2 == 1:
                entry[formats.MEASUREMENT] = 1
            else:
                entry[formats.MEASUREMENT] = 2

            if row['Sample'] < 3:
                entry[formats.PSEUDO_REP] = 1
            elif row['Sample'] < 5:
                entry[formats.PSEUDO_REP] = 2
            elif row['Sample'] < 7:
                entry[formats.PSEUDO_REP] = 3
            else:
                raise Exception("Unhandled sample number %d" % row['Sample'])

            entry[formats.TIME] = row['Time']
            entry[formats.DATE] = row['Date']
            entry[formats.TRANSMISSION] = row['PAR']/row['Total']
            entry = formats.on_read(entry, self._location, "%d%m%y")

            # check if there are previous records for that plant
            prev_records = [x for x in data if x[formats.UID] ==
                            entry[formats.UID] and
                            x[formats.PSEUDO_REP] == entry[formats.PSEUDO_REP]
                            and
                            (x[formats.DATE] - entry[formats.DATE]).days < 2]

            if len(prev_records) >= 2:
                import ipdb
                ipdb.set_trace()

            if entry[formats.TRANSMISSION] > 1.0:
                logging.warn("Dropping invalid transmission measurement")
            else:
                data.append(entry)

        data.sort(key=lambda x: (x[formats.DATE], x[formats.UID]))

        # calibrate if data is available
        if calibration:
            ratios = []
            for entry in calibration:
                ratios.append(entry['PAR'] / entry['Total'])

            ratio = sum(ratios) / len(ratios)
            for entry in data:
                entry[formats.TRANSMISSION] = entry[formats.TRANSMISSION]/ratio

        return data

    def _fix_synonyms(self, entry):
        synonyms = self._config.items("synonyms")
        for synonym in synonyms:
            if synonym[0] not in entry.keys():
                # then one of the synonyms will be present
                possible_values = synonym[1].split(",")
                for val in possible_values:
                    if val in entry.keys():
                        entry[synonym[0]] = entry.pop(val)
                        break
                else:
                    raise Exception("Could not find variable %s " % synonym[0])

        return entry

    def get_records(self):
        return self._data
