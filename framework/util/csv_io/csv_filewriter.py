# Utility class that facilitates csv writing

import csv
import copy
from framework.data import formats


class CSVFileWriter():
    def __init__(self, filename, content, custom_header=[], write_mode='w'):
        content = copy.deepcopy(content)
        f = open(filename, write_mode)

        content = formats.on_write(content)

        if custom_header and content:
            # custom_header and content are present
            # so we want to order the existing columns(keys) in
            # the specified order
            keys = content[0].keys()
            if len(keys) != len(custom_header):
                print("custom_header %d, keys %d" % (len(custom_header),
                                                     len(keys)))
                raise Exception("Wrong number of headers on custom header")

            keys = custom_header
        elif custom_header:
            # no content, so that we will just write down the header
            keys = custom_header
        elif content:
            keys = content[0].keys()

        writer = csv.DictWriter(f, keys)

        if write_mode != 'a':
            writer.writeheader()

        writer.writerows(content)

        f.close()
