import framework.data.formats as formats
from framework.data.met_data.degree_days import degree_days
import logging
from copy import deepcopy


class MetDataAccumulator:
    def __init__(self, emergence_date, met_data, t_base):
        self._emergence_date = emergence_date
        self._t_base = t_base

        met_data = deepcopy(met_data)
        self._met_data = self.accumulate(met_data)
        self.warn_missing()

    def get_record(self, date):
        try:
            return next(x for x in self._met_data if x[formats.DATE] == date)
        except:
            # date not available, replace with the closest one
            dates = [x[formats.DATE] for x in self._met_data]
            new_date = min(dates, key=lambda x: abs((x - date).days))

            logging.warning("Giving bogus met data for record on %s"
                            % date.strftime("%d/%m/%Y"))

            return self.get_record(new_date)

    def accumulate(self, met_data):
        self._missing_dates = []

        met_data = [x for x in met_data if x[formats.DATE] >=
                    self._emergence_date]
        met_data.sort(key=lambda x: x[formats.DATE])

        accum_dd = 0
        accum_vars = [formats.PAR, formats.RAINFALL]
        accum_dict = {}
        for key in accum_vars:
            accum_dict[key] = 0

        for met_entry in met_data:
            # DD accumulation part
            met_entry[formats.DD] = accum_dd
            delta_dd = degree_days(met_entry[formats.T_MAX],
                                   met_entry[formats.T_MIN],
                                   self._t_base)

            if delta_dd is None:
                self._missing_dates.append(met_entry[formats.DATE])
            else:
                accum_dd += delta_dd

            # rest accumulation part
            for key in accum_vars:
                delta = met_entry[key]
                if delta is None:
                    self._missing_dates.append(met_entry[formats.DATE])
                else:
                    met_entry[key] = accum_dict[key]
                    accum_dict[key] += delta

        self._missing_dates = sorted(set(self._missing_dates))
        return met_data

    def warn_missing(self):
        for date in self._missing_dates:
            logging.warning("Missing met data on %s" %
                            date.strftime("%d/%m/%Y"))
