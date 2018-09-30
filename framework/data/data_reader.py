# Class that reads all the data for a particular location

from framework.data.location import Location

from framework.data.phenotype.pheno_reader.pheno_reader_csv \
    import PhenoReaderCSV
from framework.data.phenotype.sunscan_reader.sunscan_reader \
    import SunScanReader
from framework.data.phenotype.pheno_reader.yield_reader_csv \
    import YieldReaderCSV

from framework.data.met_data.metdata_reader_csv import MetDataReaderCSV
from framework.data.met_data.metdata_accumulor import MetDataAccumulator
from framework.data.phenotype.pheno_reader.fll_reader import FLLReader
from framework.util.csv_io.csv_filereader import CSVFileReader

import framework.util.misc as misc
import framework.data.formats as formats
from framework.util.evaluator_functions import in_season, in_season_uid, \
    uid_prep_date

import os
from datetime import datetime
from copy import deepcopy


class DataReader:
    """Reads all the data for a particular location"""
    # All CD compatible variables
    _cd_variables = [formats.UID,
                     formats.DATE,
                     formats.GENOTYPE,
                     formats.TRANSMISSION,
                     formats.DD,
                     formats.PAR,
                     formats.LEAF_AREA_INDEX,
                     formats.STEM_COUNT,
                     formats.CANOPY_HEIGHT,
                     formats.DW_PLANT]

    def __init__(self, location_name, cache=False, t_base=0):
        """location_name - Any of the locations defined in the Location
        class - Location.BSBEC, Location.ABR15, etc.
        """
        self._location_name = location_name
        self._t_base = t_base

        if cache and self._cache_load():
            return

        self._read_raw_data(self._location_name)
        self._combined_data = self._combine_data()

        if cache:
            self._cache_save()

    def get_records(self):
        return deepcopy(self._combined_data)

    def get_data(self, var, level, dw_subset):
        """A new method for returning data - provides flexibility for choosing
        different scenarios:
        var - possible values ['CD', 'ML']. Whether the ML data will contain
        all ML compatible variables or just the CD compatible ones
        level - ['plot', 'plant']. Both CD and ML data will be given as either
        'plot' or 'plant' level
        subset - whether to subset only records containing dry weight
        measurements
        Returns:
        {'cd_data': data_for_cd_model,
         'ml_data': data_for_ml_model}
        """
        # handle levels
        data = dict()
        data['cd_data'] = self.get_cdmodel_data(level)
        data['ml_data'] = self.get_ML_data(level)

        # handle variable subsetting
        if var == 'CD':
            for entry in data['ml_data']:
                keys = entry.keys()
                for key in keys:
                    if key not in self._cd_variables:
                        del entry[key]

        # handle dw subsetting
        if dw_subset is True:
            for key in data.keys():
                # get rid of records without dry weight measurements
                data[key] = [x for x in data[key] if
                             x[formats.DW_PLANT] is not None]

                # get rid of off season data
                data[key] = [x for x in data[key] if
                             x[formats.DATE].month > 3 and
                             x[formats.DATE].month < 11]

        return data

    def get_delta_data(self, var, level, dw_subset):
        """An ever newer method to get the delta data format, needed for
        training the hybrid process/ML model. The method returns
        data['ml_data'] in the old format + additional delta version of
        the variables, and does nothing on data['cd_data']
        """
        frequencies = ['pheno', 'yield']
        data = self.get_data(var, level, False)
        ml_data = deepcopy(data['ml_data'])

        # filter out the out of season data
        ml_data = [x for x in ml_data if
                   x[formats.DATE].month > 3 and
                   x[formats.DATE].month < 11]

        preps = misc.uniq_key(ml_data, formats.PSEUDO_REP)
        ml_data = self._inject_fll(ml_data, preps, True)

        new_data = dict()
        new_data['init_entries'] = []
        for frequency in frequencies:
            new_data[frequency] = []
            if frequency == 'yield':
                freq_data = [x for x in ml_data if
                             x[formats.DW_PLANT] is not None]
            else:
                freq_data = ml_data

            # group by uid, pseudo_rep_no, year
            uids = misc.uniq_key(freq_data, formats.UID)
            years = misc.uniq_key(freq_data, formats.YEAR)

            for uid in uids:
                for prep in preps:
                    for year in years:
                        subset = [x for x in freq_data if
                                  x[formats.UID] == uid and
                                  x[formats.PSEUDO_REP] == prep and
                                  x[formats.YEAR] == year]

                        # make sure it is sorted by day
                        subset.sort(key=lambda x: x[formats.DATE])
                        for entry in subset:
                            if subset.index(entry) == 0:
                                # first entry so do nothing
                                prev_entry = entry

                                # make sure entry is not in init_entries
                                # before adding it
                                if len([x for x in new_data['init_entries'] if
                                        x[formats.UID] ==
                                        entry[formats.UID] and
                                        x[formats.PSEUDO_REP] ==
                                        entry[formats.PSEUDO_REP] and
                                        x[formats.DATE] ==
                                        entry[formats.DATE]]) == 0:
                                    new_data['init_entries'].append(entry)

                                continue

                            new_entry = deepcopy(entry)
                            # copy all the meta data
                            meta_keys = [formats.UID, formats.GENOTYPE,
                                         formats.DATE, formats.YEAR,
                                         formats.ROW, formats.COL,
                                         formats.PSEUDO_REP]

                            # get diff between rest
                            pheno_keys = [x for x in entry.keys()
                                          if x not in meta_keys]

                            if frequency == "pheno":
                                pheno_keys.remove(formats.DW_PLANT)

                            for pkey in pheno_keys:
                                delta_pkey = "%s_%s" % ("delta", pkey)
                                if entry[pkey] is not None and \
                                   prev_entry[pkey] is not None:
                                    delta = entry[pkey] - prev_entry[pkey]
                                    new_entry[delta_pkey] = delta
                                else:
                                    new_entry[delta_pkey] = None

                            prev_entry = entry
                            new_data[frequency].append(new_entry)

        data['ml_data'] = new_data
        return(data)

    def get_ML_data(self, level):
        """Returns the mean of the phenotypic measurements for each plot/date
        combination. Takes the mean only for the data points with dry matter
        measurement.
        """
        if level == "plant":
            data = self.get_records()
        elif level == "plot":
            data = self.mean_per_plot()
        else:
            raise Exception("Unknown level %s" % level)

        # add all the extra ML variables
        for entry in data:
            entry[formats.DOY] = int(entry[formats.DATE].strftime("%j"))
            entry[formats.YEAR] = int(entry[formats.DATE].strftime("%Y"))

        return data

    def _read_raw_data(self, location_name):
        # field specific configuration of variables
        if location_name == Location.BSBEC:
            self._years = [2011, 2012, 2014, 2015, 2016]
        else:
            raise Exception("Error! Unhandled field %s." % location_name)

        locations = self._create_locations(self._years, location_name)
        self._pheno_data = self._retreive_data(locations, PhenoReaderCSV)
        self._sunscan_data = self._retreive_data(locations, SunScanReader)
        self._yield_data = self._retreive_data(locations, YieldReaderCSV)
        self._met_data = self._retreive_data(locations, MetDataReaderCSV,
                                             t_base=self._t_base)

    def _retreive_data(self, locations, reader_class, **kwargs):
        data = []
        for location in locations:
            data += reader_class(location, **kwargs).get_records()

        return data

    def _combine_data(self):
        required_keys = [formats.UID,
                         formats.ROW,
                         formats.COL,
                         formats.GENOTYPE,
                         formats.DATE,
                         formats.PSEUDO_REP,
                         formats.CANOPY_HEIGHT,
                         formats.STEM_COUNT,
                         formats.LEAF_AREA_INDEX,
                         formats.BASE_DIAM_1,
                         formats.BASE_DIAM_2,
                         formats.STEM_DIAM,
                         formats.TRANSMISSION,
                         formats.FLOWERING_SCORE,
                         formats.DW_PLANT,
                         formats.DD,
                         formats.PAR,
                         formats.RAINFALL]

        final_data = []
        if self._location_name == Location.BSBEC:
            final_data += self._combine_BSBEC_pheno_data(self._pheno_data)
            final_data = self._BSBEC_add_yield(final_data, self._yield_data)
            final_data = self._BSBEC_add_met_data(final_data, self._met_data,
                                                  self._years)

            for entry in final_data:
                # add row and col values [new in 2017]
                entry = misc.row_col_BSBEC(entry)

                if entry[formats.DATE].year < 2013:
                    entry[formats.TRANSMISSION] = \
                        self._get_matching_pheno(entry,
                                                 formats.TRANSMISSION,
                                                 self._sunscan_data,
                                                 pseudo_rep=0)
                else:
                    entry[formats.TRANSMISSION] = \
                        self._get_matching_pheno(entry,
                                                 formats.TRANSMISSION,
                                                 self._sunscan_data)

                # remove the keys not needed
                rm_keys = [x for x in entry.keys() if x not in required_keys]
                for key in rm_keys:
                    del entry[key]

                # add the missing keys as empty data
                missing = [x for x in required_keys if x not in entry.keys()]
                for key in missing:
                    entry[key] = None

        else:
            raise Exception("Cannot combine data for %s" % self._location_name)

        final_data = self._fill_vals(final_data)
        return final_data

    def _BSBEC_add_met_data(self, data, met_data, years):
        uids = misc.uniq_key(data, formats.UID)

        for uid in uids:
            for year in years:
                fll_reader = FLLReader(Location(Location.BSBEC, year))
                emergence = fll_reader.get_plot_fll(uid)
                max_date = datetime(year + 1, 3, 10)

                subset = [x for x in data if
                          in_season_uid(x, uid, emergence, max_date)]

                met_data_subset = [x for x in met_data if
                                   in_season(x, emergence, max_date)]

                accumulator = MetDataAccumulator(emergence,
                                                 met_data_subset,
                                                 self._t_base)

                for entry in subset:
                    met_entry = accumulator.get_record(entry[formats.DATE])
                    entry[formats.DD] = met_entry[formats.DD]
                    entry[formats.PAR] = met_entry[formats.PAR]
                    entry[formats.RAINFALL] = met_entry[formats.RAINFALL]

        return data

    def _combine_BSBEC_pheno_data(self, data):
        final_data = []
        uids = misc.uniq_key(data, formats.UID)
        dates = misc.uniq_key(data, formats.DATE)
        pseudo_reps = misc.uniq_key(data, formats.PSEUDO_REP)
        if 0 in pseudo_reps:
            pseudo_reps.pop(pseudo_reps.index(0))

        for uid in uids:
            for pseudo_rep in pseudo_reps:
                # measurements from the previous date stored here
                prev_msmts = []
                prev_stem_id = 1

                for date in dates:
                    # at the beginning of new season wipe previous leaf
                    # measurements
                    if len(prev_msmts):
                        prev_date = prev_msmts[0][formats.DATE]
                        if date.year != prev_date.year:
                            prev_msmts = []
                            prev_stem_id = 1

                    # subset just the measurements of today
                    today = [x for x in data if
                             uid_prep_date(x, uid, pseudo_rep, date)]

                    today.sort(key=lambda x: x[formats.MEASUREMENT])

                    # check for no measurements
                    if not len(today):
                        continue

                    # determine if this date has any leaf area measurements
                    if (misc.uniq_key(today, formats.MEASUREMENT) == [0] and
                       len(prev_msmts) == 0):

                        # check if there is any point to the measurements
                        base_keys = [formats.CANOPY_HEIGHT,
                                     formats.STEM_COUNT]
                        useful_entries = [x for x in today if not
                                          set(base_keys).isdisjoint(x.keys())]

                        # add the entries that have no leaf measurements but
                        # have other pheno measurements
                        for u_entry in useful_entries:
                            final_data.append(u_entry)

                        # do not try to process the leaf measurements as there
                        # aren't any
                        continue

                    # first grab the header entry and populate with canopy
                    # height & stem count this has been tested and it should
                    # never give more than one match
                    header = next(x for x in today if
                                  x[formats.MEASUREMENT] == 0)

                    if date.year not in [2014, 2016]:
                        # make sure stem ids are consistent - max difference
                        # should be one
                        if header[formats.STEM_ID] - prev_stem_id == 1:
                            prev_msmts = []
                            prev_stem_id = header[formats.STEM_ID]
                        elif header[formats.STEM_ID] != prev_stem_id:
                            raise Exception("Weird")

                    if date.year < 2013:
                        # correct Chris Davey's data - find
                        # corresponding canopy heights and stem count
                        keys = [formats.STEM_COUNT, formats.CANOPY_HEIGHT]
                        for pheno in keys:
                            header[pheno] = \
                                self._get_matching_pheno(header,
                                                         pheno,
                                                         data,
                                                         pseudo_rep=0)

                    # convert leaf measurements -> leaf area
                    leaf_msmts = [x for x in today if
                                  x[formats.MEASUREMENT] != 0]

                    if date.year != 2014:
                        # add in previously measured leaves
                        for entry in prev_msmts:
                            msmt_ns = [x[formats.MEASUREMENT] for
                                       x in leaf_msmts]

                            if entry[formats.MEASUREMENT] not in msmt_ns:
                                leaf_msmts.append(entry)

                    leaf_msmts.sort(key=lambda x: x[formats.MEASUREMENT])

                    # drop dead leaves
                    if date.year < 2014:
                        leaf_msmts = [x for x in leaf_msmts if
                                      not self._is_dead(x)]

                    # new way to kill leaves - kill all leaves before the last
                    # dead leaf
                    elif date.year > 2014:
                        last_dead = None
                        for entry in leaf_msmts:
                            if self._is_dead(entry):
                                if not last_dead:
                                    last_dead = entry
                                elif (last_dead[formats.MEASUREMENT] <
                                      entry[formats.MEASUREMENT]):
                                    last_dead = entry

                        if last_dead is not None:
                            tbd = [x for x in leaf_msmts if
                                   x[formats.MEASUREMENT] <
                                   last_dead[formats.MEASUREMENT]]

                            if len(tbd) > 5 and date.month < 8:
                                # JUST IN CASE
                                raise Exception("You should see this!")

                            # get rid of all the dead leaves
                            leaf_msmts = [x for x in leaf_msmts
                                          if x[formats.MEASUREMENT] >
                                          last_dead[formats.MEASUREMENT]]

                    total_area = 0
                    for measurement in leaf_msmts:
                        total_area += misc.get_area(measurement)

                    if formats.STEM_COUNT in header.keys():
                        # convert 1 stem leaf area to m^2
                        # get total leaf area of the plant in m^2
                        # divide by 0.5 to convert to m^2 leaf / m^2 ground
                        LAI = ((total_area / 10000.0) *
                               header[formats.STEM_COUNT]) / 0.5

                        header[formats.LEAF_AREA_INDEX] = LAI
                    else:
                        header[formats.LEAF_AREA_INDEX] = None

                    final_data.append(header)
                    prev_msmts = leaf_msmts

        return final_data

    def _create_locations(self, years, location_name):
        locations = []
        for year in years:
            locations.append(Location(location_name, year))

        return locations

    def _get_matching_pheno(self, header, pheno, data, pseudo_rep=-999):
        """Gets the corresponding phenotypic measurement to the header entry. A
        'corresponding' measurment is one that is done on the same plant, or at
        least plot, with the closest date available, with maximum distance of a
        week between the header entry and the measurement.
        """
        if pseudo_rep != -999:
            subset = [x for x in data if pheno in x.keys() and
                      x[formats.UID] == header[formats.UID] and
                      x[formats.PSEUDO_REP] == pseudo_rep]
        else:
            subset = [x for x in data if pheno in x.keys() and
                      x[formats.UID] == header[formats.UID] and
                      x[formats.PSEUDO_REP] == header[formats.PSEUDO_REP]]

        min_diff = None
        for entry in subset:
            diff = abs((header[formats.DATE] - entry[formats.DATE]).days)
            if min_diff is None:
                min_diff = diff
            else:
                if diff < min_diff:
                    min_diff = diff

        matching = [x for x in subset if
                    abs((x[formats.DATE] - header[formats.DATE]).days) ==
                    min_diff]

        if min_diff > 7:
            return None

        if len(matching) > 1:
            total = 0
            for entry in matching:
                total += entry[pheno]

            total /= len(matching)
            return total
        else:
            return matching[0][pheno]

    def _is_dead(self, leaf):
        if leaf[formats.LEAF_WIDTH] == 0.0 or leaf[formats.LEAF_LENGTH] == 0.0:
            return True
        else:
            return False

    def _get_matching_entries(self, entry, data, threshold=10):
        """Searches for a matching entry to the one provided. A matching entry
        is one that is closest in date to the original entry, with a maximum of
        <threshold> days difference. Also the matching entry needs to be of a
        plant of the same plot.

        The function assumes that: entry - is a dict entry from yield data data
        - is a list of dict entries, containing the other phenotypic
        measurements.

        For entries with pseudo_rep > 3 (growing season harvests), it will
        return the matching measurements for all 3 plants from the plot.  For
        entries with pseudo_rep in [1,2,3] (end of year harvests), it will
        return the single matching plant in a list where available (only in
        2016 harvest) For entries with pseudo_rep == 0 (the 12 plant harvest)
        it will raise an Exception
        """
        if entry[formats.PSEUDO_REP] == 0:
            raise Exception("Cannot provide matching measurement "
                            "for pseudo rep 0")

        if entry[formats.PSEUDO_REP] > 3:
            # growing season measurement - pseudo_rep does not correspond to
            # any measured plant atm
            subset = [x for x in data if x[formats.UID] == entry[formats.UID]]
        else:
            # end of season measurement - a measured plant should have been
            # harvested
            subset = [x for x in data if
                      x[formats.UID] == entry[formats.UID] and
                      x[formats.PSEUDO_REP] == entry[formats.PSEUDO_REP]]

        # find the closest measurement date
        min_diff = None
        for pheno_entry in subset:
            diff = abs((entry[formats.DATE] - pheno_entry[formats.DATE]).days)

            if min_diff is None or diff < min_diff:
                min_diff = diff

        matching = [x for x in subset if
                    abs((x[formats.DATE] - entry[formats.DATE]).days) ==
                    min_diff]

        # we allow a maximum of <threshold> day difference
        if min_diff > threshold or not matching:
            return None

        if entry[formats.PSEUDO_REP] in [1, 2, 3]:
            # some sanity check - len(matching) is 1 with end of year harvests
            # only
            assert(len(matching) == 1)

        elif len(matching) > 3:
            # deal with two dates being equally away from the harvest date
            dates = misc.uniq_key(matching, formats.DATE)
            if len(dates) != 2:
                raise Exception("More than two dates!")

            # arbitrarily pick the earlier date
            matching = [x for x in matching if x[formats.DATE] == dates[0]]
            print("Making the switch for plot %d, %s at %s" %
                  (entry[formats.UID], entry[formats.GENOTYPE],
                   dates[0].strftime("%d/%m/%Y")))

        return matching

    def _BSBEC_add_yield(self, final_data, yield_data):
        """Looks for a matching pheno measurement for every yield measurement
        date. If found, if would update that pheno measurement entry with the
        dry weight value from the yield data. Otherwise, it would create a new
        entry with just the yield data added.
        """
        dates = misc.uniq_key(yield_data, formats.DATE)
        uids = misc.uniq_key(yield_data, formats.UID)

        for uid in uids:
            for date in dates:

                try:
                    entry = next(x for x in yield_data if
                                 x[formats.UID] == uid and
                                 x[formats.DATE] == date)

                except:
                    print("Missing %s, %s" % (uid, date.strftime("%d/%m/%Y")))
                    continue

                if entry[formats.PSEUDO_REP] == 0:
                    # to be handled
                    continue

                matching = self._get_matching_entries(entry, final_data)

                if matching is not None:
                    # add the yield measurement to the matched measurements
                    for match in matching:
                        match[formats.DW_PLANT] = entry[formats.DW_PLANT]
                else:
                    # no measurement matches - create a new entry
                    pseudo_reps = range(1, 4)
                    for prep in pseudo_reps:
                        final_data.append({formats.UID: uid,
                                           formats.DATE: date,
                                           formats.PSEUDO_REP: prep,

                                           formats.GENOTYPE:
                                           entry[formats.GENOTYPE],

                                           formats.DW_PLANT:
                                           entry[formats.DW_PLANT]})

        return final_data

    def _fill_vals(self, data):
        # Finally fix the dataset, so that all keys are available, even though
        # they have no value make writing to csv easier
        keys = []
        for entry in data:
            for key in entry.keys():
                if key not in keys:
                    keys.append(key)

        for entry in data:
            for key in keys:
                if key not in entry.keys():
                    entry[key] = None

        return data

    def _cache_load(self):
        cache_fname = os.path.join(os.environ.get("HOME"),
                                   ".data_reader.cache")

        if os.path.isfile(cache_fname):
            self._combined_data = CSVFileReader(cache_fname).get_content()
            for entry in self._combined_data:
                entry = formats.on_read(entry, None)

            self._combined_data = self._fill_vals(self._combined_data)

            print("Loaded cache data from %s" % cache_fname)
            return True
        else:
            return False

    def _cache_save(self):
        import os
        from framework.util.csv_io.csv_filewriter import CSVFileWriter

        cache_fname = os.path.join(os.environ.get("HOME"),
                                   ".data_reader.cache")
        CSVFileWriter(cache_fname, self._combined_data)

    def mean_per_plot(self):
        input_data = self.get_records()

        uids = misc.uniq_key(input_data, formats.UID)
        dates = misc.uniq_key(input_data, formats.DATE)
        phenos = [formats.TRANSMISSION,
                  formats.CANOPY_HEIGHT,
                  formats.DW_PLANT,
                  formats.LEAF_AREA_INDEX,
                  formats.STEM_COUNT,
                  formats.BASE_DIAM_1,
                  formats.BASE_DIAM_2,
                  formats.STEM_DIAM,
                  formats.RAINFALL,
                  formats.DD,
                  formats.PAR,
                  formats.FLOWERING_SCORE]

        output_data = []

        for uid in uids:
            for date in dates:
                subset = [x for x in input_data if x[formats.UID] == uid and
                          x[formats.DATE] == date]

                if len(subset) == 0:
                    continue

                entry = dict()
                for pheno in phenos:
                    values = [x[pheno] for x in subset if x[pheno] is not None]

                    if len(values) > 0:
                        try:
                            entry[pheno] = sum(values)/len(values)
                        except:
                            import ipdb
                            ipdb.set_trace()
                    else:
                        entry[pheno] = None

                entry[formats.UID] = uid
                entry[formats.DATE] = date
                entry[formats.GENOTYPE] = subset[0][formats.GENOTYPE]
                entry[formats.PSEUDO_REP] = 0  # 0 is plot level data
                entry = misc.row_col_BSBEC(entry)
                output_data.append(entry)

        return output_data

    def get_cdmodel_data(self, level='plot'):
        """Returns the combined dataset in the format appropriate for training
        Chris Davey's physiological model - the mean values for each plot will
        be taken
        """
        if level == 'plot':
            data = deepcopy(self.mean_per_plot())
        elif level == 'plant':
            # experimental feature - use the plant level
            data = deepcopy(self.get_records())

        # get rid of CD incompatible keys
        for entry in data:
            keys = entry.keys()
            for key in keys:
                if key not in self._cd_variables:
                    del entry[key]

        data = self._inject_fll(data)
        return sorted(data, key=lambda x: (x[formats.UID], x[formats.DATE]))

    def _inject_fll(self, data, preps=[], additional_phenos=False):
        # inject at FLL
        uids = misc.uniq_key(data, formats.UID)
        years = set([x[formats.DATE].year for x in data])

        for year in years:
            reader = FLLReader(Location(Location.BSBEC, year))
            for uid in uids:
                fll_date = reader.get_plot_fll(uid)
                entry = {formats.UID: uid,
                         formats.DATE: fll_date,
                         formats.GENOTYPE: misc.assign_geno_bsbec(uid),
                         formats.TRANSMISSION: 1.0,
                         formats.DD: 0.0,
                         formats.PAR: 0.0,
                         formats.LEAF_AREA_INDEX: 0.0,
                         formats.STEM_COUNT: 0.0,
                         formats.CANOPY_HEIGHT: 0.0,
                         formats.DW_PLANT: 0.0}

                if additional_phenos:
                    if year < 2016:
                        diam = None
                    else:
                        diam = 0

                    entry[formats.BASE_DIAM_1] = diam
                    entry[formats.BASE_DIAM_2] = diam
                    entry[formats.STEM_DIAM] = diam
                    entry[formats.YEAR] = year
                    entry = misc.row_col_BSBEC(entry)
                    entry[formats.DOY] = int(entry[formats.DATE]
                                             .strftime("%j"))
                    entry[formats.FLOWERING_SCORE] = 0
                    entry[formats.RAINFALL] = 0

                if preps != []:
                    for prep in preps:
                        new_entry = deepcopy(entry)
                        new_entry[formats.PSEUDO_REP] = prep
                        data.append(new_entry)
                else:
                    data.append(entry)

        return data

    def get_fll_date(self, condition):
        """
        This function gives the FLL info to the CDModel class,
        to be used in simulations"""
        geno = condition[formats.GENOTYPE]
        year = condition['year']
        fll_reader = FLLReader(Location(self._location_name, year))

        return fll_reader.get_genotype_fll(geno)

    def get_met_data(self, condition, t_base):
        location = Location(self._location_name, condition['year'])
        met_data = MetDataReaderCSV(location, t_base=t_base).get_records()
        return met_data
