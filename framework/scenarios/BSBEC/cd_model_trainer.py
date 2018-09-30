# This class trains a Chris Davey model on BSBEC data and performs predictions.
from framework.models.cd_model import ChrisDaveyModel
from framework.data.data_reader import DataReader
from framework.data.location import Location
import framework.util.misc as misc
import framework.data.formats as formats
from framework.data.phenotype.pheno_reader.fll_reader_missing \
        import FLLReaderMissing
from copy import deepcopy


LOCATION = Location.BSBEC


class CDModelTrainer:
    # additional scenario for cross validation training
    PAPER = ChrisDaveyModel.PAPER
    PAPER_NO_TRAINING = ChrisDaveyModel.PAPER_NO_TRAINING
    CV_BETWEEN_YEARS = ChrisDaveyModel.CUSTOM
    NAME = "APBPM"

    def __init__(self, mode, **kwargs):
        """This class trains a Chris Davey model on BSBEC data and performs
        predictions.  It implements two scenarios:
        PAPER - train on year 2011 and 2012. Need to provide 'test_years' kwarg
        argument

        CV_BETWEEN_YEARS - train on years specified in kwargs['train_years']
        and test on years specified in kwargs['test_years']
        """
        self._reader = DataReader(Location.BSBEC, cache=True, t_base=0)
        if mode in [CDModelTrainer.PAPER,
                    CDModelTrainer.PAPER_NO_TRAINING,
                    CDModelTrainer.CV_BETWEEN_YEARS]:

            if mode == self.CV_BETWEEN_YEARS:
                # train on whatever years you are told to
                kwargs_mdl = {'years': kwargs['train_years'],
                              'location': LOCATION}

                # add custom train data
                if 'train_data' in kwargs.keys():
                    kwargs_mdl['train_data'] = kwargs['train_data']

            elif mode == self.PAPER:
                # by default (PAPER scenario) we train on years 2011 and 2012
                kwargs_mdl = {'years': [2011, 2012],
                              'location': LOCATION}

            elif mode == self.PAPER_NO_TRAINING:
                kwargs_mdl = {'years': [2011, 2012],
                              'location': LOCATION}

            self._model = ChrisDaveyModel(self._reader, mode, **kwargs_mdl)

            if 'test_data' not in kwargs.keys():
                # subset the year that will be simulated
                self._test_data = misc.subset_yr(
                                    self._reader.get_cdmodel_data(),
                                    kwargs['test_years'])
            else:
                self._test_data = deepcopy(kwargs['test_data'])

        else:
            raise Exception("Unknown mode %s" % mode)

        if 'keep_all' in kwargs.keys():
            self._simulate(True)
        else:
            self._simulate()

    def _simulate(self, keep_all=False):
        years = set([x[formats.DATE].year for x in self._test_data])
        genotypes = set([x[formats.GENOTYPE] for x in self._test_data])

        conditions = []
        for year in years:
            for genotype in genotypes:
                condition = {formats.GENOTYPE: genotype,
                             'year': year}

                conditions.append(condition)

        for condition in conditions:
            all_params = self._model.get_parameters()
            params = [x for x in all_params if
                      x['conditions'] == condition[formats.GENOTYPE]]

            kwargs = dict()
            keys = ['k', 'LER', 'RUE']
            for key in keys:
                kwargs[key] = next(x['parameter'] for x in params if
                                   x['variable'] == key)

            # hack to fix EMI-11 k value for 2012 and later
            # that applies just to paper_no_training scenario
            if self._model._scenario == ChrisDaveyModel.PAPER_NO_TRAINING and \
               condition[formats.GENOTYPE] == "EMI-11" and \
               condition[formats.YEAR] >= 2012:
                kwargs['k'] = 0.5533

            if condition['year'] > 2014:
                location = Location(Location.BSBEC, condition['year'])
                kwargs['fll_reader'] = FLLReaderMissing(location)

            simulation = self._model.simulate(condition,
                                              t_base=0,
                                              **kwargs)

            for entry in self._test_data:
                match = [x for x in simulation if
                         x[formats.GENOTYPE] == entry[formats.GENOTYPE] and
                         x[formats.DATE] == entry[formats.DATE]]

                if len(match) > 1:
                    # this should not happen
                    raise Exception("How does that even?!")
                elif len(match) == 1:
                    entry['predicted'] = match[0][formats.DW_PLANT]
                    if keep_all:
                        for var_name in [formats.PAR,
                                         formats.DD,
                                         formats.LEAF_AREA_INDEX]:
                            entry[var_name] = match[0][var_name]

    def get_raw_results(self):
        return self._test_data

    def get_results(self):
        return [x for x in self._test_data if
                not x[formats.DW_PLANT] in [None, 0, '']]
