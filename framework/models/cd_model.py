import framework.data.formats as formats
from framework.util.csv_io.csv_tempfile import CSVTempFile
from framework.models.LER import LER
from framework.models.LER_linear import LERLinear
from framework.data.phenotype.pheno_reader.fll_reader_missing \
    import FLLReaderMissing
from framework.data.location import Location

import subprocess
import os
import math
import copy
from datetime import datetime, timedelta


class ChrisDaveyModel:
    """My implementation of Chris Davey's model - parameterisation
    and prediction
    """

    # scenarios
    PAPER = "paper"
    PAPER_NO_TRAINING = "paper_no_training"
    CUSTOM = "custom"

    def __init__(self, reader, scenario, **kwargs):
        """kwargs possible keys:
        'data' - custom subset of the dataset to be used for training. Useful
        for doing within data cross validation
        'years' - years from the dataset to train on
        """
        self._reader = reader
        self._scenario = scenario

        # handle custom dataset provision
        if 'train_data' in kwargs.keys() and kwargs['train_data'] is not None:
            self._data = kwargs.pop('train_data')
        else:
            self._data = reader.get_cdmodel_data()

        if scenario == self.PAPER:
            self._paper_scenario(self._data)
        elif scenario == self.PAPER_NO_TRAINING:
            self._paper_scenario(self._data, train=False)
        elif scenario == self.CUSTOM:
            self._custom_scenario(self._data, **kwargs)
        else:
            raise Exception("Unknown scenario %s" % scenario)

    def get_parameters(self):
        result = self._get_params_from_dict(self._k, 'k')
        result += self._get_params_from_dict(self._RUE, 'RUE')
        result += self._get_params_from_dict(self._LER, 'LER')

        # clean up the years from the conditions
        for res in result:
            genotype = res['conditions'].split('(')[0]
            res['conditions'] = genotype

        return result

    def _calc_k(self, data, conditions):
        subset = self._subset_data(data, conditions)
        f = CSVTempFile(subset)

        cmd = [os.path.join(os.environ['R_SCRIPTS'], 'nls.R'),
               '-i', f.get_name()]
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE)
        out = proc.communicate()
        k = float(out[0].split(' ')[1])
        return {'conditions': conditions,
                'parameter': k}

    def _calc_LER(self, data, conditions):
        subset = self._subset_data(data, conditions)
        return {'conditions': conditions,
                'parameter': LER(subset)}

    def _calc_RUE(self, data, conditions):
        subset = self._subset_data(data, conditions)
        f = CSVTempFile(subset)

        cmd = [os.path.join(os.environ['R_SCRIPTS'], 'RUE.R'),
               '-i', f.get_name()]
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE)
        out = proc.communicate()
        RUE = float(out[0].split(' ')[1])

        return {'conditions': conditions,
                'parameter': RUE}

    def _subset_data(self, data, conditions):
        return [x for x in data if self._belongs_to_group(x, conditions)]

    def _belongs_to_group(self, x, conditions):
        entry = {formats.GENOTYPE: x[formats.GENOTYPE],
                 'year': x[formats.DATE].year}
        return entry in conditions

    def _get_variable(self, parameters, condition):
        return next(x['parameter'] for x in parameters
                    if condition in x['conditions'])

    def _paper_scenario(self, data, train=True):
        conditions = [{formats.GENOTYPE: 'Giganteus', 'year': 2011},
                      {formats.GENOTYPE: 'Sac-5',     'year': 2011},
                      {formats.GENOTYPE: 'EMI-11',    'year': 2011},
                      {formats.GENOTYPE: 'Goliath',   'year': 2011}]

        self._k = []
        self._LER = []
        self._RUE = []

        if train:
            # cheat to get EMI-11 to fit well
            data = [x for x in data if not
                    ((x[formats.DATE] == datetime(2011, 8, 2)) and
                     (x[formats.GENOTYPE] == "EMI-11"))]

            for condition in conditions:
                # train k
                k_condition = [condition, copy.deepcopy(condition)]
                # add 2012 year for k training
                k_condition[1]['year'] = 2012
                self._k.append(self._calc_k(data, k_condition))

                # train LER
                self._LER.append(self._calc_LER(data, [condition]))

                # perform simulations without calculating RUE
                simulated_data = self._training_simulation(condition,
                                                           get_RUE=False)
                real_data = self._subset_data(data, [condition])
                merged_data = self._merge_data(real_data, simulated_data)

                # use simulated PAR and real dry weight data to calculate RUE
                self._RUE.append(self._calc_RUE(merged_data, [condition]))

        else:
            # manually input the parameters from the paper
            for condition in conditions:
                breakpoint = None
                if condition[formats.GENOTYPE] == "EMI-11":
                    k = 1.129
                    slopes = 0.001395
                    RUE = 1.66
                elif condition[formats.GENOTYPE] == "Sac-5":
                    k = 0.6539
                    slopes = [0.001395, 0.006225]
                    breakpoint = 1029
                    RUE = 1.66
                elif condition[formats.GENOTYPE] == "Giganteus":
                    k = 0.6539
                    slopes = 0.003931
                    RUE = 2.40
                elif condition[formats.GENOTYPE] == "Goliath":
                    k = 0.5533
                    slopes = [0.002276, 0.006225]
                    breakpoint = 866
                    RUE = 1.66

                self._k.append({'conditions': [condition],
                                'parameter': k})

                self._LER.append({'conditions': [condition],
                                  'parameter': LERLinear(slopes, breakpoint)})

                self._RUE.append({'conditions': [condition],
                                  'parameter': RUE})

        # finally do the simulation with all parameters present
        final_data = []
        for condition in conditions:
            final_data += self._training_simulation(condition, get_RUE=True)

        return final_data

    def _custom_scenario(self, data, **kwargs):
        # TODO WARNING THIS does not work yet. You will need to fix the
        # simulation to allow for multiyear simulations. Also check out whether
        # you can train LER and RUE on multiyear data we know k works for
        # multiyear data
        # TODO [08/11/2016] I think that works now, so not sure what past Emil
        # is complaining about
        # TODO [08/03/2017] I just rediscovered this comment, it all seems to
        # work...
        self._k = []
        self._LER = []
        self._RUE = []

        # build conditions
        conditions = []
        genotypes = set([x[formats.GENOTYPE] for x in data])
        for genotype in genotypes:
            condition = []
            for year in kwargs['years']:
                condition.append({formats.GENOTYPE: genotype,
                                  'year': year})

            conditions.append(condition)

        final_simulation = []
        for condition in conditions:
            # train k
            self._k.append(self._calc_k(data, condition))

            # train LER
            self._LER.append(self._calc_LER(data, condition))

            # perform simulations without calculating RUE
            merged_data = []
            for year_condition in condition:
                kwargs_sim = self._fix_fll_reader(year_condition, **kwargs)

                simulated_data = self._training_simulation(year_condition,
                                                           get_RUE=False,
                                                           **kwargs_sim)

                real_data = self._subset_data(data, [year_condition])
                merged_data += self._merge_data(real_data, simulated_data)

            # use simulated PAR and real dry weight data to calculate RUE
            self._RUE.append(self._calc_RUE(merged_data, condition))

            for year_condition in condition:
                kwargs_sim = self._fix_fll_reader(year_condition,
                                                  **kwargs)

                final_simulation += self._training_simulation(year_condition,
                                                              get_RUE=True,
                                                              **kwargs_sim)

        return final_simulation

    def _training_simulation(self, condition, get_RUE=True, **kwargs):
        kwargs['k'] = self._get_variable(self._k, condition)
        kwargs['LER'] = self._get_variable(self._LER, condition)

        if get_RUE:
            kwargs['RUE'] = self._get_variable(self._RUE, condition)

        return self.simulate(condition, t_base=0, **kwargs)

    def simulate(self, condition, t_base=0, **kwargs):
        k = kwargs['k']
        LER = kwargs['LER']
        if 'RUE' in kwargs.keys():
            RUE = kwargs['RUE']
        else:
            RUE = None

        if 'fll_reader' in kwargs.keys():
            # used when simulating years without flag leaf info,
            # user provides the FLLReaderMissing class here
            geno = condition[formats.GENOTYPE]
            start_date = kwargs['fll_reader'].get_genotype_fll(geno)
        else:
            # used when training on data with available flag leaf info
            start_date = self._reader.get_fll_date(condition)

        met_data = self._reader.get_met_data(condition, t_base)

        dd = 0
        LAI = 0
        PAR = 0
        dry_weight = 0

        simulated_data = []

        current_date = datetime.strptime("01/01/%d" % condition['year'],
                                         "%d/%m/%Y")

        while current_date < start_date:
            entry = self._build_sim_entry(condition[formats.GENOTYPE],
                                          current_date,
                                          dd, LAI, PAR, dry_weight)

            simulated_data.append(entry)
            current_date += timedelta(1)

        # accomodate for missing Nov and Dec of 2016 met data
        final_date = max([x[formats.DATE] for x in met_data if
                          x[formats.DATE].year == condition['year']])

        while current_date.year == condition['year'] and \
                current_date <= final_date:
            entry = self._build_sim_entry(condition[formats.GENOTYPE],
                                          current_date,
                                          dd, LAI, PAR, dry_weight)

            simulated_data.append(entry)

            # equation 1 - get delta dd
            delta_dd = next(x[formats.DD] for x in met_data if
                            x[formats.DATE] == current_date)

            if delta_dd is None:
                print("Warning: Missing degree day for %s" %
                      current_date.strftime("%d/%m/%Y"))
                delta_dd = 0

            # equation 2 - LAI = LER * delta_dd
            delta_LAI = LER.get_slope(dd) * delta_dd

            # equation 3 - proportion of light intercepted
            pl = 1 - math.exp(-k * LAI)

            # equation 4 - delta_PAR
            daily_PAR = next(x[formats.PAR] for x in met_data if
                             x[formats.DATE] == current_date)

            if daily_PAR is None:
                daily_PAR = 0
                print("Warning: Missing PAR data for %s" %
                      current_date.strftime("%d/%m/%Y"))

            delta_PAR = daily_PAR * pl

            # equation 5 - dry_weight
            if RUE is not None:
                delta_dw = delta_PAR * RUE
            else:
                delta_dw = 0  # not simulating dry weight increment

            # add daily values
            dd += delta_dd
            LAI += delta_LAI
            PAR += delta_PAR
            dry_weight += delta_dw

            current_date += timedelta(1)

        return simulated_data

    def _merge_data(self, real_data, simulated_data):
        keys = [formats.GENOTYPE, formats.DATE, formats.PAR, formats.DW_PLANT]
        data = copy.deepcopy(real_data)
        final_data = []
        for entry in data:
            match = [x for x in simulated_data if
                     x[formats.DATE] == entry[formats.DATE]]

            if len(match) != 1:
                raise Exception("Unexpected result")

            entry[formats.PAR] = match[0][formats.PAR]

            for key in [x for x in entry.keys() if x not in keys]:
                del entry[key]

            final_data.append(entry)

        return final_data

    def _build_sim_entry(self, genotype, date, dd, LAI, PAR, dry_weight):
        entry = dict()
        entry[formats.GENOTYPE] = genotype
        entry[formats.DATE] = date
        entry[formats.DD] = dd
        entry[formats.LEAF_AREA_INDEX] = LAI
        entry[formats.PAR] = PAR
        entry[formats.DW_PLANT] = dry_weight

        return entry

    def _get_params_from_dict(self, params, var_name):
        result = []
        for param in params:
            condition_string = ""
            genotypes = list(set([x[formats.GENOTYPE] for
                                  x in param['conditions']]))

            for geno in genotypes:
                condition_string += "%s(" % geno
                years = [x['year'] for x in param['conditions'] if
                         x[formats.GENOTYPE] == geno]

                for year in years:
                    condition_string += "%d" % year
                    if year != years[-1]:
                        condition_string += ","

                condition_string += ")"
                if geno != genotypes[-1]:
                    condition_string += ","

            result.append({'conditions': condition_string,
                           'parameter': param['parameter'],
                           'variable': var_name})

        return result

    def _fix_fll_reader(self, condition, **kwargs):
        if condition['year'] > 2014:
            location = Location(kwargs['location'], condition['year'])
            fll_reader = FLLReaderMissing(location)
            return {'fll_reader': fll_reader}
        else:
            return dict()
