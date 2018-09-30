# This is a naive process model done with machine learning
# Naive because there is no structure to the model
# It's just met_data -> pheno_data -> yield

import framework.data.formats as formats
from framework.models.ml_model_memfix import MLModelMemfix
from copy import deepcopy


class NaiveMLProcessModelMemfix:
    _META_DATA = [formats.UID, formats.PSEUDO_REP, formats.DATE]
    _INIT_DATA = [formats.ROW, formats.COL, formats.GENOTYPE,
                  formats.DD, formats.DOY, formats.PAR, formats.RAINFALL]
    _STAGE_ONE = [formats.TRANSMISSION, formats.LEAF_AREA_INDEX,
                  formats.FLOWERING_SCORE, formats.STEM_COUNT,
                  formats.CANOPY_HEIGHT]
    _STAGE_TWO = [formats.DW_PLANT]

    _DELTA_INIT_DATA = formats.delta(_INIT_DATA) +\
        [formats.ROW, formats.COL, formats.GENOTYPE]
    _DELTA_STAGE_ONE = formats.delta(_STAGE_ONE)
    _DELTA_STAGE_TWO = formats.delta(_STAGE_TWO)

    _DELTA_CUMUL_INIT_DATA = list(set(_INIT_DATA + _DELTA_INIT_DATA))
    _DELTA_CUMUL_STAGE_ONE = list(set(_STAGE_ONE + _DELTA_STAGE_ONE))
    _DELTA_CUMUL_STAGE_TWO = list(set(_STAGE_TWO + _DELTA_STAGE_TWO))

    _NAME = "NaiveMLProcessModelMemfix"

    def __init__(self, data, scenario, instructions=None):
        """
        data - {'stage_one': {'train_data': [], 'test_data': []},
                'stage_two': {'train_data': [], 'test_data': []}}

        scenario - choices are between 'delta', 'delta_cumul', 'cumul'
        instructions (optional) - list of dictionaries denoting what ML method
        and variables should be used for each submodel, i.e.
        [{'name': 'Transmission',
         'method': 'random forest',
         'variables': ['PAR', 'rainfall_mm', 'degree_days' ...]},
         ...]
        etc.
        """
        self._data = data
        if instructions is None:
            self._instructions = self._build_instructions(scenario)
        else:
            self._instructions = instructions

        self._scenario = scenario

        self._predictions = self.predict()

    def predict(self, compare=False):
        ####################################################################
        self._submodels = []
        for instruction in self._instructions:
            if instruction['stage'] == 1:
                stage_data = self._data['stage_one']
            elif instruction['stage'] == 2:
                stage_data = self._data['stage_two']
            else:
                raise Exception("Unknown stage %d" % instruction['stage'])

            self._submodels.append(MLModelMemfix(stage_data, instruction))
        ####################################################################

        # extract the INIT variables from the test dataset
        input_data = []
        for entry in self._data['stage_one']['test_data']:
            new_entry = dict()
            if self._scenario in ["cumul", "cumul_reduce"]:
                init_vars = set(self._INIT_DATA + self._META_DATA)
            else:
                init_vars = set(self._INIT_DATA + self._DELTA_INIT_DATA +
                                self._META_DATA)
            for var in init_vars:
                new_entry[var] = entry[var]

            input_data.append(new_entry)

        # go through each stage one model and predict stage two variables
        stage_one_models = self._get_models(1)
        for model in stage_one_models:
            var_name = model.get_name()
            predictions = model.get_predictions()

            # update each input_data entry with the predicted variable
            for prediction in predictions:
                id_keys = self._META_DATA
                if formats.PSEUDO_REP not in prediction.keys():
                    id_keys.remove(formats.PSEUDO_REP)

                filter_me = input_data
                for key in id_keys:
                    filter_me = [x for x in filter_me if
                                 x[key] == prediction[key]]

                # there should be just one entry if everything went well
                assert(len(filter_me) == 1)
                entry = filter_me[0]

                # now update the entry with the predicted variable
                entry[var_name] = prediction['predicted']

        # make sure all variables have been predicted
        if self._scenario == 'delta':
            required_keys = self._META_DATA + self._DELTA_INIT_DATA + \
                            self._DELTA_STAGE_ONE
        elif self._scenario == "cumul":
            required_keys = self._META_DATA + self._INIT_DATA + \
                            self._STAGE_ONE
        elif self._scenario == "cumul_reduce":
            required_keys = next(x for x in self._instructions
                                 if x['name'] ==
                                 formats.DW_PLANT)['variables']
        elif self._scenario == "delta_cumul":
            required_keys = self._META_DATA + \
                            self._DELTA_CUMUL_INIT_DATA + \
                            self._DELTA_CUMUL_STAGE_ONE

        for entry in input_data:
            for key in required_keys:
                assert(key in entry.keys())

        # get the dry weight prediction model
        stage_two_models = self._get_models(2)
        assert(len(stage_two_models) == 1)
        stage_two_model = stage_two_models[0]

        # start predicting
        predictions = stage_two_model.predict(record_real=False,
                                              data=input_data)

        if predictions[0]['pheno'] == formats.delta(formats.DW_PLANT):
            predictions = self._convert(predictions)
        else:
            # just because the compare function expects the predictions
            # to have a value for DW_PLANT
            for prediction in predictions:
                prediction[prediction['pheno']] = prediction['predicted']

        if compare:
            return self._compare(predictions)
        else:
            return predictions

    def get_predictions(self, compare=False):
        if compare:
            return self._compare(self._predictions)
        else:
            return self._predictions

    def _convert(self, predictions):
        """Converts delta predictions to cumulative values"""
        results = []
        for entry in self._data['init_entries']:
            if entry[formats.DATE].year != predictions[0][formats.DATE].year:
                continue

            new_entry = dict()
            for key in entry.keys():
                if key in self._META_DATA + self._STAGE_TWO:
                    new_entry[key] = entry[key]

            results.append(new_entry)

            if formats.PSEUDO_REP in entry.keys():
                matching = [x for x in predictions if
                            x[formats.UID] == entry[formats.UID] and
                            x[formats.PSEUDO_REP] == entry[formats.PSEUDO_REP]]
            else:
                matching = [x for x in predictions if
                            x[formats.UID] == entry[formats.UID]]

            matching.sort(key=lambda x: x[formats.DATE])
            previous = entry

            for match in matching:
                new_entry = dict()
                # copy meta data
                for key in self._META_DATA:
                    new_entry[key] = match[key]

                new_entry[formats.DW_PLANT] = previous[formats.DW_PLANT] + \
                    match['predicted']
                results.append(new_entry)
                previous = match
                previous[formats.DW_PLANT] = previous['predicted']

        return results

    def _get_models(self, stage):
        return [x for x in self._submodels if x.get_stage() == stage]

    def _compare(self, predictions):
        results = deepcopy(self._data['stage_two']['test_data'])
        for result in results:
            if formats.PSEUDO_REP in result.keys():
                prediction = [x for x in predictions if
                              x[formats.UID] == result[formats.UID] and
                              x[formats.DATE] == result[formats.DATE] and
                              x[formats.PSEUDO_REP] ==
                              result[formats.PSEUDO_REP]]
            else:
                prediction = [x for x in predictions if
                              x[formats.UID] == result[formats.UID] and
                              x[formats.DATE] == result[formats.DATE]]

            assert(len(prediction) == 1)

            prediction = prediction[0]
            result['predicted'] = prediction[formats.DW_PLANT]

        return results

    def _build_instructions(self, scenario):
        if scenario == 'delta':
            init_data = self._DELTA_INIT_DATA
            stage_one = self._DELTA_STAGE_ONE
            instructions = [{'name':
                             formats.delta(formats.CANOPY_HEIGHT),
                             'method': MLModelMemfix._LINEAR_REGRESSION,
                             'variables': init_data,
                             'stage': 1},
                            {'name':
                             formats.delta(formats.FLOWERING_SCORE),
                             'method': MLModelMemfix._RANDOM_FOREST,
                             'variables': init_data,
                             'stage': 1},
                            {'name':
                             formats.delta(formats.LEAF_AREA_INDEX),
                             'method': MLModelMemfix._SVM,
                             'variables': init_data,
                             'stage': 1},
                            {'name': formats.delta(formats.STEM_COUNT),
                             'method': MLModelMemfix._LINEAR_REGRESSION,
                             'variables': init_data,
                             'stage': 1},
                            {'name': formats.delta(formats.TRANSMISSION),
                             'method': MLModelMemfix._LINEAR_REGRESSION,
                             'variables': init_data,
                             'stage': 1},
                            {'name': formats.delta(formats.DW_PLANT),
                             'method': MLModelMemfix._GBM,
                             'variables': init_data + stage_one,
                             'stage': 2}]
        elif scenario == 'delta_cumul':
            init_data = self._DELTA_CUMUL_INIT_DATA
            stage_one = self._DELTA_CUMUL_STAGE_ONE
            instructions = [{'name': formats.CANOPY_HEIGHT,
                             'method': MLModelMemfix._RANDOM_FOREST,
                             'variables': init_data,
                             'stage': 1},
                            {'name': formats.delta(formats.CANOPY_HEIGHT),
                             'method': MLModelMemfix._LINEAR_REGRESSION,
                             'variables': init_data,
                             'stage': 1},
                            {'name': formats.delta(formats.FLOWERING_SCORE),
                             'method': MLModelMemfix._RANDOM_FOREST,
                             'variables': init_data,
                             'stage': 1},
                            {'name': formats.delta(formats.LEAF_AREA_INDEX),
                             'method': MLModelMemfix._RANDOM_FOREST,
                             'variables': init_data,
                             'stage': 1},
                            {'name': formats.delta(formats.STEM_COUNT),
                             'method': MLModelMemfix._LINEAR_REGRESSION,
                             'variables': init_data,
                             'stage': 1},
                            {'name': formats.delta(formats.TRANSMISSION),
                             'method': MLModelMemfix._KNN,
                             'variables': init_data,
                             'stage': 1},
                            {'name': formats.FLOWERING_SCORE,
                             'method': MLModelMemfix._RANDOM_FOREST,
                             'variables': init_data,
                             'stage': 1},
                            {'name': formats.LEAF_AREA_INDEX,
                             'method': MLModelMemfix._RANDOM_FOREST,
                             'variables': init_data,
                             'stage': 1},
                            {'name': formats.STEM_COUNT,
                             'method': MLModelMemfix._LINEAR_REGRESSION,
                             'variables': init_data,
                             'stage': 1},
                            {'name': formats.TRANSMISSION,
                             'method': MLModelMemfix._RANDOM_FOREST,
                             'variables': init_data,
                             'stage': 1},
                            {'name': formats.delta(formats.DW_PLANT),
                             'method': MLModelMemfix._RANDOM_FOREST,
                             'variables': init_data + stage_one,
                             'stage': 2}]
        elif scenario == 'cumul':
            init_data = self._INIT_DATA
            stage_one = self._STAGE_ONE
            instructions = [{'name': formats.CANOPY_HEIGHT,
                             'method': MLModelMemfix._RANDOM_FOREST,
                             'variables': init_data,
                             'stage': 1},
                            {'name': formats.FLOWERING_SCORE,
                             'method': MLModelMemfix._RANDOM_FOREST,
                             'variables': init_data,
                             'stage': 1},
                            {'name': formats.LEAF_AREA_INDEX,
                             'method': MLModelMemfix._RANDOM_FOREST,
                             'variables': init_data,
                             'stage': 1},
                            {'name': formats.STEM_COUNT,
                             'method': MLModelMemfix._LINEAR_REGRESSION,
                             'variables': init_data,
                             'stage': 1},
                            {'name': formats.TRANSMISSION,
                             'method': MLModelMemfix._RANDOM_FOREST,
                             'variables': init_data,
                             'stage': 1},
                            {'name': formats.DW_PLANT,
                             'method': MLModelMemfix._RANDOM_FOREST,
                             'variables': init_data + stage_one,
                             'stage': 2}]
        elif scenario == 'cumul_reduce':
            train_data_keys = self._data['stage_one']['train_data'][0].keys()
            init_data = [x for x in self._INIT_DATA if x in train_data_keys]

            stage_one = [x for x in self._INIT_DATA + self._STAGE_ONE
                         if x in train_data_keys]

            instructions = []

            for var in self._STAGE_ONE:
                if var in train_data_keys:
                    if var == formats.CANOPY_HEIGHT:
                        instruction = {'name': var,
                                       'method': MLModelMemfix._RANDOM_FOREST,
                                       'variables': init_data,
                                       'stage': 1}
                    elif var == formats.FLOWERING_SCORE:
                        instruction = {'name': var,
                                       'method': MLModelMemfix._RANDOM_FOREST,
                                       'variables': init_data,
                                       'stage': 1}
                    elif var == formats.LEAF_AREA_INDEX:
                        instruction = {'name': var,
                                       'method': MLModelMemfix._RANDOM_FOREST,
                                       'variables': init_data,
                                       'stage': 1}
                    elif var == formats.STEM_COUNT:
                        instruction = {'name': var,
                                       'method':
                                       MLModelMemfix._LINEAR_REGRESSION,
                                       'variables': init_data,
                                       'stage': 1}
                    elif var == formats.TRANSMISSION:
                        instruction = {'name': formats.TRANSMISSION,
                                       'method': MLModelMemfix._RANDOM_FOREST,
                                       'variables': init_data,
                                       'stage': 1}
                    else:
                        raise Exception("Unknown var %s" % var)

                    instructions.append(instruction)

            instructions.append({'name': formats.DW_PLANT,
                                 'method': MLModelMemfix._RANDOM_FOREST,
                                 'variables': stage_one,
                                 'stage': 2})
        else:
            raise Exception("Unknown scenario %s" % scenario)

        return instructions
