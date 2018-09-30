# This is the GA process model done with machine learning

import framework.data.formats as formats
from framework.models.ml_model_memfix import MLModelMemfix
from copy import deepcopy


class GAWinModel:
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

    _NAME = "GAWinModel"

    def __init__(self, data):
        """
        data - {'stage_one': {'train_data': [], 'test_data': []},
                'stage_two': {'train_data': [], 'test_data': []}}
        """
        self._data = data
        self._instructions = self._build_instructions()
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
            # using set because pheno_date may be present in both lists
            for var in set(self._init_vars + self._META_DATA):
                new_entry[var] = entry[var]

            input_data.append(new_entry)

        # go through each stage one model and predict stage two variables
        stage_one_models = self._get_models(1)
        for model in stage_one_models:
            var_name = model.get_name()
            predictions = model.predict(record_real=False, data=input_data)

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

        required_keys = self._init_vars + self._pheno_keys

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

    def _build_instructions(self):
        available_keys = self._data['stage_one']['train_data'][0].keys()
        self._init_vars = [x for x in self._INIT_DATA if x in available_keys]
        # pheno_keys are all pheno_keys that the GAWinModel can handle
        all_pheno_keys = [formats.LEAF_AREA_INDEX, formats.STEM_COUNT,
                          formats.TRANSMISSION, formats.FLOWERING_SCORE]
        self._pheno_keys = []

        instructions = []

        for pheno_key in all_pheno_keys:
            if pheno_key in available_keys:
                # predict: False - turn off prediction of
                # test data during training
                instruction = {'name': pheno_key,
                               'variables': self._init_vars +
                               self._pheno_keys,
                               'stage': 1,
                               'predict': False}

                if pheno_key == formats.LEAF_AREA_INDEX:
                    instruction['method'] = MLModelMemfix._LINEAR_REGRESSION
                elif pheno_key in [formats.STEM_COUNT, formats.TRANSMISSION]:
                    instruction['method'] = MLModelMemfix._SVM
                elif pheno_key == formats.FLOWERING_SCORE:
                    instruction['method'] = MLModelMemfix._KNN

                instructions.append(instruction)
                self._pheno_keys.append(pheno_key)

        instructions.append({'name': formats.DW_PLANT,
                             'method': MLModelMemfix._KNN,
                             'variables': self._init_vars + self._pheno_keys,
                             'predict': False,
                             'stage': 2})

        return instructions
