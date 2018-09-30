# MLProcessModel - a class used for building generic process ML models
# or simple ML models. Uses MLModelMemfix to build a model with or without
# submodels, given a set of instructions. This class can create the following
# models: simple ML models, Naive ML process model (2 stages) and
# the best GA ML process model.

# IMPORTANTE: Performs year-based cross validation

from framework.models.ml_model_memfix import MLModelMemfix
import framework.data.formats as formats
from framework.models.ml_train_exception import MLTrainException
from framework.models.naive_ml_process_model_memfix \
        import NaiveMLProcessModelMemfix
from framework.scenarios.BSBEC.cd_model_trainer import CDModelTrainer
from framework.models.GA_win_model import GAWinModel


class MLProcessModelSpeedTest:
    def __init__(self, method, data, variables, base):
        self._data = data  # full data for testing models
        self._method = method
        self._variables = variables
        self._base = base

        self._instructions = self._generate_instructions()
        self._predictions = self.predict()

    def predict(self):
        """This is where training happens"""
        if self._method in MLModelMemfix._METHODS:
            model_data = {'train_data': self._data['ml_data']['yield'],
                          'test_data': self._data['ml_data']['yield']}

            # We often get Exceptions here when new code executes this
            # class, so leaving that here
            try:
                model = MLModelMemfix(model_data, self._instructions,
                                      self._base)
            except MLTrainException:
                raise
            except Exception as e:
                import ipdb
                ipdb.set_trace()
                print e.message
                print "Found the error!"

        elif self._method == NaiveMLProcessModelMemfix._NAME:
            stage_one = {'train_data': self._data['ml_data']['pheno'],
                         'test_data': self._data['ml_data']['pheno']}

            stage_two_train = [x for x in self._data['ml_data']['yield']
                               if x[formats.DW_PLANT] is not None]
            stage_two_test = [x for x in self._data['ml_data']['yield']
                              if x[formats.DW_PLANT] is not None]
            stage_two = {'train_data': stage_two_train,
                         'test_data': stage_two_test}

            model_data = {'stage_one': stage_one,
                          'stage_two': stage_two}

            model = NaiveMLProcessModelMemfix(model_data, "cumul_reduce")

        elif self._method == GAWinModel._NAME:
            stage_one = {'train_data': self._data['ml_data']['pheno'],
                         'test_data': self._data['ml_data']['pheno']}

            stage_two_train = [x for x in self._data['ml_data']['yield']
                               if x[formats.DW_PLANT] is not None]
            stage_two_test = [x for x in self._data['ml_data']['yield']
                              if x[formats.DW_PLANT] is not None]
            stage_two = {'train_data': stage_two_train,
                         'test_data': stage_two_test}

            model_data = {'stage_one': stage_one,
                          'stage_two': stage_two}

            model = GAWinModel(model_data)

        return model

    def _generate_instructions(self):
        if self._method in [MLModelMemfix._RANDOM_FOREST, MLModelMemfix._KNN,
                            MLModelMemfix._GBM]:
            # there will be only one instruction
            instruction = {'name': formats.DW_PLANT,
                           'method': self._method,
                           'variables': self._variables,
                           'check_num': '',
                           'predict': False}

        elif self._method in [NaiveMLProcessModelMemfix._NAME,
                              GAWinModel._NAME]:
            # NaiveMLProcessModelMemfix generates them itself
            # this stops only the NaiveMLProcess models, it does not
            # influence the normal ML models
            return None

        elif self._method == CDModelTrainer.NAME:
            return None

        return instruction
