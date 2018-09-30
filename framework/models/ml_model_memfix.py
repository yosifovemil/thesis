# Build an ML model and use it to predict test data

from rpy2 import robjects
from framework.util.misc import dict_list_to_df
from copy import deepcopy
from framework.models.ml_train_exception import MLTrainException
from datetime import datetime


class MLModelMemfix:
    _RANDOM_FOREST = 'rf'
    _LINEAR_REGRESSION = 'lm'
    _ANN = 'ann'
    _KNN = 'knn'
    _SVM = 'svm'
    _GBM = 'gbm'
    _METHODS = [_RANDOM_FOREST, _LINEAR_REGRESSION, _ANN, _KNN,
                _SVM, _GBM]

    _FUNCTIONS = {_LINEAR_REGRESSION: '''model <- lm(%s, data = train_data)
                                         return(model)''',
                  _RANDOM_FOREST: '''model <- randomForest(%s,
                                                           data = train_data,
                                                           na.action = na.omit)
                                     return(model)''',
                  _ANN: '''train_data$genotype <-
                            as.numeric(train_data$genotype)
                            capture.output(model <- train(%s,
                                            data = train_data,
                                            method='nnet',
                                            na.action=na.omit))
                            return(model)''',
                  # _ANN: '''train_data$genotype <-
                  # as.numeric(train_data$genotype)
                  #          capture.output(model <- nnet(%s,
                  #                        data = train_data,
                  #                        size=5, decay=0.3,
                  #                        na.action=na.omit))
                  #          return(model)''',
                  _KNN: '''model <- kknn(%s, train = train_data,
                                          test = test_data)
                            return(model)''',
                  # _SVM: '''tuneResult <- tune(svm, %s,
                  #                             data = train_data)
                  #          model <- tuneResult$best.model
                  #          return(model)''',
                  _SVM: '''tc <- trainControl(method='boot', number=5)
                           capture.output(model <- train(%s,
                                          data = train_data,
                                          method='svmLinear2',
                                          trControl=tc,
                                          na.action=na.omit))
                           return(model)''',
                  _GBM: '''t_data <- train_data[
                                complete.cases(train_data[,"%s"]),]
                            capture.output(model <- gbm(%s,
                                                        data = t_data,
                                                        n.trees = 1000,
                                                        n.minobsinnode=%d))
                            return(model)'''}

    def __init__(self, data, instruction, base=None):
        """
        data - {'train_data': [], 'test_data': []}
        instructions - dictionary denoting what ML method and
        variables should be used for the model, i.e.
        {'name': 'Transmission',
        'method': 'random forest',
        'variables': ['PAR', 'rainfall_mm', 'degree_days' ...]}
        """
        self._data = data
        self._instruction = instruction
        if base is not None:
            self._base = base
        else:
            self._base = robjects.packages.importr("base")

        # import all necessary packages
        robjects.r('''
            suppressMessages(library(randomForest))
            suppressMessages(library(nnet))
            suppressMessages(library(caret))
            suppressMessages(library(mgcv))
            suppressMessages(library(kknn))
            suppressMessages(library(e1071))
            suppressMessages(library(gbm))
            suppressMessages(library(neuralnet))
            suppressMessages(library(LiblineaR))
            suppressMessages(library(plyr))
            ''')

        self._predictions = self._train(data, instruction)

    def _train(self, data, instruction):
        # build formula
        formula = "%s ~ %s" % (instruction['name'],
                               " + ".join(instruction['variables']))

        robjects.globalenv['train_data'] = dict_list_to_df(data['train_data'],
                                                           self._base)
        robjects.globalenv['test_data'] = dict_list_to_df(data['test_data'],
                                                          self._base)

        if instruction['method'] == self._GBM:
            # GBM sometimes fails if minobsinnode is too high, so decrease it
            # gradually until the model trains successfully
            run_model = False
            minobsinnode = 10
            finished = False
            while not finished:
                call = self._FUNCTIONS[instruction['method']] %\
                    (instruction['name'], formula, minobsinnode)
                try:
                    rmodel = robjects.r(call)
                    finished = True
                except:
                    minobsinnode -= 1
                    if minobsinnode < 3:
                        raise Exception("Minobsinnode too low")

        else:
            run_model = True
            call = self._FUNCTIONS[instruction['method']] % formula

        #####################################################################
        # perform speed test if needed
        if 'speed_test' in instruction.keys() and instruction['speed_test']:
            start = datetime.now()
            for i in range(10):
                rmodel = robjects.r(call)

            end = datetime.now()
            print end - start

        # Move along nothing to see here
        #####################################################################

        if run_model:
            try:
                rmodel = robjects.r(call)
            except:
                import ipdb
                ipdb.set_trace()
                print("About to raise MLTrainException")
                raise MLTrainException()

        # knn has a weird way of predicting test data
        if instruction['method'] == self._KNN:
            # build formula
            formula = "%s ~ %s" % (self._instruction['name'],
                                   " + ".join(self._instruction['variables']))

            robjects.globalenv['train_data'] = \
                dict_list_to_df(self._data['train_data'], self._base)

            predictions = robjects.r('''
                model <- kknn(%s, train = train_data, test = test_data)
                return(predict(model))''' % formula)
        elif instruction['method'] == self._GBM:
            predictions = robjects.r('''
                return(predict(model, test_data, n.trees=1000))''')
        elif instruction['method'] == self._ANN:
            predictions = robjects.r('''
                test_data$genotype <- as.numeric(test_data$genotype)
                return(predict(model, test_data))
                ''')
        # elif instruction['method'] == self._SVM:
        #     predictions = robjects.r('''
        #                 hack_phenos <- attr(model$terms, "term.labels")
        #                 hack_data <- test_data[hack_phenos]
        #                 return(predict(model, newdata=hack_data))''')
        else:
            # all other methods
            predictions = robjects.r('''
                return(predict(model, test_data))''')

        # save the model
        self._model = {'name': self._instruction['name'],
                       'method': self._instruction['method'],
                       'model': rmodel}
        predictions = tuple(predictions)

        if 'predict' in instruction.keys() and not instruction['predict']:
            return []

        assert(len(predictions) == len(data['test_data']))

        if 'check_num' in self._instruction.keys():
            if self._instruction['check_num'] == "BSBEC":
                # special test for the BSBEC dataset - prevent the number
                # of test data points from varying
                year = data['test_data'][0]['pheno_date'].year
                if year == 2011:
                    assert(len(predictions) == 96)
                elif year == 2015:
                    assert(len(predictions) == 173)
                elif year == 2016:
                    assert(len(predictions) == 209)

        results = []
        pheno = self._instruction['name']
        for i in range(len(data['test_data'])):
            entry = deepcopy(data['test_data'][i])
            entry['real'] = data['test_data'][i][pheno]
            entry['predicted'] = predictions[i]
            entry['method'] = instruction['method']
            entry['pheno'] = pheno
            results.append(entry)

        return results

    def get_name(self):
        return self._instruction['name']

    def get_stage(self):
        return self._instruction['stage']

    def get_method(self):
        return self._instruction['method']

    def get_predictions(self):
        return self._predictions

    def predict(self, record_real=True, data=None, submodel_test=False):
        results = []

        if data is None:
            data = self._data['test_data']

        # remove all the None cases before predicting the data
        if submodel_test:
            filtered_data = []
            for entry in data:
                skip = False
                for key in self._instruction['variables']:
                    if entry[key] is None:
                        skip = True

                if not skip:
                    filtered_data.append(entry)

            data = filtered_data

        robjects.globalenv['test_data'] = dict_list_to_df(data)

        robjects.globalenv['current_model'] = self._model['model']

        # knn has a weird way of predicting test data
        if self._model['method'] == self._KNN:
            # build formula
            formula = "%s ~ %s" % (self._instruction['name'],
                                   " + ".join(self._instruction['variables']))

            robjects.globalenv['train_data'] = \
                dict_list_to_df(self._data['train_data'], self._base)

            predictions = robjects.r('''
                model <- kknn(%s, train = train_data, test = test_data)
                return(predict(model))''' % formula)
        elif self._model['method'] == self._GBM:
            predictions = robjects.r('''
                return(predict(current_model, test_data, n.trees=1000))''')
        elif self._model['method'] == self._ANN:
            predictions = robjects.r('''
                test_data$genotype <- as.numeric(test_data$genotype)
                return(predict(current_model, test_data))
                ''')
        # elif self._model['method'] == self._SVM:
        #     predictions = robjects.r('''
        #             hack_phenos <- attr(current_model$terms, "term.labels")
        #                 hack_data <- test_data[hack_phenos]
        #                 return(predict(current_model, newdata=hack_data))''')
        else:
            # all other methods
            predictions = robjects.r('''
                return(predict(current_model, test_data))''')

        predictions = tuple(predictions)

        assert(len(predictions) == len(data))

        pheno = self._instruction['name']
        for i in range(len(data)):
            entry = deepcopy(data[i])
            if record_real:
                entry['real'] = data[i][pheno]
            entry['predicted'] = predictions[i]
            entry['method'] = self._model['method']
            entry['pheno'] = pheno
            results.append(entry)

        return results
