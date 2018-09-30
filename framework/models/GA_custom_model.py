# A randomly generated model using genes, used for GA

from framework.models.naive_ml_process_model import NaiveMLProcessModel
from framework.models.gene import Gene
from framework.util.misc import split_dataset, calculate_rmse
import random
import pickle
from copy import deepcopy
from rpy2 import robjects


class GACustomModel:
    def __init__(self, meta_data, data, years, genes=None):
        if genes is not None:
            self._genes = genes
        else:
            # generate new individual
            self._genes = self._generate_genes()

        self._years = years
        self._data = data
        self._meta_data = meta_data
        self._instructions = self._convert_genes(self._genes)
        self._rmse = None

    def get_genes(self):
        return self._genes

    def get_instructions(self):
        return self._instructions

    def get_rmse(self):
        if self._rmse is None:
            self._rmse = self._model_accuracy(self._instructions)

        return self._rmse

    def same(self, model):
        for i in range(len(self._instructions)):
            if not self._instructions[i] == model.get_instructions()[i]:
                return False

        return True

    def set_probability(self, probability):
        self._probability = probability

    def get_probability(self):
        return self._probability

    def crossover(self, model2, mutation_probability):
        genes1 = self.get_genes()
        genes2 = model2.get_genes()
        switches = []
        target_len = random.choice([1, 2])
        while len(switches) < target_len:
            choice = random.randint(1, len(genes1) - 1)
            if choice not in switches:
                switches.append(choice)

        switches.sort()
        if len(switches) == 2:
            new_genes = genes1[0:switches[0]] + \
                        genes2[switches[0]:switches[1]] +\
                        genes1[switches[1]:]
        else:
            new_genes = genes1[0:switches[0]] +\
                        genes2[switches[0]:]

        if random.random() <= mutation_probability:
            new_genes = self._mutate_gene(new_genes)

        new_model = GACustomModel(self._meta_data, self._data,
                                  self._years, new_genes)
        return new_model

    def _mutate_gene(self, genes):
        gene = random.choice(genes)
        index = genes.index(gene)
        genes[index] = self._generate_gene(gene.get_type(),
                                           gene.get_key())
        return(genes)

    def _convert_genes(self, genes):
        """Converts genes into instructions for MLModes,
        and populate the instructions with references to the R models
        """
        # first grab the method genes
        method_genes = [x for x in genes if x.get_type() == Gene._METHOD]
        methods = [x.get_value() for x in method_genes]

        # now deal with var genes
        var_genes = [x for x in genes if x.get_type() == Gene._VARIABLE and
                     x.get_value() != 0]

        current_stage = 1
        i = 1
        instructions = []
        while i <= Gene._VARIABLE_VALUES[-1]:
            stage_variables = [x.get_key() for x in var_genes
                               if x.get_value() == i]
            if len(stage_variables) > 0:
                prev_vars = [x['name'] for x in instructions
                             if x['stage'] < current_stage]

                train_variables = prev_vars + NaiveMLProcessModel._INIT_DATA
                for variable in stage_variables:
                    instruction = {'name': variable,
                                   'method': methods[current_stage - 1],
                                   'variables': train_variables,
                                   'stage': current_stage}
                    instructions.append(instruction)

                current_stage += 1

            i += 1

        instructions.append({'name': NaiveMLProcessModel._STAGE_TWO[0],
                             'method': methods[-1],
                             'variables': [x.get_key() for x in var_genes] +
                             NaiveMLProcessModel._INIT_DATA,
                             'stage': current_stage})

        return instructions

    def _model_accuracy(self, instructions):
        instructions.sort(key=lambda x: x['stage'])

        robjects.r('''
            suppressMessages(library(randomForest))
            suppressMessages(library(LiblineaR))
            suppressMessages(library(nnet))
            suppressMessages(library(caret))
            suppressMessages(library(mgcv))
            suppressMessages(library(kknn))
            suppressMessages(library(e1071))
            suppressMessages(library(gbm))
            ''')

        predictions = []
        for year in self._years:
            split_data = deepcopy(split_dataset(self._data, year)['ml_data'])
            for instruction in instructions:
                # get matching R model
                temp_instruction = deepcopy(instruction)
                temp_instruction['year'] = year
                submodel = self._get_model(temp_instruction)
                prediction = submodel.predict(data=split_data['test_data'])

                if instruction != instructions[-1]:
                    for entry in prediction:
                        entry[entry['pheno']] = entry['predicted']
                        for key in ['real', 'pheno', 'predicted', 'method']:
                            del(entry[key])

                    split_data['test_data'] = prediction

                else:
                    # last stage
                    predictions += prediction

        final_predictions = []
        for prediction in predictions:
            entry = dict()
            entry['real'] = prediction['real']
            entry['predicted'] = prediction['predicted']
            final_predictions.append(entry)

        return(calculate_rmse(final_predictions))

    def _generate_gene(self, gene_type, key):
        if gene_type == Gene._VARIABLE:
            value = random.choice(Gene._VARIABLE_VALUES)
        elif gene_type == Gene._METHOD:
            value = random.choice(Gene._METHOD_VALUES)
        else:
            raise Exception("Unknown gene type %s" % gene_type)

        return(Gene(gene_type, key, value))

    def _generate_genes(self):
        genes = []

        # add genes for stages
        phenos = NaiveMLProcessModel._STAGE_ONE

        for key in phenos:
            genes.append(self._generate_gene(Gene._VARIABLE, key))

        # add genes for methods
        for i in range(6):
            genes.append(self._generate_gene(Gene._METHOD, ''))

        return genes

    def _get_model(self, instruction):
        match = [x for x in self._meta_data if
                 set(x['variables']) == set(instruction['variables']) and
                 x['name'] == instruction['name'] and
                 x['method'] == instruction['method'] and
                 x['year'] == instruction['year']]

        if len(match) != 1:
            raise Exception("Cannot find model")

        entry = match[0]

        f = open(entry['location'])
        model = pickle.load(f)
        f.close()

        return model
