import framework.data.formats as formats
from rpy2 import robjects
import rpy2.robjects.packages as rpackages
from copy import deepcopy
from datetime import datetime
import math


def assign_geno_bsbec(plot):
    plot = int(plot)
    if plot in [35, 48, 50, 61]:
        return "EMI-11"
    elif plot in [36, 42, 56, 62]:
        return "Giganteus"
    elif plot in [39, 46, 57, 53]:
        return "Goliath"
    elif plot in [40, 51, 58, 44]:
        return "Sac-5"
    else:
        raise Exception("Unknown plot %d" % plot)


def uniq_uids(data):
    return set([x[formats.UID] for x in data])


def uniq_key(data, key):
    """Get all unique values under the key. The function sorts the output"""
    return sorted(set([x[key] for x in data]))


def get_species(plot):
    geno = assign_geno_bsbec(plot)
    if geno == "EMI-11":
        return "sinensis"
    elif geno in ["Giganteus", "Goliath"]:
        return "sacc/sin"
    elif geno == "Sac-5":
        return "sacchariflorus"
    else:
        raise Exception("Unknown genotype %s" % geno)


def assign_geno_aber(uid):
    if uid in [27829, 27908, 28092]:
        return "Mx 1553#32"

    elif uid in [27844, 27986, 28080]:
        return "Mx 1553#36"

    elif uid in [27850, 27918, 28088]:
        return "Mx 1553#53"

    elif uid in [27874, 27958, 28035]:
        return "Mx 1553#22"

    elif uid in [27880, 27963, 28040]:
        return "Mx 1553#73"


def assign_geno_bsbec_new(uid):
    plot = int(uid) / 10
    return assign_geno_bsbec(plot)


def calculate_leaf_areas(data):
    # data needs to be single uid - check if that is true
    uids = set([x[formats.UID] for x in data])
    if len(uids) > 1:
        raise Exception("More than one UID provided!")

    try:
        data.sort(key=lambda x: (x[formats.DATE], x[formats.PSEUDO_REP]))
    except:
        raise Exception("Could not sort data, I don't remember why!")

    output = []
    dates = sorted(set([x[formats.DATE] for x in data]))

    prev_measurements = []
    for date in dates:
        entry = dict()
        measurements = [x for x in data if x[formats.DATE] == date]
        if len([x for x in measurements if x[formats.PSEUDO_REP] == 0]) == 0:
            raise Exception("Missing pseudo rep 0")

        for measurement in measurements:
            if measurement[formats.PSEUDO_REP] == 0:
                entry = measurement

            else:
                # update the leaf records with the new measurements
                prev_reps = [x[formats.PSEUDO_REP] for x in prev_measurements]
                pseudo_rep = measurement[formats.PSEUDO_REP]
                dead_leaf = True if measurement[formats.LEAF_LENGTH] == 0.0 \
                    or measurement[formats.LEAF_WIDTH] == 0.0 else False

                if dead_leaf and prev_reps:
                    # filter out the dead leaves
                    max_dead_rep = measurement[formats.PSEUDO_REP]

                    if date.year < 2013:
                        # drop single measurement
                        prev_measurements = [x for x in prev_measurements if
                                             x[formats.PSEUDO_REP] !=
                                             max_dead_rep]
                        continue
                    else:
                        prev_measurements = [x for x in prev_measurements if
                                             x[formats.PSEUDO_REP] >
                                             max_dead_rep]
                        continue

                if pseudo_rep not in prev_reps:
                    prev_measurements.append(measurement)
                else:
                    match = next(x for x in prev_measurements if
                                 x[formats.PSEUDO_REP] == pseudo_rep)

                    prev_measurements[prev_measurements.index(match)] = \
                        measurement

        # calculate the leaf area based on the update leaves
        total_area = 0
        for leaf in prev_measurements:
            total_area += leaf[formats.LEAF_WIDTH] * leaf[formats.LEAF_LENGTH]

        entry[formats.LEAF_AREA] = total_area
        output.append(entry)

    return output


def get_area(measurement):
    genotype = assign_geno_bsbec(measurement[formats.UID])
    if genotype in ["EMI-11", "Sac-5", "Goliath"]:
        factor = 0.684
    elif genotype == "Giganteus":
        factor = 0.745
    else:
        raise Exception("Unknown genotype %s" % genotype)

    return (factor * measurement[formats.LEAF_LENGTH] *
            measurement[formats.LEAF_WIDTH])


def get_BSBEC_uid(MSCAN_uid):
    conversion_table = {10967: 35,
                        10968: 36,
                        10971: 39,
                        10972: 40,
                        10974: 42,
                        10976: 44,
                        10978: 46,
                        10980: 48,
                        10982: 50,
                        10983: 51,
                        10985: 53,
                        10988: 56,
                        10989: 57,
                        10990: 58,
                        10993: 61,
                        10994: 62}

    return conversion_table[MSCAN_uid]


def subset_yr(data, years):
    return [x for x in data if x[formats.DATE].year in years]


def row_col_BSBEC(entry):
    if entry[formats.UID] == 35:
        row = 8
        col = 7
    elif entry[formats.UID] == 36:
        row = 8
        col = 8
    elif entry[formats.UID] == 39:
        row = 7
        col = 2
    elif entry[formats.UID] == 40:
        row = 7
        col = 1
    elif entry[formats.UID] == 42:
        row = 6
        col = 4
    elif entry[formats.UID] == 44:
        row = 6
        col = 8
    elif entry[formats.UID] == 46:
        row = 5
        col = 5
    elif entry[formats.UID] == 48:
        row = 5
        col = 1
    elif entry[formats.UID] == 50:
        row = 4
        col = 4
    elif entry[formats.UID] == 51:
        row = 4
        col = 5
    elif entry[formats.UID] == 53:
        row = 3
        col = 8
    elif entry[formats.UID] == 56:
        row = 3
        col = 1
    elif entry[formats.UID] == 57:
        row = 2
        col = 1
    elif entry[formats.UID] == 58:
        row = 2
        col = 2
    elif entry[formats.UID] == 61:
        row = 1
        col = 8
    elif entry[formats.UID] == 62:
        row = 1
        col = 7

    entry['row'] = row
    entry['col'] = col

    return entry


def substitute_NA(var, key):
    if key in formats._INTS:
        NA_val = robjects.NA_Integer
    elif key in formats._FLOATS:
        NA_val = robjects.NA_Real
    else:
        NA_val = robjects.NA_Character

    result = [x if x is not None else NA_val for x in var]

    return result


def dict_list_to_df(data, base=None):
    """Converts a list of dictionaries to an rpy2 data frame"""
    vectors = dict()
    keys = data[0].keys()
    if base is None:
        base = rpackages.importr('base')

    for key in keys:
        # First extract each key into its own list
        var = []
        for entry in data:
            var.append(entry[key])

        # convert None to R's NA
        var = substitute_NA(var, key)

        # next convert it into a rpy2 vector
        if key in formats._INTS:
            vect = robjects.IntVector(var)
        elif key in formats._FLOATS:
            vect = robjects.FloatVector(var)
        elif key == formats.DATE:
            var_str = [x.strftime(formats.DMY) for x in var]
            vect = robjects.StrVector(var_str)
            vect = base.as_Date(vect, formats.DMY)
        else:
            vect = robjects.StrVector(var)

        vectors[key] = vect

    return(robjects.DataFrame(vectors))


def split_dataset(data, year):
    new_data = dict()
    for key in data.keys():
        if key == 'init_entries':
            continue

        test_data = [x for x in data[key] if x[formats.DATE].year == year]
        train_data = [x for x in data[key] if x not in test_data]

        new_data[key] = {'train_data': train_data,
                         'test_data': test_data}

    if 'init_entries' in data.keys():
        new_data['init_entries'] = deepcopy(data['init_entries'])

    return new_data


def dataframe_to_listdict(dataframe):
    # convert dataframe to dictionary
    data_dict = dict(zip(dataframe.names, map(list, list(dataframe))))
    keys = data_dict.keys()
    output = list()
    for i in range(len(data_dict[keys[0]])):
        entry = dict()
        for key in keys:
            entry[key] = data_dict[key][i]
            if key == formats.DATE:
                entry[key] = datetime.strptime(entry[key], "%d/%m/%Y")

        output.append(entry)

    return output


def calculate_rmse(data):
    temp_data = [x for x in data if x['real'] is not None]
    data = temp_data

    # calculate RMSE
    error = 0
    for x in data:
        error += (x['predicted'] - x['real'])**2

    rmse = math.sqrt(error / len(data))
    return rmse


def cdf(start, divisor, count):
    results = [start]
    cumul = start
    current = start
    for i in range(count - 1):
        current = current / divisor
        cumul += current
        results.append(cumul)

    return results
