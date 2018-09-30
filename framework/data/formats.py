# Some utility functions to get entry in the right format

from datetime import datetime
import framework.util.genotype as genotype

DMY = "%d/%m/%Y"

UID = 'EuroPheno.stock_id'
PSEUDO_REP = 'pseudo_rep_no'
MEASUREMENT = 'measurement_no'
FLOWERING_SCORE = 'field_flowering_score'
LEAF_LIGULE = 'leaf_ligule'
STEM_ID = 'stem_id'
CANOPY_HEIGHT = 'canopy_height'
TALLEST_STEM = 'tallest_stem'
LEAF_LENGTH = 'leaf_length_cm'
LEAF_WIDTH = 'leaf_width_cm'
STEM_COUNT = 'stem_count'
FW_PLANT = 'fresh_weight_plant'
FW_SUB = 'fresh_weight_sub_sample'
DW_SUB = 'dry_weight_sub_sample'
DW_PLANT = 'dry_weight'
TIME = 'time_taken'
WHO = 'who'
DATE = 'pheno_date'
SPECIES = 'species'
GENOTYPE = 'genotype'
LEAF_AREA = 'leaf_area'
LEAF_AREA_INDEX = 'leaf_area_index'
TRANSMISSION = 'Transmission'
STEM_DIAM = 'stem_diam_half_height'
ROW = 'row'
COL = 'col'
DOY = 'doy'
YEAR = 'year'

# JKI 20 PT pheno measurements
SHOOT_HEIGHT_AU = 'shoot_height_autumn'
MOISTURE_CONTENT = 'moisture_content'
FLOWERED_AU = 'flowered_autumn'
STEM_COUNT_HALF = 'stem_count_half'
BASE_DIAM_1 = 'base_diameter_1'
BASE_DIAM_2 = 'base_diameter_2'
DIE_OFF_HEIGHT = 'die_off_height'

# MetData measurements
PAR = "PAR"
T_MAX = "t_max"
T_MIN = "t_min"
DD = 'degree_days'
RAINFALL = 'rainfall_mm'


# I'll just going to leave that here...
def delta(entries):
    if type(entries) == list:
        blacklist = [GENOTYPE, UID, PSEUDO_REP,
                     ROW, COL, DATE,
                     MEASUREMENT]
        result = []
        for entry in entries:
            if entry not in blacklist:
                result.append("%s_%s" % ("delta", entry))
    elif type(entries) == str:
        result = "%s_%s" % ("delta", entries)
    else:
        raise Exception("Unhandled type!")

    return(result)


# do not put stem_counts here, see BSBEC 2011 data
_INTS = [UID, PSEUDO_REP, MEASUREMENT, FLOWERING_SCORE, STEM_COUNT_HALF,
         ROW, COL, DOY, YEAR]
_INTS += delta(_INTS)

_FLOATS = [CANOPY_HEIGHT, LEAF_LENGTH, LEAF_WIDTH, STEM_COUNT, FW_PLANT,
           FW_SUB, DW_SUB, DW_PLANT, LEAF_AREA, LEAF_AREA_INDEX,
           SHOOT_HEIGHT_AU, MOISTURE_CONTENT, BASE_DIAM_1, BASE_DIAM_2,
           DIE_OFF_HEIGHT, TRANSMISSION, PAR, T_MAX, T_MIN, DD, RAINFALL,
           TALLEST_STEM, STEM_DIAM]
_FLOATS += delta(_FLOATS)

_NO_CONVERSION = [TIME, WHO, DATE, SPECIES, GENOTYPE, FLOWERED_AU]

_BOOL = [LEAF_LIGULE]

_SPECIAL = [STEM_ID]


def convert(entry, function, keys):
    for key in keys:
        if key in entry.keys():
            if entry[key] != "" and entry[key] is not None:
                entry[key] = function(entry[key])
            else:
                del entry[key]


def on_read(entry, location, date_format="%d/%m/%Y"):
    """This function converts all known variables to their corresponding types
    for use inside python when doing statistics. Pass your rows through this
    function after reading them from a csv file.
    """

    # make sure we know what we are doing
    for key in entry.keys():
        if key not in get_all_pheno_names():
            raise Exception("Unhandled key %s" % key)

    # special cases first
    # throws exception if it is empty
    entry[DATE] = datetime.strptime(entry[DATE], date_format)

    # these do not throw exceptions for empty strings
    convert(entry, int, _INTS)
    convert(entry, float, _FLOATS)
    convert(entry, bool_convert, _BOOL)
    # present values just pass through, missing values get deleted
    convert(entry, no_conversion, _NO_CONVERSION)

    if STEM_ID in entry.keys():
        if entry[STEM_ID] == "" and entry[DATE].year in [2014, 2016]:
            # stem id was not used in 2014 and 2016
            pass

        elif entry[STEM_ID] == "" and entry[MEASUREMENT] != 0:
            # this is a leaf measurement, no need to convert stem_id
            pass

        elif entry[STEM_ID] == "" and \
                entry[DATE].year in [2011, 2012] and \
                entry[PSEUDO_REP] == 0:
            # this would be a height entry for the whole plot,
            # found in the 2011 and 2012 dataset for BSBEC,
            # nothing to worry about
            pass
        else:
            try:
                entry[STEM_ID] = int(entry[STEM_ID])
            except:
                import pprint
                pprint.pprint(entry)
                import ipdb
                ipdb.set_trace()

    # add genotype
    if GENOTYPE not in entry.keys():
        entry[GENOTYPE] = genotype.assign_genotype(entry[UID], location)

    if (FW_PLANT in entry.keys() and DW_SUB in entry.keys() and
            entry[FW_PLANT] is not None and entry[DW_SUB] is not None):

        # this conditional drops the 2016 harvest entries that have no DW_SUB
        if entry[FW_SUB] is not None:
            entry[DW_PLANT] = entry[FW_PLANT] * (entry[DW_SUB] / entry[FW_SUB])
        else:
            entry[DW_PLANT] = entry[DW_SUB]

        # convert from kg/plant to grams of dry weight / m^2 area
        area = location.get_plant_area()
        entry[DW_PLANT] = (entry[DW_PLANT] * 1000) / area

    return entry


def get_all_pheno_names():
    return _NO_CONVERSION + _INTS + _FLOATS + _BOOL + _SPECIAL


def strfdate_standard(date):
    return date.strftime(DMY)


def strfdate_standard_list(dates):
    for index in range(len(dates)):
        dates[index] = strfdate_standard(dates[index])

    return dates


def on_write(entries):
    """Takes a list of dictionaries and prepares it for writing into csv
    file.
    """
    for entry in entries:
        if DATE in entry.keys() and type(entry[DATE]) == datetime:
            entry[DATE] = strfdate_standard(entry[DATE])

    return entries


def bool_convert(value):
    if value in [0, False, "0", "False", "false"]:
        return 0
    elif value in [1, True, "1", "True", "true"]:
        return 1
    else:
        raise Exception("Unknown bool value")


def no_conversion(entry):
    return entry


class UselessMeasurementException(Exception):
    pass
