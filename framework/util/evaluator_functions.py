# A collection of evaluator functions used throughout the framework. Useful
# for making these [x for x in <list> if <condition>] statements a bit shorter

import framework.data.formats as formats


def in_season_uid(entry, uid, emergence, max_date):
    """
    Checks if entry is in season (between emergence and end of season, say
    march next year. UID is also verified.
    Known uses: DataReader._BSBEC_add_met_data()"""
    if entry[formats.UID] == uid and \
       entry[formats.DATE] >= emergence and \
       entry[formats.DATE] <= max_date:
        return True
    else:
        return False


def in_season(entry, emergence, max_date):
    """
    Checks if entry is in season (between emergence and end of season, say
    march next year.
    Known uses: DataReader._BSBEC_add_met_data()"""
    return(entry[formats.DATE] >= emergence and
           entry[formats.DATE] <= max_date)


def uid_prep_date(entry, uid, pseudo_rep, date):
    """
    Checks if UID, pseudo rep and date match.
    Known uses: DataReader._combine_BSBEC_pheno_data"""
    return(entry[formats.UID] == uid and
           entry[formats.PSEUDO_REP] == pseudo_rep and
           entry[formats.DATE] == date)
