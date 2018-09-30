def degree_days(tmax, tmin, tbase):
    """
    Returns degree day value
    This uses the McVicker Degree day formula
    Input temp. max, temp. min, temp. base
    """
    if tmax is None or tmin is None:
        return None

    if tmax <= tbase:
        degday = 0
    elif (tmin + tmax)/2 < tbase:
        degday = ((tmax - tbase)/4)
    elif tmin < tbase:
        degday = ((tmax - tbase)/2) - ((tbase - tmin)/4)
    else:
        degday = ((tmin + tmax)/2) - tbase

    return degday
