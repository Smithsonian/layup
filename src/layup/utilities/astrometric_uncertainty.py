EARLY_703_JD = 2456658.5
EARLY_691_JD = 2452640.5
EARLY_644_JD = 2452883.5


def data_weight_Veres2017(obsCode, jd_tdb, catalog=None, program=None):
    """Return data weighting given an observatory, Julian date, and optionally
    catalog and program. The weighting is based on the work of Veres et al.

    Parameters
    ----------
    obsCode : str
        The observatory code
    jd_tdb : float
        Julian date in TDB of the observation
    catalog : str, optional
        The survey catalog that included the observation, by default None
    program : str, optional
        Not entirely sure, by default None #TODO: Correct this docstring!!!

    Returns
    -------
    float
        Astrometric uncertainty in arcseconds
    """

    dw = 1.5
    if obsCode == "703":
        if jd_tdb <= EARLY_703_JD:
            dw = 1.0
        else:
            dw = 0.8
    elif obsCode == "691":
        if jd_tdb <= EARLY_691_JD:
            dw = 0.6
        else:
            dw = 0.5
    elif obsCode == "644":
        if jd_tdb <= EARLY_644_JD:
            dw = 0.6
        else:
            dw = 0.4
    elif obsCode == "704":
        dw = 1.0
    elif obsCode == "G96":
        dw = 0.5
    elif obsCode == "F51":
        dw = 0.2
    elif obsCode == "G45":
        dw = 0.6
    elif obsCode == "699":
        dw = 0.8
    elif obsCode == "D29":
        dw = 0.75
    elif obsCode == "C51":
        dw = 1.0
    elif obsCode == "E12":
        dw = 0.75
    elif obsCode == "608":
        dw = 0.6
    elif obsCode == "J75":
        dw = 1.0
    elif obsCode == "645":
        dw = 0.3
    elif obsCode == "673":
        dw = 0.3
    elif obsCode == "689":
        dw = 0.5
    elif obsCode == "950":
        dw = 0.5
    elif obsCode == "H01":
        dw = 0.3
    elif obsCode == "J04":
        dw = 0.4
    elif obsCode == "W84":
        dw = 0.5
    elif obsCode == "645":
        dw = 0.3
    elif obsCode == "G83" and program == "2":
        if catalog:
            if catalog in ["UCAC4", "PPMXL"]:
                dw = 0.3
            elif catalog in ["Gaia1", "Gaia2", "Gaia3", "Gaia3E"]:
                dw = 0.2
            else:
                dw = 0.3
        else:
            dw = 1.0
    elif obsCode in ["K92", "K93", "Q63", "Q64", "V37", "W84", "W85", "W86", "W87", "K91", "E10", "F65"]:
        dw = 0.4  # Careful with W84
    elif obsCode == "Y28":
        if catalog:
            if catalog in ["PPMXL", "Gaia1"]:
                dw = 0.3
            else:
                dw = 1.5
        else:
            dw = 1.5
    elif obsCode == "568":
        if catalog:
            if catalog in ["USNOB1", "USNOB2"]:  # TODO Unsure if "USNOB2" is correct abbreviation!!!
                dw = 0.5
            elif catalog in ["Gaia1", "Gaia2", "Gaia3", "Gaia3E"]:
                dw = 0.1
            elif catalog in ["PPMXL"]:
                dw = 0.2
            else:
                dw = 1.5
        else:
            dw = 1.5
    elif obsCode in ["T09", "T12", "T14"]:
        if catalog:
            if catalog in ["Gaia1", "Gaia2", "Gaia3", "Gaia3E"]:
                dw = 0.1
            else:
                dw = 1.5
        else:
            dw = 1.5
    elif obsCode == "309" and program == "&":  # Micheli
        if catalog:
            if catalog in ["UCAC4", "PPMXL"]:
                dw = 0.3
            elif catalog in ["Gaia1", "Gaia2", "Gaia3", "Gaia3E"]:
                dw = 0.2
        else:
            dw = 1.5
    elif catalog:
        dw = 1.0
    else:
        dw = 1.5

    return dw
