EARLY_703_JD = 2456658.5
EARLY_691_JD = 2452640.5
EARLY_644_JD = 2452883.5


def astrometric_uncertainty_Veres2017(obsCode, jd_tdb, catalog=None, program=None):
    """Return the astrometric uncertainty (sigma, in arcseconds) for an
    observatory, Julian date, and optionally catalog and program, based on the
    work of Veres et al. This is a one-sigma uncertainty, not a 1/sigma^2
    weight -- the fit squares-and-inverts it into the weight matrix.

    See: Veres et al. 2017 https://arxiv.org/pdf/1703.03479

    Parameters
    ----------
    obsCode : str
        The observatory code
    jd_tdb : float
        Julian date in TDB of the observation
    catalog : str, optional
        The stellar catalog used for measuring the astrometry, by default None
    program : str, optional
        The program code assigned by the Minor Planet Center, by default None

    Returns
    -------
    float
        Astrometric uncertainty in arcseconds
    """

    sigma_arcsec = 1.5
    if obsCode == "703":
        if jd_tdb <= EARLY_703_JD:
            sigma_arcsec = 1.0
        else:
            sigma_arcsec = 0.8
    elif obsCode == "691":
        if jd_tdb <= EARLY_691_JD:
            sigma_arcsec = 0.6
        else:
            sigma_arcsec = 0.5
    elif obsCode == "644":
        if jd_tdb <= EARLY_644_JD:
            sigma_arcsec = 0.6
        else:
            sigma_arcsec = 0.4
    elif obsCode == "704":
        sigma_arcsec = 1.0
    elif obsCode == "G96":
        sigma_arcsec = 0.5
    elif obsCode == "F51":
        sigma_arcsec = 0.2
    elif obsCode == "G45":
        sigma_arcsec = 0.6
    elif obsCode == "699":
        sigma_arcsec = 0.8
    elif obsCode == "D29":
        sigma_arcsec = 0.75
    elif obsCode == "C51":
        sigma_arcsec = 1.0
    elif obsCode == "E12":
        sigma_arcsec = 0.75
    elif obsCode == "608":
        sigma_arcsec = 0.6
    elif obsCode == "J75":
        sigma_arcsec = 1.0
    elif obsCode == "645":
        sigma_arcsec = 0.3
    elif obsCode == "673":
        sigma_arcsec = 0.3
    elif obsCode == "689":
        sigma_arcsec = 0.5
    elif obsCode == "950":
        sigma_arcsec = 0.5
    elif obsCode == "H01":
        sigma_arcsec = 0.3
    elif obsCode == "J04":
        sigma_arcsec = 0.4
    elif obsCode == "W84":
        sigma_arcsec = 0.5
    elif obsCode == "645":
        sigma_arcsec = 0.3
    elif obsCode == "G83" and program == "2":
        if catalog:
            if catalog in ["UCAC4", "PPMXL"]:
                sigma_arcsec = 0.3
            elif catalog in ["Gaia1", "Gaia2", "Gaia3", "Gaia3E"]:
                sigma_arcsec = 0.2
            else:
                sigma_arcsec = 0.3
        else:
            sigma_arcsec = 1.0
    elif obsCode in ["K92", "K93", "Q63", "Q64", "V37", "W84", "W85", "W86", "W87", "K91", "E10", "F65"]:
        sigma_arcsec = 0.4  # Careful with W84
    elif obsCode == "Y28":
        if catalog:
            if catalog in ["PPMXL", "Gaia1"]:
                sigma_arcsec = 0.3
            else:
                sigma_arcsec = 1.5
        else:
            sigma_arcsec = 1.5
    elif obsCode == "568":
        if catalog:
            if catalog in ["USNOB1", "USNOB2"]:  # TODO Unsure if "USNOB2" is correct abbreviation!!!
                sigma_arcsec = 0.5
            elif catalog in ["Gaia1", "Gaia2", "Gaia3", "Gaia3E"]:
                sigma_arcsec = 0.1
            elif catalog in ["PPMXL"]:
                sigma_arcsec = 0.2
            else:
                sigma_arcsec = 1.5
        else:
            sigma_arcsec = 1.5
    elif obsCode in ["T09", "T12", "T14"]:
        if catalog:
            if catalog in ["Gaia1", "Gaia2", "Gaia3", "Gaia3E"]:
                sigma_arcsec = 0.1
            else:
                sigma_arcsec = 1.5
        else:
            sigma_arcsec = 1.5
    elif obsCode == "309" and program == "&":  # Micheli
        if catalog:
            if catalog in ["UCAC4", "PPMXL"]:
                sigma_arcsec = 0.3
            elif catalog in ["Gaia1", "Gaia2", "Gaia3", "Gaia3E"]:
                sigma_arcsec = 0.2
        else:
            sigma_arcsec = 1.5
    elif catalog:
        sigma_arcsec = 1.0
    else:
        sigma_arcsec = 1.5

    return sigma_arcsec


# Backwards-compatible alias. Despite the historical "weight" name, this returns
# an astrometric *uncertainty* (sigma, in arcseconds), not a 1/sigma^2 weight --
# the fit forms the weight as 1/sigma^2 in get_weight_matrix. Prefer
# astrometric_uncertainty_Veres2017 in new code.
data_weight_Veres2017 = astrometric_uncertainty_Veres2017
