early_703 = 2456658.5
early_691 = 2452640.5
early_644 = 2452883.5

cat_codes = {
    "a": "USNO-A1.0",
    "b": "USNO-SA1.0",
    "c": "USNO-A2.0",
    "d": "USNO-SA2.0",
    "e": "UCAC-1",
    "f": "Tycho-1",
    "g": "Tycho-2",
    "h": "GSC-1.0",
    "i": "GSC-1.1",
    "j": "GSC-1.2",
    "k": "GSC-2.2",
    "l": "ACT",
    "m": "GSC-ACT",
    "n": "SDSS-DR8",
    "o": "USNO-B1.0",
    "p": "PPM",
    "q": "UCAC-4",
    "r": "UCAC-2",
    "s": "USNO-B2.0",
    "t": "PPMXL",
    "u": "UCAC-3",
    "v": "NOMAD",
    "w": "CMC-14",
    "x": "Hipparcos 2",
    "y": "Hipparcos",
    "z": "GSC (version unspecified)",
    "A": "AC",
    "B": "SAO 1984",
    "C": "SAO",
    "D": "AGK 3",
    "E": "FK4",
    "F": "ACRS",
    "G": "Lick Gaspra Catalogue",
    "H": "Ida93 Catalogue",
    "I": "Perth 70",
    "J": "COSMOS/UKST Southern Sky Catalogue",
    "K": "Yale",
    "L": "2MASS",
    "M": "GSC-2.3",
    "N": "SDSS-DR7",
    "O": "SST-RC1",
    "P": "MPOSC3",
    "Q": "CMC-15",
    "R": "SST-RC4",
    "S": "URAT-1",
    "T": "URAT-2",
    "U": "Gaia-DR1",
    "V": "Gaia-DR2",
    "W": "Gaia-DR3",
    "X": "Gaia-EDR3",
    "Y": "UCAC-5",
    "Z": "ATLAS-2",
    "0": "IHW",
    "1": "PS1-DR1",
    "2": "PS1-DR2",
    "3": "Gaia_Int  ",
    "4": "GZ",
    "5": "USNO-UBAD",
    "6": "Gaia2016",
}


def data_weight_Veres2017(obsCode, jd_tdb, catalog=None, program=None):
    # Need to put in the photographic and other observation
    # types

    dw = 1.5
    if obsCode == "703":
        if jd_tdb <= early_703:
            dw = 1.0
        else:
            dw = 0.8
    elif obsCode == "691":
        if jd_tdb <= early_691:
            dw = 0.6
        else:
            dw = 0.5
    elif obsCode == "644":
        if jd_tdb <= early_644:
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
            if catalog in ["USNOB1", "USNOB2"]:  #! Unsure if "USNOB2" is correct abbreviation
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
