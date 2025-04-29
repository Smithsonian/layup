def convert_tdb_date_to_julian_date(input_tdb_date: str, spice_kernel_dir: str = "") -> tuple[float, str]:
    """
    Convert a TDB date string to Julian Date.

    Parameters
    ----------
    input_tdb_date : str
        The input TDB date string in the format 'YYYY-MM-DD'.

    spice_kernel_dir : str, optional
        The directory containing SPICE kernel files. If not provided, the
        function will not load any SPICE kernels.

    Returns
    -------
    tuple(float, str)
        The Julian Date, as a float, corresponding to the input TDB date. Also
        returns the UTC date string in ISO format.
    """
    from datetime import datetime
    from pathlib import Path
    from astropy.time import Time
    import spiceypy as spice
    import pooch

    if not spice_kernel_dir:
        spice_kernel_dir = pooch.os_cache("layup")

    # Load SPICE kernels
    kernel_file = Path(spice_kernel_dir) / "naif0012.tls"
    if kernel_file.exists():
        # Look at all the currently loaded kernels. If `naif0012.tls` is already
        # loaded, we won't attempt to load it again.
        load_kernel = True
        for i in range(spice.ktotal("ALL")):
            this_kernel = spice.kdata(i, "ALL")
            if this_kernel[0] == kernel_file:
                load_kernel = False
                break

        if load_kernel:
            spice.furnsh(str(kernel_file))
    else:
        raise FileNotFoundError(
            f"SPICE kernel file 'naif0012.tls' not found in directory: {spice_kernel_dir}. "
            "Run `layup bootstrap` to ensure the file is present."
        )

    et = spice.str2et(input_tdb_date)
    spice_utc = spice.et2utc(et, "ISOC", prec=6)
    date_JD_TDB = Time(
        datetime.strptime(spice_utc, "%Y-%m-%dT%H:%M:%S.%f"), format="datetime", scale="utc"
    ).tdb.jd

    # Return the Julian Date as a float
    return date_JD_TDB, spice_utc
