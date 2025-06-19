# flake8: noqa

# TODO: fix linting

import sys

# Read data from an opendap server
#import netCDF4

# from cdo import *
# import requests
import numpy as np
import numpy.ma as ma

# from datetime import date, datetime, timedelta
from dateutil.relativedelta import relativedelta

# from function_read import *
# cdo = Cdo()
import math
from glob import glob

# from netCDF4 import num2date, date2num


def lon_index(longitude, lon_bnds):
    # if longitude are in -180 move to 0 - 360
    lon_bnds = np.array(lon_bnds)
    lons = np.array(longitude, copy=True)
    # print('lons : ', lons)
    # print True in list(lons<0)
    if True in list(lons < 0):
        lons[lons < 0] = lons[lons < 0] + 360
    # if indices in -180, 180 move to 0 -360

    if True in list(lon_bnds < 0):
        lon_bnds[lon_bnds < 0] = lon_bnds[lon_bnds < 0] + 360

    # if

    # check if box is over separation
    # (most of the time greenwitch pero sometimes longitude can be splitted somewhere else)
    # not done.... Problem for grid_T
    if lon_bnds[0] > lon_bnds[1] - 1:
        # return a list with index before greenwitch and indexes after
        # print('Details : ')
        # print('1 : ', np.where((lons >= lon_bnds[0]))[0])
        # print('2 : ', np.where((lons <= lon_bnds[1]))[0])
        lon_inds = [
            np.where((lons >= lon_bnds[0]))[0],
            np.where((lons < lon_bnds[1]))[0],
        ]
        # print(lon_inds)
        if list(lon_inds[1]) == []:
            lon_inds = [lon_inds[0]]
        if list(lon_inds[0]) == []:
            lon_inds = [lon_inds[1]]
    else:
        lon_inds = [np.where((lons >= lon_bnds[0]) & (lons <= lon_bnds[1]))[0]]
    # print('lon_inds : ', lon_inds)
    return lon_inds


def lonlat_index(latitude, longitude, lat_bnds, lon_bnds):
    # handle 2D latitude array

    if len(latitude.shape) == 2:
        lat1D = np.array(latitude[:, 0], copy=True)
    else:
        lat1D = np.array(latitude, copy=True)

    # print lats
    # print lat_bnds
    # print np.where((lats >= lat_bnds[0]) & (lats <= lat_bnds[1]))[0]
    lat_inds = np.where((lat1D >= lat_bnds[0]) & (lat1D <= lat_bnds[1]))[0]

    # handle 2D longitude array (we base the longitude selection on the center of the latitude box)
    if len(longitude.shape) == 2:
        centerlat = lat_inds[len(lat_inds) // 2]
        # print(lat_inds[len(lat_inds)/2])
        lon1D = np.array(longitude[centerlat, :], copy=True)
        # print lons
    else:
        lon1D = np.array(longitude, copy=True)
    # print('lon1D : ', lon1D)
    lon_inds = lon_index(lon1D, lon_bnds)
    # print(lon_inds)

    return (lat1D, lon1D, lat_inds, lon_inds)


def earth_radius(lat):
    """
    calculate radius of Earth assuming oblate spheroid defined by WGS84
    Input
    ---------
    lat: vector or latitudes in degrees

    Output
    ----------
    r: vector of radius in meters

    Notes
    -----------
    WGS84: https://earth-info.nga.mil/GandG/publications/tr8350.2/tr8350.2-a/Chapter%203.pdf
    """

    from numpy import deg2rad

    # from numpy import sin, cos

    # define oblate spheroid from WGS84
    a = 6378137
    b = 6356752.3142
    e2 = 1 - (b**2 / a**2)
    # convert from geodecic to geocentric
    # see equation 3-110 in WGS84
    lat = deg2rad(lat)
    lat_gc = np.arctan((1 - e2) * np.tan(lat))
    # radius equation
    # see equation 3-107 in WGS84
    r = (a * (1 - e2) ** 0.5) / (1 - (e2 * np.cos(lat_gc) ** 2)) ** 0.5
    return r


def area_av(
    array, pos_lat, pos_lon, lats, lons, opt="mean", weightcalc=True, weights2D=None
):
    lons[lons > 180] = lons[lons > 180] - 360
    dim = array.shape
    if weightcalc:
        weights2D = area_grid(lats, lons)
    # weights=np.swapaxes(extend_table(area_grid(lats, lons), np.delete(np.delete(dim, pos_lon), pos_lat), len(dim)-1, pos_lat))  # noqa
    weights = extend_table(weights2D, np.delete(np.delete(dim, pos_lon), pos_lat))
    # plt.imshow(area_grid(lats, lons))
    weights = np.ma.array(weights, mask=array.mask)
    # print(area_grid(lats, lons).shape)
    # print(array.shape)
    # print(weights.shape)
    if opt == "mean":
        sumweigth = np.ma.sum(np.ma.sum(weights, axis=pos_lon), axis=pos_lat)
        array_av = (
            np.ma.sum(np.ma.sum(weights * array, axis=pos_lon), axis=pos_lat)
            / sumweigth
        )
    if opt == "sum":
        array_av = np.ma.sum(np.ma.sum(weights * array, axis=pos_lon), axis=pos_lat)
    return array_av


def area_grid(lat, lon):
    """
    Calculate the area of each grid cell
    Area is in square meters

    Input
    -----------
    lat: vector of latitude in degrees
    lon: vector of longitude in degrees

    Output
    -----------
    area: grid-cell area in square-meters with dimensions, [lat,lon]

    Notes
    -----------
    Based on the function in
    https://github.com/chadagreene/CDT/blob/master/cdt/cdtarea.m
    """
    from numpy import meshgrid, abs, deg2rad, gradient, cos

    # from numpy import rad2deg, sin
    from xarray import DataArray

    lon[lon > 180] = lon[lon > 180] - 360

    xlon, ylat = meshgrid(lon, lat)
    # R = earth_radius(ylat)
    R = 6378137
    dlat = abs(deg2rad(gradient(ylat, axis=0)))
    dlon = abs(deg2rad(gradient(xlon, axis=1)))

    if np.any(dlon > 300):
        print("issue with jumps in longitude")
        sys.exit(1)

    dy = dlat * R
    dx = dlon * R * cos(deg2rad(ylat))

    # print(dy)
    # print(dx)

    area = dy * dx
    xda = DataArray(
        area,
        dims=["latitude", "longitude"],
        coords={"latitude": lat, "longitude": lon},
        attrs={
            "long_name": "area_per_pixel",
            "description": "area per pixel",
            "units": "m^2",
        },
    )
    return xda


# obsreg=obs
# modreg=mod


def area_av_old(array, pos_lat, pos_lon, lats, lons, opt="mean"):

    # Use the cosine of the converted latitudes as weights for the average

    # First find the zonal mean SST by averaging along the latitude circles

    dlon = list(lons[1:] - lons[:-1])
    dlon.append(lons[-2] - lons[-1])
    dlat = list(lats[1:] - lats[:-1])
    dlat.append(lats[-2] - lats[-1])

    dlon = np.array(dlon) % 360
    dlat = np.array(dlat)

    # extend the lats, and cos over all the dimensions
    dim = array.shape
    dlon_ext = np.swapaxes(
        extend_table(dlon, np.delete(dim, pos_lon)), len(dim) - 1, pos_lon
    )
    dlat_ext = np.swapaxes(
        extend_table(dlat, np.delete(dim, pos_lat)), len(dim) - 1, pos_lat
    )
    coslat_ext = np.swapaxes(
        extend_table(np.cos(lats * math.pi / 180), np.delete(dim, pos_lat)),
        len(dim) - 1,
        pos_lat,
    )

    # create a weighted array
    weights = ma.array(dlon_ext * dlat_ext * coslat_ext, mask=array.mask)
    # print weights

    # make the weighted sum
    if opt == "mean":
        sumweigth = np.ma.sum(np.ma.sum(weights, axis=pos_lon), pos_lat)
        array_av = (
            np.ma.sum(np.ma.sum(weights * array, axis=pos_lon), pos_lat) / sumweigth
        )
    if opt == "sum":
        array_av = np.ma.sum(np.ma.sum(weights * array, axis=pos_lon), pos_lat)
    return array_av


def createdatelst(sdate1, sdate2, smonlstint):
    sdatelst = []
    sdate = sdate1
    while sdate < sdate2:
        sdatelst.append(
            [sdate + relativedelta(months=+(mon - 1)) for mon in smonlstint]
        )
        sdate = sdate + relativedelta(months=+12)
    sdatelst = np.ndarray.flatten(np.array(sdatelst))
    return sdatelst


def extend_table(array, dims_expend):
    array_ext = array
    dims_expend = list(dims_expend)
    dims_expend.reverse()
    for dim in dims_expend:
        array_ext = np.expand_dims(array_ext, axis=0).repeat(dim, axis=0)

    return array_ext


def getvarlat(varf):
    for varlat in ["Y", "lat", "latitude", "nav_lat"]:
        if varlat in varf.variables.keys():
            return varlat


def getvarlon(varf):
    for varlat in ["X", "lon", "longitude", "nav_lon"]:
        if varlat in varf.variables.keys():
            return varlat


def getvarmask(varf):
    for varmask in ["LSM", "land"]:
        if varmask in varf.variables.keys():
            return varmask


def getvarens(varf):
    for varens in ["ensemble", "ensembles", "M", "realization"]:  # , "lev", "height"]:
        if varens in varf.variables.keys():
            return varens


def getdimens(varf, varname):
    for varens in ["ensemble", "ensembles", "M"]:  # , "lev", "height"]:
        if varens in varf.variables[varname].dimensions:
            return varens


def getdimlon(varf, varname):
    for varlon in ["x", "lon"]:
        if varlon in varf.variables[varname].dimensions:
            return varlon


def getdimlat(varf, varname):
    for varlat in ["y", "lat"]:
        if varlat in varf.variables[varname].dimensions:
            return varlat

"""
def read_mask(url, lat_bnds, lon_bnds, mask):
    # url="/esarchive/exp/ecearth/constant/land_sea_mask_512x256.nc"
    varf = netCDF4.Dataset(url)

    lat_bnds = np.array(lat_bnds)
    lon_bnds = np.array(lon_bnds)

    varlat = getvarlat(varf)
    varlon = getvarlon(varf)
    maskvar = getvarmask(varf)
    # print maskvar

    # define subset

    lats = varf.variables[varlat][:]
    lons = varf.variables[varlon][:]
    # print(lats)

    lat1D, lon1D, lat_inds, lon_inds = lonlat_index(lats, lons, lat_bnds, lon_bnds)
    # print lat_inds
    # print  lon_inds
    # print lon_inds[0][0],(lon_inds[0][-1]+1)

    if len(lon_inds) == 2:
        # print varf.variables[maskvar].shape
        # print lat_inds[0],(lat_inds[-1]+1),lon_inds[0][0],(lon_inds[0][-1]+1)
        lsmaskW = varf.variables[maskvar][
            lat_inds[0] : (lat_inds[-1] + 1), lon_inds[0][0] : (lon_inds[0][-1] + 1)
        ]
        lsmaskE = varf.variables[maskvar][
            lat_inds[0] : (lat_inds[-1] + 1), lon_inds[1][0] : (lon_inds[1][-1] + 1)
        ]
        lsmask = np.ma.concatenate((lsmaskW, lsmaskE), axis=1)

        lons_regW = lons[lon_inds[0]]
        lons_regE = lons[lon_inds[1]]
        lons_reg = np.ma.concatenate((lons_regW, lons_regE), axis=0)
    else:
        lsmask = varf.variables[maskvar][
            lat_inds[0] : (lat_inds[-1] + 1), lon_inds[0][0] : (lon_inds[0][-1] + 1)
        ]
        lons_reg = lons[lon_inds[0]]

    lats_reg = lats[lat_inds]

    varf.close()
    # reverse latitude  if needed
    if lats[0] < 0:
        lats = lats[::-1]
        # lsmak = lsmask[::-1, :]

    if mask == "oce":
        lsmask = 1 - lsmask
    elif mask == "all":
        lsmask = lsmask * 0 + 1
    elif mask != "land":
        print("mask value can only be oce, land or  all")
    # return(1)
    return (lsmask, lats_reg, lons_reg)

"""
def ReadMMMF(
    modlst,
    varname,
    sdatelst,
    nmon,
    lat_bnds,
    lon_bnds,
    mask,
    printurl=False,
    interp=False,
    firstlead=0,
):
    nmod = len(modlst)
    nsdate = len(sdatelst)
    nmembmax = 60
    varMM = np.zeros((nmod, nsdate, nmembmax, nmon)) + 1e20
    for imod in range(nmod):
        mod = modlst[imod]
        print(mod)
        if ("chfp" in mod) & (varname == "tos"):
            varnameaux = "ts"
        else:
            varnameaux = varname

        varmod, lat, lon = Readfor(
            mod,
            varnameaux,
            sdatelst,
            nmon,
            lat_bnds,
            lon_bnds,
            True,
            mask,
            printurl=False,
            interp=interp,
        )
        nmemb = varmod.shape[2]
        varMM[imod, :, :nmemb, :] = varmod
        varMM = ma.array(varMM, mask=varMM > 1e19)
    return varMM


def ReadData(varname, modlst, sdatelst, nmon, lat_bnds, lon_bnds, mask, interp=False):

    dictobs = {
        "prec": ReadGPCP2Opendab,
        "sst": ReadERSSTOpendab,
        "tos": ReadERSSTOpendab,
        "uas": ReadERAINTOpendab,
    }
    funcobs = dictobs.get(varname)
    obs = funcobs(
        sdatelst,
        nmon,
        np.array(lat_bnds),
        np.array(lon_bnds),
        True,
        mask="oce",
        interp=interp,
    )
    MM = ReadMMBSC(
        modlst,
        varname,
        sdatelst,
        nmon,
        lat_bnds,
        lon_bnds,
        True,
        mask="oce",
        interp=interp,
    )

    # print(MM.shape)
    MM.shape = (
        MM.shape[0],
        MM.shape[1] / nstartmon,
        nstartmon,
        MM.shape[2],
        MM.shape[3],
    )
    # MM=np.swapaxes(MM,3,4)
    # print(MM.shape)
    obs.shape = (obs.shape[0] / nstartmon, nstartmon, obs.shape[1])
    return (MM, obs)


def extract_array(
    varf, varname, ntimesteps, lon_bnds, lat_bnds, start_time=0, level="all"
):
    """
    varf : np.array // tensors containing variables
    varname : string // name of the variable we want to extract
    ntimesteps : int // number of timesteps we want to extract (days for diary data, month for monthly data etc.)
    lon_bnds : int list of lenght 2 [a,b] // 0<=a<b<360
    lat_bnds : int list of lenght 2 [a,b] // -90<=a<b<90
    start_time : int // time step where we want to start the extraction
    level :
    """
    # print('level : ', level)
    varfvar = varf.variables[varname]
    varlat = getvarlat(varf)
    varlon = getvarlon(varf)

    lats = varf.variables[varlat][:]
    lons = varf.variables[varlon][:]
    # print lons[0]
    # lat_inds = np.where((lats >= lat_bnds[0]) & (lats <= lat_bnds[1]))[0]
    # lon_inds = lon_index(lons, lon_bnds )
    # print('entering lonlat_index with lons : ', lons)
    lat1D, lon1D, lat_inds, lon_inds = lonlat_index(lats, lons, lat_bnds, lon_bnds)
    # print
    # print(varf.variables[varname])
    try:
        unit = varf.variables[varname].units
    except Exception:
        defaultdic = {
            "tos": "K",
            "ts": "K",
            "sst": "K",
            "tauuo": "N m**-2",
            "tauu": "N m**-2",
        }
        unit = defaultdic.get(varname)
        if unit != None:
            print("warning: unit not found in the file, set to default: " + unit)
        else:
            print("warning: unit not found in the file, cannot set to default.")
    # print(unit)
    offsetdic = {
        "Celsius_scale": 0,
        "K": -273.15,
        "mm/day": 0,
        "m s-1": 0,
        "m s**-1": 0,
        "m/s": 0,
        "Kelvin_scale": -273.15,
        "N m**-2 s": 0,
        "degC": 0,
        "N m**-2": 0,
        "N/m2": 0,
        "m s**-1": 0,
        "m/s": 0,
        "m": 0,
        "Pa": 0,
    }
    scaledic = {
        "Celsius_scale": 1,
        "K": 1,
        "mm/day": 1,
        "Kelvin_scale": 1,
        "N m**-2 s": 1.0 / 21600,
        "degC": 1,
        "N m**-2": 1,
        "N/m2": 1,
        "m s**-1": 86400 * 1000,
        "m/s": 86400 * 1000,
        "m s-1": 86400 * 1000,
        "m": 1000,
        "Pa": 1.0 / 100,
    }
    if varname in ["uas", "vas"]:
        scaledic = {"m s-1": 1, "m s**-1": 1, "m/s": 1, "m s**-1": 1}

    offset = offsetdic.get(unit)
    scale = scaledic.get(unit)

    if scale == None:
        scale = 1
    if offset == None:
        offset = 0
    # print(scale, offset)
    ndim = len(varf.variables[varname].shape)
    # print(varf.variables[varname].shape)
    # if str(varf.variables[varname].dimensions[1])==getdimlat(varf, varname):
    #    varfvar=np.swapaxes(vararray,1,2)
    # print varfvar.shape
    # print "lon inds", lon_inds
    # print len(lon_inds)
    if len(lon_inds) == 2:
        if ndim == 5:
            if level != "all":
                vararrayW = varfvar[
                    start_time : ntimesteps + start_time,
                    :,
                    level,
                    lat_inds[0] : (lat_inds[-1] + 1),
                    lon_inds[0][0] : (lon_inds[0][-1] + 1),
                ]
                vararrayE = varfvar[
                    start_time : ntimesteps + start_time,
                    :,
                    level,
                    lat_inds[0] : (lat_inds[-1] + 1),
                    lon_inds[1][0] : (lon_inds[1][-1] + 1),
                ]
                # print unit,scaledic.get(unit),offsetdic.get(unit)
                vararray = (
                    np.ma.concatenate((vararrayW, vararrayE), axis=3) * scale + offset
                )
            else:
                vararrayW = varfvar[
                    start_time : ntimesteps + start_time,
                    :,
                    :,
                    lat_inds[0] : (lat_inds[-1] + 1),
                    lon_inds[0][0] : (lon_inds[0][-1] + 1),
                ]
                vararrayE = varfvar[
                    start_time : ntimesteps + start_time,
                    :,
                    :,
                    lat_inds[0] : (lat_inds[-1] + 1),
                    lon_inds[1][0] : (lon_inds[1][-1] + 1),
                ]
                # print unit,scaledic.get(unit),offsetdic.get(unit)
                vararray = (
                    np.ma.concatenate((vararrayW, vararrayE), axis=4) * scale + offset
                )
        elif ndim == 4:
            if level != "all":
                vararrayW = varfvar[
                    start_time : ntimesteps + start_time,
                    level,
                    lat_inds[0] : (lat_inds[-1] + 1),
                    lon_inds[0][0] : (lon_inds[0][-1] + 1),
                ]
                vararrayE = varfvar[
                    start_time : ntimesteps + start_time,
                    level,
                    lat_inds[0] : (lat_inds[-1] + 1),
                    lon_inds[1][0] : (lon_inds[1][-1] + 1),
                ]
                # print unit,scaledic.get(unit),offsetdic.get(unit)
                vararray = (
                    np.ma.concatenate((vararrayW, vararrayE), axis=2) * scale + offset
                )
            else:
                vararrayW = varfvar[
                    start_time : ntimesteps + start_time,
                    :,
                    lat_inds[0] : (lat_inds[-1] + 1),
                    lon_inds[0][0] : (lon_inds[0][-1] + 1),
                ]
                vararrayE = varfvar[
                    start_time : ntimesteps + start_time,
                    :,
                    lat_inds[0] : (lat_inds[-1] + 1),
                    lon_inds[1][0] : (lon_inds[1][-1] + 1),
                ]
                # print unit,scaledic.get(unit),offsetdic.get(unit)
                vararray = (
                    np.ma.concatenate((vararrayW, vararrayE), axis=3) * scale + offset
                )

        elif ndim == 3:
            vararrayW = varfvar[
                start_time : ntimesteps + start_time,
                lat_inds[0] : (lat_inds[-1] + 1),
                lon_inds[0][0] : (lon_inds[0][-1] + 1),
            ]
            vararrayE = varfvar[
                start_time : ntimesteps + start_time,
                lat_inds[0] : (lat_inds[-1] + 1),
                lon_inds[1][0] : (lon_inds[1][-1] + 1),
            ]
            vararray = (
                np.ma.concatenate((vararrayW, vararrayE), axis=2) * scale + offset
            )

        if len(lons.shape) == 1:
            lons_regW = lons[lon_inds[0]]
            lons_regE = lons[lon_inds[1]]
            lons_reg = np.ma.concatenate((lons_regW, lons_regE), axis=0)
            lats_reg = lats[lat_inds]
        elif len(lons.shape) == 2:
            lonsW = lons[
                lat_inds[0] : (lat_inds[-1] + 1), lon_inds[0][0] : (lon_inds[0][-1] + 1)
            ]
            lonsE = lons[
                lat_inds[0] : (lat_inds[-1] + 1), lon_inds[1][0] : (lon_inds[1][-1] + 1)
            ]

            lons_reg = np.ma.concatenate((lonsW, lonsE), axis=1)

            latsW = lats[
                lat_inds[0] : (lat_inds[-1] + 1), lon_inds[0][0] : (lon_inds[0][-1] + 1)
            ]
            latsE = lats[
                lat_inds[0] : (lat_inds[-1] + 1), lon_inds[1][0] : (lon_inds[1][-1] + 1)
            ]
            # print("I am here")
            # print(latsW.shape, latsE.shape)
            lats_reg = np.ma.concatenate((latsW, latsE), axis=1)
            # print(lats_reg.shape)

    else:
        # print(lon_inds[0])
        if ndim == 5:
            if level != "all":
                vararray = (
                    varfvar[
                        start_time : ntimesteps + start_time,
                        :,
                        level,
                        lat_inds[0] : (lat_inds[-1] + 1),
                        lon_inds[0][0] : (lon_inds[0][-1] + 1),
                    ]
                    * scale
                    + offset
                )
            else:
                vararray = (
                    varfvar[
                        start_time : ntimesteps + start_time,
                        :,
                        :,
                        lat_inds[0] : (lat_inds[-1] + 1),
                        lon_inds[0][0] : (lon_inds[0][-1] + 1),
                    ]
                    * scale
                    + offset
                )
        if ndim == 4:
            if level != "all":
                # print(varfvar.shape)
                # print(lon_inds,lat_inds)
                # print([ntimesteps,level,lat_inds[0],(lat_inds[-1]+1),lon_inds[0][0],(lon_inds[0][-1]+1)])
                vararray = (
                    varfvar[
                        start_time : ntimesteps + start_time,
                        level,
                        lat_inds[0] : (lat_inds[-1] + 1),
                        lon_inds[0][0] : (lon_inds[0][-1] + 1),
                    ]
                    * scale
                    + offset
                )
            else:
                print(lon_inds, lat_inds)
                vararray = (
                    varfvar[
                        start_time : ntimesteps + start_time,
                        :,
                        lat_inds[0] : (lat_inds[-1] + 1),
                        lon_inds[0][0] : (lon_inds[0][-1] + 1),
                    ]
                    * scale
                    + offset
                )
        elif ndim == 3:
            vararray = (
                varfvar[
                    start_time : ntimesteps + start_time,
                    lat_inds[0] : (lat_inds[-1] + 1),
                    lon_inds[0][0] : (lon_inds[0][-1] + 1),
                ]
                * scale
                + offset
            )

        # print lons9.44473297e+21

        if len(lons.shape) == 1:
            # print("I am here")
            # print(len(lons.shape))
            lons_reg = lons[lon_inds[0]]
            lats_reg = lats[lat_inds]

        elif len(lons.shape) == 2:
            lats_reg = lats[
                lat_inds[0] : (lat_inds[-1] + 1), lon_inds[0][0] : (lon_inds[0][-1] + 1)
            ]
            lons_reg = lons[
                lat_inds[0] : (lat_inds[-1] + 1), lon_inds[0][0] : (lon_inds[0][-1] + 1)
            ]

    # print('vararray shape : ',vararray.shape)

    return (vararray, lats_reg, lons_reg)


def geturllistBSC(urlbase, forcastname, varname, dateformat):
    urllist = []
    # if urlbase=="":from pathlib import Path

    urlbase = "/cnrm/pastel/USERS/prodhommec/NO_SAVE/" + forcastname + "/"
    urllist = glob(urlbase + varname + "*" + dateformat + "*")

    print(urlbase + varname + "*" + dateformat + "*")
    if len(urllist) == 0:
        urllist = glob(urlbase + varname + "*" + dateformat + "*")
    if len(urllist) == 0:
        print("no files found in directory: " + urlbase)

    urllist.sort()
    return urllist


# def ReadMFfor(forcastname, varname, sdatelst, nmon, lat_bnds, lon_bnds, areaav, mask, urlbase="",
#               printurl=False, interp=False, grid="r360x180", level="all"):
#     nmod=1
#     imod=0
#
#     #read data gridto read the appropriate mask
#     dateformat0='{0.year:4d}{0.month:02d}'.format(sdatelst[0])
#     #sdatelst[0].strftime("%Y%m")
#     #urlbase='http://earth.bsc.es/thredds/dodsC/exp/'+forcastname+'/monthly_mean/'
#     urllist=geturllistBSC(urlbase,forcastname,varname,dateformat0)
#     if (varname=="tauu")&(len(urllist)==0):
#         varname="tauuo"
#         urllist=geturllistBSC(urlbase,forcastname,varname,dateformat0)
#
#     #print urlbase
#     url0=urllist[0]
#     if interp:
#         url0=cdo.remapbil(grid, input="-selvar,"+varname+" "+url0)
#
#     print(url0)
#     varf0 = netCDF4.Dataset(url0)
#
#
#     varlat=getvarlat(varf0)
#     varlon=getvarlon(varf0)
#     varens=getvarens(varf0)
#
#     #print varf0.variables.keys()
#
#     nlat = varf0.variables[varlat].shape[0]
#     nlon = varf0.variables[varlon].shape
#
#     #print nlat,nlon
#     if len(nlon)==2:
#         nlon=nlon[1]
#     else:
#         nlon=nlon[0]
#
#     try:
#         nmembmax = varf0.variables[varens].shape[0]
#     except:
#         nmembmax = 20
#     #print nmembmax
#
#     #read a test array to find if correponding to the mask
#     arraytest,lats_reg, lons_reg=extract_array(varf0, varname, nmon, lon_bnds, lat_bnds, level=level)
#     varf0.close()
#
#     #print nlon, nlat
#
#     nsdates=len(sdatelst)
#     #read land sea mask and define size of the matrix
#
#     dims_expend=[nmod, nsdates, nmembmax, nmon]
#
#     #maskurl="/esarchive/exp/ecearth/constant/land_sea_mask_%ix%i.nc"%(nlon, nlat)
#     #print(maskurl)
#     #lsmask, lats_reg_mask, lons_reg_mask=read_mask(maskurl, lat_bnds, lon_bnds, mask)
#     if len(arraytest.shape)==3:
#         nlat, nlon = arraytest[0,:,:].shape
#     #        lsmask=(arraytest[0,:,:]!=1e20)*1
#             #print lsmask
#     #        print "warning: size of the mask different of size of the array no mask is applied"
#     elif len(arraytest.shape)==4:
#         nlat, nlon= arraytest[0,0,:,:].shape
#     #    if (lsmask.shape)!=arraytest[0,0,:,:].shape:
#     #        lsmask=(arraytest[0,0,:,:]!=1e20)*1
#     #        print "warning: size of the mask different of size of the array no mask is applied"
#
#
#     #lsmaskMM=extend_table(lsmask, dims_expend)
#     lsmaskMM=ma.zeros((nmod, nsdates, nmembmax, nmon, nlat, nlon))+1
#     varMM=ma.zeros((nmod, nsdates, nmembmax, nmon, nlat, nlon))+1e20
#
#
#     for idate in range(nsdates):
#         sdate=sdatelst[idate]
#         #dateformat=sdate.strftime("%Y%m")
#         dateformat='{0.year:4d}{0.month:02d}'.format(sdate)
#         #url=url0.replace(dateformat0, dateformat)
#         #print(urlbase)
#         #print(dateformat)
#         urllist=geturllistBSC(urlbase,forcastname,varname,dateformat)
#         #print(urllist)
#         #print(urllist)
#         #loop over members, needed for unaggregated
#         for imemb,url in enumerate(urllist):
#             if printurl:
#                 print(url)
#
#             if interp:
#                 url=cdo.remapbil(grid, input=url)
#             varf = netCDF4.Dataset(url)
#             vararray,lats_reg, lons_reg = extract_array(varf, varname, nmon, lon_bnds, lat_bnds, level=level)
#             #print lats_reg
#             #print lons_reg
#
#             #print urllist
#             if len(urllist)>1:
#                 #print url
#                 #print varMM.shape
#                 #print vararray.shape
#                 #print imemb
#                 vararray=vararray.squeeze()
#                 if len(vararray.shape)==2:
#                     varMM[imod,idate,imemb,:,:]=vararray
#                 elif len(vararray.shape)==3:
#                     varMM[imod,idate,imemb,:,:,:]=vararray
#
#         if len(urllist)==1:
#             #print varf.variables[varname].dimensions
#             #print getdimens(varf, varname)
#             #print str(varf.variables[varname].dimensions[1])==getdimens(varf, varname)
#             if str(varf.variables[varname].dimensions[1])==getdimens(varf, varname):
#                 vararray=np.swapaxes(vararray,0,1)
#             nmembidate=vararray.shape[0]
#             if nmembidate!=nmembmax:
#                 print("WARNING: number of member different for start date: "+dateformat)
#                 print("%i members instead of %i"%(nmembidate,nmembmax))
#                 nmembidate=min(nmembidate,nmembmax)
#             #print  varMM.shape,  vararray.shape
#             varMM[imod,idate,:nmembidate,:,:]=vararray[:nmembidate,:,:]
#
#     varMM.mask=1-lsmaskMM
#     #print lsmaskMM
#     if(areaav):
#         varMM=area_av(varMM, 4,5, lats_reg, lons_reg)
#         varMM.mask=varMM>1e19
#     #print(varMM.shape)
#     return(varMM,lats_reg, lons_reg)
#
# def ReadObs(varname, sdatelst, nmon, lat_bnds, lon_bnds, areaav, path, mask="oce", interp = False, grid="r360x180"):
# """
# """
# a=(glob(path))
# a.sort()
#
# url=cdo.mergetime(input=" ".join(a))
# if interp:
#     url=cdo.remapbil(grid, input=url)
# print(url)
# varf = netCDF4.Dataset(url)
#
#
# varlat=getvarlat(varf)
# varlon=getvarlon(varf)
# lats = varf.variables[varlat][:]
# lons = varf.variables[varlon][:]
#
# lat_inds = np.where((lats > lat_bnds[0]) & (lats < lat_bnds[1]))[0]
# lon_inds = lon_index(lons, lon_bnds)
# #contruct index list corresponding to the forecast startdates
#
# tname = "time"
# nctime = varf.variables[tname][:] # get values
# t_unit = varf.variables[tname].units # get unit  "days since 1950-01-01T00:00:00Z"
# t_cal = varf.variables[tname].calendar
# time = num2date(nctime,units = t_unit,calendar = t_cal)
# #print time
#
# forcastime=np.ndarray.flatten(np.transpose(np.array([sdatelst+relativedelta(months=m) for m in range(nmon)])))
# #print [s for s in list(forcastime)]
# #print forcastime
# s = forcastime[0]
# time = np.array([date(t.year, t.month, 1) for t in time])
# #print time
# forcastimeindex=np.array([np.where(time==date(s.year, s.month, 1))[0][0] for s in list(forcastime)])
# sdmin=forcastimeindex.min()
# sdmax=forcastimeindex.max()+1
#
# #download data subset
# #in case selection is over greenwitch selet 2subsets
# if len(lon_inds)==2:
#     vararrayW=varf.variables[varname][sdmin:sdmax,lat_inds,lon_inds[0]][forcastimeindex-sdmin,:,:]
#     vararrayE=varf.variables[varname][sdmin:sdmax,lat_inds,lon_inds[1]][forcastimeindex-sdmin,:,:]
#     varobs=np.ma.concatenate((vararrayW, vararrayE), axis=2)
#     lons_regW=lons[lon_inds[0]]
#     lons_regE=lons[lon_inds[1]]
#     lons_reg=np.ma.concatenate((lons_regW, lons_regE), axis=0)
# else:
#     varobs=varf.variables[varname][sdmin:sdmax,lat_inds,lon_inds[0]][forcastimeindex-sdmin,:,:]
#     lons_reg=lons[lon_inds[0]]
#
# varf.close()
# lats_reg=lats[lat_inds]
#
# varobs.shape=(len(sdatelst), nmon, varobs.shape[1], varobs.shape[2])
# varobs=ma.array(varobs, mask=varobs>1e20)
#
# if(areaav):
#     varobs=area_av(varobs, 2, 3, lats_reg, lons_reg)
#
# return(varobs, lats_reg, lons_reg)
