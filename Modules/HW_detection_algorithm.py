import numpy as np
from function_read import *
from scipy.stats import gaussian_kde

def runningmean(field, ndrun):
    nday = field.shape[0]
    run=np.zeros((nday))
    run.shape
    run = np.convolve(field, np.ones((ndrun,))/ndrun)[(ndrun-1-ndrun/2):(-ndrun+ndrun/2)]
    return(run)

# use climatology subsection of data as "target"
def find_subHW(target, climyears, years, percent_thresh, duration_min, cross_valid, opt):
    """function that find all the sub heatwave from an array
    target: numpy array of 3dimension: nyear, nday,  members
    percent_thresh: Chosen percentile (usually 90, 95 or 99)
    duration_min: Minimum length of a HW or a sub HW
    return: all sub heatwaves in a list of list of list and ecfmax to construct the fit"""

    #Do we have interest in keeping cross validation? Yes we have

    targetaux=np.array(target)
    ind1=climyears[0]-years[0]
    ind2=climyears[-1]-years[0]
    targetclim=target[ind1:ind2+1,:,:]

    #print(target.shape)
    (nyear, nday, nmemb) = target.shape
    nyearclim= targetclim.shape[0]

    if cross_valid==True: # Not used for Data4UAH
        #if cross_validation, only use of the first memb
        perc_thresh_climlst=[]
        mean_thresh_climlst=[]
        for iyear in range(nyear):
            #print(targetaux.shape)
            #targetaux=np.delete(targetaux[:,:,0], iyear, axis=0)
            perc_thresh_clim_not_smoothed=np.percentile(np.delete(targetaux[:,:,0], iyear, axis=0), percent_thresh, axis=0)
            mean_clim_not_smoothed=np.mean(np.delete(targetaux[:,:,0], iyear, axis=0), axis=0)
            if opt=="runningmean":
                ndrun=30
                perc_thresh_clim=runningmean(perc_thresh_clim_not_smoothed,ndrun)
                mean_clim=runningmean(mean_clim_not_smoothed,ndrun)
            #compute the smoothed climatology using a polyfit of order 6                                
            if opt=="polyfit":
                perc_thresh_clim = np.polyval(np.polyfit(range(nday), perc_thresh_clim_not_smoothed, deg=6), range(nday))
                mean_clim = np.polyval(np.polyfit(range(nday), mean_clim_not_smoothed, deg=6), range(nday))
            perc_thresh_climlst.append(perc_thresh_clim)
            mean_climlst.append(mean_clim)
            
        perc_thresh_clim=np.array(perc_thresh_climlst)
        perc_thresh_clim=extend_table(perc_thresh_clim, [nmemb])
        perc_thresh_clim=np.swapaxes(np.swapaxes(perc_thresh_clim,1,0), 2,1)
        mean_clim=np.array(mean_climlst)
        mean_clim=extend_table(mean_clim, [nmemb])
        mean_clim=np.swapaxes(np.swapaxes(mean_clim,1,0), 2,1)

    else: # Changes to climatology calculation
        targetaux2 = np.swapaxes(targetclim, 1,2)
        targetaux2 = np.reshape(targetaux2, (nyearclim * nmemb, nday))
        perc_thresh_clim_not_smoothed = np.percentile(targetaux2[:,:], percent_thresh, axis=0) # CLIM      
        mean_clim_not_smoothed = np.mean(targetaux2[:,:], axis=0)
        if opt=="runningmean":
            ndrun=30
            perc_thresh_clim = runningmean(perc_thresh_clim_not_smoothed,ndrun)
            mean_clim = runningmean(mean_clim_not_smoothed,ndrun)
        #compute the smoothed climatology using a polyfit of order 6
        if opt=="polyfit":
            perc_thresh_clim = np.polyval(np.polyfit(range(nday), perc_thresh_clim_not_smoothed, deg=6), range(nday)) # CLIM
            mean_clim = np.polyval(np.polyfit(range(nday), mean_clim_not_smoothed, deg=6), range(nday))
        perc_thresh_clim = extend_table(perc_thresh_clim, [nyear, nmemb]) # keep as nyear, NOT nyearclim
        perc_thresh_clim = np.swapaxes(perc_thresh_clim, 1,2)
        mean_clim = extend_table(mean_clim, [nyear, nmemb]) # keep as nyear, NOT nyearclim
        mean_clim = np.swapaxes(mean_clim, 1,2) 
        
    #array to keep subHW values
    ndayseas = nday//duration_min + 1
    subHWarray = np.zeros((nyear, ndayseas, nmemb))

    #check days exceeding the percentile
    #exed_percentile = (target[:,limseas1:limseas2, :] - perc_thresh_clim[:, limseas1:limseas2, :])
    exed_percentile = (target - perc_thresh_clim)
    exed_percentile = np.ma.array(exed_percentile, mask=exed_percentile<0)

    ndaysexed_percentile = np.sum(exed_percentile>0, axis = 1)
    DD_threshold = np.ma.sum(exed_percentile, axis = 1) #np.ma.sum?, np.ma?

    #find heatwave (>x days exceeding threshold percentile)
    mask_duration_min = np.array(1-exed_percentile.mask, copy=True)
    #print('mask_duration_min.shape : ', mask_duration_min.shape)
    for ilen in range(1,duration_min):
        mask_duration_min[:,0:-(ilen),:] = mask_duration_min[:,0:-(ilen),:] + (1-exed_percentile.mask[:,ilen:,:])
    #remove heat wave shorter than duration_min
    mask_duration_min[mask_duration_min<duration_min] = 0


    HWMI = np.zeros((nyear,nmemb)) #Heat wave magnitude
    #fit the accumulative distribution function 
    ecdfmax=[]
    HWlstmembyear=[]
    HWstartmembyear = []
    HWendmembyear = []
    for iyear in range(nyear):
        HWlstmemb=[]
        HWstartmemb = []
        HWendmemb = []
        for imemb in range(nmemb):
            HWlst = []
            HWstart = []
            HWend = []
            maxtmp = []
            for itime in range(nday):
                #check if an heatwave has been detected for this grid point and not dealed before
                if mask_duration_min[iyear,itime,imemb]:
                    #compute the magnitude of the sub heatwave
                    subHWlst = []
                    subHW = np.sum(exed_percentile[iyear,itime:itime+duration_min,imemb])
                    subHWarray[iyear, itime//duration_min, imemb] = subHW
                    subHWlst.append(subHW)

                    #check if the heatwave is longer than duration_min
                    subHW = 0 #variable to keep subheatwave 
                    ilen = 1 #variable to compute the length of the big Heatwave
                    
                    while (itime+ilen+duration_min-1<nday) & (mask_duration_min[iyear,itime+ilen,imemb] == duration_min):
                        #print "big heatwave"
                        mask_duration_min[iyear,itime,imemb] += 1
                        #remove subheawes that are part of the big heat wave
                        mask_duration_min[iyear,itime+ilen,imemb] = 0
                        #cumulate subheatwave magnitude 
                        subHW=subHW+exed_percentile[iyear,itime+ilen+duration_min-1,imemb]
                        #when we reach a duration_min length subHW keep it
                        if ilen%duration_min == 0:
                            subHWlst.append(subHW)
                            subHWarray[iyear, itime//duration_min+ilen//duration_min, imemb] = subHW
                            subHW = 0
                        ilen = ilen+1
                        
                    #keep subheatwave of length <duration_min if included in a larger heatwave
                    if subHW > 0:
                        subHWlst.append(subHW)
                        subHWarray[iyear, itime//duration_min+(ilen-1)//duration_min+1, imemb] = subHW
                    #print("itime",itime)

                    #keep the subHWlst list in the HW list
                    HWlst.append(subHWlst)
                    # keep the beginning and end of the HW
                    HWstart.append(itime)
                    HWend.append(itime + ilen + duration_min-1)
                    #keep the maximum to fit the ecdf
                    maxtmp.append(max(subHWlst))
            #keep the max for the considered season
            if len(maxtmp) != 0:
                if imemb == 0:
                    ecdfmax.append(max(maxtmp))
                HWlstmemb.append(HWlst)
                HWstartmemb.append(HWstart)
                HWendmemb.append(HWend)
            else:
                #print imemb, ilat, ilon
                #print "no heatwave"
                if imemb==0:
                    ecdfmax.append(-1)
                HWlstmemb.append([[0]])
                HWstartmemb.append([0])
                HWendmemb.append([0])
        #keep HW for years
        HWlstmembyear.append(HWlstmemb)
        HWstartmembyear.append(HWstartmemb)
        HWendmembyear.append(HWendmemb)

    return(HWlstmembyear, HWstartmembyear, HWendmembyear, ecdfmax, ndaysexed_percentile, DD_threshold, subHWarray, perc_thresh_clim, exed_percentile, mask_duration_min, mean_clim)

def fitforHWMI(ecdfmax, cross_valid, yeartoexclude = None):
    """compute the gaussian fit 
    ecdfmax: list of yearly maximum of heatwave for function find_subHW
    cross_valid: True/False
    year_to_exclude: int (only if using cross validation
    """
    ecdfmax=np.array(ecdfmax)
    if cross_valid:
        ecdfmax = np.delete(ecdfmax, yeartoexclude)
    
    #remove -1 from ecdf max (corresponding to years with no heatwave)
    ecdfmaxfit = ecdfmax[ecdfmax>-1]
    try:
        kde = gaussian_kde(ecdfmaxfit)
    except:
        #print("impossible to fit")#  : lon %i, lat%i"%(ilon,ilat))
        #print ecdfmax
        kde = False
    return(kde)

def calcHWMImemb(HWlstmemb, subHWarrayyear, nday, duration_min, nmemb, kde):
    """
    calc the HWMI for a given year
    HWlstmemb: list of list containing the list of subheatwave for each members
    kde: gaussian fit from the function fitforHWMI
    return: list """
    #print HWlstmemb
    HWMI = []
    HWmembfit = []
    fitsubHWarray = np.zeros((nday//duration_min+1, nmemb))
    #print nmemb
    for imemb in range(nmemb):
        HWlstfit = []
        maxHWnorm=0
        HWlst=HWlstmemb[imemb]
        #print HWlst
        #compute HW magnitude based on normalized sub heatwave magnitude
        HWlstfit = []
        #print mask_duration_min[iyear,:,imemb,ilat,ilon]
        for iHW in range(len(HWlst)):
            sublst=HWlst[iHW]
            #find the problability corresponding to the subHW on the fitted cdf
            #print sublst
            if kde!=False:
                fitsum=sum([kde.integrate_box_1d(0,i) for i in sublst])                
                #print fit
                #keep the maximum
                maxHWnorm=max(fitsum,maxHWnorm)
                #keep all the heatwave
                HWlstfit.append(fitsum)
                #feet the subHW from the array to keep all subHW values (not optimal but easier)
                for imemb in range(nmemb):
                    for iday in range(nday//duration_min+1):
                        subaux = subHWarrayyear[iday, imemb]
                        if subaux>0:
                            fitsubHWarray[iday, imemb] = kde.integrate_box_1d(0,subaux)
            #print imemb, iHW, fitsum
            #store HW magnitude
        HWMI.append(maxHWnorm)
        HWmembfit.append(HWlstfit)
        #print HWMI   
    return(HWMI, HWmembfit, fitsubHWarray)

def calc_HWMIyear(target, climyears, years, cross_valid, percent_thresh, duration_min, opt):
    """
    function computing the HWMI index for a 3D array at a certain_loc ilat, ilon
        target: numpy array of 3dimension: nyear, nday, nmembs
        cross_valid: True/False, in case of True, exclude the year
        return the HWMI for each year
    """

    (nyear,nday,nmemb) = target.shape
    sstMeanarrayyear = np.mean(target, axis=1) # CLIM
    HWlstmembyear, HWstartmembyear, HWendmembyear, ecdfmax, ndaysexed_percentile, DD_threshold, subHWarrayyear, perc_thresh_clim, exed_percentile, mask_duration_min, mean_clim = find_subHW(target, climyears, years, percent_thresh=percent_thresh, duration_min=duration_min, cross_valid=cross_valid, opt=opt)
    HWMIyear = []
    HWlstyear = []
    fitsubHWarrayyear = np.zeros((nyear, nday//duration_min+1, nmemb))
    count_impossible_fit = 0
    if cross_valid==False:
        kde = fitforHWMI(ecdfmax, cross_valid = False)
        if kde == False:
            count_impossible_fit = 1
    for iyear in range(nyear):
        if cross_valid==True:
            kde = fitforHWMI(ecdfmax, cross_valid = True, yeartoexclude = iyear)
            #print HWlstmembyear[iyear]
            #print len(HWlstmembyear[iyear])
            #exit(1)
        #print(subHWarrayyear.shape)
            if kde == False:
                count_impossible_fit += 1/nyear
        HWMI, HWlstfit, fitsubHWarray = calcHWMImemb(HWlstmembyear[iyear], subHWarrayyear[iyear, :, :], nday, duration_min, nmemb, kde)
        HWMIyear.append(HWMI)
        HWlstyear.append(HWlstfit)
        fitsubHWarrayyear[iyear, :, :] = fitsubHWarray

    HW_arg=np.argwhere(mask_duration_min.flatten()>0)
    HW_intensity_daily=np.zeros(((nyear)*nday))
    for arg in HW_arg:
        arg=int(arg)
        dur=int(mask_duration_min.flatten()[arg])
        HW_intensity_daily[arg:arg+dur]=exed_percentile.flatten()[arg:arg+dur]

    HW_occurrence=np.copy(HW_intensity_daily)
    HW_occurrence[HW_occurrence>0]=1
    HW_intensity_daily=np.ma.masked_where(HW_intensity_daily==0,HW_intensity_daily)
    HW_occurrence=np.reshape(HW_occurrence,(nyear,nday))
    HW_intensity_daily=np.reshape(HW_intensity_daily,(nyear,nday))
    
    return {"clim":mean_clim[0,:,0],"pc_90":perc_thresh_clim[0,:,0],"HW_intensity_daily":HW_intensity_daily,"HW_occurrence":HW_occurrence,"HWMI_year":sum(HWMIyear, [])}
    #return(HWMIyear, HWlstyear, HWstartmembyear, HWendmembyear, ndaysexed_percentile, DD_threshold, subHWarrayyear, fitsubHWarrayyear, sstMeanarrayyear, count_impossible_fit, perc_thresh_clim[0], exed_percentile, mask_duration_min, mean_clim[0], HW_intensity_daily, HW_occurrence)