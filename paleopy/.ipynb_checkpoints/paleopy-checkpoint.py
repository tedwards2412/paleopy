from scipy.interpolate import interp1d
from scipy.integrate import cumtrapz, quad
from scipy.special import erf
from paleo.paleopy_classes import *

def window(x, x_a, x_b, sigma):
    return 0.5*(erf((x - x_a)/(np.sqrt(2.0)*sigma)) - erf((x - x_b)/(np.sqrt(2.0)*sigma)))

def calcBins(sigma):
    x0 = sigma/2.0
    x_bins = np.logspace(np.log10(x0), 3, 70)
    return x_bins
    
def calcBins_1nm():
    x0 = 0.5
    x_bins = np.arange(x0, 100.5)
    x_bins = np.append(x_bins, np.arange(100.5, 1001,10))
    return x_bins
    
def GetBackground(mineral, sigma, x_bins=None):
    x0 = sigma/2.0
    
    x_bins_all = np.logspace(-1, 3,200)
    x_width_all = np.diff(x_bins_all)
    x_c_all = x_bins_all[:-1] + x_width_all/2

    if (x_bins is None):
        x_bins = calcBins(sigma)
    N_bins = len(x_bins) - 1
    
    Nevents_BG = []
    
    T_exp = 1e7 #Set exposure time for the Thorium background
    
    dRdx_BG = mineral.dRdx_nu(x_bins_all, components=True, gaussian=False)
    dRdx_BG.append(mineral.dRdx_neutrons(x_bins_all))
    
    for dRdx in dRdx_BG:
        #dRdx_smooth = gaussian_filter1d(dRdx, sigma, mode='constant',cval = 1e-30)
        dRdx_interp = interp1d(x_c_all, dRdx, bounds_error=False, fill_value=0.0)
        
        N_events_ind = np.zeros(N_bins)
        for i in range(N_bins):
            xmean = 0.5*(x_bins[i] + x_bins[i+1])
            x1 = xmean - 5.0*sigma
            x2 = xmean + 5.0*sigma
            x1 = np.clip(x1, 0.1, 1e5)
            #print(xmean, x1, x2)
            integ = lambda y: dRdx_interp(y)*window(y, x_bins[i], x_bins[i+1], sigma)
            #print(integ(xmean))
            N_events_ind[i] = quad(integ, x1, x2, epsrel=1e-4)[0] + 1e-30
        
        Nevents_BG.append(N_events_ind)
        
        #dRdx *= x_width
        
    
    #Add the (smeared) delta-function for Thorium recoils
    Nevents_Th = np.zeros(N_bins)
    x_Th = mineral.xT_Thorium()
    for i in range(N_bins):
        Nevents_Th[i] = window(x_Th, x_bins[i], x_bins[i+1], sigma)
    
    Nevents_BG.append(mineral.norm_Thorium(T=T_exp)*Nevents_Th)
    
    return Nevents_BG

def GetSignal(mineral, sigma, m_DM, xsec, x_bins=None):
    x0 = sigma/2.0
    
    x_bins_all = np.logspace(-1, 3,200)
    x_width_all = np.diff(x_bins_all)
    x_c_all = x_bins_all[:-1] + x_width_all/2
    
    if (x_bins is None):
        x_bins = calcBins(sigma)
    N_bins = len(x_bins) - 1
    
    dRdx_sig = mineral.dRdx(x_bins_all, xsec, m_DM, gaussian=False)
    #dRdx_smooth = gaussian_filter1d(dRdx_sig,sigma, mode='constant',cval = 1e-30)
    dRdx_interp = interp1d(x_c_all, dRdx_sig, bounds_error=False, fill_value=0.0)
    
    Nevents_sig = np.zeros(N_bins)
    
    for i in range(N_bins):
        xmean = 0.5*(x_bins[i] + x_bins[i+1])
        x1 = xmean - 5.0*sigma
        x2 = xmean + 5.0*sigma
        x1 = np.clip(x1, 0.1, 1e5)
        #x1 = x_bins[i]
        #x2 = x_bins[i+1]
        integ = lambda y: dRdx_interp(y)*window(y, x_bins[i], x_bins[i+1], sigma)

        
        
        Nevents_sig[i] = quad(integ, x1, x2,epsrel=1e-4)[0]
    
    #Nevents_sig = np.array([quad(dRdx_interp, x_bins[i], x_bins[i+1])[0] for i in range(N_bins)])
    
    return Nevents_sig + 1e-30
    