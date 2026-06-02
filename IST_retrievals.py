#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 18 12:45:26 2019

@author: Mai Winstrup, DMI -> modified by gd
"""

import numpy as np

"""
Coefficients apply to the NORTHERN hemisphere.
Following the numbering scheme in ATBD, Table 3.
o: oblique, n: nadir view
theta: view angle (in degrees)

Brightness temperatures correspond to:
Tb37 = S7 (3.742 micron)
Tb11 = S8 (10.85 micron)
Tb12 = S9 (12.02 micron)

Using a value of the land cover variable, lc, equal to 1.5 for sea ice.

An error estimate is computed accounting for the uncertainty in measured 
brightness temperatures and for the uncertainty in equation used to compute the 
IST. The latter is set equal to the residual of the fit for the regression. 

nedt are the noise equivalent error uncertainty of the brightness channels. 
All measurements uncertainties are uncorrelated. Theta is assumed known without 
uncertainty.

"""

#Algorithm calibration uncertainty - dummy
sigma_dummy=999.9
var_nedt_dummy=999.9

def ist_retrievals(equation_type, 
                   Tb11n=0., Tb12n=0., Tb37n=0.,
                   Tb11o=0., Tb12o=0., Tb37o=0., 
                   thetan=0., lc=1.5, Tb11nsim=0., Tb12nsim=0.,
                   nedt_Tb11=0., nedt_Tb12=0., nedt_Tb37=0.,
                   nedt_Tb11o=0., nedt_Tb12o=0., nedt_Tb37o=0.): 
    ## FIXME: nedt_Tb11o, nedt_Tb12o, and nedt_Tb37o are missing -> USE DEFAULT (0.02)) now...
    

    # Convert angles to radians:
##    thetan = (thetan/100.)*2*np.pi/360.
#    thetan = (thetan)*2*np.pi/360.
    thetan = np.deg2rad(thetan)*2*np.pi/360.
        
#     elif equation_type == 2:
    if equation_type == 0:
        # Equation 12 in ATBD (needs correction in ATBD)
        #Version: 20200604 -0.9504     2.5061    -1.5024      0.072
        a = [-0.9504, 2.5061, -1.5024]
        ist = a[0] + a[1]*Tb11o + a[2]*Tb12o

        # Uncertainty due to noise in brightness temperatures (uncorrelated):
        var_nedt = a[1]**2*nedt_Tb11o**2 + a[2]**2*nedt_Tb12o**2
        sigma_nedt = np.sqrt(var_nedt)
        
        # Residual of the fit for the regression:
        std_of_residuals = 0.072
        sigma_algo = std_of_residuals        
            
    elif equation_type == 1:
        # Equation 11 in ATBD
        #Version: 20200604 -1.4567     1.0077      0.296
        a = [-1.4567, 1.0077]
        ist = a[0] + a[1]*Tb11o
        
        # Uncertainty due to noise in measured brightness temperatures:
        var_nedt = a[1]*nedt_Tb11o
        sigma_nedt = np.sqrt(var_nedt)

        # Uncertainty due to equation:
        # Given as the residual of the fit for the regression:
        std_of_residuals = 0.296
        sigma_algo = std_of_residuals
        
#     elif equation_type == 3:
    elif equation_type == 2:
        # All temperature intervals (eq. 13 in ATBD) 
        #Version: 20200604  -0.6384     2.4444    -1.4417    -0.0633      0.072

        ## METOP B OPR - OSI 205a/b - TEMPORARY (MEDIUM values from eq3...)!!  - virker ikke da koefficienterne er formuleret anderledes...
        #WAITING FOR NEW KOEFF FROM STEINAR...
#        a = [-4.01702, 1.01615, 1.41726, -0.03038]
        ## SLSTR
        a = [-0.6384, 2.4444, -1.4417,-0.0633]
        ist = a[0] + a[1]*Tb11n + a[2]*Tb12n + a[3]*(Tb11n-Tb12n)*(1./np.cos(thetan)-1.)

        # Uncertainty due to noise in brightness temperatures (uncorrelated):
        k = 1./np.cos(thetan)-1.
        var_nedt = (a[1]+a[3]*k)**2*nedt_Tb11**2 + (a[2]-a[3]*k)**2*nedt_Tb12**2
        sigma_nedt = np.sqrt(var_nedt)

        # Residual of the fit for the regression:
        std_of_residuals = 0.072
        sigma_algo = std_of_residuals        
    
    
#     elif equation_type == 0: 
    elif equation_type == 3: 
        # Version "3b"
        # Different coefficients depending on temperature interval of Tb11n.

        ## METOP B OPR - OSI 205a/b - virker ikke da koefficienterne er formuleret anderledes...
#        a_cold = [-3.29453, 1.01404, 0.74924, 0.01508]
        #Version: 20200604 -0.5197     2.3888    -1.3865    -0.1443      0.059
#        a_med = [-4.01702, 1.01615, 1.41726, -0.03038]
        #Version: 20200604 -0.3008     2.4573    -1.4560     0.0317      0.080
#        a_warm = [-4.61195, 1.01815, 1.37783, 0.30656]
        
        
        # Equation 13 in ATBD    SLSTR
        #Version: 20200604 -2.5381     2.0746    -1.0638    -0.0814      0.049
        a_cold = [-2.5381, 2.0746, -1.0638, -0.0814]
        #Version: 20200604 -0.5197     2.3888    -1.3865    -0.1443      0.059
        a_med = [-0.5197, 2.3888, -1.3865, -0.1443]
        #Version: 20200604 -0.3008     2.4573    -1.4560     0.0317      0.080
        a_warm = [-0.3008, 2.4573, -1.456, 0.0317]
    
        ist = np.where(Tb11n<240., 
                       a_cold[0] + a_cold[1]*Tb11n + a_cold[2]*Tb12n + a_cold[3]*(Tb11n-Tb12n)*(1./np.cos(thetan)-1),
                       a_med[0] + a_med[1]*Tb11n + a_med[2]*Tb12n + a_med[3]*(Tb11n-Tb12n)*(1./np.cos(thetan)-1))
        ist = np.where(Tb11n>=260., 
                       a_warm[0] + a_warm[1]*Tb11n + a_warm[2]*Tb12n + a_warm[3]*(Tb11n-Tb12n)*(1./np.cos(thetan)-1), 
                       ist)
        
        # Keep mask:
#        ist_masked = np.ma.masked_where(Tb11n.mask, ist)
#        ist = ist_masked
        
        # Uncertainty due to noise in brightness temperatures (uncorrelated):
        k = 1./np.cos(thetan)-1.
        
        var_nedt = np.where(Tb11n<240., 
                            (a_cold[1]+a_cold[3]*k)**2*nedt_Tb11**2 + (a_cold[2]-a_cold[3]*k)**2*nedt_Tb12**2, 
                            (a_med[1]+a_med[3]*k)**2*nedt_Tb11**2 + (a_med[2]-a_med[3]*k)**2*nedt_Tb12**2)
        var_nedt = np.where(Tb11n>=260., 
                            (a_warm[1]+a_warm[3]*k)**2*nedt_Tb11**2 + (a_warm[2]-a_warm[3]*k)**2*nedt_Tb12**2, 
                            var_nedt)
        sigma_nedt = np.sqrt(var_nedt)
        
        # Residual of the fit for the regression:
        std_of_residuals_cold = 0.049
        std_of_residuals_med = 0.059
        std_of_residuals_warm = 0.080        
        
        # Uncertainty from the algorithm:
#        sigma_algo = np.where(Tb11n<240., std_of_residuals_cold, std_of_residuals_med)
#        sigma_algo = np.where(Tb11n>260., std_of_residuals_warm, sigma_algo)
 
#        if (Tb11n[10][10]<240.):
#            sigma_algo=std_of_residuals_cold
#        elif (Tb11n[10][10]>260.):
#            sigma_algo=std_of_residuals_warm
#        else:
#            sigma_algo=std_of_residuals_med
        if (Tb11n<240.):
            sigma_algo=std_of_residuals_cold
        elif (Tb11n>260.):
            sigma_algo=std_of_residuals_warm
        else:
            sigma_algo=std_of_residuals_med


    elif equation_type == 4: 
        # Equation 14 in ATBD 
        #Version: 20200604 0.5277     4.3933    -2.1706    -2.3728     1.1484      0.032
        a = [0.5277, 4.3933, -2.1706, -2.3728, 1.1484]
        ist = a[0] + a[1]*Tb11n + a[2]*Tb12n + a[3]*Tb11o + a[4]*Tb12o
        
        # Uncertainty due to noise in brightness temperatures (all channels assumed uncorrelated):
        var_nedt = a[1]**2*nedt_Tb11**2 + a[2]**2*nedt_Tb12**2 + a[3]**2*nedt_Tb11o**2 + a[4]**2*nedt_Tb12o**2
        sigma_nedt = np.sqrt(var_nedt)
    
        # Uncertainty from the algorithm:
        std_of_residuals = 0.032
        sigma_algo = std_of_residuals
    
    
    elif equation_type == 5: 
        # Equation 15 in ATBD 
        # lc is a land-cover specific term, equal to 1.5 for sea ice
        #Version: 20200604 -0.7760     1.8935     1.0033      0.091
        a = [-0.776, 1.8935, 1.0033]
        ist = a[0] + a[1]*(Tb11n-Tb12n)**(1./np.cos(thetan/lc)) + a[2]*Tb12n
        
        # Uncertainty due to noise in brightness temperatures (all channels assumed uncorrelated):
        # Linear expansion as Taylor polynomial:
        k = 1./np.cos(thetan/lc)
        df_dTb11 = a[1]*k*(Tb11n-Tb12n)**(k-1)
        df_dTb12 = -a[1]*k*(Tb11n-Tb12n)**(k-1) + a[2]
        # The variance formula:
        var_nedt = df_dTb11**2*nedt_Tb11**2 + df_dTb12**2*nedt_Tb12**2
        sigma_nedt = np.sqrt(var_nedt)

        # Uncertainty from the algorithm:
        std_of_residuals = 0.091
        sigma_algo = std_of_residuals
        
        
    elif equation_type == 6: 
        # Equation 16 in ATBD
        #Version: 20200604 0.5002     4.2244    -1.6861    -2.6661     1.1258     2.0787      0.023
        a = [0.5002, 4.2244, -1.6861, -2.6661, 1.1258, 2.0787]
        ist = a[0] + a[1]*Tb11n + a[2]*Tb12n + a[3]*Tb11o +a[4]*Tb12o + a[5]*(Tb11n-Tb12n)*(1./np.cos(thetan)-1.)
        
        # Uncertainty due to noise in brightness temperatures (all channels assumed uncorrelated):
        k = 1./np.cos(thetan)-1.
        var_nedt = (a[1]+a[5]*k)**2*nedt_Tb11**2 + (a[2]-a[5]*k)**2*nedt_Tb12**2 + a[3]**2*nedt_Tb11o**2 + a[4]**2*nedt_Tb12o**2
        sigma_nedt = np.sqrt(var_nedt)

        # Uncertainty from the algorithm:
        std_of_residuals = 0.023
        sigma_algo = std_of_residuals

        
    elif equation_type == 7: 
        # Equation 17 in ATBD
        #Version: 20200604 0.3481     5.6683    -2.1445    -3.3352     1.2080    -1.2486     0.8506      0.025
        a = [0.3481, 5.6683, -2.1445, -3.3352, 1.2080, -1.2486, 0.8506]
        ist = a[0] + a[1]*Tb11n + a[2]*Tb12n +a[3]*Tb11o + a[4]*Tb12o + a[5]*Tb37n + a[6]*Tb37o

        # Uncertainty due to noise in brightness temperatures (all channels assumed uncorrelated):
        # FIXME: USE nedt from oblique views!!
        var_nedt = a[1]**2*nedt_Tb11**2 + a[2]**2*nedt_Tb12**2 + a[3]**2*nedt_Tb11o**2 + \
            a[4]**2*nedt_Tb12o**2 + a[5]**2*nedt_Tb37**2 + a[6]**2*nedt_Tb37o**2
        sigma_nedt = np.sqrt(var_nedt)
        
        # Uncertainty from the algorithm:
        std_of_residuals = 0.025
        sigma_algo = std_of_residuals

   
    elif equation_type == 8:
        # Equation 11 in ATBD
        #Version: 20200604 -1.2541     1.0065      0.294
        a = [-1.2541, 1.0065]
        ist = a[0] + a[1]*Tb11n
        
        # Uncertainty due to noise in brightness temperatures:
        var_nedt = a[1]*nedt_Tb11
        sigma_nedt = np.sqrt(var_nedt)

        # Uncertainty from the algorithm:
        std_of_residuals = 0.294
        sigma_algo = std_of_residuals

        
    elif equation_type == 9: 
        #Version: 20200604 -1.2910     1.0059     0.0016      0.262
        a = [-1.291, 1.0059, 0.0016]
        ist = a[0] + a[1]*Tb11n + a[2]*Tb11n*(1./np.cos(thetan)-1)
        
        # Uncertainty due to noise in brightness temperatures:
        k = 1./np.cos(thetan)-1
        var_nedt = (a[1]+a[2]*k)**2*nedt_Tb11**2
        sigma_nedt = np.sqrt(var_nedt)

        # Uncertainty from the algorithm:
        std_of_residuals = 0.262
        sigma_algo = std_of_residuals
    
    
    elif equation_type == 10:
        #Version: 20200604 -0.6514     2.3952    -1.3924      0.073
        a = [-0.6514, 2.3952, -1.3924]
        ist = a[0] + a[1]*Tb11n + a[2]*Tb12n

        # Uncertainty due to noise in brightness temperatures (uncorrelated):
        var_nedt = a[1]**2*nedt_Tb11**2 + a[2]**2*nedt_Tb12**2
        sigma_nedt = np.sqrt(var_nedt)        

        # Uncertainty from the algorithm:
        std_of_residuals = 0.073
        sigma_algo = std_of_residuals


    elif equation_type == 11:
        #Version: 20200604 -0.6099     2.4722    -1.4696    -0.0003      0.068
        a = [-0.6099, 2.4722, -1.4696, -0.0003]
        ist = a[0] + a[1]*Tb11n + a[2]*Tb12n + a[3]*Tb11n*(1./np.cos(thetan)-1)
        
        # Uncertainty due to noise in brightness temperatures (uncorrelated):
        k = 1./np.cos(thetan)-1
        var_nedt = (a[1]+a[3]*k)**2*nedt_Tb11**2 + a[2]**2*nedt_Tb12**2
        sigma_nedt = np.sqrt(var_nedt)

        # Uncertainty from the algorithm:
        std_of_residuals = 0.068
        sigma_algo = std_of_residuals

    elif equation_type == 12:
        # ISTnc ~ T11nc + T11oc
        #Version: 20200604 -0.3952     2.3205    -1.3193      0.137
        a = [-0.3952, 2.3205, -1.3193]
        ist = a[0] + a[1]*Tb11n + a[2]*Tb11o

        # Uncertainty due to noise in brightness temperatures (uncorrelated):
        var_nedt = a[1]**2*nedt_Tb11**2 + a[2]**2*nedt_Tb11o**2
        sigma_nedt = np.sqrt(var_nedt)        

        # Uncertainty from the algorithm:
        std_of_residuals = 0.137
        sigma_algo = std_of_residuals
        
    elif equation_type == 13:
        #Version: 20200604 -0.1144     2.4738    -1.4744     5.5904      0.015
        a = [-0.1144, 2.4738, -1.4744, 5.5904]
        k = 1./np.cos(thetan)-1.
        ist = a[0] + a[1]*Tb11n + a[2]*Tb11o + a[3]*(Tb11n-Tb11o)*k

        # Uncertainty due to noise in brightness temperatures (uncorrelated):
        #k = 1./np.cos(thetan)-1.
        var_nedt = (a[1]+a[3]*k)**2*nedt_Tb11**2 + (a[2]-a[3]*k)**2*nedt_Tb11o**2
        sigma_nedt = np.sqrt(var_nedt)        

        # Uncertainty from the algorithm:
        std_of_residuals = 0.015
        sigma_algo = std_of_residuals

    elif equation_type == 14:
        #Version: 20200604 -0.1319     0.1862     2.5277    -1.5282     4.9642      0.014
        a = [-0.1319,0.1862,2.5277,-1.5282,4.9642]
        k = 1./np.cos(thetan)-1.
        ist = a[0] + a[1]*k + a[2]*Tb11n + a[3]*Tb11o + a[4]*(Tb11n-Tb11o)*k

        # Uncertainty due to noise in brightness temperatures (uncorrelated):
        var_nedt = (a[2]+a[4]*k)**2*nedt_Tb11**2 + (a[3]-a[4]*k)**2*nedt_Tb11o**2
        sigma_nedt = np.sqrt(var_nedt)        

        # Uncertainty from the algorithm:
        std_of_residuals = 0.014
        sigma_algo = std_of_residuals
        
    elif equation_type == 15: 
        ## DER ER ROD I KOEFFICIENTERNE - SIMULEREDE TB'ER BRUGER EQ3 - TROR KOEFFICIENTERNE PASSER DERTIL..
        # Version OSISAF 205b - with 3 sets of coefficients
        # Different coefficients depending on temperature interval of Tb11n.

        ## METOP B OPR - OSI 205a/b - KOEFFICIENTER ANDERLEDES END EQ3!!
        a_cold = [-3.29453, 1.01404, 0.74924, 0.01508]
        #Version: 20200604 -0.5197     2.3888    -1.3865    -0.1443      0.059
        a_med = [-4.01702, 1.01615, 1.41726, -0.03038]
        #Version: 20200604 -0.3008     2.4573    -1.4560     0.0317      0.080
        a_warm = [-4.61195, 1.01815, 1.37783, 0.30656]
        
        ## OSI-205a atbd
        #ST = a0 + a1*T11 + a2*(T11 − T12) + a3*(T11 − T12)*steta
        k = 1./np.cos(thetan)-1
        ist = np.where(Tb11n<240., 
                       a_cold[0] + a_cold[1]*Tb11n + a_cold[2]*(Tb11n-Tb12n) + a_cold[3]*(Tb11n-Tb12n)*k,
                       a_med[0] +  a_med[1]*Tb11n +  a_med[2]*(Tb11n-Tb12n) +  a_med[3]*(Tb11n-Tb12n)*k)
        ist = np.where(Tb11n>=260., 
                       a_warm[0] + a_warm[1]*Tb11n + a_warm[2]*(Tb11n-Tb12n) + a_warm[3]*(Tb11n-Tb12n)*k,ist)
        sigma_nedt = np.sqrt(var_nedt_dummy)
        sigma_algo = sigma_dummy

    elif equation_type == 16:
        ## NEW METOP B OSI 205c/d/g - with 1 set of coefficients - OBS FORMULATION IS DIFFERENT THAT EQ2
        
        a = [1.0113, 1.8714, -0.4176, -2.8579]
        k = 1./np.cos(thetan)-1
        ist = a[0]*Tb11n + (a[1] + a[2]*k)*(Tb11n-Tb12n) + a[3]
        sigma_nedt = np.sqrt(var_nedt_dummy)
        sigma_algo = sigma_dummy

    elif equation_type == 17:
        ## NEW METOP B OSI 205c/d/g - with 1 set of coefficients - OBS FORMULATION IS DIFFERENT THAT EQ2
        ## WITH BIAS CORRECTION POSSIBILITIES
        
        a = [1.0113, 1.8714, -0.4176, -2.8579]
        k = 1./np.cos(thetan)-1
#!! ORIGINAL        ist = a[0]*Tb11n + (a[1] + a[2]*k)*(Tb11n-Tb12n) + a[3]
        ist = a[0]*Tb11n + (a[1] + a[2]*k)*(Tb11nsim-Tb12nsim) + a[3]

        sigma_nedt = np.sqrt(var_nedt_dummy)
        sigma_algo = sigma_dummy

#    print("IST: ", ist)
#    print("ALGORITHM: ", equation_type)
#    print("ISTshape: ", ist.shape)
#    print("sigma_nedt: ", sigma_nedt.shape)
#    print("sigma_algo: ", sigma_algo)
#    return ist, [sigma_nedt, sigma_algo]
    return ist, sigma_algo, sigma_nedt
