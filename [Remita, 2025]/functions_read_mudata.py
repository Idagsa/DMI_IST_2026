#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 28 17:59:27 2025

@author: zoerem

"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy.stats import linregress
import matplotlib.dates as mdates
from metpy.units import units
import metpy.calc as mpcalc
from metpy.calc import dewpoint_from_relative_humidity


def extract_station(station_id, ds):
    
    """
    Create a new dataframe with the data for only one station

    station_id = "EGP","KAN_U","KPC_U""NUK_U","QAS_U","SCO_U","UPE_U"
    ds = the big mashup dataframe filtered
    """
    
    mu0 = ds[ds["station_id"]==station_id]
    mu = mu0.sort_values('lev1_obstime', ascending=True)
    return mu

# Convert pressure to approximate altitude (Standard atmosphere formula, rough estimate)
def pressure_to_altitude(p_hpa):
    return 44330 * (1 - (p_hpa / 1013.25)**0.1903) / 1000# in km

def prom_dewp(station_id, ds, Kelvin = True):
    '''
    Computes the dew temperature(C) from PROMICE's Air temperature and Relative humidity,
    at a given station, for the whole time periode, using metpy library.

    station_id = "EGP","KAN_U","KPC_U", "UPE_U"
    ds: the big mashup dataframe filtered
    '''
    
    mu = extract_station(station_id, ds)

    airt = mu["AirTemperature(C)"]
    rh = mu["RelativeHumidity(%)"]
    
    airt_q = np.array(airt) * units.degC
    rh_q = np.array(rh) * units.percent
    
    if Kelvin == True:
        dewp = dewpoint_from_relative_humidity(airt_q, rh_q).to(units.kelvin)
    else:
        dewp = dewpoint_from_relative_humidity(airt_q, rh_q)
    return dewp




def all_station_atmo_profile(start_date, end_date, stations, ds, potential_temp=True): # , dew_temp=True
    """
    Plot the mean atmospheric profile of temperature/potential temperature and specific humidity
    between a start and an end date. For multiple stations
    
    if potential_temp = True, Plot the potential temperature
    
    Parameters:
    - start_date: string in format 'YYYY-MM-DD'
    - end_date: string in format 'YYYY-MM-DD'
    - stations: list of stations ["EGP", "KAN_U", "KPC_U", "UPE_U"]
    - ds: the big mashup dataframe filtered
    """

    # Pressure levels
    p_levels = [1000, 950, 925, 900, 850, 800, 700, 600, 500, 400, 
                300, 250, 200, 150, 100, 70, 50, 30, 20, 10, 7, 5, 3, 2, 1]
    
    # temp & specific humidity
    tcolumns = [f'nwp_{"t"}_{p}_hPa' for p in p_levels]
    qcolumns = [f'nwp_{"q"}_{p}_hPa' for p in p_levels]

    alt = pressure_to_altitude(np.array(p_levels))
    
    # Accumulate all profiles across stations
    all_temp_profiles = []
    all_q_profiles = []
    all_theta_profiles = []
    # all_dew_profies = []
    total_profiles = 0
    
    pressures = np.array(p_levels) * units.hPa

    
    for station_id in stations:
        mu = extract_station(station_id, ds)
        # Ensure datetime
        mu['lev1_obstime'] = pd.to_datetime(mu['lev1_obstime'])
        start_dt = pd.to_datetime(start_date)
        end_dt = pd.to_datetime(end_date)

       # Filter data within range
        mask = (mu['lev1_obstime'] >= start_dt) & (mu['lev1_obstime'] <= end_dt)
        subset = mu[mask]

        if subset.empty:
            continue
        print(f"{len(subset)} profiles at {station_id}")
        
        total_profiles += len(subset)
        
        # Computation of the potential temperature (theta)
        for _, row in subset.iterrows():
            temps = row[tcolumns].values * units.kelvin
            theta = mpcalc.potential_temperature(pressures, temps)
            
            # qvals = row[qcolumns].values * units('kg/kg')
            # dew = mpcalc.dewpoint_from_specific_humidity(pressures,  qvals)
            
            qvals = row[qcolumns].values
            
            all_temp_profiles.append(temps.magnitude)
            all_q_profiles.append(qvals)
            all_theta_profiles.append(theta.magnitude)
            # all_dew_profies.append(dew.magnitude)
            
    if total_profiles == 0:
        print("No profiles available in the selected date range for the selected stations.")
        return
    
    print(f"{total_profiles} total profiles available between {start_date} and {end_date} for all stations.")


    # convert to array:
    all_temp_profiles = np.array(all_temp_profiles, dtype=float)
    all_q_profiles = np.array(all_q_profiles, dtype=float)
    all_theta_profiles = np.array(all_theta_profiles, dtype=float)
    # all_dew_profiles = np.array(all_dew_profiles, dtype=float)
    
    # Compute mean and std
    temp_mean = all_temp_profiles.mean(axis=0)
    temp_std = all_temp_profiles.std(axis=0)
    
    q_mean = all_q_profiles.mean(axis=0)
    q_std = all_q_profiles.std(axis=0)
    
    theta_mean = all_theta_profiles.mean(axis=0)
    theta_std = all_theta_profiles.std(axis=0)
    
    # dew_mean = all_dew_profiles.mean(axis=0)
    # dew_std = all_dew_profiles.std(axis=0)

    # Plotting
    fig, ax1 = plt.subplots(figsize=(6, 8))

    if potential_temp == True:
        ax1.plot(theta_mean, alt, color='tab:red', label='Mean Profile of Potential Temperature')
        ax1.fill_betweenx(alt, theta_mean - theta_std, theta_mean + theta_std, color='tab:red', alpha=0.3)
        ax1.set_xlabel("Potential Temperature (K)")
    else :
        ax1.plot(temp_mean, alt, color='tab:red', label='Mean Profile of Temperature')
        ax1.fill_betweenx(alt, temp_mean - temp_std, temp_mean + temp_std, color='tab:red', alpha=0.3) #, label='Std Dev'
        ax1.set_xlabel("Temperature (K)")

    ax1.tick_params(axis='x', colors='tab:red')
    ax1.spines['bottom'].set_color('tab:red')
    ax1.xaxis.label.set_color('tab:red')
    
    # Specific humidity on secondary axis
    ax2 = ax1.twiny()
    
    # if dew_temp == True:
    #     ax2.plot(dew_mean, alt, color='tab:blue', label='Mean Profile of Dew Temperature')
    #     ax2.fill_betweenx(alt, dew_mean - dew_std, dew_mean + dew_std, color='tab:blue', alpha=0.3) #, label='Std Dev'
    #     ax2.set_xlabel("Dew temperature (C)")
    # else:
    #     ax2.plot(q_mean, alt, color='tab:blue', label='Mean Profile of Specific Humidity')
    #     ax2.fill_betweenx(alt, q_mean - q_std, q_mean + q_std, color='tab:blue', alpha=0.3) #, label='Std Dev'
    #     ax2.set_xlabel("Specific Humidity (kg water/kg total mass)")
    
    ax2.plot(q_mean, alt, color='tab:blue', label='Mean Profile of Specific Humidity')
    ax2.fill_betweenx(alt, q_mean - q_std, q_mean + q_std, color='tab:blue', alpha=0.3) #, label='Std Dev'
    ax2.set_xlabel("Specific Humidity (kg water/kg total mass)")
    ax2.xaxis.set_label_position('top')
    ax2.tick_params(axis='x', labeltop=True, labelbottom=False, colors='tab:blue')
    ax2.spines['top'].set_color('tab:blue')
    ax2.xaxis.label.set_color('tab:blue')

    # Common
    ax1.set_title(f"Mean NWP Atmospheric Profiles ({start_date} to {end_date})\nStations: {stations}")
    ax1.set_ylabel("Altitude (km)")
    ax1.grid(True, linestyle='--', alpha=0.6)
    
    # Fusion des légendes
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right')

    plt.show()


def clc_r2(x, y):
    """
    Compute the correlation's coefficient between x and y
    x and y must be of equal length
    """
    coefficients = np.polyfit(x, y, 1) # Ajuste une ligne droite aux données x et y
    polynomial = np.poly1d(coefficients) # Crée une fonction polynomiale avec les coefficients trouvés.
    y_pred = polynomial(x) # Les valeurs prédites par le modèle.
    ss_res = np.sum((y - y_pred) ** 2) # La somme des carrés des résidus (différence entre les valeurs observées et prédites).
    ss_tot = np.sum((y - np.mean(y)) ** 2) # La somme des carrés totaux (différence entre les valeurs observées et la moyenne des valeurs observées).
    r2 = 1 - (ss_res / ss_tot)
    return r2*100

def stat_temp(promice,temp, ds):
    """
    Compute the correlation stats 
    between the truth PROMICE data and the computed temp data
    promice and temp must be of equal length
    """
    temp_err = ds[temp]-promice
    temp_stat = f'{temp}-truth: mean={temp_err.mean():.2f}, std={temp_err.std():.2f},\nR²={clc_r2(ds[temp], promice):.0f}%, counts={temp_err.count()}'
    return temp_stat

def plot_temp_corr(ax, temp, promtemp, ds, colorbar):
    """
    Plot the correlation between the truth PROMICE data and the computed temp data
    showing the stations with different colors.

    'temp_stat' is taken from: stat_temp(promice,temp)
    'temp' is: "lev2_surface_temperature", "ist_1tdomain", "ist_3tdomain", "istsim_1tdomain", "istsim_3tdomain", "nwp_skt"
    or: "nwp_t_1000_hPa", "nwp_t2m"
    promtemp is "SurfaceTemperature(C)" or "AirTemperature(C)"

    atmo_var: "t", "q", "o3"
    xlabel: e.g., "Temperature (K)", "Specific humidity (kg/kg)", "Ozone (?)"
    target_time: the datetime you want to plot (string or pd.Timestamp)
    ds: the big mashup dataframe filtered

    plotting on a given axis 'ax' (for subplots)
    """
    # true data PROMICE
    promice = ds[promtemp]+273.15
    
    # stat title
    temp_stat = stat_temp(promice,temp, ds)
    

    # Sation "colorbar"
    stations = ds["station_id"].unique()
    cmap = plt.get_cmap('tab10')
    colors = [cmap(i % 10) for i in range(len(stations))] 
    station_to_color = {station: color for station, color in zip(stations, colors)}
    point_colors = ds["station_id"].map(station_to_color)
    
    # PLOT
    
    # # Fit linear regression
    # slope, intercept, r_value, p_value, std_err = linregress(promice, ds[temp])
    # x_vals = np.array(ax.get_xlim())  # take x axis limits
    # y_vals = intercept + slope * x_vals
    # fit1, = ax.plot(x_vals, y_vals, color="k", label="Linear fit")
    # fit2, = ax.plot([min(promice), max(promice)], [min(promice), max(promice)], 'k--', label='Ideal')


    # scatter Plot
    
    if colorbar == 'Stations':
        sc = ax.scatter(promice, ds[temp], c=point_colors)
        slope, intercept, r_value, p_value, std_err = linregress(promice, ds[temp])
        x_vals = np.array(ax.get_xlim())  # take x axis limits
        y_vals = intercept + slope * x_vals
        fit1, = ax.plot(x_vals, y_vals, color="k", label="Linear fit")
        fit2, = ax.plot([min(promice), max(promice)], [min(promice), max(promice)], 'k--', label='Ideal')
        # Sations ID legend
        handles_stations = [mpatches.Patch(color=station_to_color[station], label=station) for station in stations]
        handles_all = handles_stations + [fit1, fit2]
        ax.legend(handles=handles_all, title="Station ID and Fits", loc='upper left')
        
    elif colorbar == 'CloudCover':
        sc = ax.scatter(promice, ds[temp], c=ds['nwp_tcc']) # CloudCover
        cbar = plt.colorbar(sc, ax=ax)
        cbar.set_label("CloudCover", rotation=90)
        slope, intercept, r_value, p_value, std_err = linregress(promice, ds[temp])
        x_vals = np.array(ax.get_xlim())  # take x axis limits
        y_vals = intercept + slope * x_vals
        fit1, = ax.plot(x_vals, y_vals, color="k", label="Linear fit")
        fit2, = ax.plot([min(promice), max(promice)], [min(promice), max(promice)], 'k--', label='Ideal')
        handles_all = [fit1, fit2]
        ax.legend(handles=handles_all, title="Fits", loc='upper left')
    elif colorbar == 'RelativeHumidity':
        sc = ax.scatter(promice, ds[temp], c=ds['RelativeHumidity(%)'])
        cbar = plt.colorbar(sc, ax=ax)
        cbar.set_label("RelativeHumidity(%)", rotation=90)
        slope, intercept, r_value, p_value, std_err = linregress(promice, ds[temp])
        x_vals = np.array(ax.get_xlim())  # take x axis limits
        y_vals = intercept + slope * x_vals
        fit1, = ax.plot(x_vals, y_vals, color="k", label="Linear fit")
        fit2, = ax.plot([min(promice), max(promice)], [min(promice), max(promice)], 'k--', label='Ideal')
        handles_all = [fit1, fit2]
        ax.legend(handles=handles_all, title="Fits", loc='upper left')
    else:
        raise ValueError("mode must be 'Stations' or 'CloudCover' or 'RelativeHumidity'")


    
    
    ax.set_ylabel(f"{temp} (K)", fontsize = 14)
    ax.set_xlabel("PROMICE reference (K)", fontsize = 14)
    ax.set_title(temp_stat, fontsize = 14)
    ax.grid(True, linestyle='--', alpha=0.6)

def sation_stat_temp(temp, promtemp, ds):
    """
    Compute the correlation stats 
    between the truth PROMICE data and the computed temp data for each stations
    for a given temperature.
    
    temp surf: "lev2_surface_temperature", "ist_1tdomain", "ist_3tdomain", "istsim_1tdomain", "istsim_3tdomain", "nwp_skt"
    promtemp surface: "SurfaceTemperature(C)"
    
    temp atmo: "nwp_t_1000_hPa", "nwp_t2m"
    promtemp atmo: "AirTemperature(C)"

    """
    stations = ["EGP", "KAN_U", "KPC_U", "NUK_U", "QAS_U", "SCO_U", "UPE_U"]

    for station_id in stations:
        mu = extract_station(station_id, ds)
        
        # if mu.empty:
        #     print(f"⚠️ No data for station '{station_id}' — skipping.\n")
        #     continue
    
        promice = mu[promtemp] + 273.15  # convert to Kelvin
        temp_stat = stat_temp(promice, temp, mu)
        
        print(f"At '{station_id}':\n  {temp_stat}\n")
    
    
def plot_station_temp_corr(ax, temp, promtemp, mu, station_id, colorbar):
    """
    Plot the correlation between the truth PROMICE data and the computed temp data
    AT a GIVEN STATION showing the stations with different colors.

    'temp' is: "lev2_surface_temperature", "ist_1tdomain", "ist_3tdomain", "istsim_1tdomain", "istsim_3tdomain", "nwp_skt"
    or: "nwp_t_1000_hPa", "nwp_t2m"
    promtemp is "SurfaceTemperature(C)" or "AirTemperature(C)"

    atmo_var: "t", "q", "o3"
    xlabel: e.g., "Temperature (K)", "Specific humidity (kg/kg)", "Ozone (?)"
    target_time: the datetime you want to plot (string or pd.Timestamp)
    mu: dataframe for the station
    station_id: the name of the station "EGP","KAN_U","KPC_U""NUK_U","QAS_U","SCO_U","UPE_U"
        
    plotting on a given axis 'ax' (for subplots)
    """
    # true data PROMICE
    promice = mu[promtemp]+273.15
    
    # stat title
    temp_stat = stat_temp(promice,temp, mu)
    
    
    # Scatter plot

    if colorbar == 'CloudCover':
        sc = ax.scatter(promice, mu[temp], c=mu['CloudCover'])
        cbar = plt.colorbar(sc, ax=ax)
        cbar.set_label("CloudCover", rotation=90)
        
    elif colorbar == 'RelativeHumidity':
        sc = ax.scatter(promice, mu[temp], c=mu['RelativeHumidity(%)'])
        cbar = plt.colorbar(sc, ax=ax)
        cbar.set_label("RelativeHumidity(%)", rotation=90)
    else:
        raise ValueError("mode must be 'CloudCover' or 'RelativeHumidity'")
    # ax.scatter(promice, mu[temp])
    
    # Fit linear regression
    slope, intercept, r_value, p_value, std_err = linregress(promice, mu[temp])
    x_vals = np.array(ax.get_xlim())  # take x axis limits
    y_vals = intercept + slope * x_vals
    ax.plot(x_vals, y_vals, color="red")
    
    ax.set_ylabel(f"{temp} (K)")
    ax.set_xlabel("PROMICE reference (K)")
    ax.set_title(f"{temp_stat} \n {station_id}")
    ax.grid(True)
    
    
    
def plot_ts_temp(stations, temp, ds, start_date, end_date, save = True):
    """
    Plot the time serei of temperature for each stations
    
    stations is a list of the stations we want to look at: ["EGP", "KAN_U", "KPC_U", "NUK_U", "QAS_U", "SCO_U", "UPE_U"]
    
    'temp' ican be "lev2_surface_temperature", "ist_1tdomain", "ist_3tdomain", "istsim_1tdomain", "istsim_3tdomain", "nwp_skt"
    or: "nwp_t_1000_hPa", "nwp_t2m"
    or even "SurfaceTemperature(C)", "AirTemperature(C)"
    ds: the big mashup dataframe filtered
    """

    plt.figure(figsize=(12,6))
    
    for station_id in stations:
        mu = extract_station(station_id, ds)
        
        # Ensure datetime and set as index
        mu["lev1_obstime"] = pd.to_datetime(mu["lev1_obstime"])
        mu = mu.set_index("lev1_obstime")
        
        # # Plot
        # plt.plot(mu.index, mu[temp], label=station_id)
        
        # Filter date
        mu_dec = mu.loc[start_date:end_date]
        
        if not mu_dec.empty:
            plt.plot(mu_dec.index, mu_dec[temp], label=station_id)
       
       
    # Format x-axis
    
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%b %Y')) # ex : Dec 2021
    plt.gca().xaxis.set_major_locator(mdates.MonthLocator(interval=2))  # show every 2 months
    
    # plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%d %b'))  # ex : 01 Dec
    # plt.gca().xaxis.set_major_locator(mdates.DayLocator(interval=2))  # every day
    
    
    plt.legend(loc="lower left")
    plt.xlabel("Date")
    plt.ylabel("Temperature (K)")
    plt.title(f"{temp} ({start_date} to {end_date})\nStations: {stations}", fontsize=20)
    plt.grid(True)
    plt.xticks(rotation=45)
    plt.tight_layout()
    if save == True:
        plt.savefig("/home/zoerem/Documents/STAGE/figures/time_serie.png")
    else:
        pass
    
    plt.show()