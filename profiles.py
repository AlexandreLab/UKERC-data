import numpy as np
import pandas as pd
import os
import sys
from calendar import monthrange
import functions
import calendar

import pickle

# work with Xgboost models
global_path_models = r"D:\OneDrive - Cardiff University\05 - Python\Machine learning models"

def export_model(model, name):
    pickle.dump(model, open(global_path_models + os.path.sep+ name +"_pickle.dat", "wb"))
    
def import_model(name):
    loaded_model = pickle.load(open(global_path_models + os.path.sep+ name +"_pickle.dat", "rb"))
    return loaded_model
   
def get_additional_models(model_name):
    quantiles_list = []
    arr = os.listdir(global_path_models)
    arr = [x for x in arr if model_name in x]
    quantile_models = [x for x in arr if "Quantile" in x]
    quantile_models
    for n in quantile_models:
        quantiles_list.append(float(n.split('_pickle.dat')[0].split('_')[-1]))
    return quantiles_list
    
def get_predictions(df, model_name, new_X=[]):  
    print(model_name)
    if new_X==[]:
        X=["HH", "Hour", "Temperature", "DayOfWeek", 'Cos_Hour', 'Sin_Hour', 'Cos_DayOfWeek', 'Sin_DayOfWeek',
           'Cos_Month', 'Sin_Month', 'Cos_Season num', 'Sin_Season num']
    else:
        X=new_X
    model = import_model(model_name)
    
    predictions_df = pd.DataFrame(index=df.index)
    
    predictions = model.predict(df[X].values)
    
    predictions_df["Original prediction"] = predictions
    predictions_df["Final prediction"] = predictions
    
    quantiles_list = get_additional_models(model_name)
    upper_quantiles = [x for x in quantiles_list if x>0.5]
    lower_quantiles = [x for x in quantiles_list if x<0.5]
    
    print("Quantiles_list:", quantiles_list)
    
    ## creating peak models
    for ii in range(0,len(quantiles_list)):
        temp_name = "Quantile_"+str(quantiles_list[ii])
        
        peak_model = import_model(model_name+"_"+temp_name)

        peak_predictions = peak_model.predict(df[X].values)
        
        if quantiles_list[ii]>0.5:
            predictions_df["Peak prediction "+str(ii)] = peak_predictions
            predictions_df.loc[predictions_df["Peak prediction "+str(ii)]>1, "Peak prediction "+str(ii)]=1
            predictions_df.loc[predictions_df["Original prediction"]<predictions_df["Peak prediction "+str(ii)], "Final prediction"] = predictions_df["Peak prediction "+str(ii)]
        else:
            predictions_df["Min prediction "+str(ii)] = peak_predictions
            predictions_df.loc[predictions_df["Min prediction "+str(ii)]<0, "Min prediction "+str(ii)]=0
            predictions_df.loc[(predictions_df["Original prediction"]>predictions_df["Min prediction "+str(ii)]) & (predictions_df["Min prediction "+str(ii)]>0)
                               , "Final prediction"] = predictions_df["Min prediction "+str(ii)]
    
    # If some predictions are negatives they are set to zero.
    predictions_df.loc[predictions_df["Final prediction"]<0, "Final prediction"] = 0
    return predictions_df["Final prediction"].values    


def get_skeleton_df(year):
    start=pd.to_datetime(str(year)+'0101-0000', format='%Y%m%d-%H%M%', errors='ignore')
    end=pd.to_datetime(str(year)+'1231-2330', format='%Y%m%d-%H%M%', errors='ignore')

    df = pd.DataFrame()
    df["Date"] = pd.date_range(start=start, end=end, freq='30min')
    df.set_index("Date", inplace=True)
    df.index.name = "index"
    
    if calendar.isleap(year):
        df = df[~((df.index.month == 2) & (df.index.day == 29))] # remove the 29th of feb of leap years
        
    functions.addTimePeriod(df)
    
    extra_features(df, ["HH","Hour", "DayOfWeek", "Month", "Season num"])
    
    if calendar.isleap(year):
        df.loc[df["Month"]>2, "Day"] = df.loc[df["Month"]>2, "Day"]-1 # remove one day to the count
        
    df.reset_index(inplace=True)
    
    df['Date'] = pd.to_datetime(df['Date']) 
    return df

def extra_features(df, list_features = ["Hour", "DayOfWeek"]):
    for f in list_features:
        df["Cos_"+f] = np.cos(2*np.pi*df[f]/df[f].max())
        df["Sin_"+f] = np.sin(2*np.pi*df[f]/df[f].max())
    
    
def get_scaled_profiles(list_areas, technology_profile, df, target, area, correction_factor=1):
    df_profiles = pd.DataFrame(columns=list_areas)
    df_profiles.iloc[:,0] = technology_profile
    df_profiles = df_profiles.ffill(axis=1)
    
    for c in list_areas:
        temp_heat_demand = df.loc[df[area]==c, target].values[0]*correction_factor
        df_profiles[c] = df_profiles[c]/df_profiles[c].sum()*temp_heat_demand
        
    return df_profiles


def get_nb_degree_days(temperatures_month, year):

    # from sap 2012
    #temperatures_month_UK = [4.3, 4.9, 6.5, 8.9, 11.7, 14.6, 16.6, 16.4, 14.1, 10.6, 7.1, 4.2]
    #temperatures_month_Wales = [5, 5.3, 6.5, 8.5, 11.2, 13.7, 15.3, 15.3, 13.5, 10.7, 7.8, 5.2]
    base_temperature = 15.5
    
    nb_days_month = [monthrange(year, x)[1] for x in range(1,13)]
    nb_degree_days = 0
    for ii, temp in enumerate(temperatures_month):
        if temp<base_temperature:
            nb_degree_days = nb_degree_days + (base_temperature-temp)*nb_days_month[ii]
    return nb_degree_days



def add_technology_profiles(df): # only individual heating technologies for domestic customers
    design_temperature = -3.2
    
    # Add domestic profiles
    model_names = ["boiler", "ASHP_heat", "resistance_heater", "GSHP_heat", "LargeHP_heat"]
    for ii, name_model in enumerate(model_names):
        df[name_model] = get_predictions(df, name_model)

    model_names = ["ASHP_elec", "GSHP_elec", "LargeHP_elec"]
    for ii, name_model in enumerate(model_names):
        X_new = ["HH", "Hour", "Temperature", "DayOfWeek", 'Cos_Hour', 'Sin_Hour', 'Cos_DayOfWeek', 'Sin_DayOfWeek',
               'Cos_Month', 'Sin_Month', 'Cos_Season num', 'Sin_Season num', name_model.split('_')[0]+"_heat"]
        df[name_model] = get_predictions(df, name_model, X_new)
        
    df.loc[df["ASHP_heat"]<df["ASHP_elec"], "ASHP_elec"]=df["ASHP_heat"]/2.6 # Ensure that the system never has a COP<1, default COP is 2.6 for ASHPs
    df.loc[df["GSHP_heat"]<df["GSHP_elec"], "GSHP_elec"]=df["GSHP_heat"]/2.75 # Ensure that the system never has a COP<1, default COP is 2.75 for GSHPs
    
    #df.loc[df["Temperature"]>15, "ASHP_elec"]=df["ASHP_heat"]/2.6
    #df.loc[df["Temperature"]>15, "GSHP_elec"]=df["GSHP_heat"]/2.6
    daily_df = df.set_index("index").resample('1d').mean().copy()
    daily_df.dropna(inplace=True)
    model = import_model("required_heat_demand_ASHP")
    daily_df["Required_heat_ASHP"] = model.predict(daily_df["Temperature"].values.reshape(-1, 1))
    daily_df.loc[daily_df["Temperature"]>=design_temperature, "Required_heat_ASHP"]=daily_df["ASHP_heat"]
    daily_df["Heat from ASHP backup [%]"] = (daily_df["Required_heat_ASHP"]-daily_df["ASHP_heat"])/daily_df["ASHP_heat"]
    daily_df.loc[daily_df["Heat from ASHP backup [%]"]<0, "Heat from ASHP backup [%]"]=0
    
    model = import_model("required_heat_demand_GSHP")
    daily_df["Required_heat_GSHP"] = model.predict(daily_df["Temperature"].values.reshape(-1, 1))
    daily_df.loc[daily_df["Temperature"]>=design_temperature, "Required_heat_GSHP"]=daily_df["GSHP_heat"]
    daily_df["Heat from GSHP backup [%]"] = (daily_df["Required_heat_GSHP"]-daily_df["GSHP_heat"])/daily_df["GSHP_heat"]
    daily_df.loc[daily_df["Heat from GSHP backup [%]"]<0, "Heat from GSHP backup [%]"]=0
    daily_df.index = pd.to_datetime(daily_df.index)

    df = pd.merge(df, daily_df[["Heat from ASHP backup [%]","Heat from GSHP backup [%]" ]], left_on="Date", right_index=True, how='left')
    
    
    return df