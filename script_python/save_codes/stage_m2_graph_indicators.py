# -*- coding: utf-8 -*-
"""
Created on Mon Mar 10 12:46:05 2025

@author: vanleene valentine
"""

#%% Explaining what the script does
"""
This script reads an excel file containing all results (mae, rmse and pbiais) calculated in 
"stage_m2_tide_v2".

"""

#%% Needed librairies imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#%% Extracting data from the file
def load_excel_sheets(filepath, sheets):
    """
    Load specified sheets from an Excel file into a dictionary.

    Parameters
    ----------
    filepath : str
        The path to the Excel file to load the sheets from.
    sheets : list
        A list of sheet names (str) to be loaded from the Excel file.

    Returns
    -------
    dict
        A dictionary where the keys are the sheet names and the values are
        the corresponding data loaded into pandas DataFrames. Any missing or
        invalid values in the sheets are replaced by NaN, with 999 being treated as missing.
    """
    return {sheet: pd.read_excel(filepath, sheet, na_values=999) for sheet in sheets}

# Using the above function
fullFilepath = '../excel/indicators_ks_for_python.xlsx'
sheets = ['compa_all', 'compa_best', 'combi_all', 'combi_best']
dfs = load_excel_sheets(fullFilepath, sheets)

# Separating the data from each sheet in dataframes
df_all_compa = dfs['compa_all']
df_best_compa = dfs['compa_best']
df_all_combi = dfs['combi_all']
df_best_combi = dfs['combi_best']

#%% Graphical representation
def graph_indicators(df, type_best):
    """
    Plots graphs of the Ks indicators based on the data type.
    
    Parameters
    ----------
    df : DataFrame
        DataFrame containing the data to be plotted..
    type_best : str
        Type of indicator to display: 'compa' for comparison, 'combi' for combination.
    -------
    """
    for i in range(1,4):
        column_names = df.columns
        if i == 3:
            unit = '%'
        else:
            unit = 'm'
        
        # Graphical representation
        plt.figure(figsize=(10, 6), dpi=300)
        # Changing ticks size
        plt.rcParams['xtick.labelsize'] = 14
        plt.rcParams['ytick.labelsize'] = 14
        
        if type_best == 'combi':
            # Rotation of x-ticks
            plt.xticks(rotation=25)  
        # Plotting the predicted tide
        plt.scatter(df.iloc[:,0], df.iloc[:,i], color='forestgreen', s=60)
    
        # Adding legend
        plt.title(f'Ks indicator ({column_names[i]})', fontsize=20, fontweight='bold')
        if type_best == 'compa':
            plt.xlabel('Ks (m**(1/3)/s)', fontsize=16)
        elif type_best == 'combi':
            plt.xlabel('Combination of Ks (m**(1/3)/s)', fontsize=16)
        plt.ylabel(f'{column_names[i]} ({unit})', fontsize=16)
        
        # Adding horizontal lines
        mean_indicator = np.mean(df.iloc[:,i])
        plt.axhline(mean_indicator, color='seagreen', linestyle='--', 
                    label=f'Mean {column_names[i]}: {mean_indicator:.2f} {unit}')
        if i == 3:
            plt.axhline(0, color='lightcoral', linestyle='--')
        
        # Graphical option and plotting
        plt.legend(fontsize=16, loc='lower right')
        plt.grid(color='gray', linestyle='--', linewidth=0.5, alpha=0.7) 
        plt.tight_layout()
        plt.show()

graph_indicators(df_best_compa, 'compa')
# graph_indicators(df_best_combi, 'combi')

