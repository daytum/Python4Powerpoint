#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import scipy.optimize
from intake import cat
import yaml



def xnpv(rate, df, inputs):
    cashflows = np.insert(df['cash flow'].to_numpy(), 0, -inputs['capex'])
    years = np.insert(df.index.to_numpy() + 1, 0, 0) / 12
    
    return (np.sum(cashflows / (1 + rate) ** ( years)))

def xirr(df, inputs):
    return scipy.optimize.fsolve(lambda r: xnpv(r, df, inputs), 0.5)

def compute_oil_revenue(df, inputs):
    og_inputs = inputs['oil']
    return (og_inputs['price'] + og_inputs['differential']) * inputs['net revenue interest'] * df['oil']

def compute_gas_revenue(df, inputs):
    og_inputs = inputs['gas']
    return (og_inputs['price'] + og_inputs['differential']) * inputs['net revenue interest'] * df['gas'] * (1 - og_inputs['shrink'])

def compute_NGL_revenue(df, inputs):
    return df['gas'] * inputs['NGL']['yield'] / 1000 * inputs['NGL']['price'] * inputs['net revenue interest']

def compute_taxes(df, inputs):
    return -(inputs['taxes']['severence']['gas/NGL'] * (df['gas revenue'] + df['NGL revenue']) + 
             inputs['taxes']['severence']['oil'] * df['oil revenue'] + 
             df[['oil revenue','gas revenue', 'NGL revenue']].sum() * inputs['taxes']['ad valorem']) * inputs['working interest']

def compute_opex(df, inputs):
    return -(df['oil'] * inputs['opex']['oil'] + df['gas'] * inputs['opex']['gas'] + inputs['opex']['fixed']) * inputs['working interest']


def create_full_df(filename, inputs, return_df=False):
    
    df = pd.read_csv(filename)
    df.index.rename('month', inplace=True)

    gas_ultimate_recovery = df['gas'].sum()
    oil_ultimate_recovery = df['oil'].sum()


    df['% EUR gas'] = df['gas'] / gas_ultimate_recovery * 100
    df['% EUR oil'] = df['oil'] / oil_ultimate_recovery * 100


    df['oil revenue'] = df.apply(compute_oil_revenue, args=(inputs,), axis=1)
    df['gas revenue'] = df.apply(compute_gas_revenue, args=(inputs,), axis=1)
    df['NGL revenue'] = df.apply(compute_NGL_revenue, args=(inputs,), axis=1)
    df['taxes'] = df.apply(compute_taxes, args=(inputs,), axis=1)
    df['opex'] = df.apply(compute_opex, args=(inputs,), axis=1)

    df['cash flow'] = df.loc[:, 'oil revenue':'opex'].sum(axis=1)
    df.loc[df['cash flow'] < 0, 'cash flow'] = 0.0

    df['Cum cash flow'] = df['cash flow'].cumsum() - inputs['capex']


    df['Economic'] = df['cash flow'] > 0


    format_dict = {}
    for name in list(df)[4:-1]:
        format_dict[name] = lambda f: '${:>10}'.format(('({:,.0f})' if f < 0 else '{:,.0f}').format(abs(f)))

    styled_df = (df.style.format(format_dict)
             .bar(color=['red','green'], vmin=df['Cum cash flow'].min(), vmax=df['Cum cash flow'].max(), subset=['Cum cash flow'], align='zero')
             .highlight_max(subset=['Economic'], color='lightgreen')
             .highlight_min(subset=['Economic'], color='red'))
    
    if return_df:
        return df, styled_df
    else:
        return styled_df


def create_summary_df(filename, inputs, return_df=False):
    
    df, _ = create_full_df(filename, inputs, return_df=True)

    summary = {'IRR (%)': xirr(df, inputs) * 100, 'NPV10 (Million $)': xnpv(0.1, df, inputs) / 1e6}

    temp_array = np.insert(df['cash flow'].to_numpy(), 0, -inputs['capex'])

    summary['ROI (yrs)'] = -temp_array[temp_array > 0].sum() / temp_array[temp_array <= 0].sum()

    summary['Payout (yrs)'] = df[df['Cum cash flow'] > 0].index[0] / 12

    summary_df = pd.DataFrame(summary)
    
    styled_df = summary_df.style.format("{:,.2f}".format).hide_index()
    
    if return_df:
        return summary_df, styled_df
    else:
        return styled_df


def fit_type_curve(api='05123341850000'):
    
    df = cat.production.production_by_api(api=api).read()
    
    months = np.arange(200, dtype=np.double)
    oil = df['volume_oil_formation_bbls'].interpolate(method='polynomial', order=1).to_numpy()
    gas = df['volume_gas_formation_mcf'].interpolate(method='polynomial', order=1).to_numpy()
    
    model = lambda t, q_i, D, b: q_i / (1 + b * D * t) ** (1 / b)  
        
    oil_parameters, _ = scipy.optimize.curve_fit(model, months[:oil.shape[0]], oil, bounds=[[0, 0, 0], [np.inf, np.inf, 1]])
    
    gas_parameters, _ = scipy.optimize.curve_fit(model,  months[:gas.shape[0]], gas, bounds=[[0, 0, 0], [np.inf, np.inf, 1]])
    
    oil_fill_arr = np.ones_like(months) * np.nan
    gas_fill_arr = np.ones_like(months) * np.nan
    
    oil_fill_arr[:oil.shape[0]] = oil
    gas_fill_arr[:gas.shape[0]] = gas
    
    df = pd.DataFrame({'month': months, 'oil': oil_fill_arr, 'oil type curve': model(months, *oil_parameters), 'gas': gas_fill_arr, 'gas type curve': model(months, *gas_parameters)}).set_index(['month'])
    
    return df


def type_curve_plot(api='33061025220000'):
    
    df = fit_type_curve(api=api);
    
    fig, ax1 = plt.subplots();
    ax2 = ax1.twinx();
    ax1.set_ylabel('Oil (bbls)', color='b');
    ax2.set_ylabel('Gas (Mcf)', color='r');

    df['oil'].plot(ax=ax1, color='b', linestyle='-.', label='Actual Production (Oil)');
    df['oil type curve'].plot(ax=ax1, color='b', label='Model (Oil)');
    df['gas'].plot(ax=ax2, color='r', linestyle='-.', label='Actual Production (Gas)');
    df['gas type curve'].plot(ax=ax2, color='r', label='Model (Gas)');
    
    return fig


def type_curve_summary_from_db(api='33053040060000', return_df=False):

    df = fit_type_curve(api=api)
    df[['oil type curve', 'gas type curve']].rename(columns={'oil type curve': 'oil', 'gas type curve': 'gas'}).to_csv('{}.csv'.format(api), index=False)
    
    with open("inputs.yaml") as f:
        inputs = yaml.load(f, Loader=yaml.FullLoader)
        
    return create_summary_df('{}.csv'.format(api), inputs, return_df)
    


if __name__ == "__main__":
    
    import yaml
    import glob
    from header_html import header_html
    
    with open("../inputs.yaml") as f:
        
        inputs = yaml.load(f, Loader=yaml.FullLoader)
        
    html = header_html()        

    for file in glob.glob('../data/TypeCurve*.csv'):
        
        html += "<br><h4>{}</h4>".format(file.split('/')[-1].split('.')[0])
        html += create_summary_df(file, inputs).render()
        
    with open('../summary.html', mode='w') as f:
        f.write(html)
        
