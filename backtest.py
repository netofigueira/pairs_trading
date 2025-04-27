import pandas as pd
from pair_trading import PairCointegration
import yfinance as yf
import numpy as np
import math
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from utils import tickers_yf
import itertools
#### backtest de um par cointegrado

#### 1 - de maneira sequencial, buscar sinal de entrada

"""
        12M               1M 
    aval. cointegracao |  trade 
    ++++++++++++++++++ |  -----
                       | aval. cointegracao | trade
                         ++++++++++++++++++ | trade
"""




def get_entries(resid, half_life, tresh, pair1, pair2, data):


    aux = resid.reset_index()
    aux_total = data.reset_index()
    up_entries = aux[aux[0] >= tresh].index
    out_idx = np.asarray([i+ half_life for i in up_entries])

    
    up_entries = aux.iloc[up_entries]["Date"].array
    out = aux_total.iloc[out_idx]["Date"].array
    

    prices_pair1_entry = data[pair1].loc[up_entries]
    prices_pair1_exit = data[pair1].loc[out]


    prices_pair2_entry = data[pair2].loc[up_entries]
    prices_pair2_exit = data[pair2].loc[out]
    up_entries_df= pd.DataFrame( {'entry':up_entries, 'out':out, 'entry_price_short':prices_pair1_entry.values,
                                  'exit_price_short':prices_pair1_exit.values,
                                  'entry_price_long':prices_pair2_entry.values,
                                    'exit_price_long':prices_pair2_exit.values})

    up_entries_df = pd.concat([up_entries_df.head(1),up_entries_df[up_entries_df.entry > up_entries_df.out.shift()]])
    up_entries_df["type"] = "up_signal"
    # down entries 
    down_entries = aux[aux[0] <= -tresh].index

    out_idx = np.asarray([i+ half_life for i in down_entries])

    down_entries = aux.iloc[down_entries]["Date"].array
    out = aux_total.iloc[out_idx]["Date"].array


    # down point, pair1 is long, pair2 is short
    prices_pair1_entry = data[pair1].loc[down_entries]
    prices_pair1_exit = data[pair1].loc[out]
    prices_pair2_entry = data[pair2].loc[down_entries]
    prices_pair2_exit = data[pair2].loc[out]
    down_entries_df = pd.DataFrame( {'entry':down_entries, 'out':out,
                                     'entry_price_long':prices_pair1_entry.values,
                                     "exit_price_long":prices_pair1_exit.values,
                                     'entry_price_short':prices_pair2_entry.values,
                                     "exit_price_short":prices_pair2_exit.values})


    down_entries_df = pd.concat([down_entries_df.head(1),down_entries_df[down_entries_df.entry > down_entries_df.out.shift()]])
    down_entries_df["type"] = "down_signal"
    print(down_entries_df)

    res = pd.concat([up_entries_df, down_entries_df])
    res["pair"] = pair1 + "_" + pair2
    return res


def stationarity_bool(a, cutoff = 0.05):
  a = np.ravel(a)
  if adfuller(a)[1] < cutoff:
    return True
  else:
    return False

  ## OLS implementation
def ols(y, x):

  """
  ordinary least squares regression function
  """
  n = len(x)
  beta = (n*np.sum(x * y)-np.sum(y)*np.sum(x)  )/( n*np.sum(x**2) - np.sum(x)**2 )

  alpha = np.mean(y) - beta* np.mean(x)

  resid = y - (beta*x + alpha)
  return beta, alpha, resid

def calc_half_life(resid):
  lag_resid = resid.shift(1).bfill()
  delta_resid = resid  - lag_resid

  beta, alpha, resid = ols(delta_resid, lag_resid)
  half = -1* np.log(2)/beta
  return half



if __name__ == "__main__":
    # Example usage:
    tickers_yf = ["ALPA4.SA", "BBSE3.SA"]

    all_pairs =  list(itertools.combinations(tickers_yf,2))


    start = "2015-03-01"
    end = "2024-07-09"
    base = yf.download(tickers_yf, start=start, end=end)
    base = base["Adj Close"].dropna()

    res_total = pd.DataFrame(
           columns = ["pair", "entry", "out", "type", "entry_price_long", "exit_price_long", "entry_price_short", "exit_price_short"]
        )
    start_treino_date = base.index.min()
    end_treino_date = start_treino_date + pd.DateOffset(months=12)
    start_test_date = end_treino_date
    end_test_date = start_test_date + pd.DateOffset(months=3)

    for pair1, pair2 in all_pairs:
        start_treino_date = base.index.min()
        end_treino_date = start_treino_date + pd.DateOffset(months=12)
        start_test_date = end_treino_date
        end_test_date = start_test_date + pd.DateOffset(months=3)

        while end_test_date <= (base.index[-1]+ pd.DateOffset(months=1)):
            print("periodo treino", start_treino_date, end_treino_date)
            print("periodo teste", start_test_date, end_test_date)
            print(pair1, pair2)
            eval_base, trade_base = base.loc[start_treino_date:end_treino_date], base.loc[start_test_date:end_test_date]
            beta, alpha, eval_resid = ols(eval_base[pair1], eval_base[pair2])


            try:
                is_coint = stationarity_bool(eval_resid)
            except:
                print(f"problema no {pair1}, {pair2}")

                start_treino_date = start_test_date + pd.DateOffset(months=3)
                end_treino_date = start_treino_date + pd.DateOffset(months=12)
                start_test_date = end_treino_date
                end_test_date = start_test_date + pd.DateOffset(months=3)
                continue
            half_life = math.ceil(calc_half_life(eval_resid))

            if is_coint:

                print("cointegrado")
                _, _, trade_resid = ols(trade_base[pair1], trade_base[pair2])

                sigma = eval_resid.std()
                mean = eval_resid.mean()


            # avaliacao do spread acima da média
                norm_trade_resid = (trade_resid - mean)/sigma

                res = get_entries(norm_trade_resid, half_life=half_life, tresh=2, pair1=pair1, pair2=pair2, data=base.loc[start_test_date:])

                plt.plot(eval_resid.index, (eval_resid.values - mean)/sigma)
                plt.plot(norm_trade_resid.index, norm_trade_resid.values, color="green")
                up_entries, down_entries = res[res["type"] == "up_signal"], res[res["type"] == "down_signal"]
                if len(up_entries) > 0:
                    plt.scatter(up_entries.entry, norm_trade_resid.loc[up_entries.entry].values, color="black", marker='>')
                    if max(up_entries.out) <= norm_trade_resid.index[-1]:
                        plt.scatter(up_entries.out, norm_trade_resid.loc[up_entries.out].values, color="black", marker='<')
                if len(down_entries) > 0:
                    print(down_entries)
                    plt.scatter(down_entries.entry, norm_trade_resid.loc[down_entries.entry].values, color="red" ,marker='>')
                    if max(down_entries.out) <= norm_trade_resid.index[-1]:

                        plt.scatter(down_entries.out, norm_trade_resid.loc[down_entries.out].values, color="red", marker='<')
                plt.savefig(f"{end_test_date} - {pair1} x {pair2}")
                plt.close()


                res_total = pd.concat([res_total, res])
            start_treino_date = start_test_date + pd.DateOffset(months=3)
            end_treino_date = start_treino_date + pd.DateOffset(months=12)
            start_test_date = end_treino_date
            end_test_date = start_test_date + pd.DateOffset(months=3)

quit()
res_total.to_csv("resultados_backtest.csv")
quit()
aux_res_entries["entry_price_long"] = 0
aux_res_entries["exit_price_long"] = 0

aux_res_entries["entry_price_short"] = 0
aux_res_entries["exit_price_short"] = 0

# datas para as entradas up, >=2 std acima
entry_dates_up = aux_res_entries[aux_res_entries.type == "up_entry"].entry.values
exit_dates_up = aux_res_entries[aux_res_entries.type == "up_entry"].out.values 


for i in range(len(entry_dates_up)):
   
   entry,exit = entry_dates_up[i], exit_dates_up[i]
   aux_res_entries.loc[entry, "entry_price_long"] = base.loc[entry, pair2]
   aux_res_entries.loc[exit, "exit_price_long"] = base.loc[exit, pair2]
   aux_res_entries.loc[entry, "entry_price_short"] = base.loc[exit, pair1]
   aux_res_entries.loc[exit, "entry_price_short"] = base.loc[exit, pair1]
   
entry_dates_down = aux_res_entries[aux_res_entries.type == "down_entry"].entry.values
exit_dates_down = aux_res_entries[aux_res_entries.type == "down_entry"].out.values 


### add down entry prices

aux_res_entries.loc[aux_res_entries.type == "down_entry", "entry_price_long"]\
                        = base.loc[entry_dates_up,pair1].values

aux_res_entries.loc[aux_res_entries.type == "down_entry", "exit_price_long"]\
                        = base.loc[exit_dates_up,pair1].values

aux_res_entries.loc[aux_res_entries.type == "down_entry", "entry_price_short"]\
                        = base.loc[entry_dates_up,pair2].values

aux_res_entries.loc[aux_res_entries.type == "down_entry", "exit_price_short"]\
                        = base.loc[exit_dates_up,pair2].values


print(aux_res_entries)
aux_res_entries.to_csv("res.csv")