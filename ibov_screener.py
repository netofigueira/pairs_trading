import pandas as pd
import yfinance as yf
import itertools
import numpy as np 
from statsmodels.tsa.stattools import adfuller, coint


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


tickers_ibov = [
'ALPA4',
'ABEV3',
'AMER3',
'B3SA3',
'BPAN4',
'BBSE3',
'BRML3',
'BBDC3',
'BBDC4',
'BRAP4',
'BBAS3',
'BRKM5',
'BRFS3',
'CCRO3',
'FLRY3',
'GGBR4',
'GOAU4',
'GOLL4',
'HYPE3',
'IGTA3',
'ITSA4',
'ITUB4',
'JBSS3',
'JHSF3',
'KLBN11',
'RENT3',
'LCAM3',
'LAME4',
'LREN3',
'MGLU3',
'MRFG3',
'BEEF3',
'MRVE3',
'MULT3',
#'PCAR3',
'PETR3',
'PETR4',
'PRIO3',
'QUAL3',
'RADL3',
'RAIL3',
'SBSP3',
'SANB11',
'CSNA3',
'SULA11',
'SUZB3',
'TAEE11',
'VIVT3',
'TIMS3',
'TOTS3',
'UGPA3',
'USIM5',
'VALE3',
'VIIA3',
'WEGE3',
'YDUQ3',
'BOVA11',
]

tickers_yf = [i + '.SA' for i in tickers_ibov]
ibov= yf.download(tickers_yf, period='1y')

close_prices = ibov['Close']
close_prices.drop(columns=['IGTA3.SA', 'VIIA3.SA', 'SULA11.SA', 'LAME4.SA', 'BRML3.SA', 'LCAM3.SA'], inplace=True)
print(close_prices)
pares_cointegrados = []
resid_df = pd.DataFrame()
all_pairs =  list(itertools.combinations(close_prices.columns,2))
for pair in all_pairs:

  try:
    beta, alpha, resid = ols(close_prices[pair[0]], close_prices[pair[1]])

    if stationarity_bool(resid):
      meia_vida = calc_half_life(resid)
      pares_cointegrados.append([pair[0], pair[1], meia_vida, beta, alpha])

      norm_resid = (resid - resid.mean())/resid.std()
      resid_df[pair[0] + '_'+ pair[1]] = norm_resid

  except:
    pass
print('total_de_pares', len(all_pairs))
print('pares_coint', len(pares_cointegrados))

coint_pairs_df = pd.DataFrame(pares_cointegrados, columns=['pair1', 'pair2', 'half_life', 'beta', 'alpha'])

print(coint_pairs_df)
print(resid_df)

last_values = resid_df.iloc[-1]
print(last_values)
print(last_values[last_values>2])

print(last_values[last_values<-2])

