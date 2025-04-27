import yfinance as yf
import pandas as pd
import numpy as np 
from statsmodels.tsa.stattools import adfuller
import datetime
import math
import requests
from bs4 import BeautifulSoup
import matplotlib.pyplot as plt
from hurst import compute_Hc
class PairCointegration:

    def __init__(self, pair1, pair2):
        self.pair1 = pair1
        self.pair2 = pair2
        self.data = None


    def get_dividends_agenda(self):



        # URL da página
        url = 'https://statusinvest.com.br/acoes/proventos/ibovespa'

        # Fazer a requisição GET para a página
        response = requests.get(url)

        # Verificar se a requisição foi bem-sucedida
        if response.status_code == 200:
            # Parsear o conteúdo da página com BeautifulSoup
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Encontrar a tabela de proventos
            table = soup.find('table', {'id': 'provents'})  # Usando o ID específico da tabela
            
            # Verificar se a tabela foi encontrada
            if table:
                # Extrair os dados da tabela
                headers = [th.text.strip() for th in table.find_all('th')]
                rows = []
                for tr in table.find_all('tr')[1:]:
                    cells = [td.text.strip() for td in tr.find_all('td')]
                    if cells:
                        rows.append(cells)

                # Criar um DataFrame pandas com os dados extraídos
                df = pd.DataFrame(rows, columns=headers)

                return df
            else:
                print("Tabela de proventos não encontrada na página.")
        else:
            print(f"Erro ao acessar a página. Status code: {response.status_code}")


    def get_price_data(self, start_date='', end_date=''):
        # Download data for pair1 and pair2

        if end_date == '' and start_date == '':
        
            try: 
            
                data1 = yf.download(self.pair1, period="1y")
                data2 = yf.download(self.pair2, period="1y")

                self.data = pd.DataFrame({
                    self.pair1: data1['Close'][self.pair1],
                    self.pair2: data2['Close'][self.pair2]
                })

                self.pair1_price_series = data1["Close"]
                self.pair2_price_series = data2["Close"]
            except Exception as e:
                print(e)
                print("problem downloading data")
        else:        
            
            try:
                data1 = yf.download(self.pair1, start=start_date, end=end_date)
                data2 = yf.download(self.pair2, start=start_date, end=end_date)
                
                # Check if the data is not empty
                # should check separately?
                if data1.empty or data2.empty:
                    raise ValueError("Data not available for one or both tickers")

                # Merge data on date and return a DataFrame with the close prices of pair1 and pair2
                self.data = pd.DataFrame({
                    self.pair1: data1['Close'],
                    self.pair2: data2['Close']
                })
                self.data.dropna(inplace=True)  # Remove rows with missing values
                
            
            except Exception as e:
                print(f"Error while downloading data for tickers '{self.pair1}' and '{self.pair2}': {e}")
                self.data = None  # Reset the data attribute to None
        
    def _ols(self,y, x):
        """
        Perform ordinary least squares regression.

        Parameters:
        y (array-like): The dependent variable.
        x (array-like): The independent variable.

        Returns:
        tuple: A tuple containing the slope (beta), intercept (alpha), and residuals (resid).

        Raises:
        ValueError: If the inputs are empty, have different lengths, or contain non-numeric values.
        """

        # Convert inputs to numpy arrays for consistency
        y = np.array(y)
        x = np.array(x)

        # Check for valid input lengths
        if len(y) != len(x):
            raise ValueError("Input arrays must have the same length.")

        # Check for non-empty inputs
        if len(y) == 0 or len(x) == 0:
            raise ValueError("Input arrays must not be empty.")

        # Calculate the regression parameters using numpy functions
        n = len(x)
        sum_x = np.sum(x)
        sum_y = np.sum(y)
        sum_xy = np.sum(x * y)
        sum_x2 = np.sum(x ** 2)

        # Calculate beta (slope) and alpha (intercept)
        denominator = (n * sum_x2 - sum_x ** 2)
        if denominator == 0:
            raise ValueError("Perfect multicollinearity detected.")

        beta = (n * sum_xy - sum_y * sum_x) / denominator
        alpha = (sum_y - beta * sum_x) / n

        # Calculate residuals
        resid = y - (beta * x + alpha)
        # transform to series
        resid = pd.Series(resid, index=self.data.index)
        
        # half life calc
        
        
        return beta, alpha, resid
    
    
    def _normalize_residue(self, mean=None, std=None):

        """
        normalize the residue series, given a mean and std as arguments
        """

        if mean == None and std == None:
            mean = self.resid.mean()
            std = self.resid.std()

        self.norm_resid = (self.resid - mean)/std

        return self
    
    def _calc_half_life(self):
        lag_resid = self.resid.shift(1).bfill()
        delta_resid = self.resid  - lag_resid

        beta, alpha, resid = self._ols(delta_resid, lag_resid)
        self.half_life = math.ceil(-1* np.log(2)/beta)
        
       
    def _cointegration_test(self, cutoff=0.05):

        """
            simple version of cointegration test
        """
        x = np.ravel(self.resid.values)
        if adfuller(x)[1] < cutoff:
            print("pair is cointegrated")
            return True
        else:
            print("pair not cointegrated")
            return False
        
    def cointegration_model(self):

        self.get_price_data()
        self.beta, self.alpha, self.resid = self._ols(self.data[self.pair1], self.data[self.pair2])
        self._normalize_residue()
        self._calc_half_life()
        self.pair1_rsi = self._calculate_rsi(self.pair1_price_series)
        self.pair2_rsi = self._calculate_rsi(self.pair2_price_series)

        return self

    def _calculate_rsi(self, series, period=14):
        delta = series.diff(1)
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        avg_gain = gain.rolling(window=period, min_periods=1).mean()
        avg_loss = loss.rolling(window=period, min_periods=1).mean()
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        return rsi


if __name__ == "__main__":
    # Example usage:
    pair = PairCointegration("JHSF3.SA", "QUAL3.SA")
    #df = pair.get_dividends_agenda()

    #print(df)
    pair.cointegration_model()
    pair._cointegration_test()

    #H, c, data = compute_Hc(pair.resid, kind='price')

    print("beta", pair.beta)
    print("hl", pair.half_life)
    #print("Hurst",H)

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 8))



    # First subplot
    ax1.plot(pair.norm_resid.index, pair.norm_resid.values)
    ax1.axhline(y=0, color='black', linestyle='--')
    ax1.axhline(y=2, color='black', linestyle='--')
    ax1.axhline(y=-2, color='black', linestyle='--')
    ax1.set_title(f'Resíduo normalizado do par {pair.pair1} - {pair.pair2}')
    ax1.legend()

    # Second subplot
    p1 = pair.pair1_price_series 
    p1_norm = (p1 - p1.mean())/p1.std()
    p2 = pair.pair2_price_series
    p2_norm = (p2 - p2.mean())/p2.std()
    ax2.plot(p1_norm.index, p1_norm.values, label=pair.pair1,)
    ax2.plot(p2_norm.index, p2_norm.values, label=pair.pair2,)
    
    ax2.set_title('Preços normalizados')
    ax2.set_xlabel('t')
    ax2.legend()

    ax3.plot(pair.pair1_rsi.index, pair.pair1_rsi.values, label=f'RSI {pair.pair1}')
    ax3.plot(pair.pair2_rsi.index, pair.pair2_rsi.values, label=f'RSI {pair.pair2}')
    ax3.axhline(y=70, color='red', linestyle='--', label='Overbought')
    ax3.axhline(y=30, color='green', linestyle='--', label='Oversold')
    ax3.set_title('RSI (Relative Strength Index)')
    ax3.set_xlabel('t')
    ax3.legend()
    #plt.title(pair.pair1 + '_' +pair.pair2)
    #plt.axhline(y=0, color='black', linestyle='--')
    #plt.axhline(y=2, color='black', linestyle='--')
    #plt.axhline(y=-2, color='black', linestyle='--')
    plt.show()  
