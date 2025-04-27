import unittest
from unittest.mock import patch, MagicMock
from pair_trading import PairCointegration
import pandas as pd

class TestPairCointegration(unittest.TestCase):
    def setUp(self):
        # Set up the test case with example tickers
        self.pair = PairCointegration("AAPL", "MSFT")

    @patch('yfinance.download')
    def test_get_price_data_success(self, mock_download):
        # Create a mock DataFrame with example data
        data1 = pd.DataFrame({
            'Adj Close': [150, 152, 154, 156, 158],
            'Date': pd.date_range(start='2022-01-01', periods=5)
        }).set_index('Date')
        
        data2 = pd.DataFrame({
            'Adj Close': [300, 302, 304, 306, 308],
            'Date': pd.date_range(start='2022-01-01', periods=5)
        }).set_index('Date')
        
        # Configure the mock to return the example data
        mock_download.side_effect = [data1, data2]
        
        # Call the function and assert that it returns a DataFrame with the correct columns and values
        result = self.pair.get_price_data(start_date='2022-01-01', end_date='2022-01-05')
        self.assertIsNotNone(result)
        self.assertEqual(list(result.columns), ['AAPL', 'MSFT'])
        pd.testing.assert_frame_equal(result, pd.DataFrame({
            'AAPL': [150, 152, 154, 156, 158],
            'MSFT': [300, 302, 304, 306, 308]
        }, index=pd.date_range(start='2022-01-01', periods=5)))
    
    @patch('yfinance.download')
    def test_get_price_data_failure(self, mock_download):
        # Configure the mock to raise an exception to simulate data download failure
        mock_download.side_effect = Exception("Invalid ticker symbol")
        
        # Call the function and assert that it returns None due to the exception
        with self.assertLogs(level='ERROR') as log:
            result = self.pair.get_price_data(start_date='2022-01-01', end_date='2022-01-05')
            self.assertIsNone(result)
            # Check that an error message was logged
            self.assertIn("Error while downloading data for tickers", log.output[0])

# Run the tests
if __name__ == '__main__':
    unittest.main()
