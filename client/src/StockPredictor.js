import React, { useState } from 'react';
import './StockPredictor.css';
import Highcharts from 'highcharts/highstock';
import HighchartsReact from 'highcharts-react-official';
import highchartsMore from 'highcharts/highcharts-more';
import solidGauge from 'highcharts/modules/solid-gauge';
import { FontAwesomeIcon } from '@fortawesome/react-fontawesome';
import { faSpinner } from '@fortawesome/free-solid-svg-icons';
import TrendingUpIcon from '@mui/icons-material/TrendingUp';
import TrendingDownIcon from '@mui/icons-material/TrendingDown';
import PanToolIcon from '@mui/icons-material/PanTool';
const STOCK_OPTIONS = [
    { name: 'NIFTY 50', ticker: '^NSEI' },
    { name: 'Adani Ports', ticker: 'ADANIPORTS.NS' },
    { name: 'Asian Paints', ticker: 'ASIANPAINT.NS' },
    { name: 'Axis Bank', ticker: 'AXISBANK.NS' },
    { name: 'Bajaj Auto', ticker: 'BAJAJ-AUTO.NS' },
    { name: 'Bajaj Finserv', ticker: 'BAJAJFINSV.NS' },
    { name: 'Bajaj Finance', ticker: 'BAJFINANCE.NS' },
    { name: 'Bharti Airtel', ticker: 'BHARTIARTL.NS' },
    { name: 'BPCL', ticker: 'BPCL.NS' },
    { name: 'Britannia', ticker: 'BRITANNIA.NS' },
    { name: 'CIPLA', ticker: 'CIPLA.NS' },
    { name: 'Coal India', ticker: 'COALINDIA.NS' },
    { name: 'Divis Laboratories', ticker: 'DIVISLAB.NS' },
    { name: 'Dr. Reddy’s Laboratories', ticker: 'DRREDDY.NS' },
    { name: 'Eicher Motors', ticker: 'EICHERMOT.NS' },
    { name: 'GAIL', ticker: 'GAIL.NS' },
    { name: 'Grasim Industries', ticker: 'GRASIM.NS' },
    { name: 'HCL Technologies', ticker: 'HCLTECH.NS' },
    { name: 'HDFC', ticker: 'HDFC.NS' },
    { name: 'HDFC Bank', ticker: 'HDFCBANK.NS' },
    { name: 'Hero MotoCorp', ticker: 'HEROMOTOCO.NS' },
    { name: 'Hindalco Industries', ticker: 'HINDALCO.NS' },
    { name: 'Hindustan Unilever', ticker: 'HINDUNILVR.NS' },
    { name: 'ICICI Bank', ticker: 'ICICIBANK.NS' },
    { name: 'IndusInd Bank', ticker: 'INDUSINDBK.NS' },
    { name: 'Infosys', ticker: 'INFY.NS' },
    { name: 'IOC', ticker: 'IOC.NS' },
    { name: 'ITC', ticker: 'ITC.NS' },
    { name: 'JSW Steel', ticker: 'JSWSTEEL.NS' },
    { name: 'Kotak Mahindra Bank', ticker: 'KOTAKBANK.NS' },
    { name: 'L&T', ticker: 'LT.NS' },
    { name: 'M&M', ticker: 'M&M.NS' },
    { name: 'Maruti Suzuki', ticker: 'MARUTI.NS' },
    { name: 'NTPC', ticker: 'NTPC.NS' },
    { name: 'Nestle India', ticker: 'NESTLEIND.NS' },
    { name: 'ONGC', ticker: 'ONGC.NS' },
    { name: 'Power Grid Corporation of India Limited', ticker: 'POWERGRID.NS' },
    { name: 'Reliance Industries', ticker: 'RELIANCE.NS' },
    { name: 'SBI', ticker: 'SBIN.NS' },
    { name: 'Shree Cement', ticker: 'SHREECEM.NS' },
    { name: 'Sun Pharmaceutical Industries', ticker: 'SUNPHARMA.NS' },
    { name: 'Tata Consumer Products', ticker: 'TATACONSUM.NS' },
    { name: 'Tata Motors', ticker: 'TATAMOTORS.NS' },
    { name: 'Tata Steel', ticker: 'TATASTEEL.NS' },
    { name: 'Tech Mahindra', ticker: 'TECHM.NS' },
    { name: 'Titan Company', ticker: 'TITAN.NS' },
    { name: 'UltraTech Cement', ticker: 'ULTRACEMCO.NS' },
    { name: 'UPL', ticker: 'UPL.NS' },
    { name: 'Wipro', ticker: 'WIPRO.NS' },
    { name: 'Adani Tech', ticker: 'ADANITECH.NS' },
    { name: 'Hindustan Petroleum Corporation', ticker: 'HINDPETRO.NS' }
];

const StockPredictor = () => {
        const [stockSymbol, setStockSymbol] = useState('');
        const [stockName, setStockName] = useState('');
        const [maxSentiment, setMaxSentiment] = useState("-");
        const [prediction, setPrediction] = useState({
            close: [],
            start_date: null,
        });
        const [sentiment, setSentiment] = useState({
            positive: 33.33,
            negative: 33.33,
            neutral: 33.33,
        });
        const [candlestick, setCandlestick] = useState({
            cdlrel: "",
            cdlname: "",
            cdltrend: "",
        });
        const [mltrend, setMLTrend] = useState("- Trend");
        const [isLoading, setIsLoading] = useState(false);
        const [tradecall, setTradecall] = useState("-");
        const [icon, setIcon] = useState(<PanToolIcon style={{ fontSize: '40px',paddingTop:"-12px",paddingRight:"10px"}}/>);

        const handleSubmit = async(event) => {
            event.preventDefault();
            setIsLoading(true);
            try {
                const response = await fetch(`http://localhost:5001/predict?stock_symbol=${stockSymbol}&stock_name=${stockName}`);
                const data = await response.json();
                setPrediction({ close: data.pred_dict, start_date: data.start_date });
                setSentiment({ positive: data.positive, negative: data.negative, neutral: data.neutral });
                setMLTrend(data.mltrend)
                const maxVal = Math.max(data.positive, data.negative, data.neutral);

                // Set the state of maxSentiment based on which value is the maximum
                if (maxVal === data.positive) {
                    setMaxSentiment("Positive");
                } else if (maxVal === data.negative) {
                    setMaxSentiment("Negative");
                } else {
                    setMaxSentiment("Neutral");
                }

                //set states for Candlestick module
                setCandlestick({ cdlname: data.cdlname, cdlrel: data.cdlrel, cdltrend: data.cdltrend });
                setTradecall(data.tradecall);
                if (data.tradecall === "BUY") {
                  setIcon(<TrendingUpIcon fontSize="large"style={{ fontSize: '55px',paddingTop:"-12px",paddingRight:"10px"}}/>);
                } else if (data.tradecall === "SELL") {
                  setIcon(<TrendingDownIcon fontSize="large"style={{ fontSize: '55px',paddingTop:"-12px",paddingRight:"10px"}}/>);
                }
            } catch (error) {
                console.error(error);
            } finally {
                setIsLoading(false);
            }
        };

        const closePrices = prediction.close;
        const highestPrice = Math.max(...closePrices).toFixed(2);
        const lowestPrice = Math.min(...closePrices).toFixed(2);

        const options = {
                rangeSelector: {
                    selected: 1,
                },
                title: {
                    text: `Stock Prediction for ${stockSymbol}${prediction.start_date ? ` (Start: ${prediction.start_date})` : ''}`,
    },
    xAxis: {
      type: 'datetime',
      tickInterval: 60 * 1000, // One minute
      tickPositions: [],
      labels: {
        format: '{value:%H:%M}',
        formatter: function() {
          return new Date(this.value).toLocaleTimeString('en-US', { timeZone: 'Asia/Kolkata' });
        },
      },
    },
    series: [
    {
      name: 'Close',
      data: closePrices.map((price, index) => {
        const timestamp = new Date(prediction.start_date);
        timestamp.setTime(timestamp.getTime() + (index * 60 * 1000));
        const timestampIST = new Date(timestamp.getTime() + (330 * 60 * 1000));
        return [timestampIST.getTime(), price];
      }),
    },
    ],
    subtitle: {
      text: `Highest Price: ${highestPrice}, Lowest Price: ${lowestPrice}`,
      align: 'right',
      verticalAlign: 'top',
      useHTML: true,
    }
  };

  const handleStockSymbolChange = (event) => {
    const stock = STOCK_OPTIONS.find(option => option.ticker === event.target.value);
    setStockSymbol(stock.ticker);
    setStockName(stock.name);
  };

  highchartsMore(Highcharts);
  solidGauge(Highcharts);

  const options2 = {
    chart: {
      type: 'pie',
      height: '70%'
    },
    title: {
      text: `Sentiment Analysis for`,
    },
    subtitle:{
      text:`${stockName}`,
      style: {
        fontSize:'16px',
        fontWeight: 'bold',
        fontColor:'black',
      }
    },
    series: [{
      name: 'Sentiments',
      data: [{
        name: 'Positive',
        y: sentiment.positive,
        color: '#28a745'
      }, {
        name: 'Negative',
        y: sentiment.negative,
        color: '#dc3545'
      }, {
        name: 'Neutral',
        y: sentiment.neutral,
        color: '#17a2b8'
      }]
    }]
  };
  return (
    <div className={isLoading ? 'loading' : ''}>
      <div className="upperdiv">
        <div className="uppertext">
          <div className="ut1">Candlestick Pattern Recognition Module has recognized <span className={`${candlestick.cdltrend.toLowerCase()}`}> {candlestick.cdltrend}</span> - {candlestick.cdlname} which has <span className={`${candlestick.cdlrel.toLowerCase()}`}> {candlestick.cdlrel}</span>.</div>
          <div className="ut2">Machine Learning Model has predicted a/an <span className={`${mltrend.toLowerCase()}`}> {mltrend}</span>.</div>
          <div className="ut3">Sentiment analysis has recognized a <span className={`${maxSentiment.toLowerCase()}`}>{maxSentiment} Sentiment.</span></div>
  
          <div className="ut4">
            <div className="ut41">{icon}</div> 
            <div className="ut42">WE RECOMMEND - <span className={`${tradecall.toLowerCase()}`}>{tradecall} </span></div>
          </div>
        </div>
        <form onSubmit={handleSubmit}>
          <label>
            Stock symbol:
          </label>
          <div className="navbar__search">
              <div className="navbar__searchContainer"> 
                <select value={stockSymbol} onChange={handleStockSymbolChange}>
                  <option value="">Select a stock</option>
                    {STOCK_OPTIONS.map((stock) => (
                  <option key={stock.ticker} value={stock.ticker}>
                    {stock.name}
                  </option>
                ))}
                </select>
              </div>
          </div>
          <br />
          <button type="submit" disabled={isLoading}>
            {isLoading ? (
              <>
                <FontAwesomeIcon icon={faSpinner} className="fa-spin" />
                &nbsp;Loading...
              </>
            ) : (
              'Predict'
            )}
          </button>
        </form>
        <div className="uppersentiment">
          <HighchartsReact highcharts={Highcharts} options={options2} />
        </div>
      </div>
      <div className="lowerdiv"><HighchartsReact highcharts={Highcharts} constructorType={'stockChart'} options={options} /></div>
    </div>
  );
};

export default StockPredictor;