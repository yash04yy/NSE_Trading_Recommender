import { useEffect, useState, useMemo } from 'react';
import Chart from 'react-apexcharts';
import './App.css'
import Logo from './stocklogo.svg'

const stonksUrl = 'http://localhost:5000/';

async function getStonks(ticker) {
  const response = await fetch(`${stonksUrl}${ticker}`);
  return response.json();
}

const directionEmojis = {
  up: '🚀',
  down: '💩',
  '': '',
};

const chart = {
  options: {
    chart: {
      type: 'candlestick',
      height: 350
    },
    title: {
      text: 'CandleStick Chart',
      align: 'left'
    },
    xaxis: {
      type: 'datetime',
      timezone: 'Asia/Kolkata'
    },
    yaxis: {
      tooltip: {
        enabled: true
      }
    }
  },
};

const round = (number) => {
  return number ? +(number.toFixed(2)) : null;
};

function App() {
  const [series, setSeries] = useState([{
    data: []
  }]);
  const [price, setPrice] = useState(-1);
  const [prevPrice, setPrevPrice] = useState(-1);
  const [priceTime, setPriceTime] = useState(null);
  const [stockName, setStockName] = useState('');
  const [selectedStock, setSelectedStock] = useState('^NSEI');
  const [stockList, setStockList] = useState([
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
    { name: 'Power Grid Corporation of India Limited',ticker:'POWERGRID.NS' },
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
  ]);

  useEffect(() => {
    let timeoutId;
    async function getLatestPrice() {
      try {
        const data = await getStonks(selectedStock);
        console.log(data);
        const gme = data.chart.result[0];
        setStockName(gme.meta.symbol);
        setPrevPrice(price);
        setPrice(gme.meta.regularMarketPrice.toFixed(2));
        setPriceTime(new Date(gme.meta.regularMarketTime * 1000));
        const quote = gme.indicators.quote[0];
        const prices = gme.timestamp.map((timestamp, index) => ({
          x: new Date((timestamp + (5.5 * 60 * 60)) * 1000).toLocaleString('en-US', { timeZone: 'Asia/Kolkata' }),
          y: [quote.open[index], quote.high[index], quote.low[index], quote.close[index]].map(round)
        }));
        setSeries([{
          data: prices,
        }]);
      } catch (error) {
        console.log(error);
      }
      timeoutId = setTimeout(getLatestPrice, 5000 * 2);
    }

    getLatestPrice();

    return () => {
      clearTimeout(timeoutId);
    };
  }, [selectedStock]);

  const direction = useMemo(() => prevPrice < price ? 'up' : prevPrice > price ? 'down' : '', [prevPrice, price]);

  return (
    <div>
      <div className="navbar__wrapper">
        <div className="navbar__logo">
          <img src={Logo} width={55}/>
        </div>
        <div className="navbar__search">
          <div className="navbar__searchContainer">   
            <select value={selectedStock} onChange={(e) => setSelectedStock(e.target.value)} style={{ marginLeft: '10px', position: 'relative', left: '10px' }}>
            {stockList.map((stock) => (
              <option key={stock.ticker} value={stock.ticker}>
                {stock.name}
              </option>
            ))}
            </select>
          </div>
        </div>
        <div className="navbar__menuItems">
          <a href="/">Free Stocks</a>
          <a href="/">PortFolio</a>
          <a href="/">Cash</a>
          <a href="/">Messages</a>
          <a href="/">Account</a>
        </div>
      </div>
      <div className="name-dropdown">
        <div className='stock-name'>{stockName}</div>
      </div>
      <div className={['price', direction].join(' ')}>
        ₹{price} {directionEmojis[direction]}
      </div>
      <div className="price-time">
        {priceTime && priceTime.toLocaleTimeString()}
      </div>
      <Chart options={chart.options} series={series} type="candlestick" width="100%" height={320} />
    </div>
  );
}

export default App;