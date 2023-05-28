import numpy as np
from flask_restful import Resource, Api
from flask import Flask, jsonify, request
from yahoo_fin.stock_info import get_data
from sklearn.preprocessing import MinMaxScaler
from keras.models import load_model
import pandas as pd
from datetime import datetime, timedelta
from flask_cors import CORS
from flask import Flask, send_file
import pandas as pd
import yfinance as yf
import talib
from talib import abstract
import numpy as np
from itertools import compress


app = Flask (__name__)
CORS(app)
api = Api(app)


class Prediction(Resource):
    def get(self):
        #For stock predictor
        stock_symbol = request.args.get('stock_symbol')
        stock_name=request.args.get('stock_name')
        #end_date_str = request.args.get('end_date')
        #end_date = pd.to_datetime(end_date_str)
        # get the input data from the request //for postman
        # data = request.get_json();
        # stock_symbol = data['stock_symbol']
        # stock_name = data['stock_name']
        #end_date = pd.to_datetime(data['end_date'])
        #stock_symbol = 'TATASTEEL.NS'
        #end_date_str = '04/21/2023'
        #e_date = datetime.strptime(end_date, '%m/%d/%Y')
        
        
        import datetime
        now = datetime.datetime.now()
        formatted_date = now.strftime("%m/%d/%Y")

        # Add one day
        one_day = datetime.timedelta(days=1)
        five_day = datetime.timedelta(days=5)

        new_date = datetime.datetime.strptime(formatted_date, "%m/%d/%Y") + one_day 
        old_date = datetime.datetime.strptime(formatted_date, "%m/%d/%Y") - five_day

        formatted_new_date = new_date.strftime("%m/%d/%Y")
        formatted_old_date=old_date.strftime("%m/%d/%Y")

        formatted_new_date

        data= get_data(stock_symbol, start_date=formatted_old_date, end_date=formatted_new_date, index_as_date = True, interval="1m")
        
        data.reset_index(inplace=True)
        data.rename(columns={'index': 'timestamp'}, inplace=True)

        import pandas as pd
        # localize the timestamp column to the UTC timezone
        data['timestamp'] = pd.to_datetime(data['timestamp']).dt.tz_localize('UTC')
        # convert the timestamp column to the IST timezone with the desired time zone offset
        data['timestamp'] = data['timestamp'].dt.tz_convert('Asia/Kolkata').dt.tz_localize(None)
        # format the timestamp column as a string in the desired format
        data['timestamp'] = data['timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S')
        data['timestamp'] = pd.to_datetime(data['timestamp'])
        data.set_index('timestamp', inplace=True)

        data=data.dropna()

        time_steps = 60
        #1st 60 timesteps are used to give result for 61st timestep
        data1=data.iloc[time_steps+1:,:]

        mini=data1['close'].min()
        maxi=data1['close'].max()

        from sklearn.preprocessing import MinMaxScaler

        # select only numeric columns for normalization
        numeric_cols = ['open', 'high', 'low', 'close', 'volume']
        data_numeric = data[numeric_cols]
        # normalize the data
        scaler = MinMaxScaler()
        data_normalized = scaler.fit_transform(data_numeric)
        # replace the original values with the normalized values
        data[numeric_cols] = data_normalized
        # inverse transform the prediction

        import pandas as pd
        import numpy as np
        from sklearn.preprocessing import MinMaxScaler
        from ta import add_all_ta_features
        numeric_cols = ['open', 'high', 'low', 'close', 'volume']
        data_numeric = data[numeric_cols]
        # Prepare data for LSTM
        scaler = MinMaxScaler(feature_range=(0, 1))
        data_scaled = scaler.fit_transform(data_numeric)
        # Define the number of time steps
        time_steps = 60
        # Create input sequence
        X = []
        y = []
        for i in range(time_steps, len(data_scaled)):
            X.append(data_scaled[i-time_steps:i, :])
            y.append(data_scaled[i, [0, 1, 2, 3, 4]])  # use columns 0, 1, 4, and 3 for open, high, volume, and close, respectively
        X, y = np.array(X), np.array(y)

        split_ratio = 0.7
        split_index = int(split_ratio * len(X))
        X_train, X_test = X[:split_index], X[split_index:]
        y_train, y_test = y[:split_index], y[split_index:]
        # Build LSTM model
        from keras.models import Sequential
        from keras.layers import Dense, LSTM
        model = Sequential()
        model.add(LSTM(units=25, return_sequences=True, input_shape=(time_steps, X.shape[2])))
        model.add(LSTM(units=25, return_sequences=True))
        model.add(LSTM(units=25))
        model.add(Dense(units=y.shape[1]))  # change output dimension to match the number of columns in y
        model.compile(optimizer='adam', loss='mean_squared_error')
        model.fit(X_train, y_train, epochs=100, batch_size=32)

        y_pred = model.predict(X_test)

        j=int(len(data1))
        p=int(0.7*len(data1))
        i=int(len(data1)-0.7*len(data1))
        sz=j-p
        if (y_test.shape[0]>sz):
            data2 = data1.iloc[-(sz+1):]
        else:
            data2 = data1.iloc[-sz:]

        data2.loc[:, 'YTest'] = y_test[:,3]
        data2.loc[:, 'YPred'] = y_pred[:,3]

        data2.loc[:, 'YPred'] = data2['YPred'] * (maxi - mini) + mini
        data2.loc[:, 'YTest'] = data2['YTest'] * (maxi - mini) + mini

        data2 = data2.reset_index()

        df = data2[['YTest', 'YPred']]

        last_60_minutes = X_test[-1]  # shape: (60, number of features)
        prediction_input = np.reshape(last_60_minutes, (1, time_steps, X.shape[2]))  # shape: (1, 60, number of features)
        # Make the prediction
        predicted_values = model.predict(prediction_input)  # shape: (1, y.shape[1])
        # Initialize the predicted_values_array
        predicted_values_array = predicted_values.copy()
        # Repeat the process for the next 59 minutes
        for i in range(59):
            # Shift the last_60_minutes array by one minute
            last_60_minutes[:-1] = last_60_minutes[1:]
            # Set the last row of last_60_minutes to the predicted values from the previous minute
            last_60_minutes[-1] = predicted_values[0]
            # Prepare the input data for prediction
            prediction_input = np.reshape(last_60_minutes, (1, time_steps, X.shape[2]))
            # Make the prediction for the next minute
            predicted_values = model.predict(prediction_input)
            # Append the predicted values to the output array
            predicted_values_array = np.append(predicted_values_array, predicted_values, axis=0)

        predicted_values_array[:,3]=predicted_values_array[:,3] * (maxi - mini) + mini
        pva=predicted_values_array[:,3]
        diff = pva[-1] - pva[0]

# Check if the difference is positive or negative to determine the trend
        if diff > 0:
            trendstr="Uptrend"
            ifdown=1
        elif diff < 0:
            trendstr="Downtrend"
            ifdown=-1
        else:
            trendstr="No_trend"
            ifdown=0

        import heapq
        maxi= np.max(pva)
        mini= np.min(pva)
        max_diff= maxi-mini
        second_min = heapq.nsmallest(2, pva)[1]
        min_diff= second_min-mini

        abdiff=abs(diff)

        scaled_ml=(abdiff-min_diff)/(max_diff-min_diff)

        finalml= 0.6*scaled_ml*ifdown
  
        pred_dict1 = predicted_values_array[:,3].tolist()     
        last_timestamp = data2.iloc[-1, 0]  # Get last timestamp of data2

        if last_timestamp.time() == datetime.time(hour=15, minute=30):  # Check if last timestamp is 15:30
            next_date = last_timestamp.date() + datetime.timedelta(days=1)  # Increment date by one day
            next_timestamp = datetime.datetime.combine(next_date, datetime.time(hour=9, minute=15))  # Set time to 9:00
        else:
            next_timestamp = last_timestamp + pd.Timedelta(minutes=1)  # Add one minute to last timestamp

        next_timestamp_str = next_timestamp.strftime('%Y-%m-%d %H:%M:%S')  # Convert next timestamp to string
        #next_timestamp_dt = next_timestamp.to_pydatetime()  # Convert next timestamp to datetime object
        # Create dictionary with prediction and end_date

        import requests
        import json
        import datetime
        import matplotlib.pyplot as plt
        from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

        # Define the company ticker symbol
        ticker = stock_name

        # Calculate the date range for the previous 5 days
        today = datetime.date.today()
        ten_days_ago = today - datetime.timedelta(days=6)

        # Fetch the news articles from the News API for the previous 10 days
        url = f"https://newsapi.org/v2/everything?q={ticker}&pageSize=100&apiKey=df48ac0db065403bbc7a439535cce77f"
        response = requests.get(url)
        articles = json.loads(response.text)['articles']

        # Filter the articles based on their published date
        filtered_articles = [article for article in articles if datetime.datetime.strptime(article['publishedAt'][:10], '%Y-%m-%d').date() >= ten_days_ago]

        # Perform sentiment analysis on the filtered news articles and generate a pie chart of the results
        analyzer = SentimentIntensityAnalyzer()
        positive_count = 0
        negative_count = 0
        neutral_count = 0

        for article in filtered_articles:
            title = article['title']
            description = article['description']
            vs = analyzer.polarity_scores(description)
            sentiment = vs['compound']
            #print(f"Title: {title}")
            #print(f"Description: {description}")
            #print(f"Sentiment: {sentiment}\n")
            
            if sentiment > 0:
                positive_count += 1
            elif sentiment < 0:
                negative_count += 1
            else:
                neutral_count += 1

        # Calculate the percentage of each sentiment category
        total_count = positive_count + negative_count + neutral_count

        positive_percent = round(positive_count / total_count * 100, 2)
        negative_percent = round(negative_count / total_count * 100, 2)
        neutral_percent = round(neutral_count / total_count * 100, 2)

        maxim=max(positive_percent, negative_percent, neutral_percent)
        if maxim==positive_percent:
            sign=1
        elif maxim==negative_percent:
            sign=-1
        else:
            sign=0

        finalsenti=0.1*sign
        #Candlestick
        stock = yf.Ticker('INFY.NS')
        df = stock.history(period="15d", interval="1h")

        # loading dataset and converting the price column names to lowercase so abstract API can recognize the input
        df.rename(columns={'Open': 'open', 'High': 'high','Low': 'low','Close': 'close', 'Volume': 'volume'}, inplace= True)
        df.head()
        candle_names = talib.get_function_groups()['Pattern Recognition']

        # patterns not found in the patternsite.com
        exclude_items = ('CDLCOUNTERATTACK','CDLLONGLINE','CDLSHORTLINE','CDLSTALLEDPATTERN','CDLKICKINGBYLENGTH')

        candle_names = [candle for candle in candle_names if candle not in exclude_items]
        # extract OHLC 
        op = df['open']
        hi = df['high']
        lo = df['low']
        cl = df['close']
        # create columns for each pattern
        for candle in candle_names:
            # below is same as
            # df["CDL3LINESTRIKE"] = talib.CDL3LINESTRIKE(op, hi, lo, cl)
            df[candle] = getattr(talib, candle)(op, hi, lo, cl)

        # Note - 1
        # Only some patterns have bull and bear versions. 
        # However, to make the process unified and for codability purposes 
        # all patterns are labeled with "_Bull" and "_Bear" tags.
        # Both versions of the single patterns are given same performance rank, 
        # since they will always return only 1 version.  

        # Note - 2 
        # Following TA-Lib patterns are excluded from the analysis since the corresponding ranking not found:
        # CounterAttack, Longline, Shortline, Stalledpattern, Kickingbylength


        candle_rankings = {
            "CDL3LINESTRIKE_Bull": 1,
            "CDL3LINESTRIKE_Bear": 2,
            "CDL3BLACKCROWS_Bull": 3,
            "CDL3BLACKCROWS_Bear": 3,
            "CDLEVENINGSTAR_Bull": 4,
            "CDLEVENINGSTAR_Bear": 4,
            "CDLTASUKIGAP_Bull": 5,
            "CDLTASUKIGAP_Bear": 5,
            "CDLINVERTEDHAMMER_Bull": 6,
            "CDLINVERTEDHAMMER_Bear": 6,
            "CDLMATCHINGLOW_Bull": 7,
            "CDLMATCHINGLOW_Bear": 7,
            "CDLABANDONEDBABY_Bull": 8,
            "CDLABANDONEDBABY_Bear": 8,
            "CDLBREAKAWAY_Bull": 10,
            "CDLBREAKAWAY_Bear": 10,
            "CDLMORNINGSTAR_Bull": 12,
            "CDLMORNINGSTAR_Bear": 12,
            "CDLPIERCING_Bull": 13,
            "CDLPIERCING_Bear": 13,
            "CDLSTICKSANDWICH_Bull": 14,
            "CDLSTICKSANDWICH_Bear": 14,
            "CDLTHRUSTING_Bull": 15,
            "CDLTHRUSTING_Bear": 15,
            "CDLINNECK_Bull": 17,
            "CDLINNECK_Bear": 17,
            "CDL3INSIDE_Bull": 20,
            "CDL3INSIDE_Bear": 56,
            "CDLHOMINGPIGEON_Bull": 21,
            "CDLHOMINGPIGEON_Bear": 21,
            "CDLDARKCLOUDCOVER_Bull": 22,
            "CDLDARKCLOUDCOVER_Bear": 22,
            "CDLIDENTICAL3CROWS_Bull": 24,
            "CDLIDENTICAL3CROWS_Bear": 24,
            "CDLMORNINGDOJISTAR_Bull": 25,
            "CDLMORNINGDOJISTAR_Bear": 25,
            "CDLXSIDEGAP3METHODS_Bull": 27,
            "CDLXSIDEGAP3METHODS_Bear": 26,
            "CDLTRISTAR_Bull": 28,
            "CDLTRISTAR_Bear": 76,
            "CDLGAPSIDESIDEWHITE_Bull": 46,
            "CDLGAPSIDESIDEWHITE_Bear": 29,
            "CDLEVENINGDOJISTAR_Bull": 30,
            "CDLEVENINGDOJISTAR_Bear": 30,
            "CDL3WHITESOLDIERS_Bull": 32,
            "CDL3WHITESOLDIERS_Bear": 32,
            "CDLONNECK_Bull": 33,
            "CDLONNECK_Bear": 33,
            "CDL3OUTSIDE_Bull": 34,
            "CDL3OUTSIDE_Bear": 39,
            "CDLRICKSHAWMAN_Bull": 35,
            "CDLRICKSHAWMAN_Bear": 35,
            "CDLSEPARATINGLINES_Bull": 36,
            "CDLSEPARATINGLINES_Bear": 40,
            "CDLLONGLEGGEDDOJI_Bull": 37,
            "CDLLONGLEGGEDDOJI_Bear": 37,
            "CDLHARAMI_Bull": 38,
            "CDLHARAMI_Bear": 72,
            "CDLLADDERBOTTOM_Bull": 41,
            "CDLLADDERBOTTOM_Bear": 41,
            "CDLCLOSINGMARUBOZU_Bull": 70,
            "CDLCLOSINGMARUBOZU_Bear": 43,
            "CDLTAKURI_Bull": 47,
            "CDLTAKURI_Bear": 47,
            "CDLDOJISTAR_Bull": 49,
            "CDLDOJISTAR_Bear": 51,
            "CDLHARAMICROSS_Bull": 50,
            "CDLHARAMICROSS_Bear": 80,
            "CDLADVANCEBLOCK_Bull": 54,
            "CDLADVANCEBLOCK_Bear": 54,
            "CDLSHOOTINGSTAR_Bull": 55,
            "CDLSHOOTINGSTAR_Bear": 55,
            "CDLMARUBOZU_Bull": 71,
            "CDLMARUBOZU_Bear": 57,
            "CDLUNIQUE3RIVER_Bull": 60,
            "CDLUNIQUE3RIVER_Bear": 60,
            "CDL2CROWS_Bull": 61,
            "CDL2CROWS_Bear": 61,
            "CDLBELTHOLD_Bull": 62,
            "CDLBELTHOLD_Bear": 63,
            "CDLHAMMER_Bull": 65,
            "CDLHAMMER_Bear": 65,
            "CDLHIGHWAVE_Bull": 67,
            "CDLHIGHWAVE_Bear": 67,
            "CDLSPINNINGTOP_Bull": 69,
            "CDLSPINNINGTOP_Bear": 73,
            "CDLUPSIDEGAP2CROWS_Bull": 74,
            "CDLUPSIDEGAP2CROWS_Bear": 74,
            "CDLGRAVESTONEDOJI_Bull": 77,
            "CDLGRAVESTONEDOJI_Bear": 77,
            "CDLHIKKAKEMOD_Bull": 82,
            "CDLHIKKAKEMOD_Bear": 81,
            "CDLHIKKAKE_Bull": 85,
            "CDLHIKKAKE_Bear": 83,
            "CDLENGULFING_Bull": 84,
            "CDLENGULFING_Bear": 91,
            "CDLMATHOLD_Bull": 86,
            "CDLMATHOLD_Bear": 86,
            "CDLHANGINGMAN_Bull": 87,
            "CDLHANGINGMAN_Bear": 87,
            "CDLRISEFALL3METHODS_Bull": 94,
            "CDLRISEFALL3METHODS_Bear": 89,
            "CDLKICKING_Bull": 96,
            "CDLKICKING_Bear": 102,
            "CDLDRAGONFLYDOJI_Bull": 98,
            "CDLDRAGONFLYDOJI_Bear": 98,
            "CDLCONCEALBABYSWALL_Bull": 101,
            "CDLCONCEALBABYSWALL_Bear": 101,
            "CDL3STARSINSOUTH_Bull": 103,
            "CDL3STARSINSOUTH_Bear": 103,
            "CDLDOJI_Bull": 104,
            "CDLDOJI_Bear": 104
            }

        df['candlestick_pattern'] = np.nan
        df['candlestick_match_count'] = np.nan
        for index, row in df.iterrows():

            # no pattern found
            if len(row[candle_names]) - sum(row[candle_names] == 0) == 0:
                df.loc[index,'candlestick_pattern'] = "NO_PATTERN"
                df.loc[index, 'candlestick_match_count'] = 0
            # single pattern found
            elif len(row[candle_names]) - sum(row[candle_names] == 0) == 1:
                # bull pattern 100 or 200
                if any(row[candle_names].values > 0):
                    pattern = list(compress(row[candle_names].keys(), row[candle_names].values != 0))[0] + '_Bull'
                    df.loc[index, 'candlestick_pattern'] = pattern
                    df.loc[index, 'candlestick_match_count'] = 1
                # bear pattern -100 or -200
                else:
                    pattern = list(compress(row[candle_names].keys(), row[candle_names].values != 0))[0] + '_Bear'
                    df.loc[index, 'candlestick_pattern'] = pattern
                    df.loc[index, 'candlestick_match_count'] = 1
            # multiple patterns matched -- select best performance
            else:
                # filter out pattern names from bool list of values
                patterns = list(compress(row[candle_names].keys(), row[candle_names].values != 0))
                container = []
                for pattern in patterns:
                    if row[pattern] > 0:
                        container.append(pattern + '_Bull')
                    else:
                        container.append(pattern + '_Bear')
                rank_list = [candle_rankings[p] for p in container]
                if len(rank_list) == len(container):
                    rank_index_best = rank_list.index(min(rank_list))
                    df.loc[index, 'candlestick_pattern'] = container[rank_index_best]
                    df.loc[index, 'candlestick_match_count'] = len(container)
        # clean up candle columns
        df.drop(candle_names, axis = 1, inplace = True)

        def assess_reliability(row):
            if row['candlestick_pattern'] in low_reliability:
                return 'Low Reliability'
            elif row['candlestick_pattern'] in fair_reliability:
                return 'Fair Reliability'
            elif row['candlestick_pattern'] in high_reliability:
                return 'High Reliability'
            else:
                return 'No trade call'

        low_reliability = ['CDLDOJI_Bull', 'CDLDOJI_Bear', 'CDLGAPSIDESIDEWHITE_Bull', 'CDLGAPSIDESIDEWHITE_Bear', 'CDLIDENTICAL3CROWS_Bull', 'CDLIDENTICAL3CROWS_Bear', 'CDLHAMMER_Bull', 'CDLHAMMER_Bear','CDLHANGINGMAN_Bull','CDLHANGINGMAN_Bear','CDLHARAMI_Bull', 'CDLHARAMI_Bear', 'CDLHARAMICROSS_Bull', 'CDLHARAMICROSS_Bear', 'CDLHIGHWAVE_Bull', 'CDLHIGHWAVE_Bear','CDLINVERTEDHAMMER_Bull', 'CDLINVERTEDHAMMER_Bear', 'CDLLONGLEGGEDDOJI_Bull', 'CDLLONGLEGGEDDOJI_Bear', 'CDLRICKSHAWMAN_Bull', 'CDLRICKSHAWMAN_Bear' ,'CDLSEPARATINGLINES_Bull', 'CDLSEPARATINGLINES_Bear','CDLSHOOTINGSTAR_Bull', 'CDLSHOOTINGSTAR_Bear', 'CDLSPINNINGTOP_Bull', 'CDLSPINNINGTOP_Bear', 'CDLTAKURI_Bull', 'CDLTAKURI_Bear', 'CDL3LINESTRIKE_Bull', 'CDL3LINESTRIKE_Bear', 'CDLTHRUSTING_Bull', 'CDLTHRUSTING_Bear']
        fair_reliability = ['CDLBREAKAWAY_Bull', 'CDLBREAKAWAY_Bear', 'CDLDOJISTAR_Bull', 'CDLDOJISTAR_Bear', 'CDLDRAGONFLYDOJI_Bull', 'CDLDRAGONFLYDOJI_Bear', 'CDLENGULFING_Bull', 'CDLENGULFING_Bear', 'CDLGRAVESTONEDOJI_Bull', 'CDLGRAVESTONEDOJI_Bear', 'CDLHIKKAKE_Bull', 'CDLHIKKAKE_Bear', 'CDLHIKKAKEMOD_Bull', 'CDLHIKKAKEMOD_Bear', 'CDLHOMINGPIGEON_Bull', 'CDLHOMINGPIGEON_Bear', 'CDLLADDERBOTTOM_Bull', 'CDLLADDERBOTTOM_Bear', 'CDLMATCHINGLOW_Bull', 'CDLMATCHINGLOW_Bear', 'CDLONNECK_Bull', 'CDLONNECK_Bear', 'CDLPIERCING_Bull', 'CDLPIERCING_Bear', 'CDLSTICKSANDWICH_Bull', 'CDLSTICKSANDWICH_Bear', 'CDLTASUKIGAP_Bull', 'CDLTASUKIGAP_Bear', 'CDLUNIQUE3RIVER_Bull', 'CDLUNIQUE3RIVER_Bear', 'CDL3STARSINSOUTH_Bull', 'CDL3STARSINSOUTH_Bear', 'CDL3WHITESOLDIERS_Bull', 'CDL3WHITESOLDIERS_Bear', 'CDLTRISTAR_Bull', 'CDLTRISTAR_Bear', 'CDL2CROWS_Bull', 'CDL2CROWS_Bear', 'XSIDEGAP3METHODS_Bull', 'XSIDEGAP3METHODS_Bear']
        high_reliability = ['CDLCLOSINGMARUBOZU_Bull', 'CDLCLOSINGMARUBOZU_Bear', 'CDLCONCEALBABYSWALL_Bull', 'CDLCONCEALBABYSWALL_Bear', 'CDLDARKCLOUDCOVER_Bull', 'CDLDARKCLOUDCOVER_Bear', 'CDLEVENINGDOJISTAR_Bull', 'CDLEVENINGDOJISTAR_Bear', 'CDLEVENINGSTAR_Bull', 'CDLEVENINGSTAR_Bear', 'CDLRISEFALL3METHODS_Bull', 'CDLRISEFALL3METHODS_Bear', 'CDLIDENTICAL3CROWS_Bull', 'CDLIDENTICAL3CROWS_Bear', 'CDLINNECK_Bull', 'CDLINNECK_Bear', 'CDLKICKING_Bull', 'CDLKICKING_Bear', 'CDLMARUBOZU_Bull', 'CDLMARUBOZU_Bear', 'CDLMATHOLD_Bull', 'CDLMATHOLD_Bear', 'CDLMORNINGSTAR_Bull', 'CDLMORNINGSTAR_Bull','CDLMORNINGDOJSTAR_Bull', 'CDLMORNINGSTAR_Bull', 'CDL3BLACKCROWS_Bull', 'CDL3BLACKCROWS_Bear', 'CDL3INSIDE_Bull', 'CDL3INSIDE_Bear', 'CDL3OUTSIDE_Bull', 'CDL3OUTSIDE_Bear', 'CDLUPSIDEGAP2CROWS_Bull', 'CDLUPSIDEGAP2CROWS_Bear']
        df['reliability'] = df.apply(assess_reliability, axis=1)

        last_pattern = df['candlestick_pattern'].iloc[-1]  # CDL3INSIDE_Bull

        if last_pattern == 'NO_PATTERN':
            bull_bear = 'No pattern'
            pattern_name = ' '
            updown=0
        else:
            bull_bear = last_pattern.split('_')[-1]  # Bull
            pattern_name = last_pattern[3:-len(bull_bear)-1]  # 3INSIDE
            if bull_bear=="Bull":
                updown=1
            else:
                updown=-1  
            bull_bear += 'ish'

        reliability= df['reliability'].iloc[-1]

        if reliability=="Low Reliability":
            rel=0.4
        elif reliability=="Fair Reliability":
            rel=0.7
        elif reliability=="High Reliability":
            rel=1
        else:
            rel=0    

        finalcd=0.3*rel*updown
        final=finalml+finalsenti+finalcd
        if(final<-0.2):
            res="SELL"
        elif(final>0.2):
            res="BUY"
        else:
            res="NO TRADE CALL"
        result = {'pred_dict': pred_dict1, 'start_date': next_timestamp_str,'positive':positive_percent,'negative':negative_percent,'neutral':neutral_percent, 'mltrend': trendstr,'cdltrend':bull_bear,'cdlname':pattern_name,'cdlrel':reliability,'tradecall':res}

        # Return the prediction and end_date as JSON
        return jsonify(result)

# add the Prediction resource to the API
api.add_resource(Prediction, '/predict')


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001, debug=True)
    print('Server is running...')

