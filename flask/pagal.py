import pandas as pd
import yfinance as yf
import talib
from talib import abstract
import numpy as np
from itertools import compress

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
last_pattern

if last_pattern == 'NO_PATTERN':
    bull_bear = 'No pattern'
    pattern_name = ' '
else:
    bull_bear = last_pattern.split('_')[-1]  # Bull
    pattern_name = last_pattern[3:-len(bull_bear)-1]  # 3INSIDE
    bull_bear += 'ish'

bull_bear    

reliability= df['reliability'].iloc[-1]
print(reliability)