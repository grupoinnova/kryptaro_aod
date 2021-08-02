import matplotlib.pyplot as plt
import requests
import sys
import json
import datetime
import numpy as np
import matplotlib.dates as mdates
import matplotlib.cbook as cbook

reference = "c4b99fe8-efd2-11eb-824a-adea3b7ebdfa"
orders = requests.get(f'https://api.kryptaro.com/v1/sm/tfs/orders?reference={reference}')
orders = orders.json()

change_validation = 1
symbol = 'ethereum'

if len(orders) > 0:
    if orders[0]['symbol'] == 'ethusdt':
        symbol = 'ethereum'

    if orders[0]['symbol'] == 'adausdt':
        symbol = 'cardano'

    if orders[0]['symbol'] == 'bnbusdt':
        symbol = 'binance-coin'

    if orders[0]['symbol'] == 'xrpusdt':
        symbol = 'xrp'

    if orders[0]['symbol'] == 'dogeusdt':
        symbol = 'doge-coin'

    if orders[0]['symbol'] == 'ltcusdt':
        symbol = 'litecoin'

    if orders[0]['symbol'] == 'btcusdt':
        symbol = 'bitcoin'

response = requests.get(f'https://api.kryptaro.com/v1/sm/tfs/get?symbol={symbol}&tf_range={60*15}&tf_mode=0&origin=binance_tfs&from_date=2021-07-24')
response = response.json()

orders_buy = []
orders_sell = []

for order in orders:
    if order['method'] == 1:
        orders_buy.append(order['date_open'][:-3])

    if order['method'] == 2:
        orders_sell.append(order['date_open'][:-3])

price_x1 = []
price_y1 = []

ema_60_x1 = []
ema_60_y1 = []

ema_200_x1 = []
ema_200_y1 = []

volumen_x1 = []
volumen_y1 = []

markers_up_json = []
markers_down_json = []

markers_up = []
markers_down = []

tf_prev = None
tf_prev_date = 0
tf_validation = 90

markers_orders_buy = []
markers_orders_sell = []

idx = 0
for tf in response:
    idx += 1
    # Price
    price_y1.append(tf['price_close'])
    price_x1.append(tf['to'])

    # Emas
    price_ema_60 = None
    if 'price_ema_60' in tf:
        if tf['price_ema_60'] != 0:
            price_ema_60 = tf['price_ema_60']

    price_ema_200 = None
    if 'price_ema_200' in tf:
        if tf['price_ema_200'] != 0:
            price_ema_200 = tf['price_ema_200']

    ema_60_y1.append(price_ema_60)
    ema_60_x1.append(tf['to'])

    ema_200_y1.append(price_ema_200)
    ema_200_x1.append(tf['to'])

    if not(tf_prev):
        tf_prev = tf
    
    prev_to = datetime.datetime.strptime(tf_prev['to'].split('.')[0].replace('+00:00', ''), "%Y-%m-%dT%H:%M:%S")
    current_to = datetime.datetime.strptime(tf['to'].split('.')[0].replace('+00:00', ''), "%Y-%m-%dT%H:%M:%S")

    if prev_to < (current_to - datetime.timedelta(minutes = tf_validation)):
        tf_prev = tf
        print(prev_to, current_to)

    change = ((tf['price_close'] - tf_prev['price_close']) / tf_prev['price_close'])*100

    if change >= change_validation:
        markers_up.append(tf['price_close'])
        markers_up_json.append(tf['to'])
    else:
        markers_up.append(None)

    if change <= change_validation*-1:
        markers_down.append(tf['price_close'])
        markers_down_json.append(tf['to'])
    else:
        markers_down.append(None)

    c__date = str(tf['to'].split('.')[0][:-3]).replace('+00:00', '')

    if c__date in orders_buy:
        plt.axvline(x = idx, color = 'g', linestyle=':', linewidth = 0.5)
        markers_orders_buy.append(tf['price_close'])
    else:
        markers_orders_buy.append(None)

    if c__date in orders_sell:
        plt.axvline(x = idx, color = 'r', linestyle=':', linewidth = 0.5)
        markers_orders_sell.append(tf['price_close'])
    else:
        markers_orders_sell.append(None)

date_format = "%Y-%m-%dT%H:%M:%S"
a = datetime.datetime.strptime(response[0]['to'].split('.')[0].replace('+00:00', ''), date_format)
b = datetime.datetime.strptime(response[-1]['to'].split('.')[0].replace('+00:00', ''), date_format)
delta = b - a

print(delta.days)
print(orders_buy)

with open('markers_up_json.txt', 'w') as outfile:
    json.dump(markers_up_json, outfile)

with open('markers_down_json.txt', 'w') as outfile:
    json.dump(markers_down_json, outfile)

plt.plot(markers_up, marker='+', markersize=12, color='g')
plt.plot(markers_down, marker='+', markersize=12, color='r')

plt.plot(price_x1, price_y1, label = "price")
plt.plot(ema_60_x1, ema_60_y1, label = "ema_60")
plt.plot(ema_200_x1, ema_200_y1, label = "ema_200")

plt.gca().set_axis_off()
plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0, 
            hspace = 0, wspace = 0)
plt.margins(0,0)
plt.gca().xaxis.set_major_locator(plt.NullLocator())
plt.gca().yaxis.set_major_locator(plt.NullLocator())

plt.xticks([])
plt.legend()
plt.show()
