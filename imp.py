import matplotlib.pyplot as plt
import numpy as np
import requests
import sys
import json
import datetime
import math
import hashlib
import threading

from requests.packages.urllib3.exceptions import InsecureRequestWarning
requests.packages.urllib3.disable_warnings(InsecureRequestWarning)

from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import PolynomialFeatures

class IMP(object):
    def __init__(self, symbol, timeframe, backtest):
        self.tfs = requests.get(f'https://api.kryptaro.com/v1/sm/tfs/get?symbol={symbol}&tf_range={timeframe}&tf_mode=0&origin=binance_tfs&from_date=2021-07-24')
        self.tfs = self.tfs.json()

        self.backtest_orders = requests.get(f'https://api.kryptaro.com/v1/sm/tfs/orders?reference={backtest}')
        self.backtest_orders = self.backtest_orders.json()

        date_format = "%Y-%m-%dT%H:%M:%S"
        a = datetime.datetime.strptime(self.tfs[0]['to'].split('.')[0].replace('+00:00', ''), date_format)
        b = datetime.datetime.strptime(self.tfs[-1]['to'].split('.')[0].replace('+00:00', ''), date_format)
        delta = b - a

        print('Days traveled:', delta.days)

        self.backtest_orders_buys = []
        self.backtest_orders_sells = []

        for order in self.backtest_orders:
            if order['method'] == 1:
                self.backtest_orders_buys.append(order['date_open'][:-3])

            if order['method'] == 2:
                self.backtest_orders_sells.append(order['date_open'][:-3])

        # Datos para la grafica:
        self.chart_data_x = []

        self.chart_data_prices_y = []
        self.chart_data_ema60_y = []
        self.chart_data_ema200_y = []
        self.chart_data_market_sentiment_y = []

        # Loop del tfs
        self.plt = plt
        self.initialize_indicators()

        # Variables para la IA
        self.training_clf = None
        self.training_inputs = []
        self.training_outputs = []
        self.training_max_predict = 0
        self.training_min_predict = 0

        self.training_avg_up_outputs = 0
        self.training_avg_down_outputs = 0
        self.training_data_for_trained = 80
        self.training_data_for_test = 20

        # self.training_orders = []
        # self.training_orders_closed = []

        self.training_block_to_validate = 4
        self.training_blocks_history = []

        self.positives_data = []
        self.negatives_data = []

        self.test_block_to_validate = 4
        self.test_blocks_history = []
        self.test_orders = []
        self.test_orders_closed = []

        self.run_tfs(generate_data_for_chart= True)
        
        self.run_tfs(
            on_finish_calculating_the_indicators= self.test_generate_inputs_outputs, 
            data_for_trained=self.training_data_for_trained,
        )
        
        # 'linearregression',
        # 'scaled_linear_regression',
        # 'unscaled_ridge_regression',
        # 'scaled_linear_regression2',
        # 'unscaled_decision_tree_regressor',
        # 'unscaled_gaussian_naive_bayes',

        self.run_trained(
            method= 'unscaled_decision_tree_regressor',
            datasets= (self.training_inputs, self.training_outputs)
        )

        self.run_tfs(
            on_finish_calculating_the_indicators= self.test_predict_data,
            data_for_test=self.training_data_for_test,
        )

        print(f"Day for trained: {int(delta.days*(self.training_data_for_trained/100))}")
        print(f"Day for test: {int(delta.days*(self.training_data_for_test/100))}")

        if len(self.test_orders + self.test_orders_closed) > 0:
            print(f"Open orders: {len(self.test_orders)}")
            print(f"Orders closed: {len(self.test_orders_closed)}")

            w = 0
            p = 0
            for order in self.test_orders_closed + self.test_orders:
                if order['profits'] > 0:
                    w += 1
                p += order['profits']

            print(f"Effectiveness: {(w*100)/len(self.test_orders_closed + self.test_orders)}")
            print(f"Profits: {p}")
            print(f"Max predict: {self.training_max_predict}")
            print(f"Min predict: {self.training_min_predict}")
            print(f"Max output: {self.training_avg_up_outputs}")
            print(f"Min output: {self.training_avg_down_outputs}")

        self.run_chart_plt()

    def initialize_indicators(self):
        # Indicadores
        # self._VC (UP)
        self._VC_UP = 0
        self._VC_UP_TFS = []
        self._VC_UP_G_TFS = []
        self._VC_UP_CHANGE = 0
        self._VC_UP_G_CONFIRM_ACC = 0
        self._VC_UP_IMP = 0

        self._VC_UP_N_CONFIRMATION = 1
        self._VC_UP_G_CONFIRM_TRIGGER = 1
        # ... 

        # self._VC (DOWN)
        self._VC_DOWN = 0
        self._VC_DOWN_TFS = []
        self._VC_DOWN_G_TFS = []
        self._VC_DOWN_CHANGE = 0
        self._VC_DOWN_G_CONFIRM_ACC = 0
        self._VC_DOWN_IMP = 0

        self._VC_DOWN_N_CONFIRMATION = 1
        self._VC_DOWN_G_CONFIRM_TRIGGER = 2
        # ... 
        # .....................

        # self._TC (UP)
        self._TC_UP = 0
        self._TC_UP_TFS = []
        self._TC_UP_G_TFS = []
        self._TC_UP_CHANGE = 0
        self._TC_UP_G_CONFIRM_ACC = 0
        self._TC_UP_IMP = 0

        self._TC_UP_N_CONFIRMATION = 1
        self._TC_UP_G_CONFIRM_TRIGGER = 1
        # ... 

        # self._TC (DOWN)
        self._TC_DOWN = 0
        self._TC_DOWN_TFS = []
        self._TC_DOWN_G_TFS = []
        self._TC_DOWN_CHANGE = 0
        self._TC_DOWN_G_CONFIRM_ACC = 0
        self._TC_DOWN_IMP = 0

        self._TC_DOWN_N_CONFIRMATION = 1
        self._TC_DOWN_G_CONFIRM_TRIGGER = 1
        # ... 
        # .....................

        # _IC (UP)
        self._IC_UP = 0
        self._IC_UP_TFS = []
        self._IC_UP_G_TFS = []
        self._IC_UP_CHANGE = 0
        self._IC_UP_G_CONFIRM_ACC = 0
        self._IC_UP_IMP = 0

        self._IC_UP_N_CONFIRMATION = 1
        self._IC_UP_G_CONFIRM_TRIGGER = 2
        # ... 

        # _IC (DOWN)
        self._IC_DOWN = 0
        self._IC_DOWN_TFS = []
        self._IC_DOWN_G_TFS = []
        self._IC_DOWN_CHANGE = 0
        self._IC_DOWN_G_CONFIRM_ACC = 0
        self._IC_DOWN_IMP = 0

        self._IC_DOWN_N_CONFIRMATION = 1
        self._IC_DOWN_G_CONFIRM_TRIGGER = 2
        # ... 
        # .....................

        # self._WHALE (UP)
        self._WHALE_UP = 0
        self._WHALE_UP_TFS = []
        self._WHALE_UP_G_TFS = []
        self._WHALE_UP_CHANGE = 0
        self._WHALE_UP_G_CONFIRM_ACC = 0
        self._WHALE_UP_IMP = 0

        self._WHALE_UP_LVL = 3
        self._WHALE_UP_N_CONFIRMATION = 1
        self._WHALE_UP_G_CONFIRM_TRIGGER = 1
        # ... 

        # self._WHALE (DOWN)
        self._WHALE_DOWN = 0
        self._WHALE_DOWN_TFS = []
        self._WHALE_DOWN_G_TFS = []
        self._WHALE_DOWN_CHANGE = 0
        self._WHALE_DOWN_G_CONFIRM_ACC = 0
        self._WHALE_DOWN_IMP = 0

        self._WHALE_DOWN_LVL = 3
        self._WHALE_DOWN_N_CONFIRMATION = 1
        self._WHALE_DOWN_G_CONFIRM_TRIGGER = 1
        # ... 
        # .....................

    def initialize_input(self, tf, _tf):
        return [
            self._VC_UP-self._VC_DOWN,
            self._VC_UP_IMP - (self._VC_DOWN_IMP*-1),
            self._TC_UP-self._TC_DOWN,
            self._TC_UP_IMP - self._TC_DOWN_IMP,
            self._IC_UP-self._IC_DOWN,
            self._IC_UP_IMP - (self._IC_DOWN_IMP*-1),
            self._WHALE_UP-self._WHALE_DOWN,
            self._WHALE_UP_IMP - self._WHALE_DOWN_IMP,
        ]
    
    def test_generate_inputs_outputs(self, idx, tf, _tf):
        
        if 'price_max' not in _tf:
            return None

        avg_price = ((tf['price_max'] + tf['price_min'])/2)

        i = self.initialize_input(tf= tf, _tf= _tf)

        # ---------------------------------------
        # - Esta forma de manejar los datos es con tiempo maximo de abierta la orden 1h
        # ---------------------------------------
        self.training_blocks_history.append({
            "price": avg_price,
            "input": i,
        })

        if len(self.training_blocks_history) == self.training_block_to_validate:

            input_sum = self.training_blocks_history[0]['input']
            profits = self.training_blocks_history[-1]['price'] - self.training_blocks_history[0]['price']
            profits /= self.training_blocks_history[0]['price']
            profits *= 100

            for input_data in self.training_blocks_history[1:]:
                x_idx = -1
                for x in input_data['input']:
                    x_idx += 1
                    input_sum[x_idx] += x
        
            profits_rounded = round(profits)

            if profits_rounded > 0 and profits_rounded <= 1:
                profits_rounded = 1
            elif profits_rounded >= -1 and profits_rounded <= 0:
                profits_rounded = -1

            if profits_rounded > 0:
                if profits_rounded > self.training_avg_up_outputs:
                    self.training_avg_up_outputs = profits_rounded
            
            if profits_rounded < 0:
                if profits_rounded < self.training_avg_down_outputs:
                    self.training_avg_down_outputs = profits_rounded

            self.training_inputs.append(input_sum)
            self.training_outputs.append(profits_rounded)
            self.training_blocks_history = []

    def test_predict_data(self, idx, tf, _tf):

        if 'price_max' not in _tf:
            return None
        
        avg_price = (tf['price_max'] + tf['price_min'])/2

        i = self.initialize_input(tf= tf, _tf= _tf)

        # ---------------------------------------
        # - Esta forma de manejar los datos es con tiempo maximo de abierta la orden 1h
        # ---------------------------------------
        self.test_blocks_history.append({
            "price": avg_price,
            "input": i,
        })

        input_sum = self.test_blocks_history[0]['input']
        profits = self.test_blocks_history[-1]['price'] - self.test_blocks_history[0]['price']
        profits /= self.test_blocks_history[0]['price']
        profits *= 100

        for input_data in self.test_blocks_history[1:]:
            x_idx = -1
            for x in input_data['input']:
                x_idx += 1
                input_sum[x_idx] += x
        
        _PREDICT = self.run_predict(set= input_sum)[0]
        
        # if _PREDICT >= self.training_avg_up_outputs*0.50:
        #     self.test_orders.append({
        #         "price": avg_price,
        #         "method": 1,
        #         "profits": 0,
        #         "time": tf['to'].split('.')[0].replace('+00:00', ''),
        #     })

        if _PREDICT <= self.training_avg_down_outputs*0.50:
            # if self._PRICE_STATUS == 0:
            self.test_orders.append({
                "price": avg_price,
                "method": 2,
                "profits": 0,
                "time": tf['to'].split('.')[0].replace('+00:00', ''),
            })

        if _PREDICT > self.training_max_predict:
            self.training_max_predict = _PREDICT

        if _PREDICT < self.training_min_predict:
            self.training_min_predict = _PREDICT

        r = []
        for order in self.test_orders:
            if order["method"] == 1: profits = ((avg_price - order['price'])/order['price'])*100
            if order["method"] == 2: profits = ((order['price'] - avg_price)/avg_price)*100
            
            order['profits'] = profits

            max_date_to_open = datetime.datetime.strptime(order['time'], "%Y-%m-%dT%H:%M:%S")
            max_date_to_open = max_date_to_open + datetime.timedelta(hours = 1)
            current_time = datetime.datetime.strptime(tf['to'].split('.')[0].replace('+00:00', ''), "%Y-%m-%dT%H:%M:%S")

            if current_time > max_date_to_open:
                self.plt.axvline(x = idx, color = 'r' if profits < -1 else 'g', linestyle=':', linewidth = 1)
                self.test_orders_closed.append(order)
            else: r.append(order)
        self.test_orders = r
        
        # Limpieza de variable
        if len(self.test_blocks_history) == self.test_block_to_validate:
            self.test_blocks_history = []

    def run_trained(self, method='linearregression', datasets= ([], [])):

        if method == 'linearregression':
            train_based_on_method = LinearRegression()

        if method == 'scaled_linear_regression':
            train_based_on_method = make_pipeline(StandardScaler(), LinearRegression())

        if method == 'unscaled_ridge_regression':
            train_based_on_method = Ridge()

        if method == 'scaled_linear_regression2':
            train_based_on_method = make_pipeline(StandardScaler(), Ridge())

        if method == 'unscaled_decision_tree_regressor':
            train_based_on_method = DecisionTreeRegressor()

        if method == 'unscaled_gaussian_naive_bayes':
            train_based_on_method = GaussianNB()

        self.training_clf = train_based_on_method

        inputs, outputs = datasets

        if len(inputs) > 0:

            X_train, X_test, y_train, y_test = train_test_split(
                inputs, 
                outputs, 
                test_size=0.1, 
                random_state=42
            )

            cv = cross_val_score(self.training_clf, X_test, y_test)

            self.training_clf.fit(inputs, outputs)

            score = self.training_clf.score(X_test, y_test)
            
            print(f'Method: {method}')
            print(f'Accuracy (cross_val_score): {cv.mean()}')
            print(f'Accuracy (score): {score}')
            print(f'I/O (qty): {len(inputs)}')
            print(f'\n')

    def run_predict(self, set=[]):
        return self.training_clf.predict([set])

    def run_tfs(
        self, 
        on_finish_calculating_the_indicators= None, 
        generate_data_for_chart = False,
        data_for_trained= None,
        data_for_test= None,
    ):
        idx = -1
        chart_day_separator = 0
        _tf = {}

        rows = self.tfs
        
        for tf in rows:
            idx += 1

            avg_price = (tf['price_max'] + tf['price_min'])/2

            # ----------------------------------------------------------
            # DATOS RELEVANTE PARA ARMADO DE LA GRAFICA
            # ----------------------------------------------------------
            chart_day_separator += 1

            if generate_data_for_chart: self.chart_data_x.append(tf['to'])   
            if generate_data_for_chart: self.chart_data_prices_y.append(avg_price)

            if chart_day_separator == 4*24:
                chart_day_separator = 0
                self.plt.axvline(x = idx, color = '#F0F0F0', linewidth = 5)

            # Emas
            price_ema_60 = None
            if 'price_ema_60' in tf:
                if tf['price_ema_60'] != 0:
                    price_ema_60 = tf['price_ema_60']

            price_ema_200 = None
            if 'price_ema_200' in tf:
                if tf['price_ema_200'] != 0:
                    price_ema_200 = tf['price_ema_200']

            if generate_data_for_chart: self.chart_data_ema60_y.append(price_ema_60)
            if generate_data_for_chart: self.chart_data_ema200_y.append(price_ema_200)

            market_sentiment_on_6h = None
            if 'market_sentiment_on_6h' in tf:
                market_sentiment_on_6h = ((tf['market_sentiment_on_6h']/100)+1) * avg_price

            if generate_data_for_chart: self.chart_data_market_sentiment_y.append(market_sentiment_on_6h)
            # ----------------------------------------------------------
            # /// DATOS RELEVANTE PARA ARMADO DE LA GRAFICA
            # ----------------------------------------------------------

            # ----------------------------------------------------------
            # ARMADO DE INDICADORES
            # ----------------------------------------------------------
            tf['price_avg'] = avg_price

            tf['invested_qty_ontf'] = tf['qty'] * avg_price
            tf['invested_qty_in_ontf'] = tf['qty_in'] * avg_price
            tf['invested_qty_out_ontf'] = tf['qty_out'] * avg_price
            tf['invested_qty_vs_ontf'] = tf['invested_qty_in_ontf'] - tf['invested_qty_out_ontf']

            tf['invested_ticks_ontf'] = tf['ticks']
            tf['invested_ticks_in_ontf'] = tf['ticks_in']
            tf['invested_ticks_out_ontf'] = tf['ticks_out']
            tf['invested_ticks_vs_ontf'] = tf['invested_ticks_in_ontf'] - tf['invested_ticks_out_ontf']

            tf['invested_ticks_1h'] = 1

            if idx > 4:
                for t in self.tfs[(idx-4):]:
                    tf['invested_ticks_1h'] += tf['invested_ticks_vs_ontf']

            tf['invested_buys_range'] = tf['investors']['resume_range']['buys_history_invested']
            tf['invested_buys_1h'] = tf['investors']['resume_1h']['buys_history_invested']

            tf['invested_sells_range'] = tf['investors']['resume_range']['sells_history_invested']
            tf['invested_sells_1h'] = tf['investors']['resume_1h']['sells_history_invested']

            tf['invested_1h_total'] = tf['invested_buys_1h'] + tf['invested_sells_1h']
            tf['invested_1h_vs'] = tf['invested_buys_1h'] - tf['invested_sells_1h']

            tf['invested_range_total'] = tf['invested_buys_range'] + tf['invested_sells_range']
            tf['invested_range_vs'] = tf['invested_buys_range'] - tf['invested_sells_range']

            tf['invested_by_lvl__all_vs']  = 0 
            tf['invested_by_lvl__all_vs'] += tf['investors']['totaled']['buys']['1']['invested'] - tf['investors']['totaled']['sells']['1']['invested']
            tf['invested_by_lvl__all_vs'] += tf['investors']['totaled']['buys']['2']['invested'] - tf['investors']['totaled']['sells']['2']['invested']
            tf['invested_by_lvl__all_vs'] += tf['investors']['totaled']['buys']['3']['invested'] - tf['investors']['totaled']['sells']['3']['invested']
            tf['invested_by_lvl__all_vs'] += tf['investors']['totaled']['buys']['4']['invested'] - tf['investors']['totaled']['sells']['4']['invested']
            tf['invested_by_lvl__all_vs'] += tf['investors']['totaled']['buys']['5']['invested'] - tf['investors']['totaled']['sells']['5']['invested']

            tf['invested_by_lvl__all_total']  = 0 
            tf['invested_by_lvl__all_total'] += tf['investors']['totaled']['buys']['1']['invested'] + tf['investors']['totaled']['sells']['1']['invested']
            tf['invested_by_lvl__all_total'] += tf['investors']['totaled']['buys']['2']['invested'] + tf['investors']['totaled']['sells']['2']['invested']
            tf['invested_by_lvl__all_total'] += tf['investors']['totaled']['buys']['3']['invested'] + tf['investors']['totaled']['sells']['3']['invested']
            tf['invested_by_lvl__all_total'] += tf['investors']['totaled']['buys']['4']['invested'] + tf['investors']['totaled']['sells']['4']['invested']
            tf['invested_by_lvl__all_total'] += tf['investors']['totaled']['buys']['5']['invested'] + tf['investors']['totaled']['sells']['5']['invested']

            self._PRICE_STATUS = None
            self._PRICE_DISTANCE = None
            self._MARKET_SENTIMENT = tf['market_sentiment_on_6h']
            
            if 'price_ema_200' in tf and 'price_ema_60' in tf:
                if tf['price_ema_200'] != 0 and tf['price_ema_60'] != 0:

                    if tf['price_ema_60'] > tf['price_ema_200'] and avg_price > tf['price_ema_60']:
                        self._PRICE_STATUS = 1

                    if tf['price_ema_60'] < tf['price_ema_200'] and avg_price < tf['price_ema_60']:
                        self._PRICE_STATUS = 0

                    self._PRICE_DISTANCE = ((avg_price - tf['price_ema_60'])/tf['price_ema_60'])*100
            
            if _tf:
                # self._VC (UP) -------------------------------------------------------
                if (tf['invested_qty_vs_ontf'] > 0 and _tf['invested_qty_vs_ontf'] > 0) and (tf['invested_qty_vs_ontf'] > _tf['invested_qty_vs_ontf']): 
                    self._VC_UP += 1
                    self._VC_UP_TFS.append(tf)
                    self._VC_UP_G_TFS.append(tf)
                else:
                    self._VC_UP_G_CONFIRM_ACC = 0
                    self._VC_UP = 0
                    self._VC_UP_TFS = []      

                if len(self._VC_UP_TFS) > 0 and self._VC_UP >= 1:
                    if self._VC_UP_N_CONFIRMATION == 1:
                        self._VC_UP_CHANGE = self._VC_UP_TFS[-1]['price_avg'] - self._VC_UP_TFS[0]['price_avg']
                        self._VC_UP_CHANGE /= self._VC_UP_TFS[0]['price_avg']
                        self._VC_UP_CHANGE *= 100

                        self._VC_UP_IMP = 0
                        for t in self._VC_UP_TFS: self._VC_UP_IMP += t['invested_qty_vs_ontf']
                        self._VC_UP_IMP = (self._VC_UP_IMP * 100) / tf['invested_1h_total']
                        
                    else:
                        self._VC_UP_CHANGE = self._VC_UP_G_TFS[-1]['price_avg'] - self._VC_UP_G_TFS[0]['price_avg']
                        self._VC_UP_CHANGE /= self._VC_UP_G_TFS[0]['price_avg']
                        self._VC_UP_CHANGE *= 100

                        self._VC_UP_IMP = 0
                        for t in self._VC_UP_G_TFS: self._VC_UP_IMP += t['invested_qty_vs_ontf']
                        self._VC_UP_IMP = (self._VC_UP_IMP * 100) / tf['invested_1h_total']

                    if self._VC_UP >= self._VC_UP_N_CONFIRMATION:
                        self._VC_UP_G_CONFIRM_ACC += 1     

                if self._VC_UP_G_CONFIRM_ACC >= self._VC_UP_G_CONFIRM_TRIGGER:
                    self._VC_UP_G_CONFIRM_ACC = 0
                    self._VC_UP_G_TFS = []

                # self._VC (DOWN)
                if (tf['invested_qty_vs_ontf'] < 0 and _tf['invested_qty_vs_ontf'] < 0) and (tf['invested_qty_vs_ontf'] < _tf['invested_qty_vs_ontf']): 
                    self._VC_DOWN += 1
                    self._VC_DOWN_TFS.append(tf)
                    self._VC_DOWN_G_TFS.append(tf)
                else:
                    self._VC_DOWN_G_CONFIRM_ACC = 0
                    self._VC_DOWN = 0
                    self._VC_DOWN_TFS = []      

                if len(self._VC_DOWN_TFS) > 0 and self._VC_DOWN >= 1:
                    if self._VC_DOWN_N_CONFIRMATION == 1:
                        self._VC_DOWN_CHANGE = self._VC_DOWN_TFS[-1]['price_avg'] - self._VC_DOWN_TFS[0]['price_avg']
                        self._VC_DOWN_CHANGE /= self._VC_DOWN_TFS[0]['price_avg']
                        self._VC_DOWN_CHANGE *= 100

                        self._VC_DOWN_IMP = 0
                        for t in self._VC_DOWN_TFS: self._VC_DOWN_IMP += t['invested_qty_vs_ontf']
                        self._VC_DOWN_IMP = (self._VC_DOWN_IMP * 100) / tf['invested_1h_total']
                        
                    else:
                        self._VC_DOWN_CHANGE = self._VC_DOWN_G_TFS[-1]['price_avg'] - self._VC_DOWN_G_TFS[0]['price_avg']
                        self._VC_DOWN_CHANGE /= self._VC_DOWN_G_TFS[0]['price_avg']
                        self._VC_DOWN_CHANGE *= 100

                        self._VC_DOWN_IMP = 0
                        for t in self._VC_DOWN_G_TFS: self._VC_DOWN_IMP += t['invested_qty_vs_ontf']
                        self._VC_DOWN_IMP = (self._VC_DOWN_IMP * 100) / tf['invested_1h_total']

                    if self._VC_DOWN >= self._VC_DOWN_N_CONFIRMATION:
                        self._VC_DOWN_G_CONFIRM_ACC += 1     

                if self._VC_DOWN_G_CONFIRM_ACC >= self._VC_DOWN_G_CONFIRM_TRIGGER:
                    self._VC_DOWN_G_CONFIRM_ACC = 0
                    self._VC_DOWN_G_TFS = []

                # ... self._VC (UP)

                # self._TC (UP) -------------------------------------------------------
                if (tf['invested_ticks_vs_ontf'] > 0 and _tf['invested_ticks_vs_ontf'] > 0) and (tf['invested_ticks_vs_ontf'] > _tf['invested_ticks_vs_ontf']): 
                    self._TC_UP += 1
                    self._TC_UP_TFS.append(tf)
                    self._TC_UP_G_TFS.append(tf)
                else:
                    self._TC_UP_G_CONFIRM_ACC = 0
                    self._TC_UP = 0
                    self._TC_UP_TFS = []      

                if len(self._TC_UP_TFS) > 0 and self._TC_UP >= 1:
                    if self._TC_UP_N_CONFIRMATION == 1:
                        self._TC_UP_CHANGE = self._TC_UP_TFS[-1]['price_avg'] - self._TC_UP_TFS[0]['price_avg']
                        self._TC_UP_CHANGE /= self._TC_UP_TFS[0]['price_avg']
                        self._TC_UP_CHANGE *= 100

                        self._TC_UP_IMP = 0
                        for t in self._TC_UP_TFS: self._TC_UP_IMP += t['invested_ticks_vs_ontf']
                        self._TC_UP_IMP = (self._TC_UP_IMP * 100) / tf['invested_ticks_1h']
                        
                    else:
                        self._TC_UP_CHANGE = self._TC_UP_G_TFS[-1]['price_avg'] - self._TC_UP_G_TFS[0]['price_avg']
                        self._TC_UP_CHANGE /= self._TC_UP_G_TFS[0]['price_avg']
                        self._TC_UP_CHANGE *= 100

                        self._TC_UP_IMP = 0
                        for t in self._TC_UP_G_TFS: self._TC_UP_IMP += t['invested_ticks_vs_ontf']
                        self._TC_UP_IMP = (self._TC_UP_IMP * 100) / tf['invested_ticks_1h']

                    if self._TC_UP >= self._TC_UP_N_CONFIRMATION:
                        self._TC_UP_G_CONFIRM_ACC += 1     

                if self._TC_UP_G_CONFIRM_ACC >= self._TC_UP_G_CONFIRM_TRIGGER:
                    self._TC_UP_G_CONFIRM_ACC = 0
                    self._TC_UP_G_TFS = []

                # self._TC (DOWN)
                if (tf['invested_ticks_vs_ontf'] < 0 and _tf['invested_ticks_vs_ontf'] < 0) and (tf['invested_ticks_vs_ontf'] < _tf['invested_ticks_vs_ontf']): 
                    self._TC_DOWN += 1
                    self._TC_DOWN_TFS.append(tf)
                    self._TC_DOWN_G_TFS.append(tf)
                else:
                    self._TC_DOWN_G_CONFIRM_ACC = 0
                    self._TC_DOWN = 0
                    self._TC_DOWN_TFS = []      

                if len(self._TC_DOWN_TFS) > 0 and self._TC_DOWN >= 1:
                    if self._TC_DOWN_N_CONFIRMATION == 1:
                        self._TC_DOWN_CHANGE = self._TC_DOWN_TFS[-1]['price_avg'] - self._TC_DOWN_TFS[0]['price_avg']
                        self._TC_DOWN_CHANGE /= self._TC_DOWN_TFS[0]['price_avg']
                        self._TC_DOWN_CHANGE *= 100

                        self._TC_DOWN_IMP = 0
                        for t in self._TC_DOWN_TFS: self._TC_DOWN_IMP += t['invested_ticks_vs_ontf']
                        self._TC_DOWN_IMP = (self._TC_DOWN_IMP * 100) / tf['invested_ticks_1h']
                        
                    else:
                        self._TC_DOWN_CHANGE = self._TC_DOWN_G_TFS[-1]['price_avg'] - self._TC_DOWN_G_TFS[0]['price_avg']
                        self._TC_DOWN_CHANGE /= self._TC_DOWN_G_TFS[0]['price_avg']
                        self._TC_DOWN_CHANGE *= 100

                        self._TC_DOWN_IMP = 0
                        for t in self._TC_DOWN_G_TFS: self._TC_DOWN_IMP += t['invested_ticks_vs_ontf']
                        self._TC_DOWN_IMP = (self._TC_DOWN_IMP * 100) / tf['invested_ticks_1h']

                    if self._TC_DOWN >= self._TC_DOWN_N_CONFIRMATION:
                        self._TC_DOWN_G_CONFIRM_ACC += 1     

                if self._TC_DOWN_G_CONFIRM_ACC >= self._TC_DOWN_G_CONFIRM_TRIGGER:
                    self._TC_DOWN_G_CONFIRM_ACC = 0
                    self._TC_DOWN_G_TFS = []

                # _IC (UP) -------------------------------------------------------
                if (tf['invested_by_lvl__all_vs'] > 0 and _tf['invested_by_lvl__all_vs'] > 0) and (tf['invested_by_lvl__all_vs'] > _tf['invested_by_lvl__all_vs']): 
                    self._IC_UP += 1
                    self._IC_UP_TFS.append(tf)
                    self._IC_UP_G_TFS.append(tf)
                else:
                    self._IC_UP_G_CONFIRM_ACC = 0
                    self._IC_UP = 0
                    self._IC_UP_TFS = []      

                if len(self._IC_UP_TFS) > 0 and self._IC_UP >= 1:
                    if self._IC_UP_N_CONFIRMATION == 1:
                        self._IC_UP_CHANGE = self._IC_UP_TFS[-1]['price_avg'] - self._IC_UP_TFS[0]['price_avg']
                        self._IC_UP_CHANGE /= self._IC_UP_TFS[0]['price_avg']
                        self._IC_UP_CHANGE *= 100

                        self._IC_UP_IMP = 0
                        for t in self._IC_UP_TFS: self._IC_UP_IMP += t['invested_by_lvl__all_vs']
                        self._IC_UP_IMP = (self._IC_UP_IMP * 100) / tf['invested_by_lvl__all_total']
                        
                    else:
                        self._IC_UP_CHANGE = self._IC_UP_G_TFS[-1]['price_avg'] - self._IC_UP_G_TFS[0]['price_avg']
                        self._IC_UP_CHANGE /= self._IC_UP_G_TFS[0]['price_avg']
                        self._IC_UP_CHANGE *= 100

                        self._IC_UP_IMP = 0
                        for t in self._IC_UP_G_TFS: self._IC_UP_IMP += t['invested_by_lvl__all_vs']
                        self._IC_UP_IMP = (self._IC_UP_IMP * 100) / tf['invested_by_lvl__all_total']

                    if self._IC_UP >= self._IC_UP_N_CONFIRMATION:
                        self._IC_UP_G_CONFIRM_ACC += 1     

                if self._IC_UP_G_CONFIRM_ACC >= self._IC_UP_G_CONFIRM_TRIGGER:
                    self._IC_UP_G_CONFIRM_ACC = 0
                    self._IC_UP_G_TFS = []

                # _IC (DOWN)
                if (tf['invested_by_lvl__all_vs'] < 0 and _tf['invested_by_lvl__all_vs'] < 0) and (tf['invested_by_lvl__all_vs'] < _tf['invested_by_lvl__all_vs']): 
                    self._IC_DOWN += 1
                    self._IC_DOWN_TFS.append(tf)
                    self._IC_DOWN_G_TFS.append(tf)
                else:
                    self._IC_DOWN_G_CONFIRM_ACC = 0
                    self._IC_DOWN = 0
                    self._IC_DOWN_TFS = []      

                if len(self._IC_DOWN_TFS) > 0 and self._IC_DOWN >= 1:
                    if self._IC_DOWN_N_CONFIRMATION == 1:
                        self._IC_DOWN_CHANGE = self._IC_DOWN_TFS[-1]['price_avg'] - self._IC_DOWN_TFS[0]['price_avg']
                        self._IC_DOWN_CHANGE /= self._IC_DOWN_TFS[0]['price_avg']
                        self._IC_DOWN_CHANGE *= 100

                        self._IC_DOWN_IMP = 0
                        for t in self._IC_DOWN_TFS: self._IC_DOWN_IMP += t['invested_by_lvl__all_vs']
                        self._IC_DOWN_IMP = (self._IC_DOWN_IMP * 100) / tf['invested_by_lvl__all_total']
                        
                    else:
                        self._IC_DOWN_CHANGE = self._IC_DOWN_G_TFS[-1]['price_avg'] - self._IC_DOWN_G_TFS[0]['price_avg']
                        self._IC_DOWN_CHANGE /= self._IC_DOWN_G_TFS[0]['price_avg']
                        self._IC_DOWN_CHANGE *= 100

                        self._IC_DOWN_IMP = 0
                        for t in self._IC_DOWN_G_TFS: self._IC_DOWN_IMP += t['invested_by_lvl__all_vs']
                        self._IC_DOWN_IMP = (self._IC_DOWN_IMP * 100) / tf['invested_by_lvl__all_total']

                    if self._IC_DOWN >= self._IC_DOWN_N_CONFIRMATION:
                        self._IC_DOWN_G_CONFIRM_ACC += 1     

                if self._IC_DOWN_G_CONFIRM_ACC >= self._IC_DOWN_G_CONFIRM_TRIGGER:
                    self._IC_DOWN_G_CONFIRM_ACC = 0
                    self._IC_DOWN_G_TFS = []

                # self._WHALE (UP) -------------------------------------------------------
                if (tf['investors']['totaled']['buys'][str(self._WHALE_UP_LVL)]['invested'] > 0 and _tf['investors']['totaled']['buys'][str(self._WHALE_UP_LVL)]['invested'] > 0) and (tf['investors']['totaled']['buys'][str(self._WHALE_UP_LVL)]['invested'] > _tf['investors']['totaled']['buys'][str(self._WHALE_UP_LVL)]['invested']): 
                    self._WHALE_UP += 1
                    self._WHALE_UP_TFS.append(tf)
                    self._WHALE_UP_G_TFS.append(tf)
                else:
                    self._WHALE_UP_G_CONFIRM_ACC = 0
                    self._WHALE_UP = 0
                    self._WHALE_UP_TFS = []      

                if len(self._WHALE_UP_TFS) > 0 and self._WHALE_UP >= 1:
                    if self._WHALE_UP_N_CONFIRMATION == 1:
                        self._WHALE_UP_CHANGE = self._WHALE_UP_TFS[-1]['price_avg'] - self._WHALE_UP_TFS[0]['price_avg']
                        self._WHALE_UP_CHANGE /= self._WHALE_UP_TFS[0]['price_avg']
                        self._WHALE_UP_CHANGE *= 100

                        self._WHALE_UP_IMP = 0
                        for t in self._WHALE_UP_TFS: self._WHALE_UP_IMP += t['investors']['totaled']['buys'][str(self._WHALE_UP_LVL)]['invested']
                        self._WHALE_UP_IMP = (self._WHALE_UP_IMP * 100) / tf['invested_by_lvl__all_total']
                        
                    else:
                        self._WHALE_UP_CHANGE = self._WHALE_UP_G_TFS[-1]['price_avg'] - self._WHALE_UP_G_TFS[0]['price_avg']
                        self._WHALE_UP_CHANGE /= self._WHALE_UP_G_TFS[0]['price_avg']
                        self._WHALE_UP_CHANGE *= 100

                        self._WHALE_UP_IMP = 0
                        for t in self._WHALE_UP_G_TFS: self._WHALE_UP_IMP += t['investors']['totaled']['buys'][str(self._WHALE_UP_LVL)]['invested']
                        self._WHALE_UP_IMP = (self._WHALE_UP_IMP * 100) / tf['invested_by_lvl__all_total']

                    if self._WHALE_UP >= self._WHALE_UP_N_CONFIRMATION:
                        self._WHALE_UP_G_CONFIRM_ACC += 1     

                if self._WHALE_UP_G_CONFIRM_ACC >= self._WHALE_UP_G_CONFIRM_TRIGGER:
                    self._WHALE_UP_G_CONFIRM_ACC = 0
                    self._WHALE_UP_G_TFS = []
                    
                # self._WHALE (DOWN) -------------------------------------------------------
                if (tf['investors']['totaled']['sells'][str(self._WHALE_DOWN_LVL)]['invested'] > 0 and _tf['investors']['totaled']['sells'][str(self._WHALE_DOWN_LVL)]['invested'] > 0) and (tf['investors']['totaled']['sells'][str(self._WHALE_DOWN_LVL)]['invested'] > _tf['investors']['totaled']['sells'][str(self._WHALE_DOWN_LVL)]['invested']): 
                    self._WHALE_DOWN += 1
                    self._WHALE_DOWN_TFS.append(tf)
                    self._WHALE_DOWN_G_TFS.append(tf)
                else:
                    self._WHALE_DOWN_G_CONFIRM_ACC = 0
                    self._WHALE_DOWN = 0
                    self._WHALE_DOWN_TFS = []      

                if len(self._WHALE_DOWN_TFS) > 0 and self._WHALE_DOWN >= 1:
                    if self._WHALE_DOWN_N_CONFIRMATION == 1:
                        self._WHALE_DOWN_CHANGE = self._WHALE_DOWN_TFS[-1]['price_avg'] - self._WHALE_DOWN_TFS[0]['price_avg']
                        self._WHALE_DOWN_CHANGE /= self._WHALE_DOWN_TFS[0]['price_avg']
                        self._WHALE_DOWN_CHANGE *= 100

                        self._WHALE_DOWN_IMP = 0
                        for t in self._WHALE_DOWN_TFS: self._WHALE_DOWN_IMP += t['investors']['totaled']['sells'][str(self._WHALE_DOWN_LVL)]['invested']
                        self._WHALE_DOWN_IMP = (self._WHALE_DOWN_IMP * 100) / tf['invested_by_lvl__all_total']
                        
                    else:
                        self._WHALE_DOWN_CHANGE = self._WHALE_DOWN_G_TFS[-1]['price_avg'] - self._WHALE_DOWN_G_TFS[0]['price_avg']
                        self._WHALE_DOWN_CHANGE /= self._WHALE_DOWN_G_TFS[0]['price_avg']
                        self._WHALE_DOWN_CHANGE *= 100

                        self._WHALE_DOWN_IMP = 0
                        for t in self._WHALE_DOWN_G_TFS: self._WHALE_DOWN_IMP += t['investors']['totaled']['sells'][str(self._WHALE_DOWN_LVL)]['invested']
                        self._WHALE_DOWN_IMP = (self._WHALE_DOWN_IMP * 100) / tf['invested_by_lvl__all_total']

                    if self._WHALE_DOWN >= self._WHALE_DOWN_N_CONFIRMATION:
                        self._WHALE_DOWN_G_CONFIRM_ACC += 1     

                if self._WHALE_DOWN_G_CONFIRM_ACC >= self._WHALE_DOWN_G_CONFIRM_TRIGGER:
                    self._WHALE_DOWN_G_CONFIRM_ACC = 0
                    self._WHALE_DOWN_G_TFS = []

            # ----------------------------------------------------------
            # /// ARMADO DE INDICADORES
            # ----------------------------------------------------------

            # ----------------------------------------------------------
            # POR CARGA DE VARIABLES
            # ----------------------------------------------------------
            if on_finish_calculating_the_indicators:
                
                process = True
                
                if data_for_trained:
                    if idx > int(len(rows)*(data_for_trained/100)):
                        process = False

                if data_for_test:
                    if idx < int(len(rows)*((100-data_for_test)/100)):
                        process = False

                if process:
                    on_finish_calculating_the_indicators(
                        idx= idx,
                        tf= tf,
                        _tf= _tf,
                    )
            # ----------------------------------------------------------
            # /// POR CARGA DE VARIABLES
            # ----------------------------------------------------------

            _tf = tf
    
    def run_chart_plt(self):

        self.plt.plot(
            self.chart_data_x, 
            self.chart_data_prices_y, 
            label = "price_tf15m")

        self.plt.plot(
            self.chart_data_x, 
            self.chart_data_ema60_y, 
            label = "ema_60", 
            linewidth = 0.5)

        self.plt.plot(
            self.chart_data_x, 
            self.chart_data_ema200_y, 
            label = "ema_200", 
            linewidth = 0.5)

        self.plt.plot(
            self.chart_data_x, 
            self.chart_data_market_sentiment_y, 
            label = "market_sentiment_on_6h", 
            linestyle=':', linewidth = 0.5)

        self.plt.xticks([])
        self.plt.legend()
        self.plt.show()

_imp = IMP(
    backtest= '',
    symbol= 'ethereum',
    timeframe= 60*15,
)