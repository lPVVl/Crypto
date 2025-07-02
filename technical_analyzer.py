# -*- coding: utf-8 -*-
import ccxt
import pandas as pd
import pandas_ta as ta
import time
import datetime
import numpy as np 

# --- НАСТРОЙКИ БОТА ---
SYMBOL = 'BTC/USDT'     # Торговая пара
TIMEFRAME = '15m'       # Таймфрейм
EXCHANGE = ccxt.binance() # Выбираем биржу Binance

# Интервал мониторинга в секундах.
MONITORING_INTERVAL_SECONDS = 300 # 5 минут

# --- НАСТРОЙКИ СТРАТЕГИИ: SL, TP, ATR ---
ATR_LENGTH = 14 # Длина для расчета Average True Range

# Коэффициенты для SL и TP на основе ATR
SL_MULTIPLIER = 1.5      # Стоп-лосс на 1.5 ATR от точки входа
TP1_MULTIPLIER = 1.0     # Первый тейк-профит на 1.0 ATR
TP2_MULTIPLIER = 2.0     # Второй тейк-профит на 2.0 ATR
TP3_MULTIPLIER = 3.0     # Третий тейк-профит на 3.0 ATR

# --- НОВЫЕ НАСТРОЙКИ СТРАТЕГИИ: MACD и Bollinger Bands ---
MACD_FAST_LENGTH = 12
MACD_SLOW_LENGTH = 26
MACD_SIGNAL_LENGTH = 9

BB_LENGTH = 20
BB_MULTIPLIER = 2.0 # Количество стандартных отклонений

# --- НАСТРОЙКИ УПРАВЛЕНИЯ ПОЗИЦИЕЙ ---
# Trailing Stop:
# Когда цена движется в нашу сторону, стоп-лосс подтягивается
TRAILING_STOP_PERCENTAGE = 0.5 # Процент от движения цены в нашу сторону, на который подтягивается SL (от ATR или от % цены)
                               # Для простоты, пока используем ATR_MULTIPLIER, но можно сделать % от цены

# Break-Even Stop:
# Перевод стоп-лосса в безубыток после достижения TP1
ENABLE_BREAK_EVEN_STOP = True

# Примерная ожидаемая длительность движения в свечах (на основе таймфрейма)
EXPECTED_MOVE_CANDLES_MIN = 2
EXPECTED_MOVE_CANDLES_MAX = 4

# --- ГЛОБАЛЬНЫЕ ПЕРЕМЕННЫЕ СОСТОЯНИЯ БОТА ---
bot_in_position = False 
current_position_type = None
entry_price = None
stop_loss_price = None
take_profit_prices = {}
tp_levels_hit = {'TP1': False, 'TP2': False, 'TP3': False} 
# Для Trailing Stop
last_atr_for_sl_tp = None # Сохраняем ATR на момент входа для расчета SL/TP
original_stop_loss_price = None # Для Break-Even Stop

def calculate_atr_manually(df, length=14):
    """
    Расчет Average True Range (ATR) вручную.
    Использует экспоненциальное скользящее среднее (EMA) для сглаживания True Range.
    """
    if len(df) < length + 1: 
        df[f'ATR_{length}'] = np.nan
        return df

    high_low = df['high'] - df['low']
    high_prev_close = np.abs(df['high'] - df['close'].shift(1))
    low_prev_close = np.abs(df['low'] - df['close'].shift(1))

    df['TR'] = np.maximum.reduce([high_low, high_prev_close, low_prev_close])

    df[f'ATR_{length}'] = df['TR'].ewm(span=length, adjust=False).mean()

    df.drop('TR', axis=1, inplace=True)
    return df

def fetch_data():
    """Загружает исторические данные (свечи) с биржи."""
    try:
        ohlcv = EXCHANGE.fetch_ohlcv(SYMBOL, TIMEFRAME, limit=600) 
        df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df[['open', 'high', 'low', 'close', 'volume']] = df[['open', 'high', 'low', 'close', 'volume']].astype(float)
        return df
    except Exception as e:
        print(f"Произошла ошибка при загрузке данных: {e}")
        return None

def add_indicators(df):
    """Добавляет технические индикаторы в DataFrame."""
    if df.empty:
        print("add_indicators: Входной DataFrame пуст.")
        return df
    
    print(f"add_indicators: Исходный DataFrame содержит {len(df)} строк.")
    print(f"add_indicators: Типы данных колонок до индикаторов: {df.dtypes.to_dict()}")

    # Ручной расчет ATR
    df = calculate_atr_manually(df.copy(), length=ATR_LENGTH) 
    print(f"add_indicators: Ручной расчет ATR завершен.")
    
    if f'ATR_{ATR_LENGTH}' not in df.columns:
        print(f"ВНИМАНИЕ: Колонка 'ATR_{ATR_LENGTH}' все еще не найдена после ручного расчета. Это очень странно.")
        return pd.DataFrame() 
    
    # Добавление EMA и RSI через pandas_ta
    df.ta.ema(length=50, append=True)
    df.ta.ema(length=200, append=True)
    df.ta.rsi(length=14, append=True)

    # --- НОВЫЕ ИНДИКАТОРЫ ---
    # MACD
    df.ta.macd(fast=MACD_FAST_LENGTH, slow=MACD_SLOW_LENGTH, signal=MACD_SIGNAL_LENGTH, append=True)
    # Bollinger Bands
    df.ta.bbands(length=BB_LENGTH, std=BB_MULTIPLIER, append=True)
    
    initial_rows = len(df)
    df.dropna(inplace=True) 
    rows_after_dropna = len(df)

    if rows_after_dropna == 0:
        print(f"add_indicators: DataFrame стал пустым после dropna. Исходно: {initial_rows} строк.")
    elif rows_after_dropna < max(200, ATR_LENGTH + 1, BB_LENGTH, MACD_SLOW_LENGTH + MACD_SIGNAL_LENGTH): # Увеличим минимальное количество строк
        print(f"add_indicators: DataFrame имеет очень мало строк ({rows_after_dropna}) после dropna. Возможно, недостаточно для индикаторов.")

    print(f"add_indicators: DataFrame после dropna содержит {len(df)} строк.")
    return df

def analyze_data(df):
    """
    Анализирует последние данные и генерирует торговые сигналы,
    включая расчет SL/TP на основе ATR.
    """
    global bot_in_position, current_position_type, entry_price, \
           stop_loss_price, take_profit_prices, tp_levels_hit, \
           last_atr_for_sl_tp, original_stop_loss_price
    
    if df.empty:
        print("analyze_data: DataFrame пуст после добавления индикаторов. Ожидание следующей итерации.")
        return "WAIT"

    if len(df) < 1: 
        print("analyze_data: DataFrame содержит менее одной свечи после обработки. Ожидание следующей итерации.")
        return "WAIT"

    # Обновленный список необходимых колонок
    required_columns = [
        f'EMA_50', f'EMA_200', f'RSI_14', f'ATR_{ATR_LENGTH}', 'close',
        f'MACD_{MACD_FAST_LENGTH}_{MACD_SLOW_LENGTH}_{MACD_SIGNAL_LENGTH}', # MACD линия
        f'MACDh_{MACD_FAST_LENGTH}_{MACD_SLOW_LENGTH}_{MACD_SIGNAL_LENGTH}', # MACD гистограмма
        f'MACDs_{MACD_FAST_LENGTH}_{MACD_SLOW_LENGTH}_{MACD_SIGNAL_LENGTH}', # MACD сигнальная линия
        f'BBL_{BB_LENGTH}_{BB_MULTIPLIER:.1f}', # Нижняя полоса Боллинджера
        f'BBM_{BB_LENGTH}_{BB_MULTIPLIER:.1f}', # Средняя полоса Боллинджера
        f'BBU_{BB_LENGTH}_{BB_MULTIPLIER:.1f}'  # Верхняя полоса Боллинджера
    ]
    for col in required_columns:
        if col not in df.columns:
            print(f"analyze_data: Ошибка: Не найдена необходимая колонка '{col}' в DataFrame. Проверьте расчет индикаторов.")
            return "WAIT" 

    last_candle = df.iloc[-1]
    
    current_ema_50 = last_candle[f'EMA_50']
    current_ema_200 = last_candle[f'EMA_200']
    current_rsi = last_candle[f'RSI_14']
    current_atr = last_candle[f'ATR_{ATR_LENGTH}']
    current_close_price = last_candle['close']

    # --- Новые индикаторы ---
    current_macd = last_candle[f'MACD_{MACD_FAST_LENGTH}_{MACD_SLOW_LENGTH}_{MACD_SIGNAL_LENGTH}']
    current_macd_hist = last_candle[f'MACDh_{MACD_FAST_LENGTH}_{MACD_SLOW_LENGTH}_{MACD_SIGNAL_LENGTH}']
    current_macd_signal = last_candle[f'MACDs_{MACD_FAST_LENGTH}_{MACD_SLOW_LENGTH}_{MACD_SIGNAL_LENGTH}']
    
    current_bb_lower = last_candle[f'BBL_{BB_LENGTH}_{BB_MULTIPLIER:.1f}']
    current_bb_middle = last_candle[f'BBM_{BB_LENGTH}_{BB_MULTIPLIER:.1f}']
    current_bb_upper = last_candle[f'BBU_{BB_LENGTH}_{BB_MULTIPLIER:.1f}']

    # Проверка на NaN, inf (бесконечность) и очень маленькие значения для ключевых индикаторов
    if (pd.isna(current_atr) or not np.isfinite(current_atr) or current_atr <= 0.00000001 or
        pd.isna(current_macd) or not np.isfinite(current_macd) or
        pd.isna(current_macd_signal) or not np.isfinite(current_macd_signal) or
        pd.isna(current_bb_lower) or not np.isfinite(current_bb_lower) or
        pd.isna(current_bb_upper) or not np.isfinite(current_bb_upper)):
        print(f"analyze_data: Внимание: Один или несколько индикаторов содержат NaN, Inf или нулевое значение. Пропускаем анализ сигнала.")
        print(f"  ATR: {current_atr:.4f}, MACD: {current_macd:.4f}, MACDs: {current_macd_signal:.4f}, BBL: {current_bb_lower:.2f}, BBU: {current_bb_upper:.2f}")
        return "WAIT"

    print(f"\n--- Анализ на {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')} (Таймфрейм: {TIMEFRAME}) ---")
    print(f"Последняя свеча закрыта в: {last_candle['timestamp']}")
    print(f"Цена закрытия: {current_close_price:.2f}")
    print(f"EMA_50: {current_ema_50:.2f}")
    print(f"EMA_200: {current_ema_200:.2f}")
    print(f"RSI (14): {current_rsi:.2f}")
    print(f"ATR ({ATR_LENGTH}): {current_atr:.2f}")
    print(f"MACD ({MACD_FAST_LENGTH},{MACD_SLOW_LENGTH},{MACD_SIGNAL_LENGTH}): MACD={current_macd:.2f}, Signal={current_macd_signal:.2f}, Hist={current_macd_hist:.2f}")
    print(f"Bollinger Bands ({BB_LENGTH},{BB_MULTIPLIER:.1f}): Lower={current_bb_lower:.2f}, Middle={current_bb_middle:.2f}, Upper={current_bb_upper:.2f}")


    signal = "WAIT"
    
    # --- Условия для входа (улучшенная стратегия) ---
    
    # Условия LONG
    # Тренд: EMA_50 над EMA_200
    condition_long_ema_trend = current_ema_50 > current_ema_200
    # RSI: не перекуплен (или в зоне для входа)
    condition_long_rsi = current_rsi < 50
    # MACD: MACD линия пересекла сигнальную линию снизу вверх И MACD гистограмма положительная
    condition_long_macd = (current_macd > current_macd_signal) and (current_macd_hist > 0)
    # Bollinger Bands: Цена закрытия выше средней линии BB, или отскок от нижней полосы
    # Для входа LONG, можно использовать, например, отскок от нижней полосы или пересечение средней
    condition_long_bb = current_close_price > current_bb_middle # Простейшее условие: цена выше средней BB

    # Условия SHORT
    # Тренд: EMA_50 под EMA_200
    condition_short_ema_trend = current_ema_50 < current_ema_200
    # RSI: не перепродан (или в зоне для входа)
    condition_short_rsi = current_rsi > 50
    # MACD: MACD линия пересекла сигнальную линию сверху вниз И MACD гистограмма отрицательная
    condition_short_macd = (current_macd < current_macd_signal) and (current_macd_hist < 0)
    # Bollinger Bands: Цена закрытия ниже средней линии BB, или отскок от верхней полосы
    condition_short_bb = current_close_price < current_bb_middle # Простейшее условие: цена ниже средней BB

    if not bot_in_position: 
        if (condition_long_ema_trend and condition_long_rsi and 
            condition_long_macd and condition_long_bb):
            
            signal = "BUY"
            bot_in_position = True
            current_position_type = "LONG"
            entry_price = current_close_price
            last_atr_for_sl_tp = current_atr # Сохраняем ATR на момент входа
            
            stop_loss_price = entry_price - (last_atr_for_sl_tp * SL_MULTIPLIER)
            original_stop_loss_price = stop_loss_price # Сохраняем начальный SL для безубытка
            take_profit_prices['TP1'] = entry_price + (last_atr_for_sl_tp * TP1_MULTIPLIER)
            take_profit_prices['TP2'] = entry_price + (last_atr_for_sl_tp * TP2_MULTIPLIER)
            take_profit_prices['TP3'] = entry_price + (last_atr_for_sl_tp * TP3_MULTIPLIER)
            tp_levels_hit = {'TP1': False, 'TP2': False, 'TP3': False} 
            
            expected_duration_min = EXPECTED_MOVE_CANDLES_MIN * int(TIMEFRAME[:-1])
            expected_duration_max = EXPECTED_MOVE_CANDLES_MAX * int(TIMEFRAME[:-1])
            
            print(f"Сигнал: **ПОКУПКА (LONG)** - Ожидается повышение.")
            print(f"  Вход: {entry_price:.2f}")
            print(f"  **Стоп-Лосс (SL):** ~{stop_loss_price:.2f} ({SL_MULTIPLIER} * ATR_входа)")
            print(f"  **Тейк-Профит 1 (TP1):** ~{take_profit_prices['TP1']:.2f} ({TP1_MULTIPLIER} * ATR_входа)")
            print(f"  **Тейк-Профит 2 (TP2):** ~{take_profit_prices['TP2']:.2f} ({TP2_MULTIPLIER} * ATR_входа)")
            print(f"  **Тейк-Профит 3 (TP3):** ~{take_profit_prices['TP3']:.2f} ({TP3_MULTIPLIER} * ATR_входа)")
            print(f"  **Ожидаемая длительность движения:** От {expected_duration_min:.0f} до {expected_duration_max:.0f} минут (на основе {TIMEFRAME} свечей).")
            
        elif (condition_short_ema_trend and condition_short_rsi and 
              condition_short_macd and condition_short_bb):
            
            signal = "SELL" 
            bot_in_position = True
            current_position_type = "SHORT"
            entry_price = current_close_price
            last_atr_for_sl_tp = current_atr # Сохраняем ATR на момент входа
            
            stop_loss_price = entry_price + (last_atr_for_sl_tp * SL_MULTIPLIER)
            original_stop_loss_price = stop_loss_price # Сохраняем начальный SL для безубытка
            take_profit_prices['TP1'] = entry_price - (last_atr_for_sl_tp * TP1_MULTIPLIER)
            take_profit_prices['TP2'] = entry_price - (last_atr_for_sl_tp * TP2_MULTIPLIER)
            take_profit_prices['TP3'] = entry_price - (last_atr_for_sl_tp * TP3_MULTIPLIER)
            tp_levels_hit = {'TP1': False, 'TP2': False, 'TP3': False} 
            
            expected_duration_min = EXPECTED_MOVE_CANDLES_MIN * int(TIMEFRAME[:-1])
            expected_duration_max = EXPECTED_MOVE_CANDLES_MAX * int(TIMEFRAME[:-1])
            
            print(f"Сигнал: **ПРОДАЖА (SHORT)** - Ожидается понижение.")
            print(f"  Вход: {entry_price:.2f}")
            print(f"  **Стоп-Лосс (SL):** ~{stop_loss_price:.2f} ({SL_MULTIPLIER} * ATR_входа)")
            print(f"  **Тейк-Профит 1 (TP1):** ~{take_profit_prices['TP1']:.2f} ({TP1_MULTIPLIER} * ATR_входа)")
            print(f"  **Тейк-Профит 2 (TP2):** ~{take_profit_prices['TP2']:.2f} ({TP2_MULTIPLIER} * ATR_входа)")
            print(f"  **Тейк-Профит 3 (TP3):** ~{take_profit_prices['TP3']:.2f} ({TP3_MULTIPLIER} * ATR_входа)")
            print(f"  **Ожидаемая длительность движения:** От {expected_duration_min:.0f} до {expected_duration_max:.0f} минут (на основе {TIMEFRAME} свечей).")
        else:
            print("Сигнал: ОЖИДАНИЕ - Условия для входа не выполнены.")
            # Добавим причины, почему не было входа
            reasons = []
            if not (condition_long_ema_trend or condition_short_ema_trend):
                reasons.append("Тренд по EMA_50/EMA_200 не определен.")
            if not (condition_long_rsi or condition_short_rsi):
                reasons.append("RSI в неопределенной зоне.")
            if not (condition_long_macd or condition_short_macd):
                reasons.append("MACD не дал сигнала.")
            if not (condition_long_bb or condition_short_bb):
                reasons.append("Цена не соответствует условиям Bollinger Bands.")
            print("  Причины: " + ", ".join(reasons) if reasons else "Нет явных причин (просто условия не совпали).")


    else: # Если бот уже в позиции, отслеживаем её и проверяем SL/TP
        print(f"Бот уже в позиции: {current_position_type} по цене {entry_price:.2f}")
        print(f"  SL: {stop_loss_price:.2f}")
        print(f"  TP1: {take_profit_prices['TP1']:.2f} ({"достигнут" if tp_levels_hit['TP1'] else "нет"})")
        print(f"  TP2: {take_profit_prices['TP2']:.2f} ({"достигнут" if tp_levels_hit['TP2'] else "нет"})")
        print(f"  TP3: {take_profit_prices['TP3']:.2f} ({"достигнут" if tp_levels_hit['TP3'] else "нет"})")
        print(f"  Текущая цена: {current_close_price:.2f}")

        # --- Логика Trailing Stop и Break-Even Stop ---
        if current_position_type == "LONG":
            # Break-Even Stop: Если TP1 достигнут, переводим SL в безубыток (цена входа)
            if ENABLE_BREAK_EVEN_STOP and tp_levels_hit['TP1'] and stop_loss_price < entry_price:
                print(f"  TP1 достигнут, перевод SL в безубыток: {entry_price:.2f}")
                stop_loss_price = entry_price # Или entry_price + небольшая комиссия
            
            # Trailing Stop: SL подтягивается, если цена идет выше
            # Только если текущая цена уже значительно выше предыдущего SL
            # ATR_входа используется для определения шага Trailing Stop
            new_trailing_sl = current_close_price - (last_atr_for_sl_tp * TRAILING_STOP_PERCENTAGE)
            if new_trailing_sl > stop_loss_price:
                print(f"  **Trailing SL LONG:** SL повышен с {stop_loss_price:.2f} до {new_trailing_sl:.2f}")
                stop_loss_price = new_trailing_sl

            if current_close_price <= stop_loss_price:
                print(f"**ЗАКРЫТИЕ LONG по СТОП-ЛОССУ:** Цена {current_close_price:.2f} <= SL {stop_loss_price:.2f}.")
                signal = "CLOSE_LONG_SL"
                # Сброс состояния бота
                bot_in_position = False
                current_position_type = None
                entry_price = None
                stop_loss_price = None
                take_profit_prices = {}
                tp_levels_hit = {'TP1': False, 'TP2': False, 'TP3': False}
                last_atr_for_sl_tp = None
                original_stop_loss_price = None
            else: 
                # Проверяем TP3, потом TP2, потом TP1
                # Важно: TP уровни проверяются от самого дальнего к ближнему
                if not tp_levels_hit['TP3'] and current_close_price >= take_profit_prices['TP3']:
                    print(f"**ЗАКРЫТИЕ ЧАСТИ LONG по TP3:** Цена {current_close_price:.2f} >= TP3 {take_profit_prices['TP3']:.2f}.")
                    tp_levels_hit['TP3'] = True
                    signal = "PARTIAL_CLOSE_LONG_TP3" 
                elif not tp_levels_hit['TP2'] and current_close_price >= take_profit_prices['TP2']:
                    print(f"**ЗАКРЫТИЕ ЧАСТИ LONG по TP2:** Цена {current_close_price:.2f} >= TP2 {take_profit_prices['TP2']:.2f}.")
                    tp_levels_hit['TP2'] = True
                    signal = "PARTIAL_CLOSE_LONG_TP2"
                elif not tp_levels_hit['TP1'] and current_close_price >= take_profit_prices['TP1']:
                    print(f"**ЗАКРЫТИЕ ЧАСТИ LONG по TP1:** Цена {current_close_price:.2f} >= TP1 {take_profit_prices['TP1']:.2f}.")
                    tp_levels_hit['TP1'] = True
                    signal = "PARTIAL_CLOSE_LONG_TP1"
                else:
                    print("  Позиция LONG удерживается.")

        elif current_position_type == "SHORT":
            # Break-Even Stop: Если TP1 достигнут, переводим SL в безубыток (цена входа)
            if ENABLE_BREAK_EVEN_STOP and tp_levels_hit['TP1'] and stop_loss_price > entry_price:
                print(f"  TP1 достигнут, перевод SL в безубыток: {entry_price:.2f}")
                stop_loss_price = entry_price # Или entry_price - небольшая комиссия
            
            # Trailing Stop: SL подтягивается, если цена идет ниже
            new_trailing_sl = current_close_price + (last_atr_for_sl_tp * TRAILING_STOP_PERCENTAGE)
            if new_trailing_sl < stop_loss_price:
                print(f"  **Trailing SL SHORT:** SL понижен с {stop_loss_price:.2f} до {new_trailing_sl:.2f}")
                stop_loss_price = new_trailing_sl


            if current_close_price >= stop_loss_price:
                print(f"**ЗАКРЫТИЕ SHORT по СТОП-ЛОССУ:** Цена {current_close_price:.2f} >= SL {stop_loss_price:.2f}.")
                signal = "CLOSE_SHORT_SL"
                # Сброс состояния бота
                bot_in_position = False
                current_position_type = None
                entry_price = None
                stop_loss_price = None
                take_profit_prices = {}
                tp_levels_hit = {'TP1': False, 'TP2': False, 'TP3': False}
                last_atr_for_sl_tp = None
                original_stop_loss_price = None
            else: 
                # Проверяем TP3, потом TP2, потом TP1
                if not tp_levels_hit['TP3'] and current_close_price <= take_profit_prices['TP3']:
                    print(f"**ЗАКРЫТИЕ ЧАСТИ SHORT по TP3:** Цена {current_close_price:.2f} <= TP3 {take_profit_prices['TP3']:.2f}.")
                    tp_levels_hit['TP3'] = True
                    signal = "PARTIAL_CLOSE_SHORT_TP3"
                elif not tp_levels_hit['TP2'] and current_close_price <= take_profit_prices['TP2']:
                    print(f"**ЗАКРЫТИЕ ЧАСТИ SHORT по TP2:** Цена {current_close_price:.2f} <= TP2 {take_profit_prices['TP2']:.2f}.")
                    tp_levels_hit['TP2'] = True
                    signal = "PARTIAL_CLOSE_SHORT_TP2"
                elif not tp_levels_hit['TP1'] and current_close_price <= take_profit_prices['TP1']:
                    print(f"**ЗАКРЫТИЕ ЧАСТИ SHORT по TP1:** Цена {current_close_price:.2f} <= TP1 {take_profit_prices['TP1']:.2f}.")
                    tp_levels_hit['TP1'] = True
                    signal = "PARTIAL_CLOSE_SHORT_TP1"
                else:
                    print("  Позиция SHORT удерживается.")

    return signal

# --- ОСНОВНАЯ ЛОГИКА БОТА ---
if __name__ == "__main__":
    print("Бот для мониторинга рынка запущен. Нажмите Ctrl+C для остановки.")
    print("Внимание: это демонстрационный режим, реальные ордера не исполняются.")
    try:
        while True:
            data = fetch_data()
            
            if data is not None and not data.empty:
                data_with_indicators = add_indicators(data)
                
                if not data_with_indicators.empty:
                    signal = analyze_data(data_with_indicators)
                    
                    if signal == "BUY":
                        print("!!! СИГНАЛ: ОТКРЫТЬ LONG ПОЗИЦИЮ !!!")
                    elif signal == "SELL":
                        print("!!! СИГНАЛ: ОТКРЫТЬ SHORT ПОЗИЦИЮ !!!")
                    elif signal == "CLOSE_LONG_SL":
                        print("!!! СИГНАЛ: ЗАКРЫТЬ LONG ПО СТОП-ЛОССУ !!!")
                    elif signal == "CLOSE_SHORT_SL":
                        print("!!! СИГНАЛ: ЗАКРЫТЬ SHORT ПО СТОП-ЛОССУ !!!")
                    elif "PARTIAL_CLOSE_LONG_TP" in signal:
                        print(f"!!! СИГНАЛ: ЧАСТИЧНО ЗАКРЫТЬ LONG ПО {signal.split('_')[-1]} !!!")
                    elif "PARTIAL_CLOSE_SHORT_TP" in signal:
                        print(f"!!! СИГНАЛ: ЧАСТИЧНО ЗАКРЫТЬ SHORT ПО {signal.split('_')[-1]} !!!")
                else:
                    print("DataFrame стал пустым после добавления индикаторов. Пропускаем анализ этой итерации.")
            else:
                print("Не удалось получить данные или DataFrame пуст. Повторная попытка через заданный интервал.")

            print(f"\nОжидание {MONITORING_INTERVAL_SECONDS} секунд до следующей проверки...")
            time.sleep(MONITORING_INTERVAL_SECONDS)

    except KeyboardInterrupt:
        print("\nБот остановлен пользователем.")
    except Exception as e:
        print(f"Произошла непредвиденная ошибка в основном цикле: {e}")