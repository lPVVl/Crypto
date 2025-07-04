# -*- coding: utf-8 -*-
import ccxt
import pandas as pd
import pandas_ta as ta
import time
import datetime
import numpy as np
import logging
import json
import os

# --- НАСТРОЙКИ ЛОГИРОВАНИЯ ---
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)

file_handler = logging.FileHandler('bot_log.log')
file_handler.setLevel(logging.INFO)
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

# --- ФАЙЛЫ КОНФИГУРАЦИИ И СОСТОЯНИЯ БОТА ---
CONFIG_FILE = 'config.json'
STATE_FILE = 'bot_state.json'

# --- ГЛОБАЛЬНЫЕ ПЕРЕМЕННЫЕ (будут загружены из конфига или инициализированы) ---
SYMBOLS = []
TIMEFRAME = '5m'
MONITORING_INTERVAL_SECONDS = 60

# --- ИЗМЕНЕНИЕ ЗДЕСЬ: Инициализация биржи для ФЬЮЧЕРСОВ (USDT-M) ---
try:
    EXCHANGE = ccxt.binanceusdm() # Используем binanceusdm для USDT-M фьючерсов
    EXCHANGE.load_markets() # Загружаем рынки, чтобы убедиться в подключении
    logger.info("Биржа Binance USDT-M Futures успешно инициализирована.")
except Exception as e:
    logger.critical(f"Ошибка при инициализации биржи Binance USDT-M Futures: {e}", exc_info=True)
    exit()

# --- НАСТРОЙКИ СТРАТЕГИИ: SL, TP, ATR ---
ATR_LENGTH = 14

SL_MULTIPLIER = 1.5
TP1_MULTIPLIER = 1.0
TP2_MULTIPLIER = 2.0
TP3_MULTIPLIER = 3.0

# --- НАСТРОЙКИ СТРАТЕГИИ: MACD и Bollinger Bands ---
MACD_FAST_LENGTH = 12
MACD_SLOW_LENGTH = 26
MACD_SIGNAL_LENGTH = 9

BB_LENGTH = 20
BB_MULTIPLIER = 2.0

# --- НАСТРОЙКИ УПРАВЛЕНИЯ ПОЗИЦИЕЙ ---
EXPECTED_MOVE_CANDLES_MIN = 3
EXPECTED_MOVE_CANDLES_MAX = 6

# --- НАСТРОЙКИ АНАЛИЗА РАЗВОРОТА ---
REVERSAL_CHECK_INTERVAL_SECONDS = 300 # 5 минут

# --- ГЛОБАЛЬНАЯ ПЕРЕМЕННАЯ СОСТОЯНИЯ БОТА ---
bot_state = {}

# --- ФУНКЦИИ ДЛЯ УПРАВЛЕНИЯ КОНФИГУРАЦИЕЙ И СОСТОЯНИЕМ ---
def load_config():
    """Загружает настройки бота из JSON файла конфигурации."""
    global SYMBOLS, TIMEFRAME, MONITORING_INTERVAL_SECONDS

    if not os.path.exists(CONFIG_FILE):
        logger.error(f"Файл конфигурации '{CONFIG_FILE}' не найден. Создайте его с необходимыми настройками.")
        default_config = {
            "symbols": [
                'BTC/USDT',
                'ETH/USDT',
                'BNB/USDT'
            ],
            "timeframe": "5m",
            "monitoring_interval_seconds": 60
        }
        with open(CONFIG_FILE, 'w') as f:
            json.dump(default_config, f, indent=4)
        logger.info(f"Создан дефолтный файл конфигурации '{CONFIG_FILE}'. Пожалуйста, отредактируйте его.")
        exit()

    try:
        with open(CONFIG_FILE, 'r') as f:
            config = json.load(f)

        SYMBOLS = config.get('symbols', [])
        TIMEFRAME = config.get('timeframe', '5m')
        MONITORING_INTERVAL_SECONDS = config.get('monitoring_interval_seconds', 60)

        if not SYMBOLS:
            logger.error(f"Список символов в '{CONFIG_FILE}' пуст. Пожалуйста, укажите символы для торговли.")
            exit()

        logger.info(f"Настройки бота успешно загружены из '{CONFIG_FILE}'.")
        logger.info(f"  Символы: {SYMBOLS}")
        logger.info(f"  Таймфрейм: {TIMEFRAME}")
        logger.info(f"  Интервал мониторинга: {MONITORING_INTERVAL_SECONDS} сек.")

    except json.JSONDecodeError as e:
        logger.critical(f"Ошибка декодирования JSON из файла конфигурации '{CONFIG_FILE}': {e}. Проверьте синтаксис JSON.", exc_info=True)
        exit()
    except Exception as e:
        logger.critical(f"Неизвестная ошибка при загрузке конфигурации из '{CONFIG_FILE}': {e}", exc_info=True)
        exit()


def initialize_pair_state(symbol):
    """Инициализирует или сбрасывает состояние для конкретной торговой пары."""
    return {
        'bot_in_position': False,
        'current_position_type': None,
        'entry_price': None,
        'stop_loss_price': None,
        'take_profit_prices': {},
        'tp_levels_hit': {'TP1': False, 'TP2': False, 'TP3': False},
        'last_atr_for_sl_tp': None,
        'original_stop_loss_price': None,
        'last_reversal_check_time': 0
    }

def save_bot_state():
    """Сохраняет текущее состояние бота (для всех пар) в JSON файл."""
    try:
        state_to_save = {s: bot_state[s] for s in SYMBOLS if s in bot_state}
        with open(STATE_FILE, 'w') as f:
            json.dump(state_to_save, f, indent=4)
        logger.info(f"Состояние бота успешно сохранено в {STATE_FILE}")
    except Exception as e:
        logger.error(f"Ошибка при сохранении состояния бота в {STATE_FILE}: {e}", exc_info=True)

def load_bot_state():
    """Загружает состояние бота (для всех пар) из JSON файла."""
    global bot_state

    if not os.path.exists(STATE_FILE):
        logger.info(f"Файл состояния {STATE_FILE} не найден. Запуск с начальным состоянием для каждой пары.")
        for symbol in SYMBOLS:
            bot_state[symbol] = initialize_pair_state(symbol)
        return

    try:
        with open(STATE_FILE, 'r') as f:
            loaded_state = json.load(f)

        new_bot_state = {}
        for symbol in SYMBOLS:
            if symbol in loaded_state:
                pair_state = loaded_state[symbol]
                default_state = initialize_pair_state(symbol)
                for key, value in default_state.items():
                    if key not in pair_state:
                        pair_state[key] = value
                
                # --- ЛОГИКА МИГРАЦИИ/ПРОВЕРКИ СОСТОЯНИЯ ---
                if pair_state.get('bot_in_position') and \
                   pair_state.get('tp_levels_hit', {}).get('TP1') and \
                   pair_state.get('tp_levels_hit', {}).get('TP2') and \
                   pair_state.get('tp_levels_hit', {}).get('TP3'):
                    logger.warning(f"[{symbol}] Обнаружена 'зависшая' позиция (все TP достигнуты, но позиция активна) при загрузке состояния. Сброс состояния для {symbol}.")
                    new_bot_state[symbol] = initialize_pair_state(symbol)
                else:
                    new_bot_state[symbol] = pair_state
                # --- КОНЕЦ ЛОГИКИ МИГРАЦИИ ---

            else:
                new_bot_state[symbol] = initialize_pair_state(symbol)

        bot_state = new_bot_state

        logger.info(f"Состояние бота успешно загружено из {STATE_FILE}")
        for symbol, state in bot_state.items():
            pos_info = f"Позиция {'открыта' if state['bot_in_position'] else 'нет'}, Тип: {state['current_position_type']}, Цена входа: {state['entry_price']:.8f}" if state['entry_price'] else "без активной позиции."
            logger.info(f"  Загруженное состояние для {symbol}: {pos_info}")

    except json.JSONDecodeError as e:
        logger.error(f"Ошибка декодирования JSON из файла состояния {STATE_FILE}: {e}. Файл может быть поврежден. Запуск с начальным состоянием для каждой пары.", exc_info=True)
        bot_state = {}
        for symbol in SYMBOLS:
            bot_state[symbol] = initialize_pair_state(symbol)
    except Exception as e:
        logger.error(f"Неизвестная ошибка при загрузке состояния бота из {STATE_FILE}: {e}. Запуск с начальным состоянием для каждой пары.", exc_info=True)
        bot_state = {}
        for symbol in SYMBOLS:
            bot_state[symbol] = initialize_pair_state(symbol)

# --- ВСПОМОГАТЕЛЬНАЯ ФУНКЦИЯ ДЛЯ ПОВТОРНЫХ ПОПЫТОК ---
def retry_on_exception(func, retries=3, delay=1, backoff=2):
    """
    Выполняет функцию с повторными попытками при возникновении определенных исключений.
    """
    for i in range(retries):
        try:
            return func()
        except (ccxt.NetworkError, ccxt.ExchangeNotAvailable, ccxt.RequestTimeout,
                ccxt.DDoSProtection, ccxt.ExchangeError) as e:
            logger.warning(f"Ошибка биржи (попытка {i + 1}/{retries}): {e}. Повторная попытка через {delay:.1f} сек...")
            time.sleep(delay)
            delay *= backoff
        except Exception as e:
            logger.error(f"Неизвестная ошибка (попытка {i + 1}/{retries}): {e}. Повторная попытка через {delay:.1f} сек...", exc_info=True)
            time.sleep(delay)
            delay *= backoff
    raise Exception(f"Все {retries} попыток исчерпаны. Не удалось выполнить функцию {func.__name__}.")


def calculate_atr_manually(df, length=14):
    """
    Расчет Average True Range (ATR) вручную.
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

def fetch_data(symbol):
    """Загружает исторические данные (свечи) для конкретной пары с биржи с повторными попытками."""
    def _fetch():
        logger.debug(f"[{symbol}] Запрос OHLCV данных для {symbol} с лимитом {1000} на таймфрейме {TIMEFRAME}...")
        ohlcv = EXCHANGE.fetch_ohlcv(symbol, TIMEFRAME, limit=1000)
        df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df[['open', 'high', 'low', 'close', 'volume']] = df[['open', 'high', 'low', 'close', 'volume']].astype(float)
        logger.debug(f"[{symbol}] Получено {len(df)} свечей. Последняя свеча: {df.iloc[-1]['timestamp']} Close: {df.iloc[-1]['close']:.8f}")
        return df

    try:
        return retry_on_exception(_fetch, retries=5, delay=2)
    except Exception as e:
        logger.critical(f"[{symbol}] Критическая ошибка: Не удалось получить данные для {symbol} после нескольких попыток: {e}", exc_info=True)
        return None

def add_indicators(df, symbol):
    """Добавляет технические индикаторы в DataFrame."""
    if df.empty:
        logger.warning(f"[{symbol}] add_indicators: Входной DataFrame пуст.")
        return df

    df = calculate_atr_manually(df.copy(), length=ATR_LENGTH)

    if f'ATR_{ATR_LENGTH}' not in df.columns:
        logger.error(f"[{symbol}] ВНИМАНИЕ: Колонка 'ATR_{ATR_LENGTH}' все еще не найдена после ручного расчета. Это очень странно.")
        return pd.DataFrame()

    df.ta.ema(length=50, append=True)
    df.ta.ema(length=200, append=True)
    df.ta.rsi(length=14, append=True)

    df.ta.macd(fast=MACD_FAST_LENGTH, slow=MACD_SLOW_LENGTH, signal=MACD_SIGNAL_LENGTH, append=True)
    df.ta.bbands(length=BB_LENGTH, std=BB_MULTIPLIER, append=True)

    initial_rows = len(df)
    df.dropna(inplace=True)
    rows_after_dropna = len(df)

    if rows_after_dropna == 0:
        logger.warning(f"[{symbol}] add_indicators: DataFrame стал пустым после dropna. Исходно: {initial_rows} строк.")
    elif rows_after_dropna < max(200, ATR_LENGTH + 1, BB_LENGTH, MACD_SLOW_LENGTH + MACD_SIGNAL_LENGTH):
        logger.warning(f"[{symbol}] add_indicators: DataFrame имеет очень мало строк ({rows_after_dropna}) после dropna. Возможно, недостаточно для индикаторов.")

    return df

def analyze_reversal_opportunity(symbol, df, current_position_type, current_close_price):
    """
    Анализирует возможность разворота тренда после достижения TP.
    Это демонстрационная функция, требующая доработки для реальной стратегии разворота.
    """
    current_time = time.time()
    if (current_time - bot_state[symbol]['last_reversal_check_time']) < REVERSAL_CHECK_INTERVAL_SECONDS:
        logger.debug(f"[{symbol}] Пропуск проверки разворота: слишком короткий интервал с последней проверки.")
        return

    logger.info(f"\n--- Проверка возможного разворота для {symbol} после достижения TP ({current_position_type} позиция) ---")
    
    if len(df) < max(MACD_SLOW_LENGTH + MACD_SIGNAL_LENGTH, BB_LENGTH, ATR_LENGTH) * 2:
        logger.warning(f"[{symbol}] Недостаточно данных для надежного анализа разворота. Требуется больше свечей.")
        return

    last_candle = df.iloc[-1]
    current_rsi = last_candle[f'RSI_14']
    current_macd_hist = last_candle[f'MACDh_{MACD_FAST_LENGTH}_{MACD_SLOW_LENGTH}_{MACD_SIGNAL_LENGTH}']
    current_bb_lower = last_candle[f'BBL_{BB_LENGTH}_{BB_MULTIPLIER:.1f}']
    current_bb_upper = last_candle[f'BBU_{BB_LENGTH}_{BB_MULTIPLIER:.1f}']

    reversal_found = False
    suggested_direction = "NONE"

    if current_position_type == "LONG":
        if current_rsi > 70 and current_macd_hist < 0:
            logger.info(f"[{symbol}] **Потенциальный медвежий разворот после LONG позиции:** RSI ({current_rsi:.2f}) > 70 и MACD Hist ({current_macd_hist:.4f}) < 0.")
            reversal_found = True
            suggested_direction = "SHORT"
        elif current_close_price > current_bb_upper and df.iloc[-2]['close'] < current_bb_upper:
            logger.info(f"[{symbol}] **Потенциальный медвежий разворот после LONG позиции:** Цена закрылась внутри верхней BB после выхода за нее.")
            reversal_found = True
            suggested_direction = "SHORT"

    elif current_position_type == "SHORT":
        if current_rsi < 30 and current_macd_hist > 0:
            logger.info(f"[{symbol}] **Потенциальный бычий разворот после SHORT позиции:** RSI ({current_rsi:.2f}) < 30 и MACD Hist ({current_macd_hist:.4f}) > 0.")
            reversal_found = True
            suggested_direction = "LONG"
        elif current_close_price < current_bb_lower and df.iloc[-2]['close'] > current_bb_lower:
            logger.info(f"[{symbol}] **Потенциальный бычий разворот после SHORT позиции:** Цена закрылась внутри нижней BB после выхода за нее.")
            reversal_found = True
            suggested_direction = "LONG"

    if reversal_found:
        logger.info(f"[{symbol}] Найден **потенциальный сигнал разворота** для входа в {suggested_direction} по цене {current_close_price:.8f}. (Требуется дополнительное подтверждение или ручной анализ).")
        bot_state[symbol]['last_reversal_check_time'] = current_time

    else:
        logger.info(f"[{symbol}] Признаков разворота не обнаружено после достижения TP.")
        bot_state[symbol]['last_reversal_check_time'] = current_time

def analyze_data(symbol, df):
    """
    Анализирует последние данные и генерирует торговые сигналы для конкретной пары.
    """
    current_pair_state = bot_state[symbol]

    if df.empty:
        logger.info(f"[{symbol}] analyze_data: DataFrame пуст после добавления индикаторов. Ожидание следующей итерации.")
        return "WAIT"

    if len(df) < 1:
        logger.warning(f"[{symbol}] analyze_data: DataFrame содержит менее одной свечи после обработки. Ожидание следующей итерации.")
        return "WAIT"

    required_columns = [
        f'EMA_50', f'EMA_200', f'RSI_14', f'ATR_{ATR_LENGTH}', 'close',
        f'MACD_{MACD_FAST_LENGTH}_{MACD_SLOW_LENGTH}_{MACD_SIGNAL_LENGTH}',
        f'MACDh_{MACD_FAST_LENGTH}_{MACD_SLOW_LENGTH}_{MACD_SIGNAL_LENGTH}',
        f'MACDs_{MACD_FAST_LENGTH}_{MACD_SLOW_LENGTH}_{MACD_SIGNAL_LENGTH}',
        f'BBL_{BB_LENGTH}_{BB_MULTIPLIER:.1f}',
        f'BBM_{BB_LENGTH}_{BB_MULTIPLIER:.1f}',
        f'BBU_{BB_LENGTH}_{BB_MULTIPLIER:.1f}'
    ]
    for col in required_columns:
        if col not in df.columns:
            logger.error(f"[{symbol}] analyze_data: Ошибка: Не найдена необходимая колонка '{col}' в DataFrame. Проверьте расчет индикаторов.")
            return "WAIT"

    last_candle = df.iloc[-1]

    current_ema_50 = last_candle[f'EMA_50']
    current_ema_200 = last_candle[f'EMA_200']
    current_rsi = last_candle[f'RSI_14']
    current_atr = last_candle[f'ATR_{ATR_LENGTH}']
    current_close_price = last_candle['close']

    current_macd = last_candle[f'MACD_{MACD_FAST_LENGTH}_{MACD_SLOW_LENGTH}_{MACD_SIGNAL_LENGTH}']
    current_macd_hist = last_candle[f'MACDh_{MACD_FAST_LENGTH}_{MACD_SLOW_LENGTH}_{MACD_SIGNAL_LENGTH}']
    current_macd_signal = last_candle[f'MACDs_{MACD_FAST_LENGTH}_{MACD_SLOW_LENGTH}_{MACD_SIGNAL_LENGTH}']

    current_bb_lower = last_candle[f'BBL_{BB_LENGTH}_{BB_MULTIPLIER:.1f}']
    current_bb_middle = last_candle[f'BBM_{BB_LENGTH}_{BB_MULTIPLIER:.1f}']
    current_bb_upper = last_candle[f'BBU_{BB_LENGTH}_{BB_MULTIPLIER:.1f}']

    if (pd.isna(current_atr) or not np.isfinite(current_atr) or current_atr <= 1e-10 or
        pd.isna(current_macd) or not np.isfinite(current_macd) or
        pd.isna(current_macd_signal) or not np.isfinite(current_macd_signal) or
        pd.isna(current_bb_lower) or not np.isfinite(current_bb_lower) or
        pd.isna(current_bb_upper) or not np.isfinite(current_bb_upper)):
        logger.warning(f"[{symbol}] analyze_data: Один или несколько индикаторов содержат NaN, Inf или нулевое/очень малое значение. Пропускаем анализ сигнала.")
        logger.warning(f"[{symbol}]    ATR: {current_atr:.8f}, MACD: {current_macd:.8f}, MACDs: {current_macd_signal:.8f}, BBL: {current_bb_lower:.8f}, BBU: {current_bb_upper:.8f}")
        logger.warning(f"[{symbol}]    Close: {current_close_price:.8f}")
        return "WAIT"

    should_log_details = current_pair_state['bot_in_position']

    condition_long_ema_trend = current_ema_50 > current_ema_200
    condition_long_rsi = current_rsi < 70
    condition_long_macd = (current_macd > current_macd_signal) or (current_macd_hist > 0)
    condition_long_bb = current_close_price > current_bb_middle

    condition_short_ema_trend = current_ema_50 < current_ema_200
    condition_short_rsi = current_rsi > 30
    condition_short_macd = (current_macd < current_macd_signal) or (current_macd_hist < 0)
    condition_short_bb = current_close_price < current_bb_middle

    if not current_pair_state['bot_in_position']:
        if (condition_long_ema_trend and condition_long_rsi and condition_long_macd and condition_long_bb) or \
           (condition_short_ema_trend and condition_short_rsi and condition_short_macd and condition_short_bb):
            should_log_details = True

    if should_log_details:
        logger.info(f"\n--- Анализ для {symbol} на {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')} (Таймфрейм: {TIMEFRAME}) ---")
        logger.info(f"[{symbol}] Последняя свеча закрыта в: {last_candle['timestamp']}")
        logger.info(f"[{symbol}] Цена закрытия: {current_close_price:.8f}")
        logger.info(f"[{symbol}] EMA_50: {current_ema_50:.8f}")
        logger.info(f"[{symbol}] EMA_200: {current_ema_200:.8f}")
        logger.info(f"[{symbol}] RSI (14): {current_rsi:.8f}")
        logger.info(f"[{symbol}] ATR ({ATR_LENGTH}): {current_atr:.8f}")
        logger.info(f"[{symbol}] MACD ({MACD_FAST_LENGTH},{MACD_SLOW_LENGTH},{MACD_SIGNAL_LENGTH}): MACD={current_macd:.8f}, Signal={current_macd_signal:.8f}, Hist={current_macd_hist:.8f}")
        logger.info(f"[{symbol}] Bollinger Bands ({BB_LENGTH},{BB_MULTIPLIER:.1f}): Lower={current_bb_lower:.8f}, Middle={current_bb_middle:.8f}, Upper={current_bb_upper:.8f}")
        if current_pair_state['bot_in_position']:
            logger.info(f"[{symbol}] Бот уже в позиции: {current_pair_state['current_position_type']} по цене {current_pair_state['entry_price']:.8f}")
            logger.info(f"[{symbol}]    SL: {current_pair_state['stop_loss_price']:.8f}")
            logger.info(f"[{symbol}]    TP1: {current_pair_state['take_profit_prices']['TP1']:.8f} ({"достигнут" if current_pair_state['tp_levels_hit']['TP1'] else "нет"})")
            logger.info(f"[{symbol}]    TP2: {current_pair_state['take_profit_prices']['TP2']:.8f} ({"достигнут" if current_pair_state['tp_levels_hit']['TP2'] else "нет"})")
            logger.info(f"[{symbol}]    TP3: {current_pair_state['take_profit_prices']['TP3']:.8f} ({"достигнут" if current_pair_state['tp_levels_hit']['TP3'] else "нет"})")
            logger.info(f"[{symbol}]    Текущая цена: {current_close_price:.8f}")
    else:
        logger.info(f"[{symbol}] Мониторинг. Цена: {current_close_price:.8f}. Без активной позиции и без сигнала.")


    signal = "WAIT"

    if not current_pair_state['bot_in_position']:
        if (condition_long_ema_trend and condition_long_rsi and
                condition_long_macd and condition_long_bb):

            signal = "BUY"
            current_pair_state['bot_in_position'] = True
            current_pair_state['current_position_type'] = "LONG"
            current_pair_state['entry_price'] = current_close_price
            current_pair_state['last_atr_for_sl_tp'] = current_atr

            current_pair_state['stop_loss_price'] = current_pair_state['entry_price'] - (current_pair_state['last_atr_for_sl_tp'] * SL_MULTIPLIER)
            current_pair_state['original_stop_loss_price'] = current_pair_state['stop_loss_price']
            current_pair_state['take_profit_prices']['TP1'] = current_pair_state['entry_price'] + (current_pair_state['last_atr_for_sl_tp'] * TP1_MULTIPLIER)
            current_pair_state['take_profit_prices']['TP2'] = current_pair_state['entry_price'] + (current_pair_state['last_atr_for_sl_tp'] * TP2_MULTIPLIER)
            current_pair_state['take_profit_prices']['TP3'] = current_pair_state['entry_price'] + (current_pair_state['last_atr_for_sl_tp'] * TP3_MULTIPLIER)
            current_pair_state['tp_levels_hit'] = {'TP1': False, 'TP2': False, 'TP3': False}

            save_bot_state()

            expected_duration_min = EXPECTED_MOVE_CANDLES_MIN * int(TIMEFRAME[:-1])
            expected_duration_max = EXPECTED_MOVE_CANDLES_MAX * int(TIMEFRAME[:-1])

            logger.info(f"[{symbol}] Сигнал: **ПОКУПКА (LONG)** - Открываем позицию.")
            logger.info(f"[{symbol}]    Вход: {current_pair_state['entry_price']:.8f}")
            logger.info(f"[{symbol}]    **Стоп-Лосс (SL):** ~{current_pair_state['stop_loss_price']:.8f} ({SL_MULTIPLIER} * ATR_входа)")
            logger.info(f"[{symbol}]    **Тейк-Профит 1 (TP1):** ~{current_pair_state['take_profit_prices']['TP1']:.8f} ({TP1_MULTIPLIER} * ATR_входа)")
            logger.info(f"[{symbol}]    **Тейк-Профит 2 (TP2):** ~{current_pair_state['take_profit_prices']['TP2']:.8f} ({TP2_MULTIPLIER} * ATR_входа)")
            logger.info(f"[{symbol}]    **Тейк-Профит 3 (TP3):** ~{current_pair_state['take_profit_prices']['TP3']:.8f} ({TP3_MULTIPLIER} * ATR_входа)")
            logger.info(f"[{symbol}]    **Ожидаемая длительность движения:** От {expected_duration_min:.0f} до {expected_duration_max:.0f} минут.")

        elif (condition_short_ema_trend and condition_short_rsi and
              condition_short_macd and condition_short_bb):

            signal = "SELL"
            current_pair_state['bot_in_position'] = True
            current_pair_state['current_position_type'] = "SHORT"
            current_pair_state['entry_price'] = current_close_price
            current_pair_state['last_atr_for_sl_tp'] = current_atr

            current_pair_state['stop_loss_price'] = current_pair_state['entry_price'] + (current_pair_state['last_atr_for_sl_tp'] * SL_MULTIPLIER)
            current_pair_state['original_stop_loss_price'] = current_pair_state['stop_loss_price']
            current_pair_state['take_profit_prices']['TP1'] = current_pair_state['entry_price'] - (current_pair_state['last_atr_for_sl_tp'] * TP1_MULTIPLIER)
            current_pair_state['take_profit_prices']['TP2'] = current_pair_state['entry_price'] - (current_pair_state['last_atr_for_sl_tp'] * TP2_MULTIPLIER)
            current_pair_state['take_profit_prices']['TP3'] = current_pair_state['entry_price'] - (current_pair_state['last_atr_for_sl_tp'] * TP3_MULTIPLIER)
            current_pair_state['tp_levels_hit'] = {'TP1': False, 'TP2': False, 'TP3': False}

            save_bot_state()

            expected_duration_min = EXPECTED_MOVE_CANDLES_MIN * int(TIMEFRAME[:-1])
            expected_duration_max = EXPECTED_MOVE_CANDLES_MAX * int(TIMEFRAME[:-1])

            logger.info(f"[{symbol}] Сигнал: **ПРОДАЖА (SHORT)** - Открываем позицию.")
            logger.info(f"[{symbol}]    Вход: {current_pair_state['entry_price']:.8f}")
            logger.info(f"[{symbol}]    **Стоп-Лосс (SL):** ~{current_pair_state['stop_loss_price']:.8f} ({SL_MULTIPLIER} * ATR_входа)")
            logger.info(f"[{symbol}]    **Тейк-Профит 1 (TP1):** ~{current_pair_state['take_profit_prices']['TP1']:.8f} ({TP1_MULTIPLIER} * ATR_входа)")
            logger.info(f"[{symbol}]    **Тейк-Профит 2 (TP2):** ~{current_pair_state['take_profit_prices']['TP2']:.8f} ({TP2_MULTIPLIER} * ATR_входа)")
            logger.info(f"[{symbol}]    **Тейк-Профит 3 (TP3):** ~{current_pair_state['take_profit_prices']['TP3']:.8f} ({TP3_MULTIPLIER} * ATR_входа)")
            logger.info(f"[{symbol}]    **Ожидаемая длительность движения:** От {expected_duration_min:.0f} до {expected_duration_max:.0f} минут.")
        else:
            if should_log_details:
                long_reasons = []
                short_reasons = []

                if not condition_long_ema_trend: long_reasons.append("EMA тренд не восходящий.")
                if not condition_long_rsi: long_reasons.append(f"RSI >= {70}.")
                if not condition_long_macd: long_reasons.append("MACD не бычий.")
                if not condition_long_bb: long_reasons.append("Цена не выше средней BB.")
                if long_reasons:
                    logger.info(f"[{symbol}]    Не выполнены условия для LONG: {'; '.join(long_reasons)}")

                if not condition_short_ema_trend: short_reasons.append("EMA тренд не нисходящий.")
                if not condition_short_rsi: short_reasons.append(f"RSI <= {30}.")
                if not condition_short_macd: short_reasons.append("MACD не медвежий.")
                if not condition_short_bb: short_reasons.append("Цена не ниже средней BB.")
                if short_reasons:
                    logger.info(f"[{symbol}]    Не выполнены условия для SHORT: {'; '.join(short_reasons)}")

                if not long_reasons and not short_reasons:
                    logger.info(f"[{symbol}]    Нет явных причин (просто условия не совпали для текущих настроек).")


    else: # Если бот уже в позиции, отслеживаем её и проверяем SL/TP
        
        position_closed = False

        if current_pair_state['current_position_type'] == "LONG":
            if current_close_price <= current_pair_state['stop_loss_price']:
                logger.info(f"[{symbol}] **ЗАКРЫТИЕ LONG по СТОП-ЛОССУ:** Цена {current_close_price:.8f} <= SL {current_pair_state['stop_loss_price']:.8f}.")
                signal = "CLOSE_LONG_SL"
                position_closed = True
            else:
                if not current_pair_state['tp_levels_hit']['TP3'] and current_close_price >= current_pair_state['take_profit_prices']['TP3']:
                    logger.info(f"[{symbol}] **TP 3 достигнуто (LONG)!** Цена: {current_close_price:.8f}")
                    current_pair_state['tp_levels_hit']['TP3'] = True
                    signal = "PARTIAL_CLOSE_LONG_TP3" 
                    position_closed = True
                    analyze_reversal_opportunity(symbol, df, current_pair_state['current_position_type'], current_close_price)
                elif not current_pair_state['tp_levels_hit']['TP2'] and current_close_price >= current_pair_state['take_profit_prices']['TP2']:
                    logger.info(f"[{symbol}] **TP 2 достигнуто (LONG)!** Цена: {current_close_price:.8f}")
                    current_pair_state['tp_levels_hit']['TP2'] = True
                    signal = "PARTIAL_CLOSE_LONG_TP2"
                    analyze_reversal_opportunity(symbol, df, current_pair_state['current_position_type'], current_close_price)
                elif not current_pair_state['tp_levels_hit']['TP1'] and current_close_price >= current_pair_state['take_profit_prices']['TP1']:
                    logger.info(f"[{symbol}] **TP 1 достигнуто (LONG)!** Цена: {current_close_price:.8f}")
                    current_pair_state['tp_levels_hit']['TP1'] = True
                    signal = "PARTIAL_CLOSE_LONG_TP1"
                else:
                    logger.info(f"[{symbol}]    Позиция LONG удерживается.")

        elif current_pair_state['current_position_type'] == "SHORT":
            if current_close_price >= current_pair_state['stop_loss_price']:
                logger.info(f"[{symbol}] **ЗАКРЫТИЕ SHORT по СТОП-ЛОССУ:** Цена {current_close_price:.8f} >= SL {current_pair_state['stop_loss_price']:.8f}.")
                signal = "CLOSE_SHORT_SL"
                position_closed = True
            else:
                if not current_pair_state['tp_levels_hit']['TP3'] and current_close_price <= current_pair_state['take_profit_prices']['TP3']:
                    logger.info(f"[{symbol}] **TP 3 достигнуто (SHORT)!** Цена: {current_close_price:.8f}")
                    current_pair_state['tp_levels_hit']['TP3'] = True
                    signal = "PARTIAL_CLOSE_SHORT_TP3"
                    position_closed = True
                    analyze_reversal_opportunity(symbol, df, current_pair_state['current_position_type'], current_close_price)
                elif not current_pair_state['tp_levels_hit']['TP2'] and current_close_price <= current_pair_state['take_profit_prices']['TP2']:
                    logger.info(f"[{symbol}] **TP 2 достигнуто (SHORT)!** Цена: {current_close_price:.8f}")
                    current_pair_state['tp_levels_hit']['TP2'] = True
                    signal = "PARTIAL_CLOSE_SHORT_TP2"
                    analyze_reversal_opportunity(symbol, df, current_pair_state['current_position_type'], current_close_price)
                elif not current_pair_state['tp_levels_hit']['TP1'] and current_close_price <= current_pair_state['take_profit_prices']['TP1']:
                    logger.info(f"[{symbol}] **TP 1 достигнуто (SHORT)!** Цена: {current_close_price:.8f}")
                    current_pair_state['tp_levels_hit']['TP1'] = True
                    signal = "PARTIAL_CLOSE_SHORT_TP1"
                else:
                    logger.info(f"[{symbol}]    Позиция SHORT удерживается.")

        if position_closed:
            closing_reason = "SL" if "SL" in signal else "TP3"
            logger.info(f"[{symbol}] **Полное закрытие позиции по {closing_reason}!** Состояние сброшено, поиск новой позиции.")
            bot_state[symbol] = initialize_pair_state(symbol)
            save_bot_state()

    return signal

# --- ОСНОВНАЯ ЛОГИКА БОТА ---
if __name__ == "__main__":
    logger.info("Бот для мониторинга рынка запущен. Нажмите Ctrl+C для остановки.")
    logger.info("Внимание: это демонстрационный режим, реальные ордера не исполняются.")

    logger.setLevel(logging.INFO)

    load_config()
    load_bot_state()

    try:
        while True:
            for symbol in SYMBOLS:
                data = fetch_data(symbol)

                if data is not None and not data.empty:
                    data_with_indicators = add_indicators(data, symbol)

                    if not data_with_indicators.empty:
                        signal = analyze_data(symbol, data_with_indicators)

                    else:
                        logger.warning(f"[{symbol}] DataFrame стал пустым после добавления индикаторов. Пропускаем анализ этой итерации.")
                else:
                    logger.warning(f"[{symbol}] Не удалось получить данные или DataFrame пуст после fetch_data (возможно, критическая ошибка или повторные попытки не увенчались успехом).")

            logger.info(f"\nОжидание {MONITORING_INTERVAL_SECONDS} секунд до следующей проверки всех пар...")
            time.sleep(MONITORING_INTERVAL_SECONDS)

    except KeyboardInterrupt:
        logger.info("\nБот остановлен пользователем.")
        save_bot_state()
    except Exception as e:
        logger.critical(f"Произошла непредвиденная ошибка в основном цикле: {e}", exc_info=True)
        save_bot_state()