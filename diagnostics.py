# -*- coding: utf-8 -*-

import sys

print(f"--- Диагностика Окружения ---")
print(f"Путь к интерпретатору Python: {sys.executable}")

print("\n--- Версии Библиотек ---")
try:
    import numpy
    print(f"numpy: версия {numpy.__version__} | путь {numpy.__file__}")
except Exception as e:
    print(f"numpy: НЕ НАЙДЕНО или ошибка импорта | {e}")

try:
    import pandas_ta
    print(f"pandas_ta: версия {pandas_ta.version} | путь {pandas_ta.__file__}")
except Exception as e:
    print(f"pandas_ta: НЕ НАЙДЕНО или ошибка импорта | {e}")

print("\n--- Попытка проблемного импорта ---")
try:
    from numpy import NaN
    print("Импорт 'NaN' из numpy: УСПЕШНО (старый метод)")
except ImportError:
    print("Импорт 'NaN' из numpy: НЕ УДАЛСЯ (старый метод)")

try:
    from numpy import nan
    print("Импорт 'nan' из numpy: УСПЕШНО (новый метод)")
except ImportError:
    print("Импорт 'nan' из numpy: НЕ УДАЛСЯ (новый метод)")

print("---------------------------------")