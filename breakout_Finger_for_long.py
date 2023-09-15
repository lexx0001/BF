import pandas as pd
import pytz


csv_path = "c:/Users/lexx-/OneDrive/Рабочий стол/CODING/Breakout_Finder/test/RUNEUSDT-1m-2023-09-04.csv"
df = pd.read_csv(csv_path)
ph = []
pl = []
hgst = []
bull_cup_results = []
# high_max_list = []
chwidths = []  # список для хранения вычисленных ширин каналов
window_chw = 300  # размер окна для вычисления ширины канала


# Инициализация параметров
cwidthu = 0.06  # коэффициент ширины канала
mintest = 2
prd = 4  # Размер окна для поиска "Pivot Highs"
bo_len = 200  # Максимальное количество баров для обработки


def load_csv(csv_path):
    global df
    df = pd.read_csv(csv_path)

    # Преобразуем столбцы, содержащие числа, из строк в числа с плавающей запятой
    for col in ["open", "high", "low", "close"]:
        df[col] = df[col].replace(",", ".", regex=True).astype(float)


# Преобразуем столбец "open_time" в формат даты и времени
def convert_open_time(df):
    # Проверяем, было ли уже выполнено преобразование
    if "converted" in df.columns:
        return df
    print("Преобразуем столбец open_time в формат даты и времени...")
    df["open_time"] = pd.to_datetime(df["open_time"], unit="ms")

    # Применяем часовой пояс UTC+3
    tz = pytz.timezone("Europe/Moscow")  # UTC+3 для Москвы
    df["open_time"] = df["open_time"].dt.tz_localize("UTC").dt.tz_convert(tz)

    # Преобразуем дату и время в нужный формат
    df["open_time"] = df["open_time"].dt.strftime("%d-%m-%Y %H:%M")

    # Добавляем столбец, указывающий, что преобразование выполнено
    df["converted"] = None

    return df


def save_to_csv(df, csv_path):
    df.to_csv(csv_path, index=False)


load_csv(csv_path)
convert_open_time(df)


def find_pivot_highs(df, index, prd):
    if index < prd * 2 or index > len(df) - 1:
        return None
    window = df["high"].iloc[index - prd * 2 : index + 1].values
    high_max = max(window[-prd:])
    max_value = max(window)

    if max_value == window[prd] and window[prd] > high_max:
        return window[prd]

    return None


def find_pivot_lows(df, index, prd):
    if index < prd * 2 or index > len(df) - 1:
        return None
    window = df["low"].iloc[index - prd * 2 : index + 1].values
    low_min = min(window[-prd:])
    min_value = min(window)

    if min_value == window[prd] and window[prd] < low_min:
        return window[prd]

    return None


def chwidth_func(df, index, window_chw, cwidthu):
    if index < window_chw:
        return None

    high_max = df["high"].iloc[index - window_chw : index + 1].max()
    low_min = df["low"].iloc[index - window_chw : index + 1].min()

    chwidth = (high_max - low_min) * cwidthu
    return chwidth


# Вычисляем наивысшую цену в окне prd
def hgst_func(df, index, prd):
    if index < prd:
        return None
    hgst = df["high"].iloc[index - prd : index].max()
    return hgst


# будем искать "Бычий кубок"
def bull_cup(df, ph, index, bo_len, mintest, hgst_value, chw_value):
    if index >= 300:
        ph_only_values = (
            pd.Series(ph[index - bo_len : index]).dropna().reset_index(drop=True)
        )
        ph_only_values = ph_only_values.iloc[::-1]  # Переворачиваем значения
        close = df["close"].iloc[index]
        open = df["open"].iloc[index]
        bomax = None
        num = 0
        if len(ph_only_values) >= mintest and close > open and close > hgst[index]:
            xx = 0
            bomax = ph_only_values.iloc[0]
            for x in ph_only_values:
                if x >= close:
                    break
                xx += 1
                bomax = max(bomax, x)
            if xx > mintest and open <= bomax:
                for x in range(xx):
                    ph_x = ph_only_values.iloc[x]
                    if ph_x <= bomax and ph_x >= bomax - chw_value:
                        num += 1
                if num < mintest or hgst_value >= bomax:
                    bomax = None
        if bomax is not None and num >= mintest:
            message = "Breakout"
            return message
        else:
            return None



# Главный цикл
for index in range(len(df)):
    hgst_value = hgst_func(df, index, prd)
    hgst.append(hgst_value)
    ph_value = find_pivot_highs(df, index, prd)
    ph.append(ph_value)
    pl_value = find_pivot_lows(df, index, prd)
    pl.append(pl_value)
    chw_value = chwidth_func(df, index, window_chw, cwidthu)
    chwidths.append(chw_value)
    result = bull_cup(df, ph, index, bo_len, mintest, hgst_value, chw_value)
    bull_cup_results.append(result)



df["breakout"] = bull_cup_results
# df["ph"] = ph
# df["pl"] = pl
# df["hgst"] = hgst
# df["high_max"] = high_max_list
# df["chwidth"] = chwidths
save_to_csv(df, csv_path)
