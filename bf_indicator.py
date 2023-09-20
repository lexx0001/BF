import pandas as pd
import pytz
import os

csv_input_path = "c:/Users/lexx-/OneDrive/Рабочий стол/CODING/Breakout_Finder/test/1000PEPEUSDT-1m-2023-09-17.csv"
csv_output_path = "c:/Users/lexx-/OneDrive/Рабочий стол/CODING/Breakout_Finder/test/1000PEPEUSDT-1m-2023-09-17_remaster.csv"


class BreakoutFinder:
    def __init__(
        self,
        df,
        prd=5,
        mintest=1,
        cwidthu=0.06,
        bo_len=200,
        mode="both",
        window_chw=300,
    ):
        
        self.mode = mode  # режим работы индикатора: "long", "short" или "both"
        self.df = df  # датафрейм с данными
        self.prd = prd  # размер окна для поиска
        self.mintest = mintest  # минимальное количество баров для обработки
        self.cwidthu = cwidthu  # коэффициент ширины канала (0.01 - это 1%)
        self.bo_len = bo_len  # количество баров для обработки (не более 300)
        self.ph = []  # список для хранения вычисленных высоких пиков
        self.pl = []  # список для хранения вычисленных низких пиков
        self.hgst = []  # список для хранения вычисленных наивысших цен
        self.lwst = []  # список для хранения вычисленных наименьших цен
        self.bull_cup_results = (
            []
        )  # список для хранения результатов поиска "Бычьего кубка"
        self.bear_cup_results = (
            []
        )  # список для хранения результатов поиска "Медвежьего кубка"
        # self.chwidths = []  # список для хранения вычисленных ширин каналов
        self.window_chw = window_chw  # размер окна для вычисления ширины канала
        self.main_loop() # запускаем главный цикл

    def find_pivot_highs(self, index):
        df = self.df
        prd = self.prd
        if index < prd * 2 or index > len(df) - 1:
            return None
        window = df["high"].iloc[index - prd * 2 : index + 1].values
        high_max = max(window[-prd:])
        max_value = max(window)

        if max_value == window[prd] and window[prd] > high_max:
            return window[prd]

        return None

    def find_pivot_lows(self, index):
        df = self.df
        prd = self.prd
        if index < prd * 2 or index > len(df) - 1:
            return None
        window = df["low"].iloc[index - prd * 2 : index + 1].values
        low_min = min(window[-prd:])
        min_value = min(window)

        if min_value == window[prd] and window[prd] < low_min:
            return window[prd]

        return None

    def chwidth_func(self, index):
        window_chw = self.window_chw

        if index < window_chw:
            return None

        high_max = self.df["high"].iloc[index - window_chw : index + 1].max()
        low_min = self.df["low"].iloc[index - window_chw : index + 1].min()

        chwidth = (high_max - low_min) * self.cwidthu
        return chwidth

    # Вычисляем наивысшую цену в окне prd
    def hgst_func(self, index):
        if index < self.prd:
            return None
        hgst = self.df["high"].iloc[index - self.prd : index].max()
        return hgst

    # будем искать "Бычий кубок"
    def bull_cup(self,index, hgst_value, chw_value):
        mintest = self.mintest
        if index >= 300:
            ph_only_values = (
                pd.Series(self.ph[index - self.bo_len : index]).dropna().reset_index(drop=True)
            )
            ph_only_values = ph_only_values.iloc[::-1]  # Переворачиваем значения
            close = self.df["close"].iloc[index]
            open = self.df["open"].iloc[index]
            bomax = None
            num = 0
            if (
                len(ph_only_values) >= mintest
                and close > open
                and close > self.hgst[index]
            ):
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
            
    # Вычисляем наименьшую цену в окне prd
    def lwst_func(self, index):        
        if index < self.prd:
            return None
        lwst = self.df["low"].iloc[index - self.prd : index].min()
        return lwst

    # будем искать "Медвежий кубок"
    def bear_cup(self,index, lwst_value, chw_value):
        mintest = self.mintest
        if index >= 300:
            pl_only_values = (
                pd.Series(self.pl[index - self.bo_len : index]).dropna().reset_index(drop=True)
            )
            pl_only_values = pl_only_values.iloc[::-1]  # Переворачиваем значения
            close = self.df["close"].iloc[index]
            open = self.df["open"].iloc[index]
            bomin = None
            num = 0
            if len(pl_only_values) >= mintest and close < open and close < self.lwst[index]:
                xx = 0
                bomin = pl_only_values.iloc[0]
                for x in pl_only_values:
                    if x <= close:
                        break
                    xx += 1
                    bomin = min(bomin, x)
                if xx > mintest and open >= bomin:
                    for x in range(xx):
                        pl_x = pl_only_values.iloc[x]
                        if pl_x >= bomin and pl_x <= bomin + chw_value:
                            num += 1
                    if num < mintest or lwst_value <= bomin:
                        bomin = None
            if bomin is not None and num >= mintest:
                message = "Breakdown"
                return message
            else:
                return None

    # Главный цикл
    def main_loop(self):
        for index in range(len(self.df)):
            result_high = None
            result_low = None
            chw_value = self.chwidth_func(index)
            if self.mode == "long" or self.mode == "both":
                hgst_value = self.hgst_func(index)
                self.hgst.append(hgst_value)
                ph_value = self.find_pivot_highs(index)
                self.ph.append(ph_value)
                result_high = self.bull_cup(index, hgst_value, chw_value)
            self.bull_cup_results.append(result_high)
                # self.df["breakout"] = self.bull_cup_results

            if self.mode == "short" or self.mode == "both":
                lwst_value = self.lwst_func(index)
                self.lwst.append(lwst_value)
                pl_value = self.find_pivot_lows(index)
                self.pl.append(pl_value)
                result_low = self.bear_cup(index, lwst_value, chw_value)
            self.bear_cup_results.append(result_low)
                # self.df["breakdown"] = self.bear_cup_results
        self.df["breakout"] = self.bull_cup_results
        self.df["breakdown"] = self.bear_cup_results

    # df["ph"] = ph
    # df["pl"] = pl
    # df["hgst"] = hgst
    # df["high_max"] = high_max_list
    # df["chwidth"] = chwidths


class CSVHandler:
    def __init__(self, csv_path):
        self.csv_path = csv_path
        self.load_csv(csv_path)
        self.convert_open_time()
        # self.save_to_csv(csv_path)

    def load_csv(self, csv_path):
        self.df = pd.read_csv(csv_path)

        # Преобразуем столбец "open_time" в формат даты и времени

    def convert_open_time(self):
        # Проверяем, было ли уже выполнено преобразование
        if "convert" in self.df.columns:
            return
        # Преобразуем столбцы, содержащие числа, из строк в числа с плавающей запятой
        for col in ["open", "high", "low", "close"]:
            self.df[col] = self.df[col].replace(",", ".", regex=True).astype(float)

        # "Преобразуем столбец open_time в формат даты и времени...")
        self.df["open_time"] = pd.to_datetime(self.df["open_time"], unit="ms")

        # Применяем часовой пояс UTC+3
        tz = pytz.timezone("Europe/Moscow")  # UTC+3 для Москвы
        self.df["open_time"] = (
            self.df["open_time"].dt.tz_localize("UTC").dt.tz_convert(tz)
        )

        # Преобразуем дату и время в нужный формат
        self.df["open_time"] = self.df["open_time"].dt.strftime("%d-%m-%Y %H:%M")

        # Добавляем столбец, указывающий, что преобразование выполнено
        self.df["convert"] = None
        self.save_to_csv(self.csv_path)

    def save_to_csv(self, csv_path):
        if not os.path.exists(os.path.dirname(csv_path)): #если файл не существует, создаем его
            os.makedirs(os.path.dirname(csv_path))
        self.df.to_csv(csv_path, index=False)


csv_data = CSVHandler(csv_input_path)
bf = BreakoutFinder(csv_data.df, mode="both")

csv_data.save_to_csv(csv_output_path)
