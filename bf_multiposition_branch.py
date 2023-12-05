import pandas as pd
import pytz
import os


csv_input_path = "C:/Users/lexx-/my_it_project/Breakout_Finder/project/data/SOLUSDT-1m-2023-11-17.csv"
csv_output_path = "C:/Users/lexx-/my_it_project/Breakout_Finder/project/data/SOLUSDT-1m-2023-11-17_rem.csv"


class BreakoutFinder:  # индикатор "Бычий/Медвежий кубок" - создание сизналов long и short
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
        self.len_df = len(self.df)  # длина датафрейма
        self.main_loop()  # запускаем главный цикл

    def find_pivot_highs(self, index):
        df = self.df
        prd = self.prd
        if index < prd * 2 or index > self.len_df - 1:
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
        if index < prd * 2 or index > self.len_df - 1:
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
    def bull_cup(self, index, hgst_value, chw_value):
        mintest = self.mintest
        if index >= 300:
            ph_only_values = (
                pd.Series(self.ph[index - self.bo_len : index])
                .dropna()
                .reset_index(drop=True)
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
                message = True
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
    def bear_cup(self, index, lwst_value, chw_value):
        mintest = self.mintest
        if index >= 300:
            pl_only_values = (
                pd.Series(self.pl[index - self.bo_len : index])
                .dropna()
                .reset_index(drop=True)
            )
            pl_only_values = pl_only_values.iloc[::-1]  # Переворачиваем значения
            close = self.df["close"].iloc[index]
            open = self.df["open"].iloc[index]
            bomin = None
            num = 0
            if (
                len(pl_only_values) >= mintest
                and close < open
                and close < self.lwst[index]
            ):
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
                message = True
                return message
            else:
                return None

    # Главный цикл
    def main_loop(self):
        for index in range(self.len_df):
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
            # self.df["long_sig"] = self.bull_cup_results

            if self.mode == "short" or self.mode == "both":
                lwst_value = self.lwst_func(index)
                self.lwst.append(lwst_value)
                pl_value = self.find_pivot_lows(index)
                self.pl.append(pl_value)
                result_low = self.bear_cup(index, lwst_value, chw_value)
            self.bear_cup_results.append(result_low)

        self.df["long_sig"] = self.bull_cup_results
        self.df["short_sig"] = self.bear_cup_results


class CSVHandler:  # работа по обработке csv файлов, извлечение в pandas, преобразование времени
    def __init__(self, csv_path, time_convert=False):
        self.csv_path = csv_path
        self.load_csv(csv_path)
        # self.save_to_csv(csv_output_path)
        self.convert_open_time(time_convert)

    def load_csv(
        self, csv_path, columns_for_load=["open_time", "open", "high", "low", "close"]
    ):
        # узнаём какие столбцы есть в файле
        rows_csv = pd.read_csv(csv_path, nrows=0)
        self.columns_csv = rows_csv.columns.tolist()
        # если в файле нет названных столбцов, то возвращаем ошибку
        if not set(columns_for_load).issubset(self.columns_csv):
            raise ValueError(
                f"В файле нет столбцов {set(columns_for_load) - set(self.columns_csv)}"
            )

        columns_for_load = columns_for_load
        self.df = pd.read_csv(csv_path, usecols=columns_for_load)

        # Преобразуем столбец "open_time" в формат даты и времени

    def convert_open_time(self, time_convert):
        # Проверяем, было ли уже выполнено преобразование
        if "convert" in self.columns_csv:
            return
        # Преобразуем столбцы, содержащие числа, из строк в числа с плавающей запятой
        for col in ["open", "high", "low", "close"]:
            self.df[col] = self.df[col].replace(",", ".", regex=True).astype(float)

        if time_convert:
            # "Преобразуем столбец open_time в формат даты и времени...")
            self.df["open_utc_time"] = pd.to_datetime(self.df["open_time"], unit="ms")

            # Применяем часовой пояс UTC+3
            tz = pytz.timezone("Europe/Moscow")  # UTC+3 для Москвы
            self.df["open_utc_time"] = (
                self.df["open_utc_time"].dt.tz_localize("UTC").dt.tz_convert(tz)
            )

            # Преобразуем дату и время в нужный формат
            self.df["open_utc_time"] = self.df["open_utc_time"].dt.strftime(
                "%d-%m-%Y %H:%M"
            )

            # Добавляем столбец, указывающий, что преобразование выполнено
            self.df["convert"] = None
            # self.save_to_csv(self.csv_path)

    def save_to_csv(self, csv_path):
        directory = os.path.dirname(csv_path)
        if directory and not os.path.exists(directory):
            os.makedirs(directory)
        self.df.to_csv(csv_path, index=False)


class DCAorderCalc:  # расчитывает значения для DCA ордеров и данные для анализа
    def __init__(
        self,
        bo_price,
        bo_size,
        dca_order_size,
        dca_step,
        tp,
        dca_quantity,
        long_or_short,
    ):
        self.results_long = []
        self.results_short = []
        if long_or_short == "long":
            self.take_profit_percentage_long(
                bo_price,
                bo_size,
                dca_order_size,
                dca_step,
                tp,
                dca_quantity,
            )
        elif long_or_short == "short":
            self.take_profit_percentage_short(
                bo_price,
                bo_size,
                dca_order_size,
                dca_step,
                tp,
                dca_quantity,
            )

    def take_profit_percentage_long(
        self, bo_price, bo_size, dca_order_size, so_step, tp, dca_quantity
    ):
        accumulated_size = bo_size  # Инициализируем общий размер открытых ордеров с учетом базового ордера
        total_cost_with_profit_init = bo_size * (
            1 + tp
        )  # учитываем прибыль от первого ордера
        total_shares_init = bo_size / bo_price  # количество монет от первого ордера
        order_price_init = bo_price  # цена размещения ордера

        # для всех последующих ордеров
        for dca_quantity in range(dca_quantity + 1):
            total_cost_with_profit = (
                total_cost_with_profit_init  # Используем инициализированные значения
            )
            total_shares = total_shares_init  # Используем инициализированные значения
            order_price = order_price_init  # Используем инициализированные значения

            for i in range(
                1, dca_quantity + 1
            ):  # начинаем с 1, так как базовый ордер уже учтен
                order_price = bo_price * (1 - so_step) ** i  # цена размещения ордера
                total_cost_with_profit += dca_order_size * (
                    1 + tp
                )  # учитываем прибыль от ордера
                total_shares += (
                    dca_order_size / order_price
                )  # накапливаем количество монет
            tp_price = total_cost_with_profit / total_shares
            # absolute_tp = tp_price - order_price
            acc_tp_in_usdt = accumulated_size * tp  # накапливаем прибыль

            self.results_long.append(
                {
                    "dca_number": dca_quantity,
                    "order_price": order_price,
                    "tp_price": tp_price,
                    # "absolute_tp": absolute_tp,
                    "acc_tp_in_usdt": acc_tp_in_usdt,
                    "accumulated_size": accumulated_size,
                }
            )
            if accumulated_size > 0:
                accumulated_size += (
                    dca_order_size  # Обновляем общий размер открытых ордеров
                )
        return self.results_long

    def take_profit_percentage_short(
        self, bo_price, bo_size, dca_order_size, so_step, tp, dca_quantity
    ):
        accumulated_size = bo_size  # Инициализируем общий размер открытых ордеров с учетом базового ордера
        total_cost_with_profit_init = bo_size * (
            1 - tp
        )  # учитываем прибыль от первого ордера
        total_shares_init = bo_size / bo_price  # количество монет от первого ордера
        order_price_init = bo_price  # цена размещения ордера
        # для всех последующих ордеров
        for dca_quantity in range(dca_quantity + 1):
            total_cost_with_profit = (
                total_cost_with_profit_init  # Используем инициализированные значения
            )
            total_shares = total_shares_init  # Используем инициализированные значения
            order_price = order_price_init  # Используем инициализированные значения
            for i in range(1, dca_quantity + 1):
                order_price = bo_price * (1 + so_step) ** i  # цена размещения ордера
                total_cost_with_profit += dca_order_size * (
                    1 - tp
                )  # учитываем прибыль от ордера
                total_shares += (
                    dca_order_size / order_price
                )  # количество монет от ордера
            tp_price = total_cost_with_profit / total_shares
            # absolute_tp = order_price - tp_price
            acc_tp_in_usdt = accumulated_size * tp  # накапливаем прибыль
            self.results_short.append(
                {
                    "dca_number": dca_quantity,
                    "order_price": order_price,
                    "tp_price": tp_price,
                    # "absolute_tp": absolute_tp,
                    "acc_tp_in_usdt": acc_tp_in_usdt,
                    "accumulated_size": accumulated_size,
                }
            )
            if accumulated_size > 0:
                accumulated_size += dca_order_size
        return self.results_short
        # tp_percentage = (bo_price / tp_price - 1) * 100  # изменение от БО в %


class CalcMultiposPrices:
    def __init__(
        self,
        multipos_bo_price,
        step_multiposition,
        modules_quantity,
        long_or_short,
        multiplier=1.0,
    ):
        self.prices = [multipos_bo_price]
        step = multipos_bo_price * step_multiposition / 100
        for i in range(1, modules_quantity):
            if long_or_short == "long":
                price = self.prices[-1] - step
            elif long_or_short == "short":
                price = self.prices[-1] + step
            else:
                raise ValueError(
                    f"Invalid value for long_or_short: {long_or_short}. Expected 'long' or 'short'."
                )
            self.prices.append(price)
            step *= multiplier

    def get_prices(self):
        return self.prices


class DCAandTPplaceCalc:  # расчитывает место размещения DCA ордеров и TP
    def __init__(
        self,
        df,
        bo_size,
        dca_order_size,
        dca_step,
        tp,
        dca_quantity,
    ):
        self.df = df
        self.bo_size = bo_size
        self.dca_order_size = dca_order_size
        self.dca_step = dca_step
        self.tp = tp
        self.dca_quantity = dca_quantity
        self.long_sig_id = 0
        self.short_sig_id = 0
        self.dca_calc_list_long = []
        self.dca_calc_list_short = []
        self.main_loop()

    def calculate(self, index):
        # вычисления для long
        if pd.notna(self.df["long_sig"].iloc[index]):  # <<<input signal
            bo_price = self.df["open"].iloc[index + 1]
            self.dca_calc_list_long = DCAorderCalc(
                bo_price,
                self.bo_size,
                self.dca_order_size,
                self.dca_step,
                self.tp,
                self.dca_quantity,
                "long",
            ).results_long
            len_dca_calc_list = len(self.dca_calc_list_long)
            self.long_sig_id += 1
            open_time = self.df["open_time"].iloc[index + 1]
            print(f"\n ID long signal #{self.long_sig_id} at {open_time}")
            candle_df = self.df.loc[
                index + 1 :, ["open_time", "low", "high"]
            ]  # Создаем новый DataFrame для удобства
            tp_lost = 0.0
            remaining_calc = 0
            for _, (time, cand_min, cand_max) in candle_df.iterrows():
                for calc_num in range(remaining_calc, len_dca_calc_list):
                    ord_price = self.dca_calc_list_long[calc_num]["order_price"]

                    if (
                        cand_min < ord_price < cand_max
                    ):  # <<<<<<<<< - можно добавить обрезку по краям свечи для более правдоподобного поведения
                        # open_time = self.df["open_time"].iloc[index + 1 + cand_num]
                        print(
                            f"Order {calc_num} fits in candle time {int(time)}; Order {ord_price}, Candle {cand_min, cand_max}"
                        )
                        remaining_calc += 1
                        tp_lost = self.dca_calc_list_long[calc_num]["tp_price"]

                if (
                    cand_min < tp_lost < cand_max
                    and tp_lost
                    > 0  # можно добавить фильтр по направлению свечи (например, если последний размещённый ордер был в той же свече что и ТП, и свеча лонг то ОК)
                ):  # вычисление положения TP
                    print(
                        f"TP №{remaining_calc - 1} fits in candle time {int(time)}; TP {tp_lost}, Candle {cand_min, cand_max}"
                    )
                    break
        # вычисления для short
        elif pd.notna(self.df["short_sig"].iloc[index]):
            bo_price = self.df["open"].iloc[index + 1]
            self.dca_calc_list_short = DCAorderCalc(
                bo_price,
                self.bo_size,
                self.dca_order_size,
                self.dca_step,
                self.tp,
                self.dca_quantity,
                "short",
            ).results_short
            len_dca_calc_list = len(self.dca_calc_list_short)
            self.short_sig_id += 1
            open_time = self.df["open_time"].iloc[index + 1]
            print(f"\n ID short signal #{self.short_sig_id} at {open_time}")
            candle_df = self.df.loc[
                index + 1 :, ["open_time", "low", "high"]
            ]  # Создаем новый DataFrame для удобства
            tp_lost = 0.0
            remaining_calc = 0
            for _, (time, cand_min, cand_max) in candle_df.iterrows():
                for calc_num in range(remaining_calc, len_dca_calc_list):
                    ord_price = self.dca_calc_list_short[calc_num]["order_price"]

                    if (
                        cand_min < ord_price < cand_max
                    ):  # <<<<<<<<< - можно добавить обрезку по краям свечи для более правдоподобного поведения
                        # open_time = self.df["open_time"].iloc[index + 1 + cand_num]
                        print(
                            f"Order {calc_num} fits in candle time {int(time)}; Order {ord_price}, Candle {cand_min, cand_max}"
                        )
                        remaining_calc += 1
                        tp_lost = self.dca_calc_list_short[calc_num]["tp_price"]

                if (
                    cand_min < tp_lost < cand_max and tp_lost > 0
                ):  # вычисление положения TP
                    print(
                        f"TP №{remaining_calc - 1} fits in candle time {int(time)}; TP {tp_lost}, Candle {cand_min, cand_max}"
                    )
                    break

    def main_loop(self):
        for index in range(len(self.df)):
            self.calculate(index)


# будем писать отдельный класс для расчета TP и DCA long ордеров


class DCAandTPplaceCalcLong:  # расчитывает место размещения DCA ордеров и TP
    def __init__(
        self,
        df,
        bo_size,
        dca_order_size,
        dca_step,
        tp,
        dca_quantity,
    ):
        self.df = df
        self.bo_size = bo_size
        self.dca_order_size = dca_order_size
        self.dca_step = dca_step
        self.tp = tp
        self.dca_quantity = dca_quantity
        # self.short_sig_id = 0
        self.dca_calc_list_long = []
        # self.dca_calc_list_short = []
        self.multipos_bo_price = None  # начальная цена для мультипозиционности
        # self.main_loop()
        self.multi_price = 0.0
        self.candle_df = df[["open_time", "low", "high"]]

    def calculate(self, index):
        bo_price = self.df.at[index, "open"]
        if self.multipos_bo_price is None:
            self.multipos_bo_price = bo_price
        self.dca_calc_list_long = DCAorderCalc(
            bo_price,
            self.bo_size,
            self.dca_order_size,
            self.dca_step,
            self.tp,
            self.dca_quantity,
            "long",
        ).results_long
        len_dca_calc_list = len(self.dca_calc_list_long)
        # open_time = self.df["open_time"].iloc[index]
        candle_df = self.candle_df.loc[index:]  # Создаем новый DataFrame для удобства
        tp_lost = 0.0
        remaining_calc = 0
        ord_price = 0.0

        for row in candle_df.itertuples():  # прогон до конца графика
            # прогоняем по свечке каждое значение из списка
            for calc_num in range(remaining_calc, len_dca_calc_list):
                ord_price = self.dca_calc_list_long[calc_num]["order_price"]

                if (
                    row.low < ord_price < row.high
                ):  # <<<<<<<<< - можно добавить обрезку по краям свечи для более правдоподобного поведения
                    # open_time = self.df["open_time"].iloc[index + 1 + cand_num]

                    print(
                        f"Order {calc_num} fits in candle time {int(row.open_time)}; Order {ord_price}, Candle {row.low, row.high}"
                    )
                    remaining_calc += 1
                    tp_lost = self.dca_calc_list_long[calc_num]["tp_price"]

            # отрабатываем ТП
            if (
                row.low < tp_lost < row.high
                and tp_lost
                > 0  # можно добавить фильтр по направлению свечи (например, если последний размещённый ордер был в той же свече что и ТП, и свеча лонг то ОК)
            ):  # вычисление положения TP
                print(
                    f"TP №{remaining_calc - 1} fits in candle time {int(row.open_time)}; TP {tp_lost}, Candle {row.low, row.high}"
                )
                self.tp_times = row.open_time  # список для времени срабатывания TP
                break

    # def main_loop(self):
    #     for index in range(len(self.df)):
    #         self.calculate(index)


# <------------------------------------------------------------------------------------------------------------------------------->
# Пример использования:
# bo_price = 0.8797  # начальная цена
bo_size = 50  # размер базового ордера
dca_order_size = 150  # размер СО
so_step = 0.01  # шаг DCA в% (0.02 = 2%)
tp = 0.03  # желаемая доходность в %(0.01 = 1%)
dca_quantity = 6  # количество DCA ордеров (без начального ордера)
modules_quantity = 4  # количество модулей (экземпляров DCATPplaceCalcLong) которые смогут активироваться
step_multiposition = 1  # шаг между модулями в % (1 = 1%)
long_or_short = "long"  # направление сигнала
multiplier_step_multiposition = 1.0  # множитель шага между модулями


def run():
    bigdata = CSVHandler(csv_input_path, time_convert=False)
    bf = BreakoutFinder(bigdata.df, mode="both")
    # bigdata.save_to_csv(
    #     csv_output_path
    # )  # при необходимости сохранить результаты в файл(раскомментировать)
    last_calc = DCAandTPplaceCalc(
        bigdata.df, bo_size, dca_order_size, so_step, tp, dca_quantity
    )


# print(timeit.timeit(run, number=30))

# run()


def run_long():
    # читаем данные из файла
    bigdata = CSVHandler(csv_input_path, time_convert=False)
    # обрабатываем (меняем) данные в датафрейме
    bf = BreakoutFinder(bigdata.df, mode="long")
    # создаём экземпляр класса для расчёта места размещения DCA ордеров и TP
    dca_long_instance = DCAandTPplaceCalcLong(
        bigdata.df,
        bo_size,
        dca_order_size,
        so_step,
        tp,
        dca_quantity,
    )

    sig_quantity = -1  # счётчик сигналов
    multipos_bo_price = []  # список начальных цен для мультипозиционности
    multipos_self_quantity = modules_quantity  # счётчик модулей (экземпляров DCATPplaceCalcLong) которые смогут активироваться
    prices = None
    tp_times = []  # Будем собирать время срабатывания TP

    for ind, row in enumerate(
        bigdata.df[:-1].itertuples()
    ):  # перебираем весь датафрейм (кроме последней строки)
        # увеличиваем индекс на 1 ибо сигнал срабатывает на открытии следующей свечи
        indnext = ind + 1
        next_row = bigdata.df.iloc[indnext]
        # создание флага для закрытия мультипозиций если нет текущих открытых модулей
        if not (len(tp_times) == 0 or next_row.open_time <= max(tp_times)):
            break
        # если модули закончились, то заканчиваем (по сути обрабатываем только один сигнал на срабатывание)
        if multipos_self_quantity == 0:  # <<<<<<дописать условие проверки по времени
            break

        if pd.notna(row.long_sig):  # если видим сигнал
            sig_quantity += 1
            multipos_bo_price.append(next_row.open)
            # вычисляем цены для мультипозиционности
            calc = CalcMultiposPrices(
                multipos_bo_price[-1],
                step_multiposition,
                modules_quantity,
                long_or_short,
                multiplier_step_multiposition,
            )
            prices = calc.get_prices()
            print(f"\n \n multipos_bo_price {prices}")

            print(
                f"\n сигнал №{sig_quantity} на {next_row.open_time} цена {multipos_bo_price[-1]}"
            )
        if prices is not None:
            start_index = len(prices) - multipos_self_quantity
            for price in prices[start_index:]:
                if next_row.low < price < next_row.high:
                    multipos_self_quantity -= 1
                    print(
                        f"<<<<<<<<<<<<<Multiorder #{start_index} activity! Price {price}. Time {next_row.open_time}>>>>>>>>>>>>>>>"
                    )
                    dca_long_instance.calculate(indnext)
                    # получаем время срабатывания TP
                    tp_times.append(dca_long_instance.tp_times)
                if multipos_self_quantity == 0:
                    print(
                        f"----------------End of multipositions signal #{sig_quantity}--------------------"
                    )
                    break


run_long()

# calc = CalcMultiposPrices(100, 1, 5, "short", 1)
# prices = calc.get_prices()
# print(prices)
