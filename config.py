# <------------------------------------------------------------------------------------------------------------------------------->
# Пример использования:
csv_input_path = "C:/Users/lexx-/my_it_project/Breakout_Finder/project/data/SOLUSDT-1m-2023-11-17.csv"
csv_output_path = "C:/Users/lexx-/my_it_project/Breakout_Finder/project/data/SOLUSDT-1m-2023-11-17_rem.csv"
bo_size = 50  # размер базового ордера
dca_order_size = 150  # размер СО
so_step = 0.01  # шаг DCA в% (0.02 = 2%)
tp = 0.015  # желаемая доходность в %(0.01 = 1%)
dca_quantity = 6  # количество DCA ордеров (без начального ордера)
init_modules_quantity_long = 8  # количество модулей мультипозиционности(экземпляров DCATPplaceCalc) которые смогут активироваться. Не может быть меньше 1
init_modules_quantity_short = 8  # количество модулей мультипозиционности(экземпляров DCATPplaceCalc) которые смогут активироваться. Не может быть меньше 1
step_multiposition = 0.4  # шаг между модулями в % (1 = 1%)
init_long_or_short = "both"  # направление сигнала("long", "short" или "both")
init_sig_long_id = 30  # сколько сигналов лонг будем обрабатывать из датафрейма
init_sig_short_id = 30  # сколько сигналов шорт будем обрабатывать из датафрейма
multiplier_step_multiposition = 1.0  # множитель шага между модулями
