import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import seaborn as sns
import kaggle
import os
import tempfile
import re
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta

class EconomicForecastApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Прогнозирование экономических показателей")
        self.root.geometry("1200x900")
        
        # Переменные
        self.data = None
        self.filtered_data = None
        self.target_var = tk.StringVar()
        self.date_var = tk.StringVar()
        self.model_var = tk.StringVar(value="Linear Regression")
        self.kaggle_dataset_var = tk.StringVar(value="varpit94/us-inflation-data-updated-till-may-2021")
        self.kaggle_api_configured = False
        self.forecast_years = tk.IntVar(value=5)
        self.train_start_year = tk.IntVar(value=2000)
        
        # Переменные для фильтрации
        self.filter_column_var = tk.StringVar()
        self.filter_value_var = tk.StringVar()
        self.filter_values = []
        
        # Проверка конфигурации Kaggle API
        self.check_kaggle_config()
        
        # Создание интерфейса
        self.create_widgets()
        
    def check_kaggle_config(self):
        """Проверяет, настроен ли Kaggle API"""
        kaggle_dir = os.path.expanduser('~/.kaggle')
        if os.path.exists(os.path.join(kaggle_dir, 'kaggle.json')):
            self.kaggle_api_configured = True
        else:
            self.kaggle_api_configured = False
            
    def create_widgets(self):
        # Главный контейнер
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill="both", expand=True, padx=10, pady=10)
        
        # Фрейм загрузки данных
        frame_load = ttk.LabelFrame(main_frame, text="Загрузка данных")
        frame_load.pack(fill="x", padx=5, pady=5)
        
        # Поле для ввода Kaggle dataset
        ttk.Label(frame_load, text="Датасет Kaggle:").grid(row=0, column=0, padx=5, pady=5, sticky="w")
        kaggle_entry = ttk.Entry(frame_load, textvariable=self.kaggle_dataset_var, width=50)
        kaggle_entry.grid(row=0, column=1, padx=5, pady=5, sticky="we")
        
        # Примеры популярных датасетов
        ttk.Label(frame_load, text="Примеры: ").grid(row=0, column=2, padx=5, pady=5, sticky="w")
        examples = [
             ("unitednations/global-commodity-trade-statistics", "Торговая статистика"),
            ("dwdkills/russian-demography", "РФ демография"),
            ("varpit94/us-inflation-data-updated-till-may-2021", "Инфляция")
        ]
        
        for i, (dataset, desc) in enumerate(examples):
            ttk.Button(
                frame_load, 
                text=desc, 
                width=15,
                command=lambda ds=dataset: self.kaggle_dataset_var.set(ds)
            ).grid(row=0, column=3+i, padx=2, pady=5)
        
        # Кнопки загрузки
        button_frame = ttk.Frame(frame_load)
        button_frame.grid(row=1, column=0, columnspan=5, pady=5)
        
        ttk.Button(button_frame, text="Загрузить CSV", command=self.load_csv).pack(side="left", padx=5)
        ttk.Button(button_frame, text="Скачать с Kaggle", command=self.download_kaggle).pack(side="left", padx=5)
        ttk.Button(button_frame, text="Проверить Kaggle API", command=self.check_and_configure_kaggle).pack(side="left", padx=5)
        
        # Статус Kaggle API
        self.api_status = ttk.Label(button_frame, text="Статус Kaggle API: " + ("Настроен" if self.kaggle_api_configured else "Не настроен!"))
        self.api_status.pack(side="left", padx=10)
        if not self.kaggle_api_configured:
            self.api_status.configure(foreground="red")
        
        # Фрейм фильтрации данных
        filter_frame = ttk.LabelFrame(frame_load, text="Фильтрация данных")
        filter_frame.grid(row=2, column=0, columnspan=5, sticky="we", padx=5, pady=5)
        
        ttk.Label(filter_frame, text="Колонка для фильтрации:").grid(row=0, column=0, padx=5, pady=5)
        self.filter_col_combo = ttk.Combobox(filter_frame, textvariable=self.filter_column_var, state="disabled", width=20)
        self.filter_col_combo.grid(row=0, column=1, padx=5, pady=5)
        self.filter_col_combo.bind("<<ComboboxSelected>>", self.update_filter_values)
        
        ttk.Label(filter_frame, text="Значение:").grid(row=0, column=2, padx=5, pady=5)
        self.filter_val_combo = ttk.Combobox(filter_frame, textvariable=self.filter_value_var, state="disabled", width=20)
        self.filter_val_combo.grid(row=0, column=3, padx=5, pady=5)
        
        ttk.Button(filter_frame, text="Применить фильтр", command=self.apply_filter).grid(row=0, column=4, padx=5, pady=5)
        ttk.Button(filter_frame, text="Сбросить фильтр", command=self.reset_filter).grid(row=0, column=5, padx=5, pady=5)
        
        # Фрейм настроек
        frame_settings = ttk.LabelFrame(main_frame, text="Настройки модели")
        frame_settings.pack(fill="x", padx=5, pady=5)
        
        # Выбор столбцов
        ttk.Label(frame_settings, text="Столбец с датой:").grid(row=0, column=0, padx=5, pady=5)
        self.date_combo = ttk.Combobox(frame_settings, textvariable=self.date_var, state="disabled", width=20)
        self.date_combo.grid(row=0, column=1, padx=5, pady=5, sticky="we")
        
        ttk.Label(frame_settings, text="Целевая переменная:").grid(row=0, column=2, padx=5, pady=5)
        self.target_combo = ttk.Combobox(frame_settings, textvariable=self.target_var, state="disabled", width=20)
        self.target_combo.grid(row=0, column=3, padx=5, pady=5, sticky="we")
        
        ttk.Label(frame_settings, text="Модель:").grid(row=0, column=4, padx=5, pady=5)
        model_combo = ttk.Combobox(frame_settings, textvariable=self.model_var, state="readonly", width=15)
        model_combo['values'] = ("Linear Regression", "Random Forest", "Gradient Boosting")
        model_combo.grid(row=0, column=5, padx=5, pady=5, sticky="we")
        
        ttk.Label(frame_settings, text="Начальный год обучения:").grid(row=0, column=6, padx=5, pady=5)
        ttk.Entry(frame_settings, textvariable=self.train_start_year, width=10).grid(row=0, column=7, padx=5, pady=5)
        
        ttk.Label(frame_settings, text="Прогноз на (лет):").grid(row=0, column=8, padx=5, pady=5)
        ttk.Spinbox(frame_settings, from_=1, to=20, textvariable=self.forecast_years, width=5).grid(row=0, column=9, padx=5, pady=5)
        
        # Кнопка прогноза
        ttk.Button(main_frame, text="Выполнить прогноз", command=self.run_forecast).pack(pady=10)
        
        # Фрейм результатов
        frame_results = ttk.LabelFrame(main_frame, text="Результаты")
        frame_results.pack(fill="both", expand=True, padx=5, pady=5)
        
        # Панель с вкладками
        notebook = ttk.Notebook(frame_results)
        notebook.pack(fill="both", expand=True, padx=5, pady=5)
        
        # Вкладка с текстовыми результатами
        text_frame = ttk.Frame(notebook)
        notebook.add(text_frame, text="Прогноз")
        
        self.text_results = scrolledtext.ScrolledText(text_frame, height=8)
        self.text_results.pack(fill="both", expand=True, padx=5, pady=5)
        self.text_results.insert(tk.END, "Результаты прогнозирования появятся здесь...")
        
        # Вкладка с графиком
        graph_frame = ttk.Frame(notebook)
        notebook.add(graph_frame, text="График прогноза")
        
        self.figure_frame = ttk.Frame(graph_frame)
        self.figure_frame.pack(fill="both", expand=True, padx=5, pady=5)
        
        # Вкладка с данными
        data_frame = ttk.Frame(notebook)
        notebook.add(data_frame, text="Данные")
        
        # Виджет для отображения данных
        self.data_tree = ttk.Treeview(data_frame)
        self.data_tree.pack(fill="both", expand=True, padx=5, pady=5)
        
        # Добавляем скроллбары
        vsb = ttk.Scrollbar(data_frame, orient="vertical", command=self.data_tree.yview)
        hsb = ttk.Scrollbar(data_frame, orient="horizontal", command=self.data_tree.xview)
        self.data_tree.configure(yscrollcommand=vsb.set, xscrollcommand=hsb.set)
        
        vsb.pack(side="right", fill="y")
        hsb.pack(side="bottom", fill="x")
    
    def update_filter_values(self, event=None):
        col = self.filter_column_var.get()
        if col and self.data is not None:
            unique_values = self.data[col].unique()
            self.filter_values = sorted([str(val) for val in unique_values])
            self.filter_val_combo['values'] = self.filter_values
            self.filter_val_combo['state'] = 'readonly'
    
    def apply_filter(self):
        filter_col = self.filter_column_var.get()
        filter_val = self.filter_value_var.get()
        
        if not filter_col or not filter_val:
            messagebox.showwarning("Ошибка", "Выберите колонку и значение для фильтрации!")
            return
        
        try:
            self.filtered_data = self.data[self.data[filter_col].astype(str) == filter_val]
            
            self.target_combo['values'] = list(self.filtered_data.columns)
            self.date_combo['values'] = list(self.filtered_data.columns)
            self.show_data_preview()
            
            messagebox.showinfo("Фильтр", f"Применен фильтр: {filter_col} = {filter_val}\nОсталось строк: {len(self.filtered_data)}")
        except Exception as e:
            messagebox.showerror("Ошибка фильтрации", f"Не удалось применить фильтр: {str(e)}")
    
    def reset_filter(self):
        self.filtered_data = None
        self.filter_column_var.set('')
        self.filter_value_var.set('')
        self.filter_val_combo['state'] = 'disabled'
        
        if self.data is not None:
            self.target_combo['values'] = list(self.data.columns)
            self.date_combo['values'] = list(self.data.columns)
            self.show_data_preview()
        
        messagebox.showinfo("Фильтр", "Фильтр сброшен")
    
    def check_and_configure_kaggle(self):
        kaggle_dir = os.path.expanduser('~/.kaggle')
        if not os.path.exists(kaggle_dir):
            os.makedirs(kaggle_dir)
        
        kaggle_json = os.path.join(kaggle_dir, 'kaggle.json')
        
        if os.path.exists(kaggle_json):
            messagebox.showinfo("Kaggle API", "Kaggle API уже настроен!")
            self.kaggle_api_configured = True
            self.api_status.configure(text="Статус Kaggle API: Настроен", foreground="green")
            return
        
        message = (
            "Kaggle API не настроен!\n\n"
            "1. Зарегистрируйтесь на Kaggle.com\n"
            "2. Зайдите в настройки аккаунта\n"
            "3. Создайте новый API токен\n"
            "4. Скачайте файл kaggle.json\n"
            "5. Поместите его в папку: ~/.kaggle/\n\n"
            "После настройки перезапустите приложение."
        )
        messagebox.showinfo("Настройка Kaggle API", message)
        self.kaggle_api_configured = False
        self.api_status.configure(text="Статус Kaggle API: Не настроен!", foreground="red")
    
    def convert_to_numeric(self, s):
        if pd.api.types.is_numeric_dtype(s):
            return s
        
        try:
            return pd.to_numeric(s, errors='raise')
        except:
            return s
    
    def detect_and_convert_year_columns(self, df):
        """Обнаруживает столбцы, содержащие только годы, и преобразует их в даты"""
        for col in df.columns:
            # Пропускаем столбцы, которые выглядят как даты
            if any(keyword in col.lower() for keyword in ['date', 'year', 'time']):
                # Проверяем, содержит ли столбец только 4-значные числа
                if df[col].apply(lambda x: re.match(r'^\d{4}$', str(x)) is not None).all():
                    try:
                        # Преобразуем в дату (1 января указанного года)
                        df[col] = pd.to_datetime(df[col].astype(str) + pd.offsets.YearBegin(0))
                        messagebox.showinfo("Авто-преобразование", 
                                           f"Столбец '{col}' был распознан как год и преобразован в дату (1 января)")
                    except:
                        pass
        return df
    
    def load_csv(self):
        file_path = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv"), ("All files", "*.*")])
        if file_path:
            try:
                encodings = ['utf-8', 'latin1', 'windows-1251', 'ISO-8859-1']
                for encoding in encodings:
                    try:
                        self.data = pd.read_csv(file_path, encoding=encoding, parse_dates=False)
                        break
                    except UnicodeDecodeError:
                        continue
                else:
                    self.data = pd.read_csv(file_path, encoding='utf-8', errors='replace', parse_dates=False)
                
                # Авто-преобразование столбцов с годами в даты
                self.data = self.detect_and_convert_year_columns(self.data)
                
                # Преобразуем числовые столбцы
                for col in self.data.columns:
                    if not any(keyword in col.lower() for keyword in ['date', 'year', 'time']):
                        self.data[col] = self.convert_to_numeric(self.data[col])
                
                # Сбрасываем фильтр
                self.reset_filter()
                
                # Обновляем элементы интерфейса
                self.filter_col_combo['values'] = list(self.data.columns)
                self.filter_col_combo['state'] = 'readonly'
                
                self.target_combo['values'] = list(self.data.columns)
                self.target_combo['state'] = 'readonly'
                self.date_combo['values'] = list(self.data.columns)
                self.date_combo['state'] = 'readonly'
                self.show_data_preview()
                messagebox.showinfo("Успех", f"Данные загружены!\nФайл: {os.path.basename(file_path)}\nСтрок: {len(self.data)}")
            except Exception as e:
                messagebox.showerror("Ошибка", f"Ошибка при чтении файла: {str(e)}")
    
    def download_kaggle(self):
        dataset = self.kaggle_dataset_var.get().strip()
        
        if not re.match(r"^[a-zA-Z0-9_-]+/[a-zA-Z0-9_-]+$", dataset):
            messagebox.showerror("Ошибка", "Неверный формат ссылки Kaggle!\nИспользуйте: владелец/набор-данных")
            return
        
        if not self.kaggle_api_configured:
            self.check_and_configure_kaggle()
            if not self.kaggle_api_configured:
                messagebox.showerror("Ошибка", "Kaggle API не настроен! Сначала настройте API.")
                return
        
        try:
            with tempfile.TemporaryDirectory() as tmp_dir:
                kaggle.api.dataset_download_files(dataset, path=tmp_dir, unzip=True, quiet=False)
                files = [f for f in os.listdir(tmp_dir) if f.endswith('.csv')]
                
                if files:
                    file_path = os.path.join(tmp_dir, files[0])
                    
                    encodings = ['utf-8', 'latin1', 'windows-1251', 'ISO-8859-1']
                    for encoding in encodings:
                        try:
                            self.data = pd.read_csv(file_path, encoding=encoding, parse_dates=False)
                            break
                        except UnicodeDecodeError:
                            continue
                    else:
                        self.data = pd.read_csv(file_path, encoding='utf-8', errors='replace', parse_dates=False)
                    
                    # Авто-преобразование столбцов с годами в даты
                    self.data = self.detect_and_convert_year_columns(self.data)
                    
                    for col in self.data.columns:
                        if not any(keyword in col.lower() for keyword in ['date', 'year', 'time']):
                            self.data[col] = self.convert_to_numeric(self.data[col])
                    
                    # Сбрасываем фильтр
                    self.reset_filter()
                    
                    self.filter_col_combo['values'] = list(self.data.columns)
                    self.filter_col_combo['state'] = 'readonly'
                    
                    self.target_combo['values'] = list(self.data.columns)
                    self.target_combo['state'] = 'readonly'
                    self.date_combo['values'] = list(self.data.columns)
                    self.date_combo['state'] = 'readonly'
                    self.show_data_preview()
                    messagebox.showinfo("Успех", f"Данные скачаны!\nНабор: {dataset}\nФайл: {files[0]}\nСтрок: {len(self.data)}")
                else:
                    messagebox.showwarning("Предупреждение", f"В наборе {dataset} не найдены CSV файлы! Пытаемся найти другие файлы...")
                    all_files = os.listdir(tmp_dir)
                    if all_files:
                        file_path = os.path.join(tmp_dir, all_files[0])
                        try:
                            self.data = pd.read_csv(file_path)
                            
                            # Авто-преобразование столбцов с годами в даты
                            self.data = self.detect_and_convert_year_columns(self.data)
                            
                            for col in self.data.columns:
                                if not any(keyword in col.lower() for keyword in ['date', 'year', 'time']):
                                    self.data[col] = self.convert_to_numeric(self.data[col])
                            
                            # Сбрасываем фильтр
                            self.reset_filter()
                            
                            self.filter_col_combo['values'] = list(self.data.columns)
                            self.filter_col_combo['state'] = 'readonly'
                            
                            self.target_combo['values'] = list(self.data.columns)
                            self.target_combo['state'] = 'readonly'
                            self.date_combo['values'] = list(self.data.columns)
                            self.date_combo['state'] = 'readonly'
                            self.show_data_preview()
                            messagebox.showinfo("Успех", f"Данные загружены из {all_files[0]}!\nСтрок: {len(self.data)}")
                        except Exception as e:
                            messagebox.showerror("Ошибка", f"Не удалось прочитать файл {all_files[0]}: {str(e)}")
                    else:
                        messagebox.showerror("Ошибка", f"В наборе {dataset} не найдены файлы!")
        except kaggle.rest.ApiException as e:
            if e.status == 403:
                error_msg = (
                    "Ошибка 403: Доступ запрещен!\n\n"
                    "Возможные причины:\n"
                    "1. Вы не приняли правила использования датасета на странице Kaggle\n"
                    "2. Датсет требует регистрации на соревновании\n"
                    "3. Ваш API токен неверен или устарел\n\n"
                    f"Пожалуйста, посетите страницу датасета: https://www.kaggle.com/datasets/{dataset}"
                )
                messagebox.showerror("Ошибка доступа", error_msg)
            else:
                messagebox.showerror("Ошибка Kaggle API", f"Статус: {e.status}\nОшибка: {e.reason}")
        except Exception as e:
            error_msg = str(e)
            if "404" in error_msg:
                error_msg += "\nПроверьте правильность ссылки на датасет"
            messagebox.showerror("Ошибка", f"Ошибка при загрузке данных: {error_msg}")
    
    def show_data_preview(self):
        self.data_tree.delete(*self.data_tree.get_children())
        
        current_data = self.filtered_data if self.filtered_data is not None else self.data
        
        columns = list(current_data.columns)
        self.data_tree["columns"] = columns
        self.data_tree["show"] = "headings"
        
        for col in columns:
            self.data_tree.heading(col, text=col)
            self.data_tree.column(col, width=100, anchor="center")
        
        for i, row in current_data.head(100).iterrows():
            values = [str(x) if not pd.isna(x) else "" for x in row]
            self.data_tree.insert("", "end", values=values)
    
    def run_forecast(self):
        current_data = self.filtered_data if self.filtered_data is not None else self.data
        
        if current_data is None:
            messagebox.showwarning("Ошибка", "Сначала загрузите данные!")
            return
        
        target = self.target_var.get()
        date_col = self.date_var.get()
        
        if not target or not date_col:
            messagebox.showwarning("Ошибка", "Выберите целевую переменную и столбец с датой!")
            return
        
        try:
            # Преобразуем целевую переменную
            current_data[target] = self.convert_to_numeric(current_data[target])
            
            # Преобразуем даты
            # Проверяем, является ли столбец датой в формате "только год"
            if current_data[date_col].dtype == 'int64' and current_data[date_col].between(1800, 2200).all():
                # Преобразуем год в дату (1 января)
                current_data[date_col] = pd.to_datetime(current_data[date_col].astype(str), format='%Y')
            else:
                # Пробуем автоматическое преобразование
                try:
                    current_data[date_col] = pd.to_datetime(current_data[date_col], errors='coerce')
                except:
                    # Если не удалось, пробуем разные форматы
                    for fmt in ['%Y', '%Y-%m', '%Y-%m-%d', '%d.%m.%Y', '%m/%d/%Y']:
                        try:
                            current_data[date_col] = current_data[date_col].apply(lambda x: datetime.strptime(str(x), fmt))
                            break
                        except:
                            continue
            
            # Удаляем строки с невалидными датами
            current_data = current_data.dropna(subset=[date_col])
            
            # Сортируем по дате
            current_data = current_data.sort_values(by=date_col)
            
            # Создаем числовые признаки из даты
            current_data['year'] = current_data[date_col].dt.year
            current_data['month'] = current_data[date_col].dt.month
            current_data['day'] = current_data[date_col].dt.day
            
            # Фильтруем данные для обучения по выбранному году
            start_year = self.train_start_year.get()
            train_data = current_data[current_data['year'] >= start_year]
            
            if len(train_data) == 0:
                messagebox.showerror("Ошибка", f"Нет данных для обучения, начиная с {start_year} года!")
                return
            
            # Подготовка данных для обучения
            X_train = train_data[['year', 'month', 'day']]
            y_train = train_data[target]
            
            # Проверяем наличие пропущенных значений
            if X_train.isnull().any().any() or y_train.isnull().any():
                messagebox.showwarning("Предупреждение", 
                                      "В данных есть пропущенные значения! Они будут заполнены средними.")
                X_train = X_train.fillna(X_train.mean())
                y_train = y_train.fillna(y_train.mean())
            
            # Выбор модели
            model_name = self.model_var.get()
            if model_name == "Linear Regression":
                model = LinearRegression()
            elif model_name == "Random Forest":
                model = RandomForestRegressor(n_estimators=100, random_state=42, max_depth=5)
            elif model_name == "Gradient Boosting":
                model = GradientBoostingRegressor(n_estimators=100, random_state=42, max_depth=3)
            
            # Обучение модели
            model.fit(X_train, y_train)
            
            # Прогноз на будущее
            forecast_years = self.forecast_years.get()
            last_date = current_data[date_col].max()
            
            # Определяем частоту данных (годовая, месячная и т.д.)
            freq = 'Y'  # По умолчанию годовая частота
            if len(current_data) > 1:
                time_diffs = current_data[date_col].diff().dropna()
                if not time_diffs.empty:
                    # Определяем наиболее распространенный интервал
                    common_diff = time_diffs.mode().iloc[0] if not time_diffs.mode().empty else time_diffs.iloc[0]
                    
                    if common_diff < timedelta(days=32):
                        freq = 'M'  # Месячная частота
                    elif common_diff < timedelta(days=365):
                        freq = 'D'  # Дневная частота
            
            # Создаем даты для прогноза
            future_dates = []
            if freq == 'Y':
                for i in range(1, forecast_years + 1):
                    future_dates.append(last_date + relativedelta(years=i))
            elif freq == 'M':
                for i in range(1, forecast_years * 12 + 1):
                    future_dates.append(last_date + relativedelta(months=i))
            else:
                for i in range(1, forecast_years + 1):
                    future_dates.append(last_date + relativedelta(years=i))
            
            # Создаем данные для прогноза
            future_df = pd.DataFrame({
                'date': future_dates,
                'year': [d.year for d in future_dates],
                'month': [d.month for d in future_dates],
                'day': [d.day for d in future_dates]
            })
            
            X_future = future_df[['year', 'month', 'day']]
            y_future = model.predict(X_future)
            
            # Вывод результатов
            self.text_results.delete(1.0, tk.END)
            self.text_results.insert(tk.END, f"Модель: {model_name}\n")
            self.text_results.insert(tk.END, f"Целевая переменная: {target}\n")
            self.text_results.insert(tk.END, f"Период обучения: с {start_year} года\n")
            self.text_results.insert(tk.END, f"Размер обучающей выборки: {len(X_train)}\n")
            self.text_results.insert(tk.END, f"Прогноз на {forecast_years} {'лет' if freq == 'Y' else 'месяцев'}:\n\n")
            
            for i in range(len(future_dates)):
                self.text_results.insert(tk.END, f"{future_dates[i].strftime('%Y-%m-%d')}: {y_future[i]:.2f}\n")
            
            # Визуализация
            self.plot_time_series(date_col, target, train_data, future_dates, y_future, model, model_name, freq)
            
        except Exception as e:
            import traceback
            error_details = traceback.format_exc()
            messagebox.showerror("Ошибка", f"{str(e)}\n\nДетали ошибки:\n{error_details}")
    
    def plot_time_series(self, date_col, target, train_data, future_dates, y_future, model, model_name, freq='Y'):
        # Очистка предыдущих графиков
        for widget in self.figure_frame.winfo_children():
            widget.destroy()
        
        # Создаем фигуру
        fig, ax = plt.subplots(figsize=(14, 7))
        
        # Полные исторические данные
        current_data = self.filtered_data if self.filtered_data is not None else self.data
        all_dates = current_data[date_col]
        all_values = current_data[target]
        ax.plot(all_dates, all_values, 'o-', color='gray', label="Все исторические данные", alpha=0.5)
        
        # Данные, использованные для обучения
        train_dates = train_data[date_col]
        train_values = train_data[target]
        ax.plot(train_dates, train_values, 'b-', label="Данные обучения")
        
        # Прогноз на будущее
        ax.plot(future_dates, y_future, 'g--', label="Прогноз на будущее")
        ax.scatter(future_dates, y_future, color='green', s=50)
        
        # Для линейной регрессии строим всю линию регрессии
        if model_name == "Linear Regression":
            min_date = min(all_dates.min(), future_dates[0])
            max_date = max(all_dates.max(), future_dates[-1])
            
            # Создаем последовательность дат с правильной частотой
            if freq == 'Y':
                date_range = pd.date_range(start=min_date, end=max_date, freq='YS')
            elif freq == 'M':
                date_range = pd.date_range(start=min_date, end=max_date, freq='MS')
            else:
                date_range = pd.date_range(start=min_date, end=max_date, freq='D')
            
            pred_df = pd.DataFrame({'date': date_range})
            pred_df['year'] = pred_df['date'].dt.year
            pred_df['month'] = pred_df['date'].dt.month
            pred_df['day'] = pred_df['date'].dt.day
            
            predictions = model.predict(pred_df[['year', 'month', 'day']])
            
            ax.plot(pred_df['date'], predictions, 'r-', label="Линейная регрессия", alpha=0.7)
        
        # Для других моделей
        else:
            train_predictions = model.predict(train_data[['year', 'month', 'day']])
            ax.plot(train_dates, train_predictions, 'r-', label="Предсказания модели", alpha=0.7)
            
            if len(train_dates) > 0 and len(future_dates) > 0:
                last_train_date = train_dates.iloc[-1]
                first_forecast_date = future_dates[0]
                last_value = train_predictions[-1]
                first_forecast_value = y_future[0]
                
                ax.plot([last_train_date, first_forecast_date], [last_value, first_forecast_value], 'r--')
        
        # Настройки графика
        ax.set_title(f"Прогнозирование {target} с помощью {model_name}", fontsize=14)
        ax.set_xlabel("Дата", fontsize=12)
        ax.set_ylabel(target, fontsize=12)
        ax.grid(True, linestyle='--', alpha=0.7)
        ax.legend(loc='best')
        
        # Форматирование дат на оси X
        fig.autofmt_xdate()
        
        # Встраивание в Tkinter
        canvas = FigureCanvasTkAgg(fig, master=self.figure_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill="both", expand=True)

if __name__ == "__main__":
    root = tk.Tk()
    app = EconomicForecastApp(root)
    root.mainloop()