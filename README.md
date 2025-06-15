Для начала нужно создать новую среду (я использую conda)

```
conda create -n economic_forecast python=3.9
conda activate economic_forecast
```

Туда поставить нужные библиотеки

```
conda install -c conda-forge pandas scikit-learn matplotlib seaborn
pip install kaggle tk
conda install -c anaconda tk
```
Запуск происходит таким образом

```
conda activate economic_forecast
python economic_forecast_app.py
```

Я написал .bat для запуска по клику но захардкодил там пути к оболочке anaconda и python потому что виндовс... (мне лень)
