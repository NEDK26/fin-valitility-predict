import os

import numpy as np
import pandas as pd
from datetime import datetime
from apscheduler.schedulers.blocking import BlockingScheduler
import exchange_calendars as xcals
import pytz

columns = [
    '现价', '开盘', '昨日收盘', '今日成交量', '买手', '卖手',
    '买5', '买5量', '买4', '买4量', '买3', '买3量', '买2', '买2量', '买1', '买1量',
    '卖1', '卖1量', '卖2', '卖2量', '卖3', '卖3量', '卖4', '卖4量', '卖5', '卖5量', '空列',
    'date', '涨价', '涨幅', '开盘价', '当日最低价', '现价/成交/成交金额'
]


def process_new_csv_files(data_directory, output_directory):
    today = datetime.now().strftime('%Y-%m-%d')
    todayStr = datetime.now().strftime('%Y%m%d')
    # 遍历数据目录中的所有CSV文件
    for filename in os.listdir(data_directory):
        if filename.endswith(".csv"):
            file_path = os.path.join(data_directory, filename)
            output_file_path = os.path.join(output_directory, f'{today}')
            print(output_file_path)
            # 读取CSV文件
            df = pd.read_csv(file_path, names=columns)
            df['date'] = df['date'].astype(str)
            df = df[df['date'].str.startswith(todayStr)]
            if len(df) == 0:
                continue
            df = calculate_wap(df)
            # return df
            # 执行数据处理
            # df['timestamp'] = pd.to_datetime(df['timestamp'])
            # df = df.sort_values(by='timestamp').reset_index(drop=True)
            # df.set_index('timestamp', inplace=True)
            #
            # # 重采样到每日频率，计算OHLC和成交量总和
            # df_resampled = df.resample('D').agg({
            #     'price': 'ohlc',  # 开盘价、最高价、最低价和收盘价
            #     'volume': 'sum'   # 成交量求和
            # })
            #
            # # 展平OHLC列
            # df_resampled.columns = ['_'.join(col).strip() for col in df_resampled.columns.values]
            # df_resampled = df_resampled.reset_index()
            #
            # 保存处理后的数据到新的CSV文件
            if not os.path.exists(output_file_path):
                os.makedirs(output_file_path)

            if os.path.exists(output_file_path):
                # 如果文件已存在，则追加数据
                df.to_csv(f'{output_file_path}/{filename}', mode='a', header=False, index=False)
            else:
                # 否则，创建新文件
                df.to_csv(output_file_path, index=False)

            print(f"Processed file: {filename}")


def calculate_wap(df):
    a = df
    a.set_index('date', inplace=False)
    a = a.reset_index().set_index('date')
    a = a.iloc[:, 7:27]
    # a.head()
    _, __ = 0, 0
    for i in range(1, 6):
        a[f'买{i}'].replace(0, np.nan, inplace=True)
        a[f'卖{i}'].fillna(method='ffill', inplace=True)

    for i in range(1, 6):
        # a[f'wap{i}']=(a[f'买{i}']*a[f'买{i}量']+a[f'卖{i}']*a[f'卖{i}量'])/(a[f'买{i}量']+a[f'卖{i}量'])

        _ = _ + a[f'买{i}'] * a[f'买{i}量'] + a[f'卖{i}'] * a[f'卖{i}量']
        __ = __ + a[f'买{i}量'] + a[f'卖{i}量']

        a[f'wap{i}'] = _ / __
    a['index'] = range(1, len(a) + 1)  # plot x axit

    for i in range(1, 6):
        a.loc[:, f'log_return{i}'] = log_return(a[f'wap{i}'])
        a = a[~a[f'log_return{i}'].isnull()]

    return a


def log_return(list_stock_prices):
    return np.log(list_stock_prices).diff()


def is_trading_time():
    timezone = pytz.timezone("Etc/GMT+0")
    date = datetime.now().astimezone(timezone).date()
    exchange_name = "XSHG"
    exchange = xcals.get_calendar(exchange_name)
    return exchange.is_session(date)


if __name__ == "__main__":
    data_directory = '/home/work3/xmd2/B/dataSource/Lob/OriginLob'
    output_directory = '/home/work3/xmd2/B/dataSource/Lob/LobData/'
    # process_new_csv_files(data_directory, output_directory)
    scheduler = BlockingScheduler()

    # 添加每天执行的任务
    scheduler.add_job(process_new_csv_files, 'cron', hour=16, minute=0, args=[data_directory, output_directory])
    if is_trading_time():
        # 启动调度器
        scheduler.start()
