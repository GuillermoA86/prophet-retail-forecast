import pandas as pd
import numpy as np

def load_and_merge_data(data_dir="data/"):
    print("Cargando datasets y aplicando limpieza metodológica...")
    transactions = pd.read_csv(f"{data_dir}transactions.csv")
    stores = pd.read_csv(f"{data_dir}stores.csv")
    calendar = pd.read_csv(f"{data_dir}calendar.csv")

    # Limpieza y conversión de fechas con formato latinoamericano
    transactions['date'] = pd.to_datetime(transactions['date'], format='%d/%m/%Y')
    calendar['date'] = pd.to_datetime(calendar['date'], format='%d/%m/%Y')

    # 3. Corrección de caracteres especiales (ñ, acentos) en calendar (season, holiday_name)
    for col in calendar.columns:
        if calendar[col].dtype == 'object':
            calendar[col] = (calendar[col]
                             .str.replace('Ã±', 'ñ', regex=False)
                             .str.replace('Ã¡', 'á', regex=False)
                             .str.replace('Ã©', 'é', regex=False)
                             .str.replace('Ãí', 'í', regex=False)
                             .str.replace('Ãó', 'ó', regex=False)
                             .str.replace('Ãú', 'ú', regex=False))

    # 4. Imputación lógica para resolver la "trampa" de datos faltantes en transactions.csv
    # cash_transactions = total_transactions - card_transactions
    mask_cash_t = transactions['cash_transactions'].isna()
    transactions.loc[mask_cash_t, 'cash_transactions'] = (
        transactions.loc[mask_cash_t, 'total_transactions'] - transactions.loc[mask_cash_t, 'card_transactions']
    )

    # amount_cash = amount_total - amount_card
    mask_amt_c = transactions['amount_cash'].isna()
    transactions.loc[mask_amt_c, 'amount_cash'] = (
        transactions.loc[mask_amt_c, 'amount_total'] - transactions.loc[mask_amt_c, 'amount_card']
    )

    # avg_ticket = amount_total / total_transactions
    mask_avg = transactions['avg_ticket'].isna()
    transactions.loc[mask_avg, 'avg_ticket'] = (
        transactions.loc[mask_avg, 'amount_total'] / transactions.loc[mask_avg, 'total_transactions']
    )

    # units_sold = amount_total / avg_ticket
    mask_units = transactions['units_sold'].isna()
    transactions.loc[mask_units, 'units_sold'] = (
        transactions.loc[mask_units, 'amount_total'] / transactions.loc[mask_units, 'avg_ticket']
    )

    # Merge principal con stores y calendar
    df = transactions.merge(stores, on='store_id', how='left')
    df = df.merge(calendar, on='date', how='left')

    df['replenishment_signal'] = df['replenishment_signal'].fillna(0)
    df = df.sort_values(['store_id', 'category', 'date']).reset_index(drop=True)
    return df