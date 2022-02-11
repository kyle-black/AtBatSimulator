import pandas as pd
import sqlite3 as sql

#import sqlite3

connection = sql.connect("bayesatbat.db")

df = pd.read_sql('SELECT * FROM EVENT', con=connection)

df.to_csv('MAIN.csv')

# print(df)
