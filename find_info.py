import sqlite3
import os

print("--- Searching Chrome history ---")
if os.path.exists('D:/ch.db'):
    conn = sqlite3.connect('D:/ch.db')
    for row in conn.execute("SELECT url, title, last_visit_time FROM urls WHERE url LIKE '%amazon%' OR url LIKE '%challenge%' OR url LIKE '%entity%' OR title LIKE '%amazon%' OR title LIKE '%challenge%'"):
        print(row)
    conn.close()

print("--- Searching Edge history ---")
if os.path.exists('D:/ed.db'):
    conn = sqlite3.connect('D:/ed.db')
    for row in conn.execute("SELECT url, title, last_visit_time FROM urls WHERE url LIKE '%amazon%' OR url LIKE '%challenge%' OR url LIKE '%entity%' OR title LIKE '%amazon%' OR title LIKE '%challenge%'"):
        print(row)
    conn.close()
