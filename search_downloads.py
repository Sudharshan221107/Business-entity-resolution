import sqlite3
import os

print("--- Searching Chrome downloads ---")
if os.path.exists('D:/ch.db'):
    conn = sqlite3.connect('D:/ch.db')
    for row in conn.execute("SELECT target_path, tab_url, site_url FROM downloads WHERE target_path LIKE '%amazon%' OR target_path LIKE '%6ab5628d%' OR tab_url LIKE '%amazon%' OR tab_url LIKE '%6ab5628d%'"):
        print(row)
    conn.close()

print("--- Searching Edge downloads ---")
if os.path.exists('D:/ed.db'):
    conn = sqlite3.connect('D:/ed.db')
    for row in conn.execute("SELECT target_path, tab_url, site_url FROM downloads WHERE target_path LIKE '%amazon%' OR target_path LIKE '%6ab5628d%' OR tab_url LIKE '%amazon%' OR tab_url LIKE '%6ab5628d%'"):
        print(row)
    conn.close()
