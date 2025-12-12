#!/usr/bin/env python3
"""Display HPO study status from optuna.db"""
import sqlite3
import sys

try:
    conn = sqlite3.connect('optuna.db')
    cursor = conn.cursor()

    cursor.execute('SELECT study_name FROM studies')
    studies = cursor.fetchall()

    if not studies:
        print('No HPO studies found in optuna.db')
        sys.exit(0)

    for (study_name,) in studies:
        print(f'\nStudy: {study_name}')

        cursor.execute(
            'SELECT state, COUNT(*) FROM trials WHERE study_id = '
            '(SELECT study_id FROM studies WHERE study_name = ?) '
            'GROUP BY state', (study_name,)
        )

        for state, count in cursor.fetchall():
            print(f'  {state:12s}: {count:4d}')

    conn.close()
except sqlite3.OperationalError as e:
    print(f'Error: Cannot access optuna.db - {e}')
    sys.exit(1)
except Exception as e:
    print(f'Error: {e}')
    sys.exit(1)
