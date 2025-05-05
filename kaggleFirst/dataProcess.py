import os
import numpy as np
import pandas as pd
path = r"D:\kaggleData\competitions\Data_Dictionary.xlsx"
table = pd.read_excel(path, header=2, sheet_name='train')
print(table)