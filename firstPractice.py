import pandas as pd
import numpy
names = ["ken", "sandy", "frank"]
firstCharacter = {name : name[0] for name in names}
print(firstCharacter)
print("----------------------分割线-------------------------")

table = pd.DataFrame({'Bob': ['I liked it.', 'It was awful.'],  # DataFrame
              'Sue': ['Pretty good.', 'Bland.']},
             index=['Product A', 'Product B'])
print(table)

print("----------------------分割线-------------------------")
series = pd.Series([1, 2, 3], index=['first', 'second', 'third'], name='Sequence')
print(series)