import pkg_resources
from symspellpy.symspellpy import SymSpell
import pandas as pd
from pandas import DataFrame
data=pd.read_csv("G:\\password.csv",header=None)
data=data[0:1000]
for i in data.iterrows():
    res.append(str(i[1]).split()[1])
for i in res:
    temp=sym_spell.word_segmentation(str(i))
    print(temp.corrected_string)
    q.append(temp.corrected_string)