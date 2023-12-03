import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
import statsmodels.formula.api as sm

#lê o arquivo csv a partir do caminho
df = pd.read_csv(r'C:\Users\osvaldop.CIT\Desktop\17 Data Science\Cases\Formatura (Kroton Entrevista)\formatura.csv', 
sep=';', encoding='latin-1', 
delimiter=None, header='infer', names=None, index_col=None, 
usecols=None, squeeze=False, prefix=None, mangle_dupe_cols=True, 
dtype=None, engine=None, converters=None, true_values=None, 
false_values=None, skipinitialspace=False, skiprows=None, 
nrows=None, na_values=None, keep_default_na=True, na_filter=True, 
verbose=False, skip_blank_lines=True, parse_dates=False, 
infer_datetime_format=False, keep_date_col=False, date_parser=None, 
dayfirst=False, iterator=False, chunksize=None, compression='infer', 
thousands=None, decimal=b'.', lineterminator=None, quotechar='"', 
quoting=0, escapechar=None, comment=None, dialect=None, 
tupleize_cols=None, error_bad_lines=False, 
warn_bad_lines=True, skipfooter=0, doublequote=True, 
delim_whitespace=False, low_memory=True, memory_map=False, 
float_precision=None)

#imprime o total de linhas do arquivo
print("Total rows: {0}".format(len(df)))

#lista os nomes das colunas do arquivo
#print(list(df))

#imprime o formato do arquivo
#print(df.shape)

#imprime as 5 primeiras linhas do cabeçalho
#print(df.head())

#imprime o arquivo
#print(df) 

#imprime a descrição das colunas do arquivo
#print(df.describe())
#print(df.info())


#imprime os valores únicos de todas as colunas
#for col in df:
#  print(df[col].unique())


df=df.dropna(subset=['NRO_REPRO_NORMAL', 
'NR_TOTAL_DISCIPLINAS','SEMESTRES_CURSADOS',
'DURACAO_CURSO','DURACAO_CURSO',
'PERIODOS_TRANCADOS','CH_TOTAL_MATRIZ'])

df['A'] = df['NRO_REPRO_NORMAL']
df['B'] = df['NR_TOTAL_DISCIPLINAS']
df['C'] = df['SEMESTRES_CURSADOS']
df['D'] = df['DURACAO_CURSO']
df['E'] = df['PERIODOS_TRANCADOS']
df['F'] = df['CH_TOTAL_MATRIZ']
df['G'] = df['A']/df['B']
df['H'] = df['C']/df['D']
df['I'] = df['F']/df['B']

amostra = df.sample(10000,replace='False')
#print(amostra)

lm = sm.ols(formula= 
'NOTA ~ G + B + H + F + I', data=amostra).fit()
print(lm.summary())





    
