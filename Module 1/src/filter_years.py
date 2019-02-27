import pandas as pd
spreadsheet = pd.ExcelFile('District_wise_yield.xlsx')
df2 = spreadsheet.parse('Sheet1')
df2.head()
i=0
newDf = pd.DataFrame()
for row in df2.itertuples():
    if row[3] >= 1997 and row[3] <= 2002 :
        newDf = newDf.append(df2.iloc[i-1:i,:], ignore_index = True)    
    i=i+1
newDf.head(200)
newDf.to_excel('District_wise_yield(2002).xlsx',sheet_name='Sheet_name_1')