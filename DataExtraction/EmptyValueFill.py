import openpyxl
xfile = openpyxl.load_workbook('CropPricesComplete.xlsx')

sheet = xfile.get_sheet_by_name('Sheet1')
start=int(input("Enter the starting index : "))
end=int(input("Enter the ending index : "))
for i in range(start,end+1):
    if(sheet["B"+str(i+1)].value==None):
        sheet["B" + str(i + 1)].value=sheet["B"+str(i)].value
    if (sheet["A" + str(i + 1)].value == None):
        sheet["A" + str(i + 1)].value = sheet["A" + str(i)].value
    if (sheet["C" + str(i + 1)].value == None):
        sheet["C" + str(i + 1)].value = sheet["C" + str(i)].value
    # print(sheet["B"+str(i)].value)
xfile.save('CropPricesComplete.xlsx')
# for i in range(2,9128):
#     # a=str(sheet["I"+str(i)].value)
#     # b=a.split("-")
#     # print(type(sheet["I"+str(i)].value))
#     a=sheet["I"+str(i)].value
#     b=a.split("\"")
#     c=b[1].split("-")
#     sheet["I" + str(i)].value=str(c[1]+"-"+c[1]+"-"+c[0])
#     # print(c[0]+" "+c[1]+" "+c[2])
# xfile.save('RicePrices.xlsx')