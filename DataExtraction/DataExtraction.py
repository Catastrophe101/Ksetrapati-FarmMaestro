import openpyxl
xfile = openpyxl.load_workbook('IndianNPKDataset.xlsx')

sheet = xfile.get_sheet_by_name('Sheet1')
start=True
while(True):
    print("Put state 0 to exit")
    state=input("Enter the state: ")
    if(state=="0"):
        break
    soil=input("Enter the soil: ")
    crop=input("Enter the crop: ")
    variety=input("Enter the variety: ")
    season=input("Enter the season: ")
    if(start):
        col=input("Enter the starting col(A...Z): ")
        row=input("Enter the starting row(1....100): ")
        start=False
    yeild1=input("Enter the yeild1: ")
    yeild2=input("Enter the yeild2: ")
    data=input("Enter the data: ").split(" ")
    #print(type(data))
    print(len(data))
    length=len(data)
    norow=length/9
    norow=int(norow)
    tempRow=row

    #putting the state col in place
    for j in range(1,2*norow+1):
        sheet[str(col) + str(tempRow)] = str(state)
        tempRow = int(row) + j

    col=chr(ord(col)+1)
    tempRow=row
    #putting the soil col in place
    for j in range(1,2*norow+1):
        sheet[str(col) + str(tempRow)] = str(soil)
        tempRow = int(row) + j

    col=chr(ord(col)+1)
    tempRow=row
    #putting the crop col in place
    for j in range(1,2*norow+1):
        sheet[str(col) + str(tempRow)] = str(crop)
        tempRow = int(row) + j

    col=chr(ord(col)+1)
    tempRow=row
    #putting the variety col in place
    for j in range(1,2*norow+1):
        sheet[str(col) + str(tempRow)] = str(variety)
        tempRow = int(row) + j

    col=chr(ord(col)+1)
    tempRow=row
    #putting the season col in place
    for j in range(1,2*norow+1):
        sheet[str(col) + str(tempRow)] = str(season)
        tempRow = int(row) + j


    #FEEDING NUMERIC DATA
    tempRow=row
    ccount=1
    tempCol=col
    print(tempCol)
    print(tempRow)

    for i in range(0,norow):
        for j in range(0,2):
            if(j==0):
                colcount=1
                for k in range(0,3):
                    tempCol = chr(ord(col) + colcount)
                    sheet[str(str(tempCol) + str(tempRow))] = data[(i*9)+k]
                    print(str(str(tempCol) + str(tempRow))+" "+str(data[(i*9)+k]))
                    colcount = colcount + 1
                for k in range(3, 6):
                    tempCol = chr(ord(col) + colcount)
                    sheet[str(str(tempCol) + str(tempRow))] = data[(i * 9) + k]
                    print(str(str(tempCol) + str(tempRow)) + " " + str(data[(i * 9) + k]))
                    colcount = colcount + 1
                tempCol = chr(ord(col) + colcount)
                sheet[str(str(tempCol) + str(tempRow))] =yeild1

            elif(j==1):
                colcount = 1
                tempRow = int(tempRow) + 1
                for k in range(0,3):
                    tempCol = chr(ord(col) + colcount)
                    sheet[str(str(tempCol) + str(tempRow))] = data[(i*9)+k]
                    print(str(str(tempCol) + str(tempRow))+" "+str(data[(i*9)+k]) )
                    colcount = colcount + 1
                for k in range(6, 9):
                    tempCol = chr(ord(col) + colcount)
                    sheet[str(str(tempCol) + str(tempRow))] = data[(i * 9) + k]
                    print(str(str(tempCol) + str(tempRow)) + " " + str(data[(i * 9) + k]))
                    colcount = colcount + 1
                tempCol = chr(ord(col) + colcount )
                sheet[str(str(tempCol) + str(tempRow))] = yeild2

        tempRow = int(tempRow) + 1

    row = int(row)+2*norow
    col= chr(ord(col)-4)
    print("New Row is : "+str(row))
xfile.save('../DataSets/IndianNPKDataset.xlsx')


