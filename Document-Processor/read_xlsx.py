from xlrd import open_workbook

wb = open_workbook('/Users/mohit/Downloads/output (1).xlsx')
for s in wb.sheets():
    print ('Sheet:', s.name)
    for row in range(s.nrows):
        values = []
        for col in range(s.ncols):
            values.append(s.cell(row, col).value)
        values = [str(v) for v in values]        
        # values = [v for v in values if len(v.rstrip().lstrip()) > 0]
        # print (" ".join(values))
        print (values)
    break
