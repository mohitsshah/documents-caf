import csv

docs = {}
repeats = []
csv_path = "legal_files.csv"
rows = csv.reader(open(csv_path))
next(rows)
for row in rows:
    format = row[-2]
    if format == "PDF":
        id = row[-1]
        if id != "null":
            if id not in docs:
                docs[id] = 1
            else:
                repeats.append(id)

csv_path = "credit_files.csv"
rows = csv.reader(open(csv_path))
next(rows)
for row in rows:
    format = row[-2]
    if format == "PDF":
        id = row[-1]
        if id != "null":
            if id not in docs:
                docs[id] = 1
            else:
                repeats.append(id)

print (len(repeats))
print (repeats)
print (len(docs.keys()))