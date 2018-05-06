import numpy as np
import re
from fuzzywuzzy import fuzz

pattern1 = "(?:\s|^)(?:page|pg|Page|Pg)(?:\s{0,}[a-zA-Z,.:]*){0,3}[#]?\s*(\d{1,3})(?:\s|$)"
pattern2 = ""
pattern3 = "(?:\s|^)(\d{1,3})(?:\s|$)"

page_lines = []

# page_1 = ["Approval Summary Page No: 1",
#           "Relationship: Mohit Shah 12/23/2017 12:34:56",
#           "Name: Citibank Account Number: 123",
#           "Random Text Line 1",
#           "Random Text Line 2",
#           "Random Text Line 3",
#           "what is this",
#           "page footer",
#           "nothing in here"]
#
# page_2 = ["Approval Summary",
#           "Relationship: Mohit Shah 12/23/2017 12:34:56",
#           "Name: Citibank Account Number: 123",
#           "Random Text Line 1",
#           "Random Text Line 2",
#           "Random Text Line 3",
#           "what is this",
#           "page footer",
#           "nothing in here"]
#
# page_3 = ["Approval Summary Page No: 3",
#           "Relationship: Mohit Shah 12/23/2017 12:34:56",
#           "Name: Citibank Account Number: 123",
#           "Random Text Line 1",
#           "Random Text Line 2",
#           "Random Text Line 3",
#           "what is this",
#           "page footer",
#           "nothing in here"]
#
# page_4 = ["Approval Summary Page No: 3",
#           "Relationship: Mohit Shah 12/23/2017 12:34:56",
#           "Name: Citibank Account Number: 123",
#           "Random Text Line 1",
#           "Random Text Line 2",
#           "Random Text Line 3",
#           "what is this",
#           "page footer",
#           "nothing in here"]
#
# page_5 = ["Approval Summary Page No: 3",
#           "Relationship: Mohit Shah 12/23/2017 12:34:56",
#           "Name: Citibank Account Number: 123",
#           "Random Text Line 1",
#           "Random Text Line 2",
#           "Random Text Line 3",
#           "what is this",
#           "page footer",
#           "nothing in here"]
#
# page_6 = ["Approval Summary Page No: 6",
#           "Relationship: Mohit Shah 12/23/2017 12:34:56",
#           "Name: Citibank Account Number: 123",
#           "Random Text Line 1",
#           "Random Text Line 2",
#           "Random Text Line 3",
#           "what is this",
#           "page footer",
#           "nothing in here"]
#
# page_7 = ["Approval Summary Page No: 7",
#           "Relationship: Mohit Shah 12/23/2017 12:34:56",
#           "Name: Citibank Account Number: 123",
#           "Random Text Line 1",
#           "Random Text Line 2",
#           "Random Text Line 3",
#           "what is this",
#           "page footer",
#           "nothing in here"]
#
# page_8 = ["Approval Summary Page No: 8",
#           "Relationship: Mohit Shah 12/23/2017 12:34:56",
#           "Name: Citibank Account Number: 123",
#           "Random Text Line 1",
#           "Random Text Line 2",
#           "Random Text Line 3",
#           "what is this",
#           "page footer",
#           "nothing in here"]
#
# page_9 = ["Approval Summary",
#           "Relationship: Mohit Shah 12/23/2017 12:34:56",
#           "Name: Citibank Account Number: 123",
#           "Random Text Line 1",
#           "Random Text Line 2",
#           "Random Text Line 3",
#           "what is this",
#           "page footer",
#           "nothing in here page#1"]
#
# page_10 = ["Approval Summary",
#            "Relationship: Mohit Shah 12/23/2017 12:34:56",
#            "Name: Citibank Account Number: 123",
#            "Random Text Line 1",
#            "Random Text Line 2",
#            "Random Text Line 3",
#            "what is this",
#            "page footer",
#            "nothing page:2"]
#
# page_11 = ["Approval Summary",
#            "Relationship: Mohit Shah 12/23/2017 12:34:56",
#            "Name: Citibank Account Number: 123",
#            "Random Text Line 1",
#            "Random Text Line 2",
#            "Random Text Line 3",
#            "what is this",
#            "page footer",
#            "random page 3"]
#
# page_12 = ["Approval Summary Page No:",
#            "Relationship: Mohit Shah 12/23/2017 12:34:56",
#            "Name: Citibank Account Number: 123",
#            "Random Text Line 1",
#            "Random Text Line 2",
#            "Random Text Line 3",
#            "what is this",
#            "page footer",
#            "nothing in here"]
#
# page_13 = ["Approval Summary Page No: 2",
#            "Relationship: Mohit Shah 12/23/2017 12:34:56",
#            "Name: Citibank Account Number: 123",
#            "Random Text Line 1",
#            "Random Text Line 2",
#            "Random Text Line 3",
#            "what is this",
#            "page footer",
#            "nothing in here"]
#
# page_14 = ["Approval Summary Page: 3",
#            "Relationship: Mohit Shah 12/23/2017 12:34:56",
#            "Name: Citibank Account Number: 123",
#            "Random Text Line 1",
#            "Random Text Line 2",
#            "Random Text Line 3",
#            "what is this",
#            "page footer",
#            "nothing in here"]
#
# page_15 = ["Approval Summary",
#            "Relationship: Mohit Shah 12/23/2017 12:34:56",
#            "Name: Citibank Account Number: 123",
#            "Random Text Line 1",
#            "Random Text Line 2",
#            "Random Text Line 3",
#            "what is this",
#            "1 20 230",
#            "nothing in here 1"]
#
# page_16 = ["Approval Summary",
#            "Relationship: Mohit Shah 12/23/2017 12:34:56",
#            "Name: Citibank Account Number: 123",
#            "Random Text Line 1",
#            "Random Text Line 2",
#            "Random Text Line 3",
#            "what is this",
#            "76 footer",
#            "2"]
#
# page_17 = ["Approval Summary",
#            "Relationship: Mohit Shah 12/23/2017 12:34:56",
#            "Name: Citibank Account Number: 123",
#            "Random Text Line 1",
#            "Random Text Line 2",
#            "Random Text Line 3",
#            "what is this",
#            "page footer",
#            "3"]
#
# page_18 = ["Approval Summary",
#            "Relationship: Mohit Shah 12/23/2017 12:34:56",
#            "Name: Citibank Account Number: 123",
#            "Random Text Line 1",
#            "Random Text Line 2",
#            "Random Text Line 3",
#            "what is this",
#            "page footer",
#            "1"]
#
# page_19 = ["Approval Summary",
#            "Relationship: Mohit Shah 12/23/2017 12:34:56",
#            "Name: Citibank Account Number: 123",
#            "Random Text Line 1",
#            "Random Text Line 2",
#            "Random Text Line 3",
#            "what is this",
#            "random footer 123",
#            "2"]
#
# page_20 = ["Approval Summary",
#            "Relationship: Mohit Shah 12/23/2017 12:34:56",
#            "Name: Citibank Account Number: 123",
#            "Random Text Line 1",
#            "Random Text Line 2",
#            "Random Text Line 3",
#            "what is this",
#            "line footer",
#            "nothing in here 1"]

# page_1 = [" Approval Summary ",
#           "Relationship: Mohit Shah 12/23/2017 12:34:56",
#           "Name: Citibank Account Number: 123",
#           "Random Text Line 1",
#           "Random Text Line 2",
#           "Random Text Line 3",
#           "what is this",
#           "page footer",
#           "nothing in here"]
#
# page_2 = [" CAS Approval Summary Page No 1",
#           "Relationship: Mohit Shah 12/23/2017 12:34:56",
#           "Name: Citibank Account Number: 123",
#           "Random Text Line 1",
#           "Random Text Line 2",
#           "Random Text Line 3",
#           "what is this",
#           "page footer",
#           "nothing in here"]
#
# page_3 = [" CAS Approval Summary Page No 2",
#           "Relationship: Mohit Shah 12/23/2017 12:34:56",
#           "Name: Citibank Account Number: 123",
#           "Random Text Line 1",
#           "Random Text Line 2",
#           "Random Text Line 3",
#           "what is this",
#           "page footer",
#           "nothing in here"]
#
# page_4 = [" CAS Approval Summary Page No 3",
#           "Relationship: Mohit Shah 12/23/2017 12:34:56",
#           "Name: Citibank Account Number: 123",
#           "Random Text Line 1",
#           "Random Text Line 2",
#           "Random Text Line 3",
#           "what is this",
#           "page footer",
#           "nothing in here"]
#
# page_5 = [" CAS Approval Summary Page No 4",
#           "Relationship: Mohit Shah 12/23/2017 12:34:56",
#           "Name: Citibank Account Number: 123",
#           "Random Text Line 1",
#           "Random Text Line 2",
#           "Random Text Line 3",
#           "what is this",
#           "page footer",
#           "nothing in here"]
#
# page_6 = [" CAS Approval Summary Page No 5",
#           "Relationship: Mohit Shah 12/23/2017 12:34:56",
#           "Name: Citibank Account Number: 123",
#           "Random Text Line 1",
#           "Random Text Line 2",
#           "Random Text Line 3",
#           "what is this",
#           "page footer",
#           "nothing in here"]
#
# page_7 = [" CAS Approval Summary Page No 6",
#           "Relationship: Mohit Shah 12/23/2017 12:34:56",
#           "Name: Citibank Account Number: 123",
#           "Random Text Line 1",
#           "Random Text Line 2",
#           "Random Text Line 3",
#           "what is this",
#           "page footer",
#           "nothing in here"]
#
# page_8 = [" CAS Approval Summary Page No 7",
#           "Relationship: Mohit Shah 12/23/2017 12:34:56",
#           "Name: Citibank Account Number: 123",
#           "Random Text Line 1",
#           "Random Text Line 2",
#           "Random Text Line 3",
#           "what is this",
#           "page footer",
#           "nothing in here"]
#
# page_9 = [" CAS Operational Summary Page:1",
#           "Relationship: Mohit Shah 12/23/2017 12:34:56",
#           "Name: Citibank Account Number: 123",
#           "Random Text Line 1",
#           "Random Text Line 2",
#           "Random Text Line 3",
#           "what is this",
#           "page footer",
#           "nothing in here page#1"]
#
# page_10 = ["Garbage",
#            "Relationship: Mohit Shah 12/23/2017 12:34:56",
#            "Name: Citibank Account Number: 123",
#            "Random Text Line 1",
#            "Random Text Line 2",
#            "Random Text Line 3",
#            "what is this",
#            "page footer",
#            "nothing page:2"]
#
# page_11 = ["garbage value",
#            "Relationship: ",
#            "Name: Citibank Account Number",
#            "Random Text Line 1",
#            "Random Text Line 2",
#            "Random Text Line 3",
#            "what is this",
#            "page footer",
#            "No value found"]
#
# page_12 = [" CAS Approval Summary Page No.1",
#            "Relationship: Mohit Shah 12/23/2017 12:34:56",
#            "Name: Citibank Account Number: 123",
#            "Random Text Line 1",
#            "Random Text Line 2",
#            "Random Text Line 3",
#            "what is this",
#            "page footer",
#            "nothing in here"]
#
# page_13 = [" CAS Approval Summary Page No.2",
#            "Relationship: Mohit Shah 12/23/2017 12:34:56",
#            "Name: Citibank Account Number: 123",
#            "Random Text Line 1",
#            "Random Text Line 2",
#            "Random Text Line 3",
#            "what is this",
#            "page footer",
#            "nothing in here"]
#
# page_14 = [" CAS Approval Summary Page No.3",
#            "Relationship: Mohit Shah 12/23/2017 12:34:56",
#            "Name: Citibank Account Number: 123",
#            "Random Text Line 1",
#            "Random Text Line 2",
#            "Random Text Line 3",
#            "what is this",
#            "page footer",
#            "nothing in here"]
#
# page_15 = [" CAS Approval Summary Page No.4",
#            "Relationship: Mohit Shah 12/23/2017 12:34:56",
#            "Name: Citibank Account Number: 123",
#            "Random Text Line 1",
#            "Random Text Line 2",
#            "Random Text Line 3",
#            "what is this",
#            "page footer",
#            "nothing in here 1"]
#
# page_16 = [" CAS Approval Summary ",
#            "Relationship: Mohit Shah",
#            "Name: Citibank Account Number",
#            "Random Text Line 1",
#            "Random Text Line 2",
#            "Random Text Line 3",
#            "what is this",
#            "page footer",
#            "nothing in here"]
#
# page_17 = [" CAS Approval Summary Page No.6",
#            "date and time 12/23/2017 12:34:56",
#            "Name: Citibank Account Number: 123",
#            "Random Text Line",
#            "Random Text Line",
#            "Random Text Line",
#            "what is this",
#            "page footer",
#            "nothing in here 3"]
#
# page_18 = [" Operational Summary Page No.1",
#            "Date and time 12/23/2017 12:34:56",
#            "Name: Citibank Account Number: 123",
#            "Random Text Line 1",
#            "Random Text Line 2",
#            "Random Text Line 3",
#            "what is this",
#            "page footer",
#            "nothing in here 1"]

# page_1 = ["No header",
#           "No header",
#           "No header",
#           "Random Text Line 1",
#           "Random Text Line 2",
#           "Random Text Line 3",
#           "what is this",
#           "page footer",
#           "Page:1"]
#
# page_2 = ["No header",
#           "No header",
#           "No header",
#           "Random Text Line 1",
#           "Random Text Line 2",
#           "Random Text Line 3",
#           "what is this",
#           "page footer",
#           "Page: 2"]
#
# page_3 = ["No header",
#           "No header",
#           "No header",
#           "Random Text Line 1",
#           "Random Text Line 2",
#           "Random Text Line 3",
#           "what is this",
#           "page footer",
#           "Page: 3"]
#
# page_4 = [" CAS Approval Summary Page No 1",
#           "Relationship: Mohit Shah 12/23/2017 12:34:56",
#           "Name: Citibank Account Number: 123",
#           "Random Text Line 1",
#           "Random Text Line 2",
#           "Random Text Line 3",
#           "what is this",
#           "page footer",
#           "nothing in here"]
#
# page_5 = [" No header",
#           "No header",
#           "No header",
#           "Random Text Line 1",
#           "Random Text Line 2",
#           "Random Text Line 3",
#           "what is this",
#           "page footer",
#           "1"]
#
# page_6 = ["No header",
#           "No header",
#           "No header",
#           "Random Text Line 1",
#           "Random Text Line 2",
#           "Random Text Line 3",
#           "what is this",
#           "page footer",
#           "2"]
#
# page_7 = ["No header",
#           "No header",
#           "No header",
#           "Random Text Line 1",
#           "Random Text Line 2",
#           "Random Text Line 3",
#           "what is this",
#           "page footer",
#           "3"]
#
# page_8 = [" CAS operational Summary Page No 1",
#           "Relationship: Mohit Shah 12/23/2017 12:34:56",
#           "Name: Citibank Account Number: 123",
#           "Random Text Line 1",
#           "Random Text Line 2",
#           "Random Text Line 3",
#           "what is this",
#           "page footer",
#           "4"]
#
# page_9 = ["No header",
#           "Relationship: Mohit Shah 12/23/2017 12:34:56",
#           "Name: Citibank Account Number: 123",
#           "Random Text Line 1",
#           "Random Text Line 2",
#           "Random Text Line 3",
#           "what is this",
#           "page footer",
#           "#1"]
#
# page_10 = ["Garbage",
#            "Relationship: Mohit Shah 12/23/2017 12:34:56",
#            "Name: Citibank Account Number: 123",
#            "Random Text Line 1",
#            "Random Text Line 2",
#            "Random Text Line 3",
#            "what is this",
#            "page footer",
#            "#2"]
#
# page_11 = ["garbage value 1",
#            "Relationship: ",
#            "Name: Citibank Account Number",
#            "Random Text Line 1",
#            "Random Text Line 2",
#            "Random Text Line 3",
#            "what is this",
#            "page footer",
#            "No value found"]
#
# page_12 = [" garbage 2",
#            "Relationship: Mohit Shah 12/23/2017 12:34:56",
#            "Name: Citibank Account Number: 123",
#            "Random Text Line 1",
#            "Random Text Line 2",
#            "Random Text Line 3",
#            "what is this",
#            "page footer",
#            "nothing in here"]
#
# page_13 = ["garbage 3",
#            "Relationship: Mohit Shah 12/23/2017 12:34:56",
#            "Name: Citibank Account Number: 123",
#            "Random Text Line 1",
#            "Random Text Line 2",
#            "Random Text Line 3",
#            "what is this",
#            "page footer",
#            "nothing in here"]
#
# page_14 = ["garbage 4",
#            "Relationship: Mohit Shah 12/23/2017 12:34:56",
#            "Name: Citibank Account Number: 123",
#            "Random Text Line 1",
#            "Random Text Line 2",
#            "Random Text Line 3",
#            "what is this",
#            "page footer",
#            "nothing in here"]
#
# page_15 = ["CAS Approval Summary Page No.1",
#            "Relationship: Mohit Shah ",
#            "Name: Citibank Account Number",
#            "Random Text Line 1",
#            "Random Text Line 2",
#            "Random Text Line 3",
#            "what is this",
#            "page footer",
#            "nothing in here"]

page_1 = ["no header",
          "no header",
          "no header",
          "Random Text Line 1",
          "Random Text Line 2",
          "Random Text Line 3",
          "what is this",
          "page footer",
          "Page"]

page_2 = ["no header",
          "no header",
          "no header",
          "Random Text Line 1",
          "Random Text Line 2",
          "Random Text Line 3",
          "what is this",
          "page footer",
          "Page"]

page_3 = ["no header",
          "no header",
          "no header",
          "Random Text Line 1",
          "Random Text Line 2",
          "Random Text Line 3",
          "what is this",
          "page footer",
          "Page: 1"]

page_4 = [" CAS Approval Summary Page No 1",
          "Relationship: Mohit Shah 12/23/2017 12:34:56",
          "Name: Citibank Account Number: 123",
          "Random Text Line 1",
          "Random Text Line 2",
          "Random Text Line 3",
          "what is this",
          "page footer",
          "nothing in here"]

page_5 = [" CAS Approval Summary Page No 2",
          "No header",
          "No header",
          "Random Text Line 1",
          "Random Text Line 2",
          "Random Text Line 3",
          "what is this",
          "page footer",
          "1"]

page_6 = ["CAS operational Summary Page No 1",
          "No header",
          "No header",
          "Random Text Line 1",
          "Random Text Line 2",
          "Random Text Line 3",
          "what is this",
          "page footer",
          "2"]

page_7 = ["CAS Approval Summary Page No 1",
          "No header",
          "No header",
          "Random Text Line 1",
          "Random Text Line 2",
          "Random Text Line 3",
          "what is this",
          "page footer",
          "3"]

page_8 = [" CAS Approval Summary Page No 2",
          "Relationship: Mohit Shah 12/23/2017 12:34:56",
          "Name: Citibank Account Number: 123",
          "Random Text Line 1",
          "Random Text Line 2",
          "Random Text Line 3",
          "what is this",
          "page footer",
          "footer"]

page_9 = ["CAS Approval Summary Page No 3",
          "Relationship: Mohit Shah 12/23/2017 12:34:56",
          "Name: Citibank Account Number: 123",
          "Random Text Line 1",
          "Random Text Line 2",
          "Random Text Line 3",
          "what is this",
          "page footer",
          "footers"]

page_10 = ["Garbage",
           "Relationship: Mohit Shah 12/23/2017 12:34:56",
           "Name: Citibank Account Number: 123",
           "Random Text Line 1",
           "Random Text Line 2",
           "Random Text Line 3",
           "what is this",
           "page footer",
           "Page1"]

page_11 = ["garbage value",
           "Relationship: ",
           "Name: Citibank Account Number",
           "Random Text Line 1",
           "Random Text Line 2",
           "Random Text Line 3",
           "what is this",
           "page footer",
           "Page2"]

page_12 = [" garbage ",
           "Relationship: Mohit Shah 12/23/2017 12:34:56",
           "Name: Citibank Account Number: 123",
           "Random Text Line 1",
           "Random Text Line 2",
           "Random Text Line 3",
           "what is this",
           "page footer",
           "Page3"]

page_13 = ["garbage ",
           "Relationship: Mohit Shah 12/23/2017 12:34:56",
           "Name: Citibank Account Number: 123",
           "Random Text Line 1",
           "Random Text Line 2",
           "Random Text Line 3",
           "what is this",
           "page footer",
           "nothing in here"]

page_14 = ["CAS Operational Summary Page No 1",
           "Relationship: Mohit Shah 12/23/2017 12:34:56",
           "Name: Citibank Account Number: 123",
           "Random Text Line 1",
           "Random Text Line 2",
           "Random Text Line 3",
           "what is this",
           "page footer",
           "nothing in here"]

# page_lines.extend(
#     [page_1, page_2, page_3, page_4, page_5, page_6, page_7, page_8, page_9, page_10, page_11, page_12, page_13,
#      page_14, page_15, page_16, page_17, page_18, page_19, page_20])


# page_lines.extend(
#     [page_1, page_2, page_3, page_4, page_5, page_6, page_7, page_8, page_9, page_10, page_11, page_12, page_13,
#      page_14, page_15, page_16, page_17, page_18])

# page_lines.extend(
#     [page_1, page_2, page_3, page_4, page_5, page_6, page_7, page_8, page_9, page_10, page_11, page_12, page_13,
#      page_14, page_15])

page_lines.extend(
    [page_1, page_2, page_3, page_4, page_5, page_6, page_7, page_8, page_9, page_10, page_11, page_12, page_13,
     page_14])


def find_pattern1(text):
    p1 = re.compile(pattern1)
    m = re.findall(p1, text)
    return m


def find_pattern3(text):
    p1 = re.compile(pattern3)
    m = re.findall(p1, text)
    return m


hp1 = []
hp2 = []
hp3 = []
fp1 = []
fp2 = []
fp3 = []

for lines in page_lines:
    headers = lines[0:3]
    tmp1 = []
    tmp3 = []
    for x in headers:
        tmp1.extend(find_pattern1(x))
        tmp3.extend(find_pattern3(x))
    tmp1 = [int(x) for x in tmp1]
    tmp3 = [int(x) for x in tmp3]
    hp1.append(tmp1)
    hp3.append(tmp3)

    footers = lines[-3:]
    footers.reverse()
    tmp1 = []
    for x in footers:
        tmp1.extend(find_pattern1(x))
    tmp1 = [int(x) for x in tmp1]
    fp1.append(tmp1)

    footers = footers[0:1]
    tmp3 = []
    for x in footers:
        tmp3.extend(find_pattern3(x))
    tmp3 = [int(x) for x in tmp3]
    fp3.append(tmp3)

tags = ["O" for t in page_lines]
nums = [-1 for n in page_lines]
vals = [-1 for v in page_lines]

tags[0] = "B"
nums[0] = 1
vals[0] = 1

for num in range(1, len(page_lines)):
    curr = hp1[num]
    prev = hp1[num - 1]

    if len(curr) > 0:
        val1 = curr[0]
        if val1 == 1:
            tags[num] = "B"
            nums[num] = 1
            vals[num] = val1
            continue
        if len(prev) > 0:
            val2 = prev[0]
            if val1 == 1:
                tags[num] = "B"
                nums[num] = 1
                vals[num] = val1
                continue
            if val1 == val2 + 1:
                tags[num] = "I"
                nums[num] = nums[num - 1] + 1
                vals[num] = val1
                continue
            if val1 > 2 and tags[num - 1] != "O":
                tags[num] = "I"
                nums[num] = nums[num - 1] + 1
                vals[num] = val1
                continue
        else:
            if val1 == 2:
                tags[num] = "I"
                nums[num] = 2
                vals[num] = val1
                if tags[num - 1] == "O":
                    tags[num - 1] = "B"
                    nums[num - 1] = 1
                    vals[num - 1] = 1
                continue
            if val1 > 2:
                tags[num] = "I"
                nums[num] = val1
                vals[num] = val1
                continue

    # Searching pattern 1 over footers
    curr = fp1[num]
    prev = fp1[num - 1]

    if len(curr) > 0:
        val1 = curr[0]
        if val1 == 1:
            tags[num] = "B"
            nums[num] = 1
            vals[num] = val1
            continue
        if len(prev) > 0:
            val2 = prev[0]
            if val1 == 1:
                tags[num] = "B"
                nums[num] = 1
                vals[num] = val1
                continue
            if val1 == val2 + 1:
                tags[num] = "I"
                nums[num] = nums[num - 1] + 1
                vals[num] = val1
                continue
            if val1 > 2 and tags[num - 1] != "O":
                tags[num] = "I"
                nums[num] = nums[num - 1] + 1
                vals[num] = val1
                continue
        else:
            if val1 == 2:
                tags[num] = "I"
                nums[num] = 2
                vals[num] = val1
                if tags[num - 1] == "O":
                    tags[num - 1] = "B"
                    nums[num - 1] = 1
                    vals[num - 1] = 1
                continue
            if val1 > 2:
                tags[num] = "I"
                nums[num] = val1
                vals[num] = val1
                continue

for num in range(1, len(page_lines)):
    if tags[num] != "O":
        continue
    curr = fp3[num]
    prev = fp3[num - 1]
    if len(curr) == 0:
        continue
    if len(prev) > 0:
        inc_flag = False
        tmp = []
        for c in curr:
            for p in prev:
                if c == p + 1:
                    inc_flag = True
                    tmp.append(p)
                    tmp.append(c)
            if inc_flag:
                break
        if inc_flag:
            tags[num] = "I"
            nums[num] = tmp[1]
            vals[num] = tmp[1]
            if tmp[0] == 1:
                tags[num - 1] = "B"
                nums[num - 1] = 1
                vals[num - 1] = 1
            continue

    if len(curr) == 1 and curr[0] == 1:
        tags[num] = "B"
        nums[num] = 1
        vals[num] = 1
        continue


def format_string(text):
    text = text.split(" ")
    text = [x.rstrip().lstrip() for x in text if len(x.rstrip().lstrip()) > 0]
    text = " ".join(text)
    text = re.sub(r'\d', '@', text)
    return text

print (tags)

for num in range(1, len(page_lines)):
    if tags[num] != "O":
        continue
    prev_headers = page_lines[num - 1][0:3]
    headers = page_lines[num][0:3]
    next_headers = page_lines[num + 1][0:3] if num <= len(page_lines) else []
    prev_match = -1
    for h in headers:
        h = format_string(h)
        p_scores = [fuzz.ratio(h, format(ph)) for ph in prev_headers]
        max_score = max(p_scores)
        if max_score > 80:
            prev_match = max_score

    next_match = -1
    for h in headers:
        h = format_string(h)
        n_scores = [fuzz.ratio(h, format(nh)) for nh in next_headers]
        max_score = max(n_scores)
        if max_score > 80:
            next_match = max_score
    print (prev_match, next_match)

    if prev_match > 0 and next_match > 0:
        tags[num] = "I"
        if tags[num + 1] == "O":
            tags[num + 1] = "I"
    elif prev_match > 0:
        tags[num] = "I"
    elif next_match > 0:
        tags[num] = "I"
    else:
        pass

segments = []
tmp = []
for i, t in enumerate(tags):
    if t == "B" or t == "O":
        if len(tmp) > 0:
            segments.append(tmp)
            tmp = [i]
        else:
            tmp = [i]
    elif t == "I":
        tmp.append(i)
if len(tmp) > 0:
    segments.append(tmp)

print (tags)
for i, seg in enumerate(segments, 1):
    print("Segment %d, Pages: %d-%d" % (i, seg[0] + 1, seg[-1] + 1))

import itertools
print (' '.join(c[0] for c in itertools.groupby(x)))