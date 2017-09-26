import re
import os
import sys
import gzip


def parse(path):
  g = gzip.open(path, 'r')
  for l in g:
    yield eval(l)

def parseLst(lst):
  res = ""
  for e in lst:
    res += e.strip('\'') + ' '
  return res.strip(' ')

if len(sys.argv) != 3:
  print "Parameters: "
  print "1. INPUT: Amazon review json file (.json.gz)"
  print "2. OUTPUT: Amazon simple format file (.gz)"
  sys.exit()

fw = gzip.open(sys.argv[2], 'w')
count = 0
for dict in parse(sys.argv[1]):
  count += 1
  if count % 10000 == 0:
    print count

  fw.write(dict['reviewerID'] + " ")
  fw.write(dict['asin'] + " ")
  fw.write(str(dict['overall']) + " ")
  fw.write(str(dict['unixReviewTime']) + " ")
  lst = re.findall(r"[\w']+", dict['reviewText'].lower())
  fw.write(str(len(lst)) + " ")
  fw.write(parseLst(lst))
  fw.write("\n")

fw.close()
print "Done!"