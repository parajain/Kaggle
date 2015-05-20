#!/usr/bin/python

from sys import stdin

l = []
with open("stop") as f:
    for line in f:
        line = line.strip()
        l.append(line)

print l
