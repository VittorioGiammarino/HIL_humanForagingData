#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun  3 10:12:25 2021

@author: vittorio
"""

import csv


with open("Rat_Data/processed.cvs") as f:
    data_raw = f.readlines()

agent_data = csv.reader(data_raw)

