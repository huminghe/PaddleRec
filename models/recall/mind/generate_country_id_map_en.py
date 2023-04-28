# !/usr/bin/python
# -*- coding: utf-8 -*-
# @author: huminghe
# @date: 2023/4/27
#


country_name_code_map = {}

with open("data/country-codes.csv", "r") as rf:
    for line in rf:
        data = line.split(",")
        country_code = data[9].lower()
        cn_name = data[40]
        print(country_code + "\t" + cn_name)
        country_name_code_map[cn_name] = country_code

with open("data/country_id_map.txt", "r") as rf:
    with open("data/country_code_id_map.txt", "w") as rw:
        for line in rf:
            country_name, country_id = line.split("\t")
            country_code = country_name_code_map.get(country_name, country_name)
            rw.write(country_code + "\t" + country_id)
