#!/bin/bash

input_file=$1

x2x +fa "./${input_file}.rd" > "./tmp/${input_file}_rd.txt"
x2x +fa "./${input_file}.ee" > "./tmp/${input_file}_ee.txt"
x2x +fa "./${input_file}.ra" > "./tmp/${input_file}_ra.txt"
x2x +fa "./${input_file}.rk" > "./tmp/${input_file}_rk.txt"
x2x +fa "./${input_file}.rg" > "./tmp/${input_file}_rg.txt"
x2x +fa "./${input_file}.reaper_gci" > "./tmp/${input_file}_reaper_gci.txt"
x2x +fa "./${input_file}.reaper_f0" > "./tmp/${input_file}_reaper_f0.txt"

