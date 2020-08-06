#!/usr/bin/python

cp -r Output/* Zero-DCE_code/data/test_data/test/
rm -r Output/*
python Zero-DCE_code/lowlight_test_cpu.py
cp -r Zero-DCE_code/data/result/test/* Output/
rm -r Zero-DCE_code/data/test_data/test/*
rm -r Zero-DCE_code/data/result/test/
