# -*- coding: utf-8 -*-
import openpyxl
import re
import numpy as np
import pandas as pd
import xlrd

def read_excel():
    #第一行没有存进去
    df=pd.read_csv('D:\pythonProject\PII-Learning\data\pcfg_format_password.csv')
    print(df.values)
    pcfg_list=df.values[:, 0]
    for i,pcfg in enumerate(pcfg_list):
        #将LDS全部替换为空，再判断剩下的是否全为数字即可
        if i%100==0:
            print(i)
        str0=pcfg
        str0=str0.replace('L','')
        str0=str0.replace('D','')
        str0=str0.replace('S','')
        if str0.isdigit() :
            df.drop(i, axis=0,inplace=True)
    print(df.value)
    # df.to_excel('D:\pythonProject\PII-Learning\data\pcfg_clean_password.csv')
    # print("y")

PII_type=['U','E','B','N']
def read_file(filename):
    with open(filename,'r') as f:
        lines=f.readlines()
        clear_pcfg=[]
        for line in lines:
            temp=line.strip('\n').split(',')
            for ch in temp[5]:
                if ch in PII_type:
                    clear_pcfg.append(line)
                    break
    return clear_pcfg
def write_file(filename,data):
    with open(filename,'w') as f:
        f.writelines(data)
if __name__ == '__main__':
    clear_pcfg=read_file('D:\pythonProject\PII-Learning\data\pcfg_format_password+password.csv')
    print(clear_pcfg)
    write_file('D:\pythonProject\PII-Learning\data\Only_Contain_PIIs_dataset.csv',clear_pcfg)