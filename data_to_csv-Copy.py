import re
import csv

i = 0

with open('adult1.data', 'w',encoding='utf-8') as csvfile:
    
    with open('adult.data', 'r',newline='') as filein:
        
        for line in filein:
            
            flage = 0
            #line_list1 = re.sub(r"\n*","",line)
            
            for i in range(0,len(line)):
                if(line[i] == '?'):
                    flage+=1
            
            if flage == 0:
                csvfile.write(line)
                print(line)
            #===========================================================
            
            #if flage == 0:
            #print(line['age'], line['workclass'], line['education'])
            #csvfile.write(line['age']+','+line['workclass']+','+line['fnlwgt']+','+line['education']+','+line['education-num']+','+line['marital-status']+','+line['occupation']+','+line['relationship']+','+line['sex']+','+line['hours-per-week']+','+line['income']+'\n')
                #print(line)
            
            #break