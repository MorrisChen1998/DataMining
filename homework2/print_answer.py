# -*- coding: utf-8 -*-
"""
Created on Wed Nov 25 02:15:51 2020

@author: morri
"""
def printOutAnswer(task, answer):
    file= open(task+".csv","w+")
    file.write("prediction\n")
    for i in range(len(answer)):
        file.write("%s\n"%answer[i])
        
    file.close()