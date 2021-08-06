# -*- coding: utf-8 -*-
"""
Created on Wed Jun 15 09:15:09 2016

@author: ericgrimson
"""

x = 6

if x != 5:
    print('i am here')
else:
    print('no I am not')


def fancy_divide(numbers, index):
    try:
        denom = numbers[index]
        for i in range(len(numbers)):
            numbers[i] /= denom
    except IndexError:
            print("-1")
    else:
        print("1")

    finally:
        print("0")

fancy_divide([0, 2, 4], 0)