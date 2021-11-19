#!/usr/bin/env python
# coding: utf-8

# In[5]:

import streamlit as st
import matplotlib.pyplot as mplt
import math as mt
import numpy as np
import sympy as sp
import pandas as pd


# In[4]:

def fix_f(function):
    need_fix = ["sin", "cos", "tan", "arcsin", "exp", "arccos", "arctan", "log"]
    for i in need_fix:
        if i in function:
            function = function.replace(f"{i}", f"np.{i}")
    return function
def func(f, a):
    x = a
    return eval(f)
def goldensection(a, b, f, acc, zoom):
    #find local minimum using golden section method
    #printing graph and updated interval after each iteration
    #a: lower bound (numerical)
    #b: upper bound (numerical)
    #f: function (1 variable) (string)
    #acc: accuracy
    #return final interval
    f = fix_f(f)
    a = float(a)
    b = float(b)
    acc = float(acc)
    i = 0
    x = np.linspace(a - zoom, b + zoom, 100)
    f_val = eval(f)
    p = (3-mt.sqrt(5))/2
    dic = {}
    res = pd.DataFrame(columns = ["a", "b", "m", "n", "f(m)", "f(n)"])
    while b-a > acc:
        i += 1
        m = a + p*(b-a)
        n = b - p*(b-a)
        if func(f, m) < func(f, n):
            b = n
        else:
            a = m
        mplt.plot(x, f_val)
        mplt.axvline(x = a)
        mplt.axvline(x = b, color = "r")
        plot = mplt.show()
        st.pyplot(plot)
        st.write(f"""
                 **iteration: {i} ** \n
                 updated interval: {[a, b]} \n
                 m = {m} n = {n} \n
                 f(m) = {func(f,m)} \n
                 f(n) = {func(f,n)}  \n""")
        dic = {"a": a, "b": b, "m": m, "n" : n, "f(m)": func(f, m), "f(n)": func(f, n)}
        res = res.append(dic, ignore_index = True)
    st.title("result summary".upper())
    st.dataframe(res)
    return [a,b]


# In[3]:



# In[7]:
    


# In[4]:


def bisection(f, a, b, acc, zoom):
    #find local minimum using bisection method
    #printing graph and updated interval after each iteration
    #a: lower bound (numerical)
    #b: upper bound (numerical)
    #f: function (1 variable) (string)
    #acc: accuracy
    #zoom: how narrow / wide the graph should be
    #return final interval
    fp = str(sp.diff(f))
    f = fix_f(f)
    a = float(a)
    b = float(b)
    acc = float(acc)
    x = np.linspace(a-zoom, b + zoom, 100)
    f_val = eval(f)
    i = 0
    res = pd.DataFrame(columns =["a", "b", "c", "f'(a).f'(c)"])
    while b-a > acc:
        a1 = a
        b1 = b
        i += 1
        fpa = func(fix_f(fp), a)
        fpb = func(fix_f(fp), b)
        c = (a+b)/2
        fpc = func(fix_f(fp), c)
        if fpa * fpc < 0:
            b = c
        if fpc * fpb < 0:
            a = c
        mplt.plot(x, f_val)
        mplt.axvline(x = a)
        mplt.axvline(x = b, color = "r")
        plot = mplt.show()
        st.pyplot(plot)
        st.write(f"""
                 **iteration: {i} **\n
                 updated interval {[a,b]}, c = {c} \n
                 f'(a).f'(c) = {fpa*fpc} \n
                 f'(c) = {fpc}")""")
        if a1 == a and b1 == b:
            st.write("the algorithm does not converges to local minimizer")
            break
        dic = {"a": a, "b": b, "c": c, "f'(a).f'(c)" : fpa*fpc, "f'(c)" : fpc}
        res = res.append(dic, ignore_index = True)
    st.title("result summary".upper())
    st.dataframe(res)
    return [a,b]


# In[48]:


def newton_raphson(f, x0, acc, zoom):
    #find local minimum using newton raphson method
    #printing graph and updated interval after each iteration
    #x0, x1: initial points
    #f: function (1 variable) (string)
    #acc: accuracy
    #zoom: how narrow / wide the graph should be
    #return final interval
    fp = str(sp.diff(f))
    fp2 = str(sp.diff(fp))
    f = fix_f(f)
    x0 = float(x0)
    acc = float(acc)
    x = np.linspace(x0 - zoom, x0 + zoom, 100)
    f_val = eval(f)
    i = 0
    dic = {}
    res = pd.DataFrame(columns =["x(k)", "f'(x(k))"])
    while abs(func(fix_f(fp), x0)) > acc:     
        i += 1
        x0 = x0 - (func(fix_f(fp), x0)/ func(fix_f(fp2), x0))
        mplt.plot(x, f_val)
        mplt.axvline(x = x0)
        plot = mplt.show()
        st.pyplot(plot)
        st.write(f"""**iteration: {i}**
                 \n x(k) = {x0}
                 \n f'(x(k)) = {func(fix_f(fp), x0)}
                 """)
        dic = { "x(k)": x0, "f'(x(k))": func(fix_f(fp), x0)}
        res = res.append(dic, ignore_index = True)
    st.title("result summary".upper())
    st.dataframe(res)
    return x0


# In[49]:


def secant(f, x0, x1, acc, zoom):
    #find local minimum using bisection method
    #printing graph and updated interval after each iteration
    #x0, x1: initial points
    #f: function (1 variable) (string)
    #acc: accuracy
    #zoom: how narrow / wide the graph should be
    #return final interval
    fp = str(sp.diff(f))
    f = fix_f(f)
    fp = fix_f(fp)
    x0 = float(x0)
    x1 = float(x1)
    acc = float(acc)
    x = np.linspace(x0 - zoom, x0 + zoom, 100)
    f_val = eval(f)
    i = 0
    res = pd.DataFrame(columns =["x(k)", "f'(x(k))"])
    dic = {}
    while abs(func(fp, x0)) > acc:
        i += 1
        f2x = (func(fp, x1) - func(fp, x0))/(x1 - x0)
        x0 = x1
        x1 = x1 - func(fp, x0)/f2x
        if x0 == x1:
            st.write("the algorithm does not converges to local minimizer")
            break
        mplt.plot(x, f_val)
        mplt.axvline(x = x1)
        plot = mplt.show()
        st.pyplot(plot)
        st.write(f"""
                 **iteration: {i}** \n
                 \n x(k) = {x1}
                 \n f'(x(k)) = {func(fp, x1)}""")
        dic = {"x(k)": x1, "f'(x(k))": func(fp, x1)}
        res = res.append(dic, ignore_index = True)
    st.title("result summary".upper())
    st.dataframe(res)
    return x1


# In[ ]:





# In[ ]:




