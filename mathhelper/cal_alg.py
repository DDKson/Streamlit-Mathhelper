import streamlit as st
import numpy as np
import sympy
import re
import matplotlib.pyplot as mplt
from PIL import Image
import time


#MATH SOLVING
class Matrix:
    def __init__(self, matrix):
        self._matrix = matrix
        
    def add_two_mat(self, the_other_mat):
        """Calculate sum of two matrices"""
        addition = np.add(self._matrix, the_other_mat)
        return addition
    
    def subtract_two_mat(self, the_other_mat):
        """Calculate subtraction of two matrices"""
        subtraction = self._matrix - the_other_mat
        return subtraction
    
    def multiply_two_mat(self, the_other_mat):
        """Calculate multiplication of two matrices"""
        multiplication = np.dot(self._matrix, the_other_mat)
        return multiplication
    
    def scalar_multiply(self, alpha):
        """Calculate scalar multiplication of a matrix with a number"""
        scalar_mul = alpha * self._matrix
        return scalar_mul
    
    def gauss_eliminate(self):
        """Gauss eliminate to reduced row echelon form"""
        rref_form = sympy.Matrix(self._matrix).rref()
        return rref_form
    
    def calculate_determinant(self):
        """Find determinant of given matrix"""
        det = np.linalg.det(self._matrix)
        return det
    
    def find_inverse(self):
        """Find inverse matrix of given matrix"""
        #if self.calculate_determinant() == 0:
            #return False
        inverse_mat = np.linalg.inv(self._matrix)
        return inverse_mat
    
    def find_transpose(self):
        """Find the transpose of a given matrix"""
        trans = np.transpose(self._matrix)
        return trans


class Calculus:
    def __init__(self, fx, x):
        self._x = x
        self._fx = fx
    
    def find_derivative(self, x0):
        """Find the derivative of the f(x) at x0"""
        deriva_f = sympy.diff(self._fx)
        derivative_f = sympy.lambdify(self._x, deriva_f)
        return sympy.sympify(self._fx), deriva_f, derivative_f(x0)
    
    def find_higher_derivative(self, x0, order):
        """Find higher order derivative of the f(x) at x0"""
        deriva_f = self._fx
        for i in range(0, order):
            deriva_f = sympy.diff(deriva_f)
        derivative_f = sympy.lambdify(self._x, deriva_f)
        return sympy.sympify(self._fx), deriva_f, float(derivative_f(x0))
    
    def find_integration(self, a, b):
        """Find the integration of f(x) from a to b"""
        integration = sympy.integrate(self._fx, self._x)
        integration_f = sympy.lambdify(self._x, integration)
        return sympy.sympify(self._fx), integration, integration_f(b) - integration_f(a)  
    
    def fix_f(self):
        """Turn user input into what can use in draw_graph function"""
        need_fix = ["sin", "cos", "tan", "arcsin", "exp", "arccos", "arctan", "log", "sqrt"]
        function = self._fx
        for i in need_fix:
            if i in function:
                function = function.replace(f"{i}", f"np.{i}")
        return function

    def draw_graphh(self, a, b, zoom):
        """Draw a graph of area under the curve"""
        #a: lower limit, b: upper limit, f: function, zoom: how narrow or wide the view is
        f = self.fix_f()
        x = np.linspace(a - zoom, b + zoom, 100)
        f_val = eval(f)
        mplt.plot(x, f_val)
        mplt.axvline(x = 0, color = "black")
        mplt.axhline(0, color = "black")
        v1 = mplt.axvline(x = a, color = "r")
        v2 = mplt.axvline(x = b, color = "r")
        mplt.fill_between(x, 0, f_val, where = (x >= a) & (x <= b), color = "g", alpha = 0.5)
        plot = mplt.show()
        st.pyplot(plot)