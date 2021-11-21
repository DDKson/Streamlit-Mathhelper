import streamlit as st
import numpy as np
import sympy
import time
from cal_alg import *
from one_dvis import *
from PIL import Image

st.set_page_config(page_title = 'Mathhelper', layout = 'wide')

def convert_str_to_nparr(astr, size):
    """Convert the input into a numpy 2d array/ a matrix"""
    a = np.array(astr.split( ), dtype = int)
    return a.reshape(size)

def enter_no_of_rows_and_cols(mat):
    """Enter the number of rows and columns of a matrix"""
    row = st.sidebar.text_input(f"Enter number of rows in matrix {mat}")
    col = st.sidebar.text_input(f"Enter number of columns in matrix {mat}")
    return row, col

def enter_elements_mat(mat):
    result = st.sidebar.text_input(f"Enter elements of matrix {mat}, divided by space in between")
    return result
def tutorial_how_to_input_in_calculus():
    st.latex(r" \text{ Please input by this form: }")
    st.latex(r"\bullet \space a * b : \space \text{ a * b }")
    st.latex(r" \bullet \space x^{n} : \space \text{ x**n }")
    st.latex(r" \bullet \space \sqrt{x} : \space \text{ sqrt(x) }")
    st.latex(r" \bullet \space \frac{a}{b} : \space \text{ a / b }")
    st.latex(r" \bullet \space \cos(x) : \space \text{ cos(x) }")
    st.latex(r" \bullet \space e^x : \space \text{ exp(x) }")
    st.latex(r" \bullet \space ln(x) : \space \text{ log(x) }")
    st.latex(r" \bullet \space \log_a b : \space \text{ log(b, a) }")

    
sb = st.container()
bd = st.container()

intro = st.container()
alge = st.container()
cal = st.container()
opti = st.container()
us = st.container()
a,b,d,m = st.columns(4)
bd = st.container()


add_selectbox = st.sidebar.selectbox("Which part do you want to practice?", 
                                     ('Home Page', 'Algebra', 'Calculus', 'Optimization', "Pomodoro")) 

if add_selectbox == 'Home Page':
    
#Display an image: (logo)
#Khi gắn link ảnh thì phải copy ảnh từ máy và paste lại (gồm cả nguồn)
    

    with intro:
        st.image(Image.open("img/cover.png"), width = 1000)
        st.title("WELCOME TO MATHHELPER !!!")
        st.caption("MathHelper is created by group **PP8** to help you with basic to advanced operations in **Linear Algebra, Calculus and Optimization**. Here you will get step-by-step instructions on thousands of math problems, walking you through their solutions. We always aim to provide the best user experience. Thank you so much for trusting our service!")

    with alge:
        st.header(":rocket:Algebra")
        st.caption("Algebra is a branch of mathematics that deals with symbols and the arithmetic operations across these symbols. These symbols do not have any fixed values and are called variables. Here we help you:")
        st.markdown("* **Matrix addition, Matrix subtraction, Matrix multiplication, Scalar multiplication of two matrices.**")
        st.markdown("* **Gauss Eliminate.**")
        st.markdown("* **Calculate determinant.**")
        st.markdown("* **Find transpose and inverse matrix.**")
        st.caption("Please click on 'Algebra' in the left column if you want to practice with Algebra")
    
    
    with cal:
        st.header(":rocket:Calculus")
        st.caption("Calculus in Mathematics is generally used in mathematical models to obtain optimal solutions and thus helps in understanding the changes between the values related by a function. Calculus is broadly classified into two different sections:")
        st.markdown("* **Differential Calculus.**")
        st.markdown("* **Integral Calculus.**")
        st.caption("Please click on 'Calculus' in the left column if you want to practice with Calculus")
    
    
    with opti:
        st.header(":rocket:Optimization")
        st.caption("Optimization is a mathematical procedure to select the best decision variables from a set of alternatives to get the optimum objective value. Here we can help you:")
        st.markdown("* **One - dimensional method and visualize.**")
        st.caption("Please click on 'Optimization' in the left column if you want to practice with Optimization")


    #Tạo ra hộp chứa thông tin: 
    st.info(":fire:Was this app helpful?")
    if st.button('Yes'):
        st.balloons()
        st.write("Thank you for your feedback!")
    elif st.button('No'):
        st.write("Thank you so much for your feedback. Your feedbacks will help us to improve and develop MathHelper further.")
    with us:
        st.header(":green_heart:About us: ") 
        st.markdown("**Together, we are building an interesting application. No matter how hard Math is, whenever you study with MathHelper you will feel 'Math is fun'!**")
    with a:
        st.image(Image.open("img/HA.png"))
        st.markdown("**Hồng Anh**")
    with b:
        st.image(Image.open("img/Bách.png"))
        st.markdown("**Xuân Bách**")
    with d:
        st.image(Image.open("img/Duy.png"))
        st.markdown("**Tuấn Duy**")
    with m:
        st.image("img/Minh.png")
        st.markdown("**Tuấn Minh**")

if add_selectbox == 'Algebra':
    st.image(Image.open("img/algebra.png"), width = 1000)
    with bd:
        st.write(f"# Welcome to {add_selectbox} helper!")
        option_of_algebra = st.sidebar.radio("Choose method: ", ["Sum", "Subtract", "Multiply", "Scalar Multiply", "Gauss Eliminate", "Calculate determinant", "Find inverse", "Find transpose"])
        
        if option_of_algebra == "Sum":
            st.write('A matrix can only be **added** to another matrix if the two matrices have the same **dimensions**. To add two matrices, just **add the corresponding entries**, and place this sum in the corresponding position in the matrix which results.')
            st.latex(r""" A + B = 
                     \begin{bmatrix} a_{11} & a_{12} & ... & ... & a_{1n} \\ 
                     a_{21} & a_{22} & ... & ... & a_{2n} \\
                     ... & ... & ... & ... & ... \\
                     a_{m1} & a_{m2} & ... & ... & a_{mn}
                     \end{bmatrix}
                     +
                     \begin{bmatrix} b_{11} & b_{12} & ... & ... & b_{1n} \\ 
                     b_{21} & b_{22} & ... & ... & b_{2n} \\
                     ... & ... & ... & ... & ... \\
                     b_{m1} & b_{m2} & ... & ... & b_{mn}
                     \end{bmatrix} 
                     = 
                     \begin{bmatrix} a_{11} + b_{11} & a_{12} + b_{12} & ... & ... & a_{1n} + b_{1n} \\ 
                     a_{21} + b_{21} & a_{22} + b_{22} & ... & ... & a_{2n} + b_{2n} \\
                     ... & ... & ... & ... & ... \\
                     a_{m1} + b_{m1} & a_{m2} + b_{m2} & ... & ... & a_{mn} + b_{mn}
                     \end{bmatrix} """)
            st.write('Or more concisely (assuming that A + B = C):')
            st.latex(r"c_{ij} = a_{ij} + b_{ij}")
            mat1_no_row, mat1_no_col = enter_no_of_rows_and_cols(1)
            mat2_no_row, mat2_no_col = enter_no_of_rows_and_cols(2)
            #Two matrices must have equal size
            if (mat1_no_row != mat2_no_row) or (mat1_no_col != mat2_no_col):
                if mat1_no_row and mat1_no_col and mat2_no_row and  mat2_no_col:
                    st.error("Can not add these two matrices. Two matrices must have equal size! Please re-enter.")
            else:
                mat1_add = enter_elements_mat(1)
                mat2_add = enter_elements_mat(2)
                
                if mat1_no_row and mat1_no_col and mat2_no_row and mat2_no_col and mat1_add and mat2_add:
                    if len(mat1_add.split()) != (int(mat1_no_row) * int(mat1_no_col)):
                        st.error(f"You must enter exactly {int(mat1_no_row) * int(mat1_no_col)} elements of matrix 1. Please re-enter.")
                    if len(mat2_add.split()) != (int(mat2_no_row) * int(mat2_no_col)):
                        st.error(f"You must enter exactly {int(mat2_no_row) * int(mat2_no_col)} elements of matrix 2. Please re-enter.")
                    else:
                        the_mat_size = (int(mat1_no_row), int(mat1_no_col))
                        mat1_add = convert_str_to_nparr(mat1_add, the_mat_size)
                        mat2_add = convert_str_to_nparr(mat2_add, the_mat_size)
                        A = Matrix(mat1_add)
                        result = A.add_two_mat(mat2_add)
                        st.subheader("RESULT: ")
                        st.write(mat1_add, "**+**", mat2_add, "**=**", result)
                
        if option_of_algebra == "Subtract":
            st.write('A matrix can only be **subtracted** from another matrix if the two matrices have the same **dimensions**. To get subtraction of two matrices, just **subtract the corresponding entries**, and place this sum in the corresponding position in the matrix which results.')
            st.latex(r""" A + B = 
                     \begin{bmatrix} a_{11} & a_{12} & ... & ... & a_{1n} \\ 
                     a_{21} & a_{22} & ... & ... & a_{2n} \\
                     ... & ... & ... & ... & ... \\
                     a_{m1} & a_{m2} & ... & ... & a_{mn}
                     \end{bmatrix}
                     +
                     \begin{bmatrix} b_{11} & b_{12} & ... & ... & b_{1n} \\ 
                     b_{21} & b_{22} & ... & ... & b_{2n} \\
                     ... & ... & ... & ... & ... \\
                     b_{m1} & b_{m2} & ... & ... & b_{mn}
                     \end{bmatrix} 
                     = 
                     \begin{bmatrix} a_{11} - b_{11} & a_{12} - b_{12} & ... & ... & a_{1n} - b_{1n} \\ 
                     a_{21} - b_{21} & a_{22} - b_{22} & ... & ... & a_{2n} - b_{2n} \\
                     ... & ... & ... & ... & ... \\
                     a_{m1} - b_{m1} & a_{m2} - b_{m2} & ... & ... & a_{mn} - b_{mn}
                     \end{bmatrix} """)
            st.write('Or more concisely (assuming that A - B = C):')
            st.latex(r"c_{ij} = a_{ij} - b_{ij}")
            mat1_no_row, mat1_no_col = enter_no_of_rows_and_cols(1)
            mat2_no_row, mat2_no_col = enter_no_of_rows_and_cols(2)
            #Two matrices must have equal size
            if (mat1_no_row != mat2_no_row) or (mat1_no_col != mat2_no_col):
                if mat1_no_row and mat1_no_col and mat2_no_row and  mat2_no_col:
                    st.error("Can not get subtraction. Two matrices must have equal size! Please re-enter.")
            else:
                mat1_sub = enter_elements_mat(1)
                mat2_sub = enter_elements_mat(2)
                if mat1_no_row and mat1_no_col and mat2_no_row and mat2_no_col and mat1_sub and mat2_sub:
                    if len(mat1_sub.split()) != (int(mat1_no_row) * int(mat1_no_col)):
                        st.error(f"You must enter exactly {int(mat1_no_row) * int(mat1_no_col)} elements of matrix 1. Please re-enter.")
                    if len(mat2_sub.split()) != (int(mat2_no_row) * int(mat2_no_col)):
                        st.error(f"You must enter exactly {int(mat2_no_row) * int(mat2_no_col)} elements of matrix 2. Please re-enter.")
                    else:
                        the_mat_size = (int(mat1_no_row), int(mat1_no_col))
                        mat1_sub = convert_str_to_nparr(mat1_sub, the_mat_size)
                        mat2_sub = convert_str_to_nparr(mat2_sub, the_mat_size)
                        B = Matrix(mat1_sub)
                        result = B.subtract_two_mat(mat2_sub)
                        st.subheader("RESULT: ")
                        st.write(mat1_sub, "**-**", mat2_sub, "**=**", result)
        
        if option_of_algebra == "Multiply":
            st.write("For matrix multiplication, **the number of columns in the first matrix must be equal to the number of rows in the second matrix**. The resulting matrix, known as the **matrix product**, has the number of rows of the first and the number of columns of the second matrix.")
            st.write("If **A** is an **m × n** matrix and **B** is an **n × p** matrix, the matrix product **C = AB** is defined to be the **m × p** matrix, such that")
            st.latex(r""" c_{ij} = a_{i1}b_{1j} + a_{i2}b_{2j} + ... + a_{in}b_{nj} = \sum_{k=1}^{n} a_{ik}b_{kj}""")
            st.write("for **i = 1, ..., m** and **j = 1, ..., p**.")
            st.write(r"That is, each element of C is **the dot product of the ith row of A and the jth column of B.**")
            mat1_no_row, mat1_no_col = enter_no_of_rows_and_cols(1)
            mat2_no_row, mat2_no_col = enter_no_of_rows_and_cols(2)
            if mat1_no_col != mat2_no_row:
                if mat1_no_row and mat1_no_col and mat2_no_row and  mat2_no_col:
                    st.error("Can not multiply. The number of columns in the first matrix must be equal to the number of rows in the second one. Please re-enter!")
            else:
                mat1_mul = enter_elements_mat(1)
                mat2_mul = enter_elements_mat(2)
                if mat1_no_row and mat1_no_col and mat2_no_row and mat2_no_col and mat1_mul and mat2_mul:
                    if len(mat1_mul.split()) != (int(mat1_no_row) * int(mat1_no_col)):
                        st.error(f"You must enter exactly {int(mat1_no_row) * int(mat1_no_col)} elements of matrix 1. Please re-enter.")
                    if len(mat2_mul.split()) != (int(mat2_no_row) * int(mat2_no_col)):
                        st.error(f"You must enter exactly {int(mat2_no_row) * int(mat2_no_col)} elements of matrix 2. Please re-enter.")
                    else:
                        the_mat_size1 = (int(mat1_no_row), int(mat1_no_col))
                        the_mat_size2 = (int(mat2_no_row), int(mat2_no_col))
                        mat1_mul = convert_str_to_nparr(mat1_mul, the_mat_size1)
                        mat2_mul = convert_str_to_nparr(mat2_mul, the_mat_size2)
                        C = Matrix(mat1_mul)
                        result = C.multiply_two_mat(mat2_mul)
                        st.subheader("RESULT: ")
                        st.write("**A**", mat1_mul, "**B**", mat2_mul, "**AB**", result)
        
        
        if option_of_algebra == "Scalar Multiply":
            st.write("The term **scalar multiplication** refers to the **product of a real number and a matrix**. In scalar multiplication, **each entry** in the matrix **is multiplied by the given scalar**.")
            st.latex(r""" k 
                     \begin{bmatrix} a_{11} & a_{12} & ... & ... & a_{1n} \\ 
                     a_{21} & a_{22} & ... & ... & a_{2n} \\
                     ... & ... & ... & ... & ... \\
                     a_{m1} & a_{m2} & ... & ... & a_{mn}
                     \end{bmatrix}
                     = 
                     \begin{bmatrix} ka_{11} & ka_{12} & ... & ... & ka_{1n} \\ 
                     ka_{21} & ka_{22} & ... & ... & ka_{2n} \\
                     ... & ... & ... & ... & ... \\
                     ka_{m1} & ka_{m2} & ... & ... & ka_{mn}
                     \end{bmatrix}""")
                     
            mat1_no_row, mat1_no_col = enter_no_of_rows_and_cols("")
            if mat1_no_col:
                mat1_mul_sca = enter_elements_mat("")
                alpha = st.sidebar.text_input("Enter number")
                if mat1_no_row and mat1_no_col and mat1_mul_sca and alpha:
                    if len(mat1_mul_sca.split()) != (int(mat1_no_row) * int(mat1_no_col)):
                        st.error(f"You must enter exactly {int(mat1_no_row) * int(mat1_no_col)} elements of matrix. Please re-enter.")
                    else:
                        alpha = float(alpha)
                        the_mat_size1 = (int(mat1_no_row), int(mat1_no_col))
                        mat1_mul_sca = convert_str_to_nparr(mat1_mul_sca, the_mat_size1)
                        D = Matrix(mat1_mul_sca)
                        result = D.scalar_multiply(alpha)
                        st.subheader("RESULT: ")
                        st.write(alpha, "x", mat1_mul_sca, "=", result)
        
        
        if option_of_algebra == "Gauss Eliminate":
            st.write("In mathematics, **Gaussian elimination**, also known as row reduction, is an algorithm for solving systems of linear equations.")
            st.write("Using row operations to convert a matrix into reduced row echelon form is sometimes called **Gauss–Jordan elimination**. Using these operations, a matrix can always be transformed into an upper triangular matrix, and in fact one that is in row echelon form. Once all of the **leading coefficients** (the leftmost nonzero entry in each row) are 1, and every column containing a leading coefficient has zeros elsewhere, the matrix is said to be in **reduced row echelon form**.")
            st.write('Example: ')
            st.latex(r""" 
                     \begin{bmatrix} 1 & 1 & 3 & 1 \\ 
                     1 & -1 & 1 & 1 \\
                     0 & 1 & 2 & 2
                     \end{bmatrix}
                     \to
                     ...
                     \to
                     \begin{bmatrix} \color{red} 1 & 0 & 0 & -3 \\ 
                     0 & \color{red} 1 & 0 & -2 \\
                     0 & 0 & \color{red} 1 & 2
                     \end{bmatrix}""")
            mat1_no_row, mat1_no_col = enter_no_of_rows_and_cols("")
            if mat1_no_col:
                mat1_gauss = enter_elements_mat("")
                if mat1_no_row and mat1_no_col and mat1_gauss:
                    if len(mat1_gauss.split()) != (int(mat1_no_row) * int(mat1_no_col)):
                        st.error(f"You must enter exactly {int(mat1_no_row) * int(mat1_no_col)} elements of matrix. Please re-enter.")
                    
                    else:
                        the_mat_size1 = (int(mat1_no_row), int(mat1_no_col))
                        mat1_gauss = convert_str_to_nparr(mat1_gauss, the_mat_size1)
                        E = Matrix(mat1_gauss)
                        result = E.gauss_eliminate()
                        st.subheader("YOUR MATRIX: ")
                        st.write(mat1_gauss)
                        st.subheader("RREF FORM: ")
                        #Convert the sympy matrix to numpy matrix then write
                        #.astype(float64) will cast numbers of the array into the default numpy float type, which will work with arbitrary numpy matrix manipulation functions.
                        st.write(np.array(result[0].tolist()).astype(np.float64))
        
        
        if option_of_algebra == "Calculate determinant":
            st.write("In mathematics, the determinant is a **scalar value** that is a function of the entries of a **square matrix**. It allows characterizing some properties of the matrix and the linear map represented by the matrix. The determinant of a matrix A is denoted **det(A)**, **det A**, or **|A|**.")
            st.write("In the case of a **2 × 2 matrix A** the determinant can be defined as")
            st.latex(r""" \lvert A \rvert =  
                     \begin{vmatrix} a & b \\ 
                     c & d 
                     \end{vmatrix} = ad - bc
                     """)
            st.write("Similarly, for a **3 × 3 matrix A**, its determinant is")
            st.latex(r""" \lvert A \rvert =  
                     \begin{vmatrix} a & b & c\\ 
                     d & e & f \\
                     g & h & i \\
                     \end{vmatrix} = 
                     a \begin{vmatrix} e & f \\ 
                     h & i 
                     \end{vmatrix}
                     - b \begin{vmatrix} d & f \\ 
                     g & i 
                     \end{vmatrix}
                     + c \begin{vmatrix} d & e \\ 
                     g & h 
                     \end{vmatrix} = aei + bfg + cdh - ceg - bdi - afh
                     """)
            mat1_no_row, mat1_no_col = enter_no_of_rows_and_cols("")
            if mat1_no_col != mat1_no_row:
                if mat1_no_col and mat1_no_row:
                    st.error("It must be a square matrix (n x n) to calculate determinant.")
            else:   
                mat1_det = enter_elements_mat("")
                if mat1_no_col and mat1_no_row and mat1_det:
                    if len(mat1_det.split()) != (int(mat1_no_row) * int(mat1_no_col)):
                        st.error(f"You must enter exactly {int(mat1_no_row) * int(mat1_no_col)} elements of matrix. Please re-enter.")
                    
                    else:
                        the_mat_size1 = (int(mat1_no_row), int(mat1_no_col))
                        mat1_det = convert_str_to_nparr(mat1_det, the_mat_size1)
                        F = Matrix(mat1_det)
                        result = F.calculate_determinant()
                        st.subheader("YOUR MATRIX: ")
                        st.write(mat1_det)
                        st.subheader("DETERMINANT: " )
                        st.write(result)
        
        if option_of_algebra == "Find inverse":
            st.write("The inverse of matrix is another matrix, which on multiplication with the given matrix gives the **multiplicative identity**.")
            st.write("For a matrix A, its **inverse** is: ")
            st.latex(r" A^{-1} \space \text{and} \space  A.A^{-1} = I.")
            st.write('Formula to find the inverse matrix:')
            st.latex(r"A^{-1} = \frac{1}{\lvert A \rvert} \cdot \text{Adj}A")
            st.markdown('* The given matrix A should be a **square matrix**.')
            st.markdown("* The determinant of the matrix A should **not be equal to zero**.")
            mat1_no_row, mat1_no_col = enter_no_of_rows_and_cols("")
            if mat1_no_col != mat1_no_row:
                if mat1_no_col and mat1_no_row:
                    st.error("It must be a square matrix (n x n) to find inverse.")
            else:   
                mat1_inv = enter_elements_mat("")
                if mat1_no_col and mat1_no_row and mat1_inv:
                    if len(mat1_inv.split()) != (int(mat1_no_row) * int(mat1_no_col)):
                        st.error(f"You must enter exactly {int(mat1_no_row) * int(mat1_no_col)} elements of matrix. Please re-enter.")
                    
                    else:
                        the_mat_size1 = (int(mat1_no_row), int(mat1_no_col))
                        mat1_inv = convert_str_to_nparr(mat1_inv, the_mat_size1)
                        F = Matrix(mat1_inv)
                        st.subheader("YOUR MATRIX: ")
                        st.write(mat1_inv)
                        if F.calculate_determinant() == 0:
                            st.error("Can not find the inverse matrix because determinant equals to 0")
                        else:
                            result = F.find_inverse()
                            st.subheader("INVERSE MATRIX: ")
                            st.write(result)
                    
                    
        if option_of_algebra == "Find transpose":
            st.write("In linear algebra, **the transpose of a matrix** is an operator which flips a matrix over its diagonal; that is, it **switches the row and column indices** of the matrix A by producing another matrix")
            st.latex(r"\lbrack A^{T} \rbrack_{ij} = \lbrack A \rbrack_{ji}")
            st.latex(r"\text{If A is an m × n matrix, then}\space A^T \space \text{is an n × m matrix.}")
            st.image("https://upload.wikimedia.org/wikipedia/commons/e/e4/Matrix_transpose.gif")
            
            mat1_no_row, mat1_no_col = enter_no_of_rows_and_cols("")
            if mat1_no_col:
                mat1_trans = enter_elements_mat("")
                if mat1_no_col and mat1_no_row and mat1_trans:
                    if len(mat1_trans.split()) != (int(mat1_no_row) * int(mat1_no_col)):
                        st.error(f"You must enter exactly {int(mat1_no_row) * int(mat1_no_col)} elements of matrix. Please re-enter.")
                    
                    else:
                        the_mat_size1 = (int(mat1_no_row), int(mat1_no_col))
                        mat1_trans = convert_str_to_nparr(mat1_trans, the_mat_size1)
                        G = Matrix(mat1_trans)
                        result = G.find_transpose()
                        st.subheader("YOUR MATRIX: ")
                        st.write(mat1_trans)
                        st.subheader("TRANSPOSE MATRIX: ")
                        st.write(result)
                

if add_selectbox == 'Calculus':
    st.image(Image.open("img/Calculus.png"), width = 1000)
    with bd:
        st.write(f"# Welcome to {add_selectbox} helper!")
        option_of_calculus = st.sidebar.radio("Choose method: ", ["Derivate", "Higher derivative",  "Integrate"])        
        
        if option_of_calculus == "Derivate":
            st.write("In mathematics, the derivative of a function of a real variable measures the sensitivity to change of the function value (output value) with respect to a change in its argument (input value).")
            st.write("The derivative of a function of a single variable at a chosen input value, when it exists, is the slope of the tangent line to the graph of the function at that point.")
            st.write("We denote the derivative of f(x) as f'(x) if and only if there exists an open interval I so that:")
            st.latex(r'{\displaystyle L=\lim _{h\to 0}{\frac {f(a+h)-f(a)}{h}}}')
            x = sympy.Symbol('x')
            tutorial_how_to_input_in_calculus()
            
            functionf = st.sidebar.text_input("Enter the function")
            value_x0 = st.sidebar.text_input("Enter value x0")
            if functionf and value_x0:
                value_x0 = float(value_x0)
                H = Calculus(functionf, x)
                result = H.find_derivative(value_x0)
                st.subheader("Your function: ")
                st.write(result[0])
                st.subheader("Derivative: ")
                st.write(result[1])
                st.subheader(f"Derivative at {value_x0}: ")
                st.write(result[2])
        
        if option_of_calculus == "Higher derivative":
            st.write('The process of differentiation can be applied **several times** in succession, leading in particular to the second derivative f″ of the function f, which is just the derivative of the derivative f′. The **second derivative** often has a useful physical interpretation.')
            st.write('**Third derivatives** occur in such concepts as curvature; and even **fourth derivatives** have their uses, notably in elasticity. The **nth derivative of f(x)** is denoted by')
            st.latex(r"f^{(n)}(x)")
            
            tutorial_how_to_input_in_calculus()
            
            x = sympy.Symbol('x')
            functionf = st.sidebar.text_input("Enter function".title())
            value_x0 = st.sidebar.text_input("Enter x0")
            order = st.sidebar.text_input("Enter nth order".title())
            if functionf and value_x0 and order:
                value_x0 = float(value_x0)
                order = int(order)
                K = Calculus(functionf, x)
                result = K.find_higher_derivative(value_x0, order)
                st.subheader("Your function: ")
                st.write(result[0])
                st.subheader("High order derivative: ")
                st.write(result[1])
                st.subheader(f"Derivative at {value_x0}: ")
                st.write(result[2])
                
                
                
        if option_of_calculus == "Integrate":
            st.write("In mathematics, an integral assigns numbers to functions in a way that describes displacement, area, volume, and other concepts that arise by combining infinitesimal data.")
            st.write("Along with differentiation, integration is a fundamental, essential operation of calculus,[a] and serves as a tool to solve problems in mathematics and physics involving the area of an arbitrary shape, the length of a curve, and the volume of a solid, among others.")
            st.write("The integral  of a real-valued function f(x) with respect to a real variable x on an interval [a, b] is written as:")
            st.latex(r'{\displaystyle \int _{a}^{b}f(x)\,dx.}')
            st.write("When the limits are omitted, as in")
            st.latex(r'\int f(x)\,dx,')
            st.write("the integral is called an indefinite integral.")

            x = sympy.Symbol('x')
            
            tutorial_how_to_input_in_calculus()
            
            functionf = st.sidebar.text_input("Enter the function")
            value_a = st.sidebar.text_input("Enter lower limit a")
            value_b = st.sidebar.text_input("Enter lower limit b")
            if functionf and value_a and value_b:
                value_a = float(value_a)
                value_b = float(value_b)
                I = Calculus(functionf, x)
                result = I.find_integration(value_a, value_b)
                st.subheader("Your function: ")
                st.write(result[0])
                st.subheader("Integration: ")
                st.write(result[1])
                st.subheader(f"Integration from {value_a} to {value_b}:")
                st.write(result[2])
                
                st.subheader("The integration is the area under the curve, limit by a and b: ")
                size = st.slider("Choose size of the graph: ", 0, 10)
                I.draw_graphh(value_a, value_b, int(size))
                
st.set_option('deprecation.showPyplotGlobalUse', False)               


if add_selectbox == 'Optimization':
    st.image(Image.open("img/Opti.png"), width = 1000)    
    methods = ["Goldensection", "Bisection", "Newton Raphson", "Secant"]
    main1 = st.sidebar.radio("choose desired method:", methods)
    tutorial_how_to_input_in_calculus()
    if main1 == methods[0] or main1 == methods[1]:
        with sb:
            f = st.sidebar.text_input("Enter Function:")
            a = st.sidebar.text_input("enter lower bound:".title())
            b = st.sidebar.text_input("enter upper bound:".title())
            acc = st.sidebar.text_input("enter accuracy:".title())
            zoom = st.sidebar.slider("zoom (0: narrower, 10: wider)".title(), 0., 10., step = 0.1)
        if main1 == methods[0]:
            with bd:
                st.title(f"chosen method: {main1}".upper())
                st.header(":dart: Summary of the Golden Section Algorithm")
                st.markdown("* Starting from $[a_0, b_0]$.")
                st.markdown("* Let m, n be the following intermediate points")
                st.latex(r''' m = a_0 + \rho \lparen b_0 - a_0 \rparen ,  n = b_0 - \rho \lparen b_0 - a_0 \rparen''')
                st.markdown("* Comparing $f(m)$ and $f(n)$. Then update the next interval ")
                st.latex(r'''\lbrack a_1, b_1 \rbrack = \begin{cases} \lbrack a_n, n \rbrack &\text{if } f \lparen m \rparen \lt f \lparen n \rparen, \\ \lbrack m, b_0 \rbrack &\text{if } f \lparen m \rparen \ge f \lparen n \rparen. \end{cases} ''' )
            if f and a and b and acc:
                with bd:
                    goldensection(a, b, f, acc, zoom)
        elif main1 == methods[1]:
                 st.header(":dart: Bisection Method")
                 st.markdown("* First, we locate a sub - interval $[a_1, b_1]$ of $[a, b]$ such that")
                 st.latex(r''' f' \lparen a_1 \rparen f' \lparen b_1 \rparen \lt 0''')
                 st.markdown("* The Intermediate Value Theorem ensures that there is a solution of $f'(x) = 0$  (2) inside $[a_1, b_1]$")
                 st.markdown("* Let $c_1$ be the midpoint of $[a_1, b_1]$.")
                 st.markdown(" If $f' (a_1) f'(c_1) \lt 0$ then (2) has a solution in $[a_1, c_1]$, the updated interval is $[a_1, c_1]$.")
                 st.markdown(" If $f'(c_1) f'(b_1) \lt 0$ then (2) has a solution in $[c_1, b_1]$, the updated interval is $[c_1, b_1]$.")
                 if f and a and b and acc:
                    bisection(f, a, b, acc, zoom)
    if main1 == methods[2]:
        with sb:
            f = st.sidebar.text_input("enter function:".title())
            x0 = st.sidebar.text_input("initial point".title())
            acc = st.sidebar.text_input("enter accuracy:".title())
            zoom = st.sidebar.slider("zoom (0: narrower, 10: wider)".title(), 0., 10., step = 0.1)
        with bd:
           st.header(":dart: Newton - Raphson Algorithm")
           st.markdown("* Starting from some initial approximation $x^{(0)}$ of $x^*$, if the current approximation is $x^{(k)}$, the next approximation is given by")
           st.latex(r''' x^{(k+1)} = x^{(k)} - \frac{f' \lparen x^{(k)} \rparen} {f'' \lparen x^{(k)} \rparen}\qquad \lparen 5 \rparen''')
           st.markdown("* The algorithm is terminated if some preset degree of precision is achieved.") 
           st.markdown("For example, ") 
           st.latex(r''' \lvert f' \lparen x^{(k)} \rvert \lt \epsilon ''')
           st.markdown("or")
           st.latex(r''' \lvert x^{(k + 1)} - x^{(k)} \rvert \lt \epsilon''')    
           st.markdown("where $\epsilon$ is some small number.")
        if f and x0 and acc:
            with bd:
                st.title(f"chosen method: {main1}".upper())
                newton_raphson(f, x0, acc, zoom)
    if main1 == methods[3]:
        with sb:
            f = st.sidebar.text_input("enter function:".title())
            x0 = st.sidebar.text_input("initial point".title())
            x1 = st.sidebar.text_input("initial point x(1)".title())
            acc = st.sidebar.text_input("enter accuracy:".title())
            zoom = st.sidebar.slider("zoom (0: narrower, 10: wider)".title(), 0., 10., step = 0.1)
        with bd:
             st.header(" :dart: Secant Method for Optimization ")
             st.markdown("The Secant method for optimization is therefore given by ")
             st.latex(r''' x^{(k + 1)} = \frac{x^{(k - 1)} f' \lparen x^{(k)} \rparen - x^{(k)} f' \lparen x^{(k - 1)} \rparen} { f' \lparen x^{(k)} \rparen - f' \lparen x^{(k - 1)} \rparen} \qquad \lparen 10 \rparen''')
             st.markdown("We observe that, the second derivative is no longer needed. However, we need two initial points $x^{(0)}, x^{(1)}$ to start the recursive formula. ")
             st.markdown("**Stopping criteria:** ")
             st.latex(r''' \lvert x^{(k + 1)} - x^{(k)} \rvert \lt \epsilon''')
             st.markdown("where $\epsilon$ is some small number.")
        if f and x0 and x1 and acc:
            with bd:
                st.title(f"chosen method: {main1}".upper())
                secant(f, x0, x1, acc, zoom)

if add_selectbox == 'Pomodoro':
    
    with bd:
        st.write(f"# Welcome to {add_selectbox} timer!")
        st.write("**The Pomodoro Technique** is a time management method developed by Francesco Cirillo in the late 1980s. It uses a timer to **break work into intervals**, traditionally **25 minutes** in length, separated by short breaks. Each interval is known as a pomodoro, from the Italian word for 'tomato', after the tomato-shaped kitchen timer Cirillo used as a university student.")
        st.write("The Pomodoro technique has ** five steps: **")
        st.markdown(" :dart: Pick a task")
        st.markdown(" :clock10: Set a **25-minute** timer")
        st.markdown(" :hourglass_flowing_sand: Work on your task until the time is up")
        st.markdown(" :stuck_out_tongue_closed_eyes: Take a **5-minute** break")
        st.markdown(" :star: Every **4 pomodoros**, take a longer **15-30 minute** break") 
        st.write("In this version, we let you choose your interval of study and break on your own to make you get the most of it. Take advantage of this method while studying Math or any subjects. Have a good day!")
        button_clicked = st.button("Start timer")
        
        t1 = st.sidebar.text_input("Enter length of time to study in minutes: ")
        t2 = st.sidebar.text_input("Enter length of time to break in minutes: ")
        if button_clicked:
            with st.empty():
                if t1: 
                    t1 = int(t1) * 60
                    time_in_mint1 = t1 / 60
                   
                    while t1:
                        mins, secs = divmod(t1, 60)
                        timer = '{:02d}:{:02d}'.format(mins, secs)
                        st.header(f"⏳ {timer}")
                        time.sleep(1)
                        t1 -= 1
                        st.success(f"🔔 {time_in_mint1} minutes is over! You did a great job! Time for a break.")
        
            with st.empty():
                if t2:
                    t2 = int(t2) * 60
                    time_in_mint2 = t2 / 60
                   
                    while t2:
                        # Start the break
                        
                        mins2, secs2 = divmod(t2, 60)
                        timer2 = '{:02d}:{:02d}'.format(mins2, secs2)
                        st.header(f"⏳ {timer2}")
                        time.sleep(1)
                        t2 -= 1
                        st.error(f"⏰ {time_in_mint2} minute break is over!")  

st.set_option('deprecation.showPyplotGlobalUse', False)