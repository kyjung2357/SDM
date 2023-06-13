import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['mathtext.fontset'] = 'cm'
plt.rcParams["font.family"] = "Times New Roman"

########################################################################################################################
# FUNCTIONS
########################################################################################################################

def alpha(f, x, h=1e-6) -> float:
    r"""
    The right-hand derivative for mesh size h.

    [Args]
        f : `function`
        x : `float`
        h : `float`, mesh size
    """
    return (f(x + h) - f(x))/h

def beta(f, x, h=1e-6) -> float:
    r"""
    The left-hand derivative for mesh size h.

    [Args]
        f : `function`
        x : `float`
        h : `float`, mesh size
    """
    return (f(x) - f(x - h))/h

def symmetric_derivative(f, x, h=1e-6) -> float:
    r"""
    The symmetric derivative for mesh size h. 
    Recall that the symmetric derivative can be the average of the right-hand derivative and the left-hand derivative.
    Note that symmetric derivatives belong to the subdifferential. 
    Hence, we employ symmetric derivatives in embodying the subgradient method.

    [Args]
        f : `function`
        x : `float`
        h : `float`, mesh size
    """
    return (alpha(f, x, h=h) + beta(f, x, h=h))/2

def A(f, x, h=1e-6) -> float:
    r"""
    The specular derivative for mesh size h.

    [Args]
        f : `function`
        x : `float`
        h : `float`, mesh size
    """
    a = alpha(f, x, h=h)
    b = beta(f, x, h=h)

    if a + b == 0:
        return 0.0
    else:
        return (a*b - 1 + np.sqrt(1 + a**2)*np.sqrt(1 + b**2))/(a + b)

# Lemma 4.7
def Findtheta(f, x, h=1e-6) -> float:
    r"""
    The theta which is the angle between the specular tangent line and the phototangent

    [Args]
        f : `function`
        x : `float`
        h : `float`, mesh size
    """
    a = alpha(f, x, h=h)
    b = beta(f, x, h=h)

    if a*b + 1 == 0:
        return 0.5*(np.pi - np.arctan(a) + np.arctan(b))
    else:
        return 0.5*(np.arctan(a) - np.arctan(b))

########################################################################################################################
# OPTIMIZATION with the Subgradient Method
########################################################################################################################

def SM_with_constant_step_size(objective_function, x_0, gamma=0.005, eta=1e-6, h=1e-6, N=1000) -> tuple:
    r"""
    The Subgradient Method with symmetric derivatives and the constant step size rule.

    [Args]
        objective_function : `function`
        x_0 : `float`, the initial point 
        gamma : `float`, constant step size
        eta : `float`, tolerance for stopping criterion 
        h : `float`, mesh size
        N : `int`, maximum number of iterations 

    [return]
        x_star : `float`, the point where the minimum is approximated 
        each_point : `list`, points and values at the point for each iteration
        values : `list`, values at each point for each iteration
    """
    x = x_0
    x_star = x_0  
    subgrad = symmetric_derivative(objective_function, x_0, h=h)
    
    value = objective_function(x)
    each_point = [(x, value)]           # List to store function values and its point at each iteration
    values = [value]                    # List to store function values at each iteration

    n = 1
    while n < N and np.abs(subgrad) > eta:
        x = x - gamma * subgrad
        each_point.append((x, objective_function(x)))               # Store current function values and its point
        values.append((objective_function(x)))                      # Store current function values

        if objective_function(x) < objective_function(x_star):
            x_star = x 

        subgrad = symmetric_derivative(objective_function, x, h=h)

        n += 1

    return x_star, each_point, values 

def SM_with_diminishing_step_size(objective_function, x_0, eta=1e-6, h=1e-6, N=1000) -> tuple:
    r"""
    The Subgradient Method with the square summable but not summable step rule gamma_n=1/n.

    [Args]
        objective_function : `function`
        x_0 : `float`, the initial point 
        eta : `float`, tolerance for stopping criterion 
        h : `float`, mesh size
        N : `int`, maximum number of iterations 

    [return]
        x_star : `float`, the point where the minimum is approximated 
        each_point : `list`, points and values at the point for each iteration
        values : `list`, values at each point for each iteration
    """
    x = x_0
    x_star = x_0  
    subgrad = symmetric_derivative(objective_function, x_0, h=h)
    
    value = objective_function(x)
    each_point = [(x, value)]           # List to store function values and its point at each iteration
    values = [value]                    # List to store function values at each iteration

    n = 1
    while n < N and np.abs(subgrad) > eta:
        gamma = 1/n                                                 # the square summable but not summable step rule
        x = x - gamma * subgrad
        each_point.append((x, objective_function(x)))               # Store current function values and its point
        values.append((objective_function(x)))                      # Store current function values

        if objective_function(x) < objective_function(x_star):
            x_star = x 
        
        subgrad = symmetric_derivative(objective_function, x, h=h)

        n += 1

    return x_star, each_point, values 


########################################################################################################################
# OPTIMIZATION with the Specular Gradient Method
########################################################################################################################

def SGM_with_constant_step_size(objective_function, x_0, gamma=0.005, eta=1e-6, h=1e-6, N=1000) -> tuple:
    r"""
    The Specular Gradient Method with the constant step size rule.

    [Args]
        objective_function : `function`
        x_0 : `float`, the initial point 
        gamma : `float`, constant step size
        eta : `float`, tolerance for stopping criterion 
        h : `float`, mesh size
        N : `int`, maximum number of iterations 

    [return]
        x_star : `float`, the point where the minimum is approximated 
        each_point : `list`, points and values at the point for each iteration
        values : `list`, values at each point for each iteration
    """
    x = x_0
    x_star = x_0  
    subgrad = A(objective_function, x_0, h=h)
    
    value = objective_function(x)
    each_point = [(x, value)]           # List to store function values and its point at each iteration
    values = [value]                    # List to store function values at each iteration

    n = 1
    while n < N and np.abs(subgrad) > eta:
        x = x - gamma * subgrad
        each_point.append((x, objective_function(x)))               # Store current function values and its point
        values.append((objective_function(x)))                      # Store current function values

        if objective_function(x) < objective_function(x_star):
            x_star = x 

        subgrad = A(objective_function, x, h=h)

        n += 1

    return x_star, each_point, values 

def SGM_with_diminishing_step_size(objective_function, x_0, eta=1e-6, h=1e-6, N=1000) -> tuple:
    r"""
    The Specular Gradient Method with the square summable but not summable step rule gamma_n=1/n.

    [Args]
        objective_function : `function`
        x_0 : `float`, the initial point 
        eta : `float`, tolerance for stopping criterion 
        h : `float`, mesh size
        N : `int`, maximum number of iterations 

    [return]
        x_star : `float`, the point where the minimum is approximated 
        each_point : `list`, points and values at the point for each iteration
        values : `list`, values at each point for each iteration
    """
    x = x_0
    x_star = x_0  
    subgrad = A(objective_function, x_0, h=h)
    
    value = objective_function(x)
    each_point = [(x, value)]           # List to store function values and its point at each iteration
    values = [value]                    # List to store function values at each iteration

    n = 1
    while n < N and np.abs(subgrad) > eta:
        gamma = 1/n                                                 # the square summable but not summable step rule
        x = x - gamma * subgrad
        each_point.append((x, objective_function(x)))               # Store current function values and its point
        values.append((objective_function(x)))                      # Store current function values

        if objective_function(x) < objective_function(x_star):
            x_star = x 
        
        subgrad = A(objective_function, x, h=h)

        n += 1

    return x_star, each_point, values 

def ISGM(objective_function, x_start, x_end, x_0, eta=1e-6, h=1e-6, N=1000) -> tuple:
    r"""
    The Implicit Specular Derivative Method.

    [Args]
        objective_function : `function`
        x_start : `float`, the start point of the interval 
        x_end : `float`, the start point of the interval 
        x_0 : `float`, the initial point 
        eta : `float`, tolerance for stopping criterion 
        h : `float`, mesh size
        N : `int`, maximum number of iterations 

    [return]
        x_star : `float`, the point where the minimum is approximated 
        each_point : `list`, points and values at the point for each iteration
        values : `list`, values at each point for each iteration
    """
    x = x_0
    x_star = x_0  
    t = x_end - x_start 
    a = alpha(objective_function, x, h=h)
    b = beta(objective_function, x, h=h)
    theta = Findtheta(objective_function, x, h=h)

    value = objective_function(x)
    each_point = [(x, value)]           # List to store function values and its point at each iteration
    values = [value]                    # List to store function values at each iteration
    T = [t]

    n = 1
    while n < N and np.abs(a + b) > eta:
        x = x - t * np.sign(a + b)

        each_point.append((x, objective_function(x)))               # Store current function values and its point
        values.append((objective_function(x)))                      # Store current function values

        if objective_function(x) < objective_function(x_star):
            x_star = x 
        
        a = alpha(objective_function, x, h=h)
        b = beta(objective_function, x, h=h)
        
        if np.pi / 4 <= theta < np.pi / 2:
            theta = max(theta, Findtheta(objective_function, x, h=h))
            t = t * np.sin(theta)
        elif 0 <= theta < np.pi / 4:
            theta = min(theta, Findtheta(objective_function, x, h=h))
            t = t / (2 * np.cos(theta))

        T.append(t)

        n += 1

    return x_star, each_point, values, T

########################################################################################################################
# OPTIMIZATION with the above methods
########################################################################################################################

def OPTIMIZATION(objective_function, x_start, x_end, x_0, gamma=0.005, eta=1e-6, h=1e-6, N=1000) -> tuple:
    method1 = SM_with_constant_step_size(objective_function, x_0=x_0, gamma=gamma, eta=eta, h=h, N=N)
    method2 = SM_with_diminishing_step_size(objective_function, x_0=x_0, eta=eta, h=h, N=N)
    method3 = SGM_with_constant_step_size(objective_function, x_0=x_0, gamma=gamma, eta=eta, h=h, N=N)
    method4 = SGM_with_diminishing_step_size(objective_function, x_0=x_0, eta=eta, h=h, N=N)
    method5 = ISGM(objective_function, x_start=x_start, x_end=x_end, x_0=x_0, eta=eta, h=h, N=N)

    return method1, method2, method3, method4, method5

########################################################################################################################
# VISUALIZATION
########################################################################################################################

def VISUALIZATION(objective_function, x_start, x_end, x_0, gamma=0.005, eta=1e-6, h=1e-6, N=1000, figure_size=(5, 3), filename='filename', legend=True, functionname='f'):

    result = OPTIMIZATION(objective_function=objective_function, x_start=x_start, x_end=x_end, x_0=x_0, gamma=gamma, eta=eta, h=h, N=N)

    X = list(range(0, N))
    
    H = [h]*N
    Y1 = result[0][2]
    X1 = X[:len(Y1)]
    Y2 = result[1][2]
    X2 = X[:len(Y2)]
    Y3 = result[2][2]
    X3 = X[:len(Y3)]
    Y4 = result[3][2]
    X4 = X[:len(Y4)]
    Y5 = result[4][2]
    X5 = X[:len(Y5)]

    # Create the plot
    plt.figure(figsize=figure_size)
    plt.plot(X, H, label='$h=10^{-6}$', color='red', linestyle='dotted', linewidth=1)
    plt.plot(X1, Y1, label='SM with $\gamma=0.005$', color='blue', linestyle='dashdot', linewidth=1)          
    plt.plot(X2, Y2, label='SM with $\gamma_k=\\frac{1}{k}$', color='green', linestyle='dashdot', linewidth=1)
    # plt.plot(X3, Y3, label='SGM with $\gamma=0.005$', color='blue', linewidth=1)
    # plt.plot(X4, Y4, label='SGM with $\gamma_k=\\frac{1}{k}$', color='green', linewidth=1)
    plt.plot(X5, Y5, label='ISGM', color='purple')
    plt.xlabel('Iteration $k$', fontsize='11')
    plt.ylabel(f'Value of objective function ${functionname}(x_k)$', fontsize='11')
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize='10') if legend is True else None
    plt.xlim(min(X), max(X))
    plt.grid(True)
    plt.yscale('log')  
    plt.savefig(f'{filename}.png', dpi=1000, bbox_inches='tight')
    plt.show()

    return result

def combine_images_vertically(image1_path, image2_path, output_path):
    # Open the images
    image1 = Image.open(image1_path)
    image2 = Image.open(image2_path)

    # Calculate the combined image size with aspect ratio maintained
    combined_width = image1.width
    combined_height = int(image1.height + (image1.width / image2.width) * image2.height)

    # Resize the images
    image1_resized = image1.resize((combined_width, image1.height))
    image2_resized = image2.resize((combined_width, combined_height - image1.height))

    # Create a new blank image with the combined size
    combined_image = Image.new('RGB', (combined_width, combined_height))

    # Paste the first image at the top
    combined_image.paste(image1_resized, (0, 0))

    # Paste the second image below the first image
    combined_image.paste(image2_resized, (0, image1_resized.height))

    # Save the combined image
    combined_image.save(output_path)