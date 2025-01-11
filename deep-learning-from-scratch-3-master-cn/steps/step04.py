import numpy as np


class Variable:
    """表示变量的类。
    
    Attributes:
        data: 存储变量的数值数据
    """
    def __init__(self, data):
        self.data = data


class Function:
    """表示函数的基类。
    
    所有具体的函数都应该继承自这个类，并实现forward方法。
    """
    def __call__(self, input):
        """函数调用的实现。
        
        Args:
            input (Variable): 输入变量
            
        Returns:
            Variable: 函数运算的结果
        """
        x = input.data
        y = self.forward(x)
        output = Variable(y)
        self.input = input
        self.output = output
        return output

    def forward(self, x):
        """前向计算的具体实现。
        
        Args:
            x: 输入数据
            
        Raises:
            NotImplementedError: 基类中的方法需要在子类中实现
        """
        raise NotImplementedError()


class Square(Function):
    """计算平方的函数。"""
    def forward(self, x):
        """实现平方计算。
        
        Args:
            x: 输入数据
            
        Returns:
            数据的平方值
        """
        return x ** 2


class Exp(Function):
    """计算自然指数的函数。"""
    def forward(self, x):
        """实现自然指数计算。
        
        Args:
            x: 输入数据
            
        Returns:
            数据的自然指数值
        """
        return np.exp(x)


def numerical_diff(f, x, eps=1e-4):
    """使用中心差分计算函数的数值导数。
    
    Args:
        f (Function): 要计算导数的函数
        x (Variable): 计算导数的位置
        eps (float): 微小扰动值
        
    Returns:
        float: 函数在x处的导数近似值
    """
    x0 = Variable(x.data - eps)
    x1 = Variable(x.data + eps)
    y0 = f(x0)
    y1 = f(x1)
    return (y1.data - y0.data) / (2 * eps)


f = Square()
x = Variable(np.array(2.0))
dy = numerical_diff(f, x)
print('函数f(x)=x^2在x=2.0的导数为', dy)


def f(x):
    """复合函数示例。
    
    计算 square(exp(square(x)))
    
    Args:
        x (Variable): 输入变量
        
    Returns:
        Variable: 复合函数计算结果
    """
    A = Square()
    B = Exp()
    C = Square()
    return C(B(A(x)))


x = Variable(np.array(0.5))
dy = numerical_diff(f, x)
print('复合函数[a=square(x)，b=exp(a),c=square(b), y = c(b(a(x)))]在x=0.5的导数为', dy)
