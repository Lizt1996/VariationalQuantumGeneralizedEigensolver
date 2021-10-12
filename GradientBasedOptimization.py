import numpy as np
import random


def QuasiNewtonBFGS(fun, jacobian, x: list, iters: int = 100000, tol: float = 1e-5, callback=None):
    """
    实现BFGS拟牛顿法
    :param fun: 原函数
    :param jacobian: 原函数的雅可比矩阵 -接受参数 x
    :param x0: 初始值
    :param iters: 遍历的最大epoch
    :param tol: 停止精度
    :param callback: 回调函数，在每一次迭代后执行 -接收参数 x
    :return: 最终更新完毕的x值
    """
    # 步长。
    alpha = 0.1
    # 初始化B正定矩阵
    x_len = x.__len__()
    B = np.eye(x_len)
    # 一阶导g的第二范式的最小值（阈值）
    epsilon = tol
    for i in range(1, iters):
        g = jacobian(x)
        if callback is not None:
            callback(x)
        if np.linalg.norm(g) < epsilon:
            break
        p = -np.linalg.solve(B, g)
        # 更新x值
        x_new = x + p * GoldsteinSearch(fun, jacobian, p, x, 1, 0.1, 2)
        g_new = jacobian(x_new)
        y = g_new - g
        k = x_new - x
        y_t = y.reshape([x_len, 1])
        Bk = np.dot(B, k)
        k_t_B = np.dot(k, B)
        kBk = np.dot(np.dot(k, B), k)
        # 更新B正定矩阵。完全按照公式来计算
        B = B + y_t * y / np.dot(y, k) - Bk.reshape([x_len, 1]) * k_t_B / kBk
        x = x_new
    return x


def QuasiNewtonBFGS_sto(fun, jacobian, x: list, iters: int = 100000, callback=None):
    """
    实现BFGS拟牛顿法
    :param fun: 原函数
    :param jacobian: 原函数的雅可比矩阵 -接受参数 x
    :param x0: 初始值
    :param iters: 遍历的最大epoch
    :param tol: 停止精度
    :param callback: 回调函数，在每一次迭代后执行 -接收参数 x
    :return: 最终更新完毕的x值
    """
    # 步长。
    alpha = 0.001
    # 初始化B正定矩阵
    x_len = x.__len__()
    B = np.eye(x_len)
    # 一阶导g的第二范式的最小值（阈值）
    for i in range(1, iters):
        g = jacobian(x)
        if callback is not None:
            callback(x)
        p = -np.linalg.solve(B, g)
        # 更新x值
        x_new = x + p * alpha
        g_new = jacobian(x_new)
        y = g_new - g
        k = x_new - x
        y_t = y.reshape([x_len, 1])
        Bk = np.dot(B, k)
        k_t_B = np.dot(k, B)
        kBk = np.dot(np.dot(k, B), k)
        # 更新B正定矩阵。完全按照公式来计算
        B = B + y_t * y / np.dot(y, k) - Bk.reshape([x_len, 1]) * k_t_B / kBk
        x = x_new
    return x


def GoldsteinSearch(f, df, d, x, alpham, rho, t):
    """
    线性搜索子函数
    数f，导数df，当前迭代点x和当前搜索方向d
    """
    flag = 0
    a = 0
    b = alpham
    fk = f(x)
    gk = df(x)
    phi0 = fk
    dphi0 = np.dot(gk, d)
    # print(dphi0)
    alpha = b * random.uniform(0, 1)
    count = 0
    while flag == 0 and count < 100:
        newfk = f(x + alpha * d)
        phi = newfk
        # print(phi,phi0,rho,alpha ,dphi0)
        if (phi - phi0) <= (rho * alpha * dphi0):
            if (phi - phi0) >= ((1 - rho) * alpha * dphi0):
                flag = 1
            else:
                a = alpha
                b = b
                if b < alpham:
                    alpha = (a + b) / 2
                else:
                    alpha = t * alpha
        else:
            a = a
            b = alpha
            alpha = (a + b) / 2
        count += 1
    return alpha


def GoldsteinSearch_tomax(f, df, d, x, alpham, rho, t):
    """
    线性搜索子函数
    数f，导数df，当前迭代点x和当前搜索方向d
    """
    flag = 0
    a = 0
    b = alpham
    fk = f(x)
    gk = df(x)
    phi0 = fk
    dphi0 = np.dot(gk, d)
    # print(dphi0)
    alpha = b * random.uniform(0, 1)
    count = 0
    while flag == 0 and count < 10:
        newfk = f(x + alpha * d)
        phi = newfk
        # print(phi,phi0,rho,alpha ,dphi0)
        if (phi - phi0) >= (rho * alpha * dphi0):
            if (phi - phi0) <= ((1 - rho) * alpha * dphi0):
                flag = 1
            else:
                a = alpha
                b = b
                if b > alpham:
                    alpha = (a + b) / 2
                else:
                    alpha = t * alpha
        else:
            a = a
            b = alpha
            alpha = (a + b) / 2
        count += 1
    return alpha


def steepest(fun, jacobian, x0: list, iters: int = 100000, tol: float = 1e-5, callback=None, direct='-', alpha=None):
    if direct == '-':
        sign = -1
    else:
        sign = 1
    imax = iters
    epo = np.zeros((2, imax))
    i = 1
    x = x0
    grad = jacobian(x)
    delta = sum(grad ** 2)  # 初始误差
    if callback is not None:
        callback(x)
    while i < imax and delta > tol:
        print('epoch = ' + str(i) + 'loss = ' + str(fun(x)))
        p = sign * jacobian(x)
        if direct == '-':
            alpha = GoldsteinSearch(fun, jacobian, p, x, 1, 0.1, 2)
        else:
            alpha = GoldsteinSearch_tomax(fun, jacobian, p, x, 1, 0.1, 2)
        x = x + alpha * p
        grad = jacobian(x)
        delta = sum(grad ** 2)
        i = i + 1
        if callback is not None:
            callback(x)
    return x


def steepest_sto(fun, jacobian, x0: list, alpha=0.001, iters: int = 10000, callback=None, direct='-', tol=None):
    if direct == '-':
        sign = -1
    else:
        sign = 1
    print('初始点为:')
    print(x0, '\n')
    imax = iters
    i = 0
    x = x0
    while i < imax:
        if callback is not None:
            callback(x)
        p = sign * jacobian(x)
        x = x + alpha * p
        i = i + 1
    return x


def Adam(fun, jacobian, x0: list, iters: int = 10000, tol: float = 1e-5, callback=None, width: int = 10,
         alpha: float = 0.005, beta1: float = 0.9, beta2: float = 0.999, epsilon: float = 0.00000001):
    ml = np.zeros(len(x0))
    vl = np.zeros(len(x0))
    x = x0
    conv_judge_list = [0 for i in range(width)]
    t = 0
    while t < iters:
        t += 1
        '''Adam Update'''
        # Get gradients w.r.t. stochastic objective at timestep i
        g = jacobian(x)
        # Update biased first moment estimate
        m = beta1 * ml + (1 - beta1) * g
        # Update biased second raw moment estimate
        v = beta2 * vl + (1 - beta2) * (g ** 2)
        # Compute bias-corrected first moment estimate
        m_hat = m / (1 - beta1 ** t)
        # Compute bias-corrected second raw moment estimate
        v_hat = v / (1 - beta2 ** t)
        # Update parameters
        x = x - alpha * m_hat / (np.sqrt(v_hat) + epsilon)
        '''Call back function'''
        if callback is not None:
            callback(x)
        '''Convergence Judgement'''
        # conv_judge_list 1 looped right shift
        conv_judge_list.insert(0, conv_judge_list.pop())
        conv_judge_list[0] = fun(x)
        if t > width:
            # Convergence condition: max - min < tol
            if max(conv_judge_list) - min(conv_judge_list) < tol:
                return x
    return x


def Adam_S(fun, jacobian, x0: list, iters: int = 10000, tol: float = 1e-5, callback=None, width: int = 10,
           alpha: float = 0.005, beta1: float = 0.9, beta2: float = 0.999, epsilon: float = 0.00000001):
    ml = np.zeros(len(x0))
    vl = np.zeros(len(x0))
    x = x0
    conv_judge_list = [0 for i in range(width)]
    t = 0
    while t < iters:
        t += 1
        '''Adam Update'''
        # Get gradients w.r.t. stochastic objective at timestep i
        g = jacobian(x)
        # Update biased first moment estimate
        m = beta1 * ml + (1 - beta1) * g
        # Update biased second raw moment estimate
        v = beta2 * vl + (1 - beta2) * (g ** 2)
        # Compute bias-corrected first moment estimate
        m_hat = m / (1 - beta1 ** t)
        # Compute bias-corrected second raw moment estimate
        v_hat = v / (1 - beta2 ** t)
        # Update parameters
        x = x - alpha * m_hat / (np.sqrt(v_hat) + epsilon)
        '''Call back function'''
        if callback is not None:
            callback(x)
        else:
            print('epoch = ' + str(t) + 'loss = ' + str(fun(x)))

        '''Convergence Judgement'''
        # conv_judge_list 1 looped right shift
        conv_judge_list.insert(0, conv_judge_list.pop())
        conv_judge_list[0] = fun(x)
        if t > width:
            # Convergence condition: max - min < tol
            if max(conv_judge_list) - min(conv_judge_list) < tol:
                return x
    return x


def Adam_traced(fun, jacobian, x0: list, iters: int = 10000, tol: float = 1e-5, callback=None, width: int = 10,
                alpha: float = 0.005, beta1: float = 0.9, beta2: float = 0.999, epsilon: float = 0.00000001):
    ml = np.zeros(len(x0))
    vl = np.zeros(len(x0))
    x = x0
    conv_judge_list = [0 for i in range(width)]
    t = 0
    while t < iters:
        t += 1
        '''Adam Update'''
        # Get gradients w.r.t. stochastic objective at timestep i
        g = jacobian(x)
        # Update biased first moment estimate
        m = beta1 * ml + (1 - beta1) * g
        # Update biased second raw moment estimate
        v = beta2 * vl + (1 - beta2) * (g ** 2)
        # Compute bias-corrected first moment estimate
        m_hat = m / (1 - beta1 ** t)
        # Compute bias-corrected second raw moment estimate
        v_hat = v / (1 - beta2 ** t)
        # Update parameters
        x = x - alpha * m_hat / (np.sqrt(v_hat) + epsilon)
        '''Call back function'''
        if callback is not None:
            callback(x)
        '''Convergence Judgement'''
        # conv_judge_list 1 looped right shift
        conv_judge_list.insert(0, conv_judge_list.pop())
        conv_judge_list[0] = fun(x)
        if t > width:
            # Convergence condition: max - min < tol
            if max(conv_judge_list) - min(conv_judge_list) < tol:
                return x
    return x
