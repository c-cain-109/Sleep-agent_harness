def model_function(x):
    """待优化的模型函数"""
    a, b, c = 1.0, 1.0, 1.0  # Agent 自动优化
    return a * x**2 + b * x + c
