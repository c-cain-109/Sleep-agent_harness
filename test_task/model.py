def model_function(x):
    """待优化的模型函数"""
    a, b, c = 2.5, -1.0 , 1.2  # 用户手动修改这些参数模拟 Agent 行为
    return a * x**2 + b * x + c
