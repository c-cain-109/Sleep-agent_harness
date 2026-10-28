
import torch
import torch.nn as nn
import selfeeg.models as models


# 五分类标签的固定顺序
CLASS_NAMES = ("W", "N1", "N2", "N3", "REM")


def build_model(seed: int = 42) -> nn.Module:
    """创建单通道、100 Hz、五分类的 TinySleepNet。"""
    return models.TinySleepNet(
        nb_classes=len(CLASS_NAMES),
        Chans=1,
        Fs=100,
        return_logits=True,
        seed=seed,
    )


def smoke_test() -> None:
    """用模拟数据验证前向计算、损失计算和反向传播。"""
    torch.manual_seed(42)

    model = build_model(seed=42)

    # 4 个模拟 epoch；每段 30 秒，每秒 100 个采样点
    x = torch.randn(4, 1, 3000, dtype=torch.float32)

    # 仅用于测试训练流程，不是真实睡眠标签
    y = torch.tensor([0, 1, 2, 3], dtype=torch.long)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    model.train()
    optimizer.zero_grad()

    logits = model(x)
    loss = criterion(logits, y)

    assert logits.shape == (4, 5), (
        f"输出形状错误：{tuple(logits.shape)}"
    )
    assert torch.isfinite(logits).all(), "模型输出包含非有限值"
    assert torch.isfinite(loss), "损失值不是有限数"

    loss.backward()

    # 检查是否有参数获得有效梯度
    has_valid_gradient = any(
        p.grad is not None and torch.isfinite(p.grad).all()
        for p in model.parameters()
    )
    assert has_valid_gradient, "未发现有效梯度"

    optimizer.step()

    print("输入形状:", tuple(x.shape))
    print("输出形状:", tuple(logits.shape))
    print("测试损失:", round(loss.item(), 4))
    print("反向传播: 正常")
    print("优化器更新: 正常")
    print("TinySleepNet 训练流程测试通过！")


if __name__ == "__main__":
    smoke_test()
