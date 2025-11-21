#!/usr/bin/env python
"""
深度诊断训练问题
"""
import torch
import numpy as np
from pathlib import Path
from collections import Counter
from pytorch_image_classification import (
    create_dataloader,
    create_model,
    create_loss,
    create_optimizer,
    create_scheduler,
    get_default_config,
    update_config,
)

def deep_diagnose():
    print("=" * 70)
    print("深度诊断训练问题")
    print("=" * 70)
    
    # 加载配置
    config = get_default_config()
    config.merge_from_file('configs/cabbage/shake_shake.yaml')
    config = update_config(config)
    device = torch.device(config.device)
    
    print(f"\n配置信息:")
    print(f"  学习率: {config.train.base_lr}")
    print(f"  批次大小: {config.train.batch_size}")
    print(f"  类别数: {config.dataset.n_classes}")
    print(f"  总epoch数: {config.scheduler.epochs}")
    
    # 1. 检查所有批次的标签分布
    print("\n" + "=" * 70)
    print("1. 检查所有训练批次的标签分布")
    print("=" * 70)
    train_loader, val_loader = create_dataloader(config, is_train=True)
    
    all_labels = []
    for batch_idx, (data, targets) in enumerate(train_loader):
        all_labels.extend(targets.numpy().tolist())
        if batch_idx >= 10:  # 只检查前10个批次
            break
    
    label_counts = Counter(all_labels)
    print(f"  前10个批次的标签分布:")
    for i in range(config.dataset.n_classes):
        count = label_counts.get(i, 0)
        print(f"    类别 {i}: {count} 次")
    
    # 检查是否有所有类别
    if len(label_counts) < config.dataset.n_classes:
        print(f"  ⚠️  警告: 只看到 {len(label_counts)}/{config.dataset.n_classes} 个类别")
    
    # 2. 检查模型初始输出
    print("\n" + "=" * 70)
    print("2. 检查模型初始输出分布")
    print("=" * 70)
    model = create_model(config).to(device)
    model.eval()
    
    # 获取一个批次
    data, targets = next(iter(train_loader))
    data = data.to(device)
    targets = targets.to(device)
    
    with torch.no_grad():
        output = model(data)
        probs = torch.softmax(output, dim=1)
        
        print(f"  模型输出形状: {output.shape}")
        print(f"  输出logits范围: [{output.min().item():.2f}, {output.max().item():.2f}]")
        print(f"  输出logits均值: {output.mean().item():.2f}")
        print(f"  输出logits标准差: {output.std().item():.2f}")
        
        print(f"\n  每个样本的预测概率分布:")
        for i in range(min(3, len(data))):  # 只显示前3个样本
            print(f"    样本 {i}: {probs[i].cpu().numpy()}")
        
        # 检查预测
        _, pred = output.topk(1, 1, True, True)
        pred = pred.squeeze(1)
        print(f"\n  预测值: {pred.cpu().numpy()}")
        print(f"  真实标签: {targets.cpu().numpy()}")
        
        # 计算初始准确率
        correct = (pred == targets).sum().item()
        acc = correct / len(targets)
        print(f"  初始准确率: {acc:.4f} ({correct}/{len(targets)})")
    
    # 3. 检查损失函数
    print("\n" + "=" * 70)
    print("3. 检查损失函数")
    print("=" * 70)
    train_loss, val_loss = create_loss(config)
    
    with torch.no_grad():
        loss_value = train_loss(output, targets)
        print(f"  损失值: {loss_value.item():.4f}")
        
        # 计算理论最大损失（随机猜测）
        theoretical_max_loss = -np.log(1.0 / config.dataset.n_classes)
        print(f"  理论最大损失（随机）: {theoretical_max_loss:.4f}")
        print(f"  当前损失/理论最大: {loss_value.item() / theoretical_max_loss:.4f}")
        
        if loss_value.item() > theoretical_max_loss * 0.9:
            print(f"  ⚠️  警告: 损失值接近随机猜测，模型可能没有学习")
    
    # 4. 检查梯度
    print("\n" + "=" * 70)
    print("4. 检查梯度（训练一个批次）")
    print("=" * 70)
    model.train()
    optimizer = create_optimizer(config, model)
    
    # 前向传播
    output = model(data)
    loss = train_loss(output, targets)
    
    # 反向传播
    optimizer.zero_grad()
    loss.backward()
    
    # 检查梯度
    total_grad_norm = 0
    param_count = 0
    zero_grad_count = 0
    
    for name, param in model.named_parameters():
        if param.grad is not None:
            param_grad_norm = param.grad.data.norm(2)
            total_grad_norm += param_grad_norm.item() ** 2
            param_count += 1
            if param_grad_norm.item() < 1e-8:
                zero_grad_count += 1
        else:
            zero_grad_count += 1
    
    total_grad_norm = total_grad_norm ** (1. / 2)
    
    print(f"  总梯度范数: {total_grad_norm:.6f}")
    print(f"  有梯度的参数数量: {param_count}")
    print(f"  零梯度参数数量: {zero_grad_count}")
    
    if total_grad_norm < 1e-6:
        print(f"  ❌ 错误: 梯度太小，模型可能无法学习！")
    elif total_grad_norm > 100:
        print(f"  ⚠️  警告: 梯度很大，可能需要梯度裁剪")
    else:
        print(f"  ✓ 梯度正常")
    
    # 5. 检查学习率调度器
    print("\n" + "=" * 70)
    print("5. 检查学习率调度器")
    print("=" * 70)
    scheduler = create_scheduler(config, optimizer, steps_per_epoch=len(train_loader))
    
    print(f"  初始学习率: {scheduler.get_last_lr()[0]:.6f}")
    print(f"  调度器类型: {config.scheduler.type}")
    print(f"  总epoch数: {config.scheduler.epochs}")
    
    # 模拟几个step
    print(f"\n  前5个step的学习率:")
    for step in range(5):
        lr = scheduler.get_last_lr()[0]
        print(f"    Step {step}: {lr:.6f}")
        scheduler.step()
    
    # 6. 检查一个完整的训练步骤
    print("\n" + "=" * 70)
    print("6. 检查完整训练步骤")
    print("=" * 70)
    
    model.train()
    optimizer.zero_grad()
    
    # 前向
    output = model(data)
    loss_before = train_loss(output, targets).item()
    
    # 反向
    loss = train_loss(output, targets)
    loss.backward()
    
    # 检查梯度裁剪
    if config.train.gradient_clip > 0:
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.train.gradient_clip)
        print(f"  应用梯度裁剪: {config.train.gradient_clip}")
    
    # 更新
    optimizer.step()
    scheduler.step()
    
    # 再次前向，看loss是否变化
    model.eval()
    with torch.no_grad():
        output_after = model(data)
        loss_after = train_loss(output_after, targets).item()
    
    print(f"  更新前损失: {loss_before:.4f}")
    print(f"  更新后损失: {loss_after:.4f}")
    print(f"  损失变化: {loss_after - loss_before:.4f}")
    
    if abs(loss_after - loss_before) < 1e-6:
        print(f"  ⚠️  警告: 损失几乎没有变化，可能学习率太小或梯度太小")
    
    # 7. 总结和建议
    print("\n" + "=" * 70)
    print("诊断总结和建议")
    print("=" * 70)
    
    issues = []
    suggestions = []
    
    if loss_value.item() > theoretical_max_loss * 0.9:
        issues.append("损失值接近随机猜测")
        suggestions.append("1. 增大学习率（建议从 1e-4 改为 1e-3 或 1e-2）")
    
    if total_grad_norm < 1e-6:
        issues.append("梯度太小")
        suggestions.append("2. 检查模型初始化或增大学习率")
    
    if abs(loss_after - loss_before) < 1e-3:
        issues.append("训练步骤后损失几乎不变")
        suggestions.append("3. 增大学习率或检查数据")
    
    if len(issues) == 0:
        print("\n✓ 未发现明显问题，可能需要更多训练轮次")
        print("  建议: 继续训练观察，或尝试增大学习率")
    else:
        print(f"\n发现 {len(issues)} 个潜在问题:")
        for i, issue in enumerate(issues, 1):
            print(f"  {i}. {issue}")
        
        print(f"\n建议:")
        for suggestion in suggestions:
            print(f"  {suggestion}")
    
    print("\n诊断完成！")

if __name__ == '__main__':
    deep_diagnose()







