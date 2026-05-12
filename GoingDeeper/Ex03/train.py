"""학습/검증/추론 유틸리티 함수"""

import time
import torch
import torch.nn as nn


def hourglass_loss(outputs, target, joints_vis):
    """Stacked Hourglass 손실 함수 (Intermediate Supervision)"""
    criterion = nn.MSELoss(reduction='none')
    total_loss = 0
    mask = joints_vis.unsqueeze(-1).unsqueeze(-1)

    for output in outputs:
        loss = criterion(output, target)
        loss = loss * mask
        total_loss += loss.mean()

    return total_loss


def simple_baseline_loss(output, target, joints_vis):
    """SimpleBaseline 손실 함수 (단일 MSE)"""
    criterion = nn.MSELoss(reduction='none')
    mask = joints_vis.unsqueeze(-1).unsqueeze(-1)

    loss = criterion(output, target)
    loss = loss * mask
    return loss.mean()


def train_one_epoch(model, loader, optimizer, loss_fn, device, model_type='hourglass'):
    """1 epoch 학습"""
    model.train()
    total_loss = 0

    for images, heatmaps, vis in loader:
        images = images.to(device)
        heatmaps = heatmaps.to(device)
        vis = vis.to(device)

        optimizer.zero_grad()
        outputs = model(images)

        if model_type == 'hourglass':
            loss = loss_fn(outputs, heatmaps, vis)
        else:
            loss = loss_fn(outputs, heatmaps, vis)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(loader)


def validate(model, loader, loss_fn, device, model_type='hourglass'):
    """검증"""
    model.eval()
    total_loss = 0

    with torch.no_grad():
        for images, heatmaps, vis in loader:
            images = images.to(device)
            heatmaps = heatmaps.to(device)
            vis = vis.to(device)

            outputs = model(images)

            if model_type == 'hourglass':
                loss = loss_fn(outputs, heatmaps, vis)
            else:
                loss = loss_fn(outputs, heatmaps, vis)

            total_loss += loss.item()

    return total_loss / len(loader)


def fit(model, train_loader, val_loader, optimizer, scheduler, loss_fn,
        num_epochs, device, model_type='hourglass', print_every=5):
    """전체 학습 루프"""
    history = {'train_loss': [], 'val_loss': []}

    start_time = time.time()
    for epoch in range(num_epochs):
        train_loss = train_one_epoch(model, train_loader, optimizer,
                                      loss_fn, device, model_type)
        val_loss = validate(model, val_loader, loss_fn, device, model_type)

        scheduler.step()

        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)

        if (epoch + 1) % print_every == 0 or epoch == 0:
            elapsed = time.time() - start_time
            print(f"Epoch [{epoch+1:3d}/{num_epochs}] | "
                  f"Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f} | "
                  f"Time: {elapsed:.1f}s")

    total_time = time.time() - start_time
    print(f"\n학습 완료! 총 시간: {total_time:.1f}초 ({total_time/num_epochs:.1f}초/epoch)")

    return history


def get_predictions(model, image_tensor, device, model_type='hourglass'):
    """모델로부터 키포인트 예측을 수행합니다."""
    from preprocess import heatmaps_to_keypoints

    model.eval()
    with torch.no_grad():
        input_tensor = image_tensor.unsqueeze(0).to(device)
        output = model(input_tensor)

        if model_type == 'hourglass':
            heatmaps = output[-1][0]
        else:
            heatmaps = output[0]

        heatmaps = heatmaps.cpu().numpy()

    keypoints, confidences = heatmaps_to_keypoints(heatmaps)
    return keypoints, confidences, heatmaps


def measure_inference_time(model, loader, device, num_runs=3):
    """모델의 평균 추론 시간을 측정합니다."""
    model.eval()
    times = []

    with torch.no_grad():
        for run in range(num_runs):
            start = time.time()
            for images, _, _ in loader:
                images = images.to(device)
                _ = model(images)
            elapsed = time.time() - start
            times.append(elapsed)

    import numpy as np
    avg_time = np.mean(times)
    samples = len(loader.dataset)
    return avg_time, avg_time / samples
