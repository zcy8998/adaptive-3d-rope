# --------------------------------------------------------
# References:
# MAE: https://github.com/facebookresearch/mae
# --------------------------------------------------------
import math
from contextlib import nullcontext
from typing import Iterable

import torch

import util.lr_sched as lr_sched
import util.misc as misc

# import wandb

def train_one_epoch(model: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler,
                    log_writer=None,
                    args=None):
    model.train(True)
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 100

    accum_iter = args.accum_iter

    optimizer.zero_grad()

    if log_writer is not None:
        print('log_dir: {}'.format(log_writer.log_dir))

    for data_iter_step, samples in enumerate(metric_logger.log_every(data_loader, print_freq, header)):

        # we use a per iteration (instead of per epoch) lr scheduler
        if data_iter_step % accum_iter == 0:
            lr_sched.adjust_learning_rate(optimizer, data_iter_step / len(data_loader) + epoch, args)

        samples = samples.to(device, non_blocking=True)

        autocast_context = torch.amp.autocast("cuda") if device.type == "cuda" else nullcontext()
        with autocast_context:
            loss, _, _ = model(samples, mask_ratio=args.mask_ratio)

        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            print(f"Sample is {samples}")
            raise ValueError(f"Loss is {loss_value}, stopping training")
            # sys.exit(1)

        loss /= accum_iter
        loss_scaler(loss, optimizer, parameters=model.parameters(),
                    update_grad=(data_iter_step + 1) % accum_iter == 0)
        if (data_iter_step + 1) % accum_iter == 0:
            optimizer.zero_grad()

        if device.type == "cuda":
            torch.cuda.synchronize()

        metric_logger.update(loss=loss_value)

        lr = optimizer.param_groups[0]["lr"]
        metric_logger.update(lr=lr)

        loss_value_reduce = misc.all_reduce_mean(loss_value)
        if log_writer is not None and (data_iter_step + 1) % accum_iter == 0:
            """ We use epoch_1000x as the x-axis in tensorboard.
            This calibrates different curves when batch size changes.
            """
            epoch_1000x = int((data_iter_step / len(data_loader) + epoch) * 1000)
            log_writer.add_scalar('train_loss', loss_value_reduce, epoch_1000x)
            log_writer.add_scalar('lr', lr, epoch_1000x)

            # Wandb logging
            # if args.local_rank == 0 and args.wandb is not None:
            #     try:
            #         wandb.log({'train_loss_step': loss_value_reduce,
            #                    'train_lr_step': lr, 'epoch_1000x': epoch_1000x})
            #     except ValueError:
            #         pass

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


def train_one_epoch_csi(model: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler,
                    log_writer=None,
                    args=None):
    model.train(True)
    metric_logger = misc.MetricLogger(delimiter=" ")
    metric_logger.add_meter('lr', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 100

    accum_iter = args.accum_iter

    optimizer.zero_grad()

    if log_writer is not None:
        print('log_dir: {}'.format(log_writer.log_dir))
    for data_iter_step, batch in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        if len(batch) == 4:
            samples, token_length, input_size, phys_meta = batch
            phys_meta = phys_meta.to(device, non_blocking=True)
        else:
            samples, token_length, input_size = batch
            phys_meta = None
        # we use a per iteration (instead of per epoch) lr scheduler
        if data_iter_step % accum_iter == 0:
            lr_sched.adjust_learning_rate(optimizer, data_iter_step / len(data_loader) + epoch, args)

        samples = samples.to(device, non_blocking=True)
        token_length = token_length.to(device)

        autocast_context = torch.amp.autocast("cuda") if device.type == "cuda" else nullcontext()
        with autocast_context:
            loss, _, _ = model(samples, token_length, input_size=input_size, mask_ratio=args.mask_ratio,
                               mask_strategy=args.mask_type, phys_meta=phys_meta)

        loss_value = loss.item()

        if not math.isfinite(loss_value):
            raise ValueError(f"Loss is {loss_value}, Sample is {samples}, stopping training")

        loss /= accum_iter
        loss_scaler(loss, optimizer, clip_grad=1.0, parameters=model.parameters(),
                    update_grad=(data_iter_step + 1) % accum_iter == 0)
        if (data_iter_step + 1) % accum_iter == 0:
            optimizer.zero_grad()

        if device.type == "cuda":
            torch.cuda.synchronize()

        metric_logger.update(loss=loss_value)

        lr = optimizer.param_groups[0]["lr"]
        metric_logger.update(lr=lr)

        loss_value_reduce = misc.all_reduce_mean(loss_value)

        if (data_iter_step + 1) % 100 == 0:
            print({'train_loss_step': loss_value_reduce})

        if log_writer is not None and (data_iter_step + 1) % accum_iter == 0:
            """ We use epoch_1000x as the x-axis in tensorboard.
            This calibrates different curves when batch size changes.
            """
            epoch_1000x = int((data_iter_step / len(data_loader) + epoch) * 1000)
            log_writer.add_scalar('train_loss', loss_value_reduce, epoch_1000x)
            log_writer.add_scalar('lr', lr, epoch_1000x)

            # Wandb logging
            # if args.local_rank == 0 and args.wandb is not None:
            #     try:
            #         wandb.log({'train_loss_step': loss_value_reduce,
            #                    'train_lr_step': lr, 'epoch_1000x': epoch_1000x})
            #     except ValueError:
            #         pass

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


def train_one_epoch_3mask(model: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler,
                    log_writer=None,
                    args=None):
    model.train(True)
    metric_logger = misc.MetricLogger(delimiter=" ")
    metric_logger.add_meter('lr', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 100

    accum_iter = args.accum_iter

    optimizer.zero_grad()

    if log_writer is not None:
        print('log_dir: {}'.format(log_writer.log_dir))
    mask_list = {'random': 0.85, 'temporal': 0.5, 'freq': 0.5}
    
    for data_iter_step, batch in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        if len(batch) == 4:
            samples, token_length, input_size, phys_meta = batch
            phys_meta = phys_meta.to(device, non_blocking=True)
        else:
            samples, token_length, input_size = batch
            phys_meta = None
        # 学习率调度
        if data_iter_step % accum_iter == 0:
            lr_sched.adjust_learning_rate(optimizer, data_iter_step / len(data_loader) + epoch, args)

        samples = samples.to(device, non_blocking=True)
        token_length = token_length.to(device)

        # 用于日志记录的标量 loss
        total_loss_value = 0.0
        
        # 归一化因子
        loss_norm_factor = len(mask_list) * accum_iter

        # ==========================================
        # 1. 串行前向与反向传播 (显存优化)
        # ==========================================
        for mask_strategy, mask_ratio in mask_list.items():
            autocast_context = torch.amp.autocast("cuda") if device.type == "cuda" else nullcontext()
            with autocast_context:
                temp_loss, _, _ = model(samples, token_length, input_size=input_size, 
                                        mask_strategy=mask_strategy, mask_ratio=mask_ratio,
                                        phys_meta=phys_meta)
                
                # 检查 NaN
                loss_val_item = temp_loss.item()
                if not math.isfinite(loss_val_item):
                    raise ValueError(f"Loss is {loss_val_item}, Sample is {samples}, stopping training")
                
                total_loss_value += loss_val_item

                # 归一化 Loss
                weighted_loss = temp_loss / loss_norm_factor
            
            # 【适配点 1】：利用 Scaler 的 __call__ 进行反向传播
            # 传入 update_grad=False，这样它只做 scale().backward()，不更新参数，不清空梯度
            loss_scaler(weighted_loss, optimizer, parameters=model.parameters(), update_grad=False)

        # ==========================================
        # 2. 梯度更新阶段
        # ==========================================
        if (data_iter_step + 1) % accum_iter == 0:
            # 【适配点 2】：直接访问内部 _scaler 进行手动更新
            # 因为 loss_scaler.__call__ 强制要求 backward，这里我们只能绕过它
            
            # 2.1 Unscale 梯度
            loss_scaler._scaler.unscale_(optimizer)
            
            # 2.2 梯度裁剪 (Clip Grad)
            # 注意：unscale 后才能正确 clip
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            # 2.3 更新参数
            loss_scaler._scaler.step(optimizer)
            
            # 2.4 更新 Scaler 缩放因子
            loss_scaler._scaler.update()
            
            # 2.5 清空梯度
            optimizer.zero_grad(set_to_none=True)

        if device.type == "cuda":
            torch.cuda.synchronize()

        # 日志记录逻辑保持不变
        loss_value = total_loss_value
        metric_logger.update(loss=loss_value)

        lr = optimizer.param_groups[0]["lr"]
        metric_logger.update(lr=lr)

        loss_value_reduce = misc.all_reduce_mean(loss_value)

        if (data_iter_step + 1) % 100 == 0:
            print({'train_loss_step': loss_value_reduce})

        if log_writer is not None and (data_iter_step + 1) % accum_iter == 0:
            epoch_1000x = int((data_iter_step / len(data_loader) + epoch) * 1000)
            log_writer.add_scalar('train_loss', loss_value_reduce, epoch_1000x)
            log_writer.add_scalar('lr', lr, epoch_1000x)
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


def train_one_epoch_save_grads(model,
                               data_loader,
                               optimizer,
                               device,
                               epoch,
                               loss_scaler,
                               log_writer,
                               args=None,
                               grad_saver=None,
                               dataset_name=None):
    """
    Training loop that saves normalized full-gradient vectors per update step.

    Important notes:
    - grad_saver: if provided, `grad_saver.save_from_model(model, dataset_name, epoch, global_step)`
                 will be called AFTER loss.backward() and BEFORE optimizer.step().
    - dataset_name: string indicating which dataset is being trained in this run (used for folder).
    """

    model.train()
    # metric_logger and lr_sched referenced in original snippet; user should provide them in scope
    metric_logger = misc.MetricLogger(delimiter=" ")
    metric_logger.add_meter('lr', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = f'Epoch: [{epoch}]'
    print_freq = 10

    accum_iter = args.accum_iter if hasattr(args, "accum_iter") else 1
    optimizer.zero_grad()

    # global step counter (useful for filenames)
    global_step = 0

    for data_iter_step, batch in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        # unpack batch: assume (samples, token_length, input_size) typical
        if isinstance(batch, (list, tuple)):
            samples = batch[0]
            token_length = batch[1] if len(batch) > 1 else None
            input_size = batch[2] if len(batch) > 2 else None
        else:
            samples = batch
            token_length = None
            input_size = None

        # lr scheduling if present
        if data_iter_step % accum_iter == 0 and hasattr(lr_sched, "adjust_learning_rate"):
            lr_sched.adjust_learning_rate(optimizer, data_iter_step / len(data_loader) + epoch, args)

        samples = samples.to(device, non_blocking=True)
        if token_length is not None:
            token_length = token_length.to(device, non_blocking=True)

        # Forward
        # We assume model returns (loss, *rest) or a scalar loss
        outputs = model(samples, token_length=token_length) if token_length is not None else model(samples)
        loss = outputs[0] if isinstance(outputs, (tuple, list)) else outputs
        if loss is None:
            raise RuntimeError("Model returned no loss.")

        # gradient accumulation support
        loss_value = loss / accum_iter
        # if you use a loss scaler (amp), use it here
        # if loss_scaler is not None:
        #     # typical usage: loss_scaler(loss, optimizer, clip_grad=None, parameters=model.parameters(), create_graph=False)
        #     loss_scaler(loss_value, optimizer, parameters=model.parameters(), update_grad=( (data_iter_step + 1) % accum_iter == 0 ))
        # else:
        loss_value.backward()

        # when ready to step
        if (data_iter_step + 1) % 5 == 0:
            # Save gradients BEFORE optimizer.step()
            if grad_saver is not None and dataset_name is not None:
                try:
                    grad_saver.save_from_model(model, dataset_name, epoch, global_step)
                except Exception as e:
                    print(f"[GradientSaver] failed to save at step {global_step}: {e}")

            optimizer.step()
            optimizer.zero_grad()
        
        metric_logger.update(loss=loss_value)
        lr = optimizer.param_groups[0]["lr"]
        metric_logger.update(lr=lr)

        global_step += 1

        # optional debug break if provided
        if hasattr(args, "debug_max_batches") and args.debug_max_batches is not None:
            if global_step >= int(args.debug_max_batches):
                break

    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}
