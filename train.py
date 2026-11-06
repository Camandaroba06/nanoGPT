import os
import time
import math
import random
import csv
import argparse
from contextlib import nullcontext

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group

from model import GPTConfig, GPT


def get_batch(split, data_dir, batch_size, block_size, device, device_type):
    filename = 'train.bin' if split == 'train' else 'validation.bin'
    filepath = os.path.join(data_dir, filename)

    data = np.memmap(filepath, dtype=np.uint16, mode='r')
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([torch.from_numpy((data[i:i+block_size]).astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy((data[i+1:i+1+block_size]).astype(np.int64)) for i in ix])

    if device_type == 'cuda':
        x, y = x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
    else:
        x, y = x.to(device), y.to(device)

    return x, y


@torch.no_grad()
def estimate_loss(model, eval_iters, ctx, data_dir, batch_size, block_size, device, device_type):
    out = {}
    inference_latency_ms = 0.0

    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)

        t_start = time.time()
        for k in range(eval_iters):
            X, Y = get_batch(split, data_dir, batch_size, block_size, device, device_type)
            with ctx:
                logits, loss = model(X, Y)
            losses[k] = loss.item()
        t_end = time.time()

        out[split] = losses.mean()

        if split == 'val':
            inference_latency_ms = ((t_end - t_start) / eval_iters) * 1000

    model.train()
    return out, inference_latency_ms


def get_lr(it, warmup_iters, lr_decay_iters, learning_rate, min_lr):
    if it < warmup_iters:
        return learning_rate * (it + 1) / (warmup_iters + 1)
    if it > lr_decay_iters:
        return min_lr

    decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))

    return min_lr + coeff * (learning_rate - min_lr)

def get_rng_state():
    state = {
        'python': random.getstate(),
        'numpy': np.random.get_state(),
        'torch': torch.get_rng_state(),
    }

    if torch.cuda.is_available():
        state['cuda'] = torch.cuda.get_rng_state_all()

    return state


def set_rng_state(state):
    random.setstate(state['python'])
    np.random.set_state(state['numpy'])
    torch.set_rng_state(state['torch'])

    if torch.cuda.is_available() and 'cuda' in state:
        torch.cuda.set_rng_state_all(state['cuda'])

def collect_rank_states(X, Y, ddp):
    local_state = {
        'rng': get_rng_state(),
        'X': X.detach().cpu(),
        'Y': Y.detach().cpu(),
    }

    if ddp:
        rank_states = [None for _ in range(dist.get_world_size())]
        dist.all_gather_object(rank_states, local_state)
        return rank_states

    return [local_state]


# ALTERAÇÃO: salvamento atômico do checkpoint
def save_checkpoint(checkpoint_path, checkpoint):
    tmp_path = checkpoint_path + '.tmp'
    torch.save(checkpoint, tmp_path)
    os.replace(tmp_path, checkpoint_path)


# ALTERAÇÃO: preparação do CSV para scratch ou resume
def prepare_metrics_file(metrics_file, init_from, resume_iter):
    columns = [
        'iter',
        'train_loss',
        'val_loss',
        'train_ppl',
        'val_ppl',
        'step_time_ms',
        'val_latency_ms'
    ]

    if init_from == 'scratch':
        with open(metrics_file, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=columns)
            writer.writeheader()
        return

    previous_rows = []

    if os.path.exists(metrics_file):
        with open(metrics_file, 'r', newline='') as f:
            reader = csv.DictReader(f)

            for row in reader:
                try:
                    if int(row['iter']) < resume_iter:
                        previous_rows.append(row)
                except (ValueError, KeyError):
                    continue

    with open(metrics_file, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerows(previous_rows)


def train_loop(model, optimizer, scaler, ctx, cfg, master_process, device, device_type, raw_model):
    if cfg.resume_rank_state is not None:
        X = cfg.resume_rank_state['X'].to(device)
        Y = cfg.resume_rank_state['Y'].to(device)
    else:
        X, Y = get_batch(
            'train', cfg.data_dir, cfg.batch_size,
            cfg.block_size, device, device_type
        )

    t0 = time.time()
    local_iter_num = 0
    iter_num = cfg.iter_num
    best_val_loss = cfg.best_val_loss
    running_mfu = -1.0

    metrics_file = os.path.join(
        cfg.out_dir,
        f'metrics_{cfg.block_type}.csv'
    )

    if master_process:
        prepare_metrics_file(
            metrics_file,
            cfg.init_from,
            iter_num
        )

    dt = 0.0

    while True:
        lr = (
            get_lr(
                iter_num,
                cfg.warmup_iters,
                cfg.lr_decay_iters,
                cfg.learning_rate,
                cfg.min_lr
            )
            if cfg.decay_lr else cfg.learning_rate
        )

        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        should_eval = (
            iter_num % cfg.eval_interval == 0
            or iter_num == cfg.max_iters
        )

        if should_eval:

            if cfg.ddp:
                dist.barrier()

            rank_states = collect_rank_states(X, Y, cfg.ddp)

            if master_process:
                losses, val_latency_ms = estimate_loss(
                    raw_model,
                    cfg.eval_iters,
                    ctx,
                    cfg.data_dir,
                    cfg.batch_size,
                    cfg.block_size,
                    device,
                    device_type
                )

                train_ppl = math.exp(losses['train'])
                val_ppl = math.exp(losses['val'])
                step_time_ms = dt * 1000

                print(
                    f"step {iter_num}: val loss {losses['val']:.4f} | "
                    f"val ppl {val_ppl:.2f} | "
                    f"latência inf. {val_latency_ms:.2f}ms"
                )

                with open(metrics_file, 'a', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow([
                        iter_num,
                        f"{losses['train']:.4f}",
                        f"{losses['val']:.4f}",
                        f"{train_ppl:.2f}",
                        f"{val_ppl:.2f}",
                        f"{step_time_ms:.2f}",
                        f"{val_latency_ms:.2f}"
                    ])

                if losses['val'] < best_val_loss or cfg.always_save_checkpoint:
                    best_val_loss = losses['val']

                    if iter_num > 0:
                        checkpoint = {
                            'model': raw_model.state_dict(),
                            'optimizer': optimizer.state_dict(),
                            'model_args': cfg.model_args,
                            'iter_num': iter_num,
                            'best_val_loss': best_val_loss,
                            'scaler': scaler.state_dict(),
                            'rank_states': rank_states,
                        }

                        print(f"Salvo checkpoint em {cfg.out_dir}")
                        save_checkpoint(
                            cfg.checkpoint_path,
                            checkpoint
                        )

            if cfg.ddp:
                dist.barrier()

        if iter_num == 0 and cfg.eval_only:
            break

        if iter_num >= cfg.max_iters:
            break

        loss_accum = 0.0

        for micro_step in range(cfg.gradient_accumulation_steps):
            if cfg.ddp:
                model.require_backward_grad_sync = (
                    micro_step == cfg.gradient_accumulation_steps - 1
                )

            with ctx:
                logits, loss = model(X, Y)
                loss = loss / cfg.gradient_accumulation_steps
                loss_accum += loss.detach().float()

            X, Y = get_batch(
                'train',
                cfg.data_dir,
                cfg.batch_size,
                cfg.block_size,
                device,
                device_type
            )

            scaler.scale(loss).backward()

        if cfg.grad_clip != 0.0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                cfg.grad_clip
            )

        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)

        t1 = time.time()
        dt = t1 - t0
        t0 = t1

        if iter_num % cfg.log_interval == 0 and master_process:
            lossf = loss_accum.item()

            if local_iter_num >= 5:
                mfu = raw_model.estimate_mfu(
                    cfg.batch_size * cfg.gradient_accumulation_steps,
                    dt
                )
                running_mfu = (
                    mfu if running_mfu == -1.0
                    else 0.9 * running_mfu + 0.1 * mfu
                )

            print(
                f"iter {iter_num}: loss {lossf:.4f}, "
                f"time {dt*1000:.2f}ms, mfu {running_mfu*100:.2f}%"
            )

        iter_num += 1
        local_iter_num += 1


def main():
    parser = argparse.ArgumentParser(
        description="Treinamento nanoGPT Refatorado"
    )

    parser.add_argument(
        '--block_type',
        type=str,
        default='seq',
        choices=['seq', 'inv', 'par']
    )

    parser.add_argument(
        '--init_from',
        type=str,
        default='scratch',
        choices=['scratch', 'resume']
    )

    args = parser.parse_args()

    class Config:
        block_type = args.block_type
        init_from = args.init_from

        base_dir = os.path.dirname(os.path.abspath(__file__))
        out_dir = os.path.join(base_dir, f'out_{block_type}')
        dataset = 'c4'
        data_dir = os.path.join(base_dir, 'data', dataset)
        checkpoint_path = os.path.join(out_dir, 'ckpt.pt')

        eval_interval = 250
        log_interval = 1
        eval_iters = 200
        eval_only = False
        always_save_checkpoint = True

        gradient_accumulation_steps = 5 * 8
        batch_size = 12
        block_size = 1024

        n_layer = 12
        n_head = 12
        n_embd = 768
        dropout = 0.0
        bias = False

        learning_rate = 6e-4
        min_lr = 6e-5

        max_iters = 6100
        lr_decay_iters = 6100
        warmup_iters = 200

        weight_decay = 1e-1
        beta1 = 0.9
        beta2 = 0.95
        grad_clip = 1.0
        decay_lr = True

        backend = 'nccl'
        device = 'cuda'
        dtype = (
            'bfloat16'
            if torch.cuda.is_available()
            and torch.cuda.is_bf16_supported()
            else 'float16'
        )
        compile = True

        iter_num = 0
        best_val_loss = 1e9

        resume_rank_state = None

    cfg = Config()

    ddp = int(os.environ.get('RANK', -1)) != -1
    cfg.ddp = ddp

    if ddp:
        init_process_group(backend=cfg.backend)

        ddp_rank = int(os.environ['RANK'])
        ddp_local_rank = int(os.environ['LOCAL_RANK'])
        ddp_world_size = int(os.environ['WORLD_SIZE'])

        device = f'cuda:{ddp_local_rank}'
        torch.cuda.set_device(device)

        master_process = ddp_rank == 0
        seed_offset = ddp_rank

        assert cfg.gradient_accumulation_steps % ddp_world_size == 0
        cfg.gradient_accumulation_steps //= ddp_world_size

    else:
        ddp_rank = 0
        ddp_local_rank = 0
        master_process = True
        seed_offset = 0
        ddp_world_size = 1
        device = cfg.device

    tokens_per_iter = (
        cfg.gradient_accumulation_steps
        * ddp_world_size
        * cfg.batch_size
        * cfg.block_size
    )

    if master_process:
        print("==== Iniciando Treinamento ====")
        print(f"Variante arquitetural: {cfg.block_type.upper()}")
        print(f"Tokens por iteração: {tokens_per_iter:,}")
        os.makedirs(cfg.out_dir, exist_ok=True)

    seed = 1337 + seed_offset
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    device_type = 'cuda' if 'cuda' in str(device) else 'cpu'

    ptdtype = {
        'float32': torch.float32,
        'bfloat16': torch.bfloat16,
        'float16': torch.float16
    }[cfg.dtype]

    ctx = (
        nullcontext()
        if device_type == 'cpu'
        else torch.amp.autocast(
            device_type=device_type,
            dtype=ptdtype
        )
    )

    model_args = dict(
        n_layer=cfg.n_layer,
        n_head=cfg.n_head,
        n_embd=cfg.n_embd,
        block_size=cfg.block_size,
        bias=cfg.bias,
        vocab_size=50304,
        dropout=cfg.dropout,
        block_type=cfg.block_type
    )

    cfg.model_args = model_args

    checkpoint = None

    if cfg.init_from == 'resume':
        if not os.path.exists(cfg.checkpoint_path):
            raise FileNotFoundError(
                f"Checkpoint não encontrado: {cfg.checkpoint_path}"
            )

        checkpoint = torch.load(
            cfg.checkpoint_path,
            map_location=device,
            weights_only=False
        )

        checkpoint_model_args = checkpoint['model_args']

        if checkpoint_model_args.get('block_type') != cfg.block_type:
            raise ValueError(
                "O block_type do checkpoint não corresponde "
                "à variante solicitada."
            )

        model_args = checkpoint_model_args
        cfg.model_args = model_args

        cfg.iter_num = checkpoint['iter_num']
        cfg.best_val_loss = checkpoint['best_val_loss']

        print(
            f"Retomando treinamento a partir da iteração "
            f"{cfg.iter_num}."
        ) if master_process else None

    gptconf = GPTConfig(**model_args)
    model = GPT(gptconf)

    if checkpoint is not None:
        state_dict = checkpoint['model']

        unwanted_prefix = '_orig_mod.'
        state_dict = {
            key[len(unwanted_prefix):] if key.startswith(unwanted_prefix) else key: value
            for key, value in state_dict.items()
        }

        model.load_state_dict(state_dict)

    model.to(device)

    num_params = sum(p.numel() for p in model.parameters())

    planned_train_tokens = cfg.max_iters * tokens_per_iter
    tokens_per_parameter = planned_train_tokens / num_params

    if master_process:
        print(f"Parâmetros do modelo: {num_params:,}")
        print(f"Tokens planejados: {planned_train_tokens:,}")
        print(f"Tokens por parâmetro: {tokens_per_parameter:.2f}")

    scaler = torch.cuda.amp.GradScaler(
        enabled=(cfg.dtype == 'float16')
    )

    optimizer = model.configure_optimizers(
        cfg.weight_decay,
        cfg.learning_rate,
        (cfg.beta1, cfg.beta2),
        device_type
    )

    if checkpoint is not None:
        optimizer.load_state_dict(checkpoint['optimizer'])

        if 'scaler' in checkpoint:
            scaler.load_state_dict(checkpoint['scaler'])

    if cfg.compile:
        if master_process:
            print("Compilando o modelo...")
        model = torch.compile(model)

    if ddp:
        model = DDP(model, device_ids=[ddp_local_rank])

    raw_model = model.module if ddp else model

    if cfg.compile:
        raw_model = raw_model._orig_mod

    if checkpoint is not None:
        if 'rank_states' in checkpoint:
            rank_states = checkpoint['rank_states']

            if ddp:
                if ddp_rank >= len(rank_states):
                    raise RuntimeError(
                        "O checkpoint não contém o estado deste rank."
                    )

                cfg.resume_rank_state = rank_states[ddp_rank]

            else:
                if len(rank_states) != 1:
                    raise RuntimeError(
                        "O checkpoint foi salvo com uma configuração "
                        "DDP diferente da execução atual."
                    )

                cfg.resume_rank_state = rank_states[0]

            set_rng_state(cfg.resume_rank_state['rng'])

        else:
            print(
                "Aviso: checkpoint antigo sem estados aleatórios. "
                "A retomada não reproduzirá a sequência aleatória."
            ) if master_process else None

    train_loop(
        model,
        optimizer,
        scaler,
        ctx,
        cfg,
        master_process,
        device,
        device_type,
        raw_model
    )

    if ddp:
        destroy_process_group()


if __name__ == "__main__":
    main()
