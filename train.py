import os
import time
import math
import argparse
from contextlib import nullcontext

import numpy as np
import torch
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
            # Tempo total dividido pelo número de iterações = Latência média por batch (em milissegundos)
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

def train_loop(model, optimizer, scaler, ctx, cfg, master_process, device, device_type, raw_model):
    X, Y = get_batch('train', cfg.data_dir, cfg.batch_size, cfg.block_size, device, device_type)
    t0 = time.time()
    local_iter_num = 0
    iter_num = cfg.iter_num
    best_val_loss = cfg.best_val_loss
    running_mfu = -1.0

    metrics_file = os.path.join(cfg.out_dir, f'metrics_{cfg.block_type}.csv')
    if master_process and not os.path.exists(metrics_file):
        with open(metrics_file, 'w') as f:
            f.write("iter,train_loss,val_loss,train_ppl,val_ppl,step_time_ms,val_latency_ms\n")

    dt = 0.0 
    
    while True:
        lr = get_lr(iter_num, cfg.warmup_iters, cfg.lr_decay_iters, cfg.learning_rate, cfg.min_lr) if cfg.decay_lr else cfg.learning_rate
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        if iter_num % cfg.eval_interval == 0 and master_process:
            losses, val_latency_ms = estimate_loss(model, cfg.eval_iters, ctx, cfg.data_dir, cfg.batch_size, cfg.block_size, device, device_type)
            
            train_ppl = math.exp(losses['train'])
            val_ppl = math.exp(losses['val'])
            step_time_ms = dt * 1000 # Tempo gasto no último batch de treino
            
            print(f"step {iter_num}: val loss {losses['val']:.4f} | val ppl {val_ppl:.2f} | latência inf. {val_latency_ms:.2f}ms")
            
            with open(metrics_file, 'a') as f:
                f.write(f"{iter_num},{losses['train']:.4f},{losses['val']:.4f},{train_ppl:.2f},{val_ppl:.2f},{step_time_ms:.2f},{val_latency_ms:.2f}\n")

            if losses['val'] < best_val_loss or cfg.always_save_checkpoint:
                best_val_loss = losses['val']
                if iter_num > 0:
                    checkpoint = {
                        'model': raw_model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'model_args': cfg.model_args,
                        'iter_num': iter_num,
                        'best_val_loss': best_val_loss,
                    }
                    print(f"Salvo checkpoint em {cfg.out_dir}")
                    torch.save(checkpoint, os.path.join(cfg.out_dir, 'ckpt.pt'))
        
        if iter_num == 0 and cfg.eval_only:
            break

        for micro_step in range(cfg.gradient_accumulation_steps):
            if cfg.ddp:
                model.require_backward_grad_sync = (micro_step == cfg.gradient_accumulation_steps - 1)
            with ctx:
                logits, loss = model(X, Y)
                loss = loss / cfg.gradient_accumulation_steps
            
            X, Y = get_batch('train', cfg.data_dir, cfg.batch_size, cfg.block_size, device, device_type)
            scaler.scale(loss).backward()
        
        if cfg.grad_clip != 0.0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)

        t1 = time.time()
        dt = t1 - t0
        t0 = t1
        if iter_num % cfg.log_interval == 0 and master_process:
            lossf = loss.item() * cfg.gradient_accumulation_steps
            if local_iter_num >= 5: 
                mfu = raw_model.estimate_mfu(cfg.batch_size * cfg.gradient_accumulation_steps, dt)
                running_mfu = mfu if running_mfu == -1.0 else 0.9*running_mfu + 0.1*mfu
            print(f"iter {iter_num}: loss {lossf:.4f}, time {dt*1000:.2f}ms, mfu {running_mfu*100:.2f}%")
        
        iter_num += 1
        local_iter_num += 1

        if iter_num > cfg.max_iters:
            break

def main():
    parser = argparse.ArgumentParser(description="Treinamento nanoGPT Refatorado")
    parser.add_argument('--block_type', type=str, default='seq', choices=['seq', 'inv', 'par'])
    args = parser.parse_args()

    class Config:
        block_type = args.block_type
        out_dir = f'out_{block_type}'
        dataset = 'c4'
        data_dir = os.path.join('data', dataset)
        
        eval_interval = 250
        log_interval = 1
        eval_iters = 200
        eval_only = False
        always_save_checkpoint = True
        init_from = 'scratch'
        
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
        dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16'
        compile = True
        
        iter_num = 0
        best_val_loss = 1e9

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
        master_process = True
        seed_offset = 0
        ddp_world_size = 1
        device = cfg.device

    tokens_per_iter = cfg.gradient_accumulation_steps * ddp_world_size * cfg.batch_size * cfg.block_size
    if master_process:
        print(f"==== Iniciando Treinamento ====")
        print(f"Variante Arquitetural: {cfg.variant.upper()}")
        print(f"Tokens por iteração: {tokens_per_iter:,}")
        os.makedirs(cfg.out_dir, exist_ok=True)
        
    torch.manual_seed(1337 + seed_offset)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    device_type = 'cuda' if 'cuda' in device else 'cpu'
    
    ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[cfg.dtype]
    ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

    model_args = dict(n_layer=cfg.n_layer, n_head=cfg.n_head, n_embd=cfg.n_embd, block_size=cfg.block_size, bias=cfg.bias, vocab_size=50304, dropout=cfg.dropout, block_type=cfg.block_type)
    cfg.model_args = model_args
    
    if cfg.init_from == 'scratch':
        if master_process: print("Inicializando novo modelo do zero (scratch)...")
        gptconf = GPTConfig(**model_args)
        model = GPT(gptconf)
        
    model.to(device)
    scaler = torch.cuda.amp.GradScaler(enabled=(cfg.dtype == 'float16'))
    optimizer = model.configure_optimizers(cfg.weight_decay, cfg.learning_rate, (cfg.beta1, cfg.beta2), device_type)
    
    if cfg.compile:
        if master_process: print("Compilando o modelo...")
        model = torch.compile(model)
        
    if ddp:
        model = DDP(model, device_ids=[ddp_local_rank])
        
    raw_model = model.module if ddp else model

    train_loop(model, optimizer, scaler, ctx, cfg, master_process, device, device_type, raw_model)

    if ddp:
        destroy_process_group()

if __name__ == "__main__":
    main()
