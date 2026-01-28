import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp.wrap import size_based_auto_wrap_policy

from model import OptiMultiVideo, VisionEncoder, TextDecoder

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()

def train_func(rank, world_size):
    setup(rank, world_size)
    
    # 1. Create Model
    # Move to device strictly before wrapping for FSDP (usually) 
    # but FSDP handles CPU init too. 
    # Let's Init on CPU then move to Rank
    torch.cuda.set_device(rank)
    
    vision = VisionEncoder(hidden_dim=768).to(rank)
    text = TextDecoder(hidden_dim=768).to(rank)
    model = OptiMultiVideo(vision, text, hidden_dim=768).to(rank)
    
    # 2. Wrap FSDP
    # Standard AutoWrap Policy
    my_auto_wrap_policy = size_based_auto_wrap_policy
    model = FSDP(model, auto_wrap_policy=my_auto_wrap_policy)
    
    optimizer = optim.AdamW(model.parameters(), lr=1e-4)
    criterion = nn.CrossEntropyLoss()
    
    # 3. Dummy Data Loop
    batch_size = 16 
    seq_len = 32
    num_steps = 20
    
    print(f"[Rank {rank}] Addr: {model}")
    
    # Warmup
    for _ in range(5):
        inputs = torch.randn(batch_size, seq_len, 3, 224, 224).to(rank)
        prompts = torch.randint(0, 1000, (batch_size, seq_len)).to(rank)
        
        optimizer.zero_grad()
        output = model(inputs, prompts)
        loss = criterion(output.view(-1, 1000), prompts.view(-1))
        loss.backward()
        optimizer.step()
        
    dist.barrier()
    
    # Timing
    start_time = time.time()
    
    for step in range(num_steps):
        inputs = torch.randn(batch_size, seq_len, 3, 224, 224).to(rank)
        prompts = torch.randint(0, 1000, (batch_size, seq_len)).to(rank)
        
        optimizer.zero_grad()
        
        output = model(inputs, prompts)
        loss = criterion(output.view(-1, 1000), prompts.view(-1))
        
        loss.backward()
        optimizer.step()
        
        if rank == 0 and step % 5 == 0:
            print(f"Step {step}, Loss: {loss.item():.4f}")
            
    torch.cuda.synchronize()
    end_time = time.time()
    
    if rank == 0:
        avg_time = (end_time - start_time) / num_steps
        print(f"Training Finished. Avg Step Time: {avg_time:.4f} s")
        print(f"Throughput: {batch_size * world_size / avg_time:.2f} samples/s")

    cleanup()

def main():
    world_size = torch.cuda.device_count()
    if world_size == 0:
        print("No CUDA devices found. Running in CPU mode (Mock) or exiting.")
        # FSDP doesn't support CPU well in this config logic, 
        # but for safety in non-GPU envs:
        return
        
    print(f"Spawning {world_size} processes for FSDP training...")
    mp.spawn(train_func,
             args=(world_size,),
             nprocs=world_size,
             join=True)

if __name__ == "__main__":
    main()
