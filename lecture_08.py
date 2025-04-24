import torch
import time
import os
from typing import List, Callable
import torch.nn.functional as F
import torch.distributed as dist
import torch.distributed.fsdp
from execute_util import text, image, link, system_text
from torch_util import get_device
from lecture_util import article_link
from lecture_08_utils import spawn, int_divide, summarize_tensor, get_init_params

def main():
    text("Last week: parallelism within a single GPU")
    text("This week: parallelism across multiple GPUs")
    image("images/gpu-node-overview.png", width=600)

    text("In both cases, **compute** (arithmetic logic units) is far from inputs/outputs (**data**).")
    text("Unifying theme: orchestrate computation to avoid data transfer bottlenecks")

    text("Last week: reduce memory accesses via fusion/tiling")
    text("This week: reduce communication across GPUs/nodes via replication/sharding")

    text("Generalized hierarchy (from small/fast to big/slow):")
    text("- Single node, single GPU: L1 cache / shared memory")
    text("- Single node, single GPU: HBM")
    text("- Single node, multi-GPU: NVLink")
    text("- Multi-node, multi-GPU: NVSwitch")

    text("## Part 1: building blocks of distributed communication/computation")
    hardware()                 # How nodes actually communicate
    collective_operations()    # Conceptual programming interface
    torch_distributed()        # How this is implemented in NCCL/PyTorch
    benchmarking()             # Estimate NCCL bandwidth

    text("## Part 2: distributed training")
    text("Walk through bare-bones implementations of each strategy on deep MLPs.")
    text("Recall that MLPs are the compute bottleneck in Transformers.")
    data_parallelism()         # Cut up along the batch dimension
    tensor_parallelism()       # Cut up along the width dimension
    pipeline_parallelism()     # Cut up along the depth dimension

    text("## Summary")

    text("The game: trading off")
    text("- memory usage (store locally) and")
    text("- communication (send across GPUs)")

    text("- Hardware is getting faster, but will always have this hierarchical structure")
    text("- Many ways to parallelize: data, tensor/expert, pipeline, sequence")


def hardware():
    text("## Single GPU")
    image("https://miro.medium.com/v2/resize:fit:2000/format:webp/1*6xoBKi5kL2dZpivFe1-zgw.jpeg", width=800)
    text("Memory bandwidth for HBM for H100 NVL is 3.9 TB/s "), article_link("https://resources.nvidia.com/en-us-tensor-core/nvidia-tensor-core-gpu-datasheet")

    text("## Multi-node, multi-GPU")
    text("Traditionally:")
    image("https://media.springernature.com/lw685/springer-static/image/art%3A10.1186%2Fs42774-021-00098-3/MediaObjects/42774_2021_98_Fig1_HTML.png?as=webp", width=400)
    text("- GPUs on same node communicate via a PCI(e) bus (v7.0, 16 lanes => 242 GB/s) "), article_link("https://en.wikipedia.org/wiki/PCI_Express")
    text("- GPUs on different nodes communicate via Ethernet (~200 MB/sec)")

    text("Both are too slow...")
    text("Key hardware advance: have GPUs connect **directly**, bypassing CPU")

    text("## InfiniBand")
    text("Standard developed in 1999; Mellanox created InfiniBand hardware, acquired by NVIDIA in 2019")
    text("Idea: Remote Direct Memory Access (RDMA) to connect nodes directly")
    image("https://lambdalabs.com/hubfs/Imported_Blog_Media/nvlink-diagram-update.png", width=600)

    text("## NVLink/NVSwitch")
    text("NVIDIA developed proprietary protocol since 2014")
    text("4.5x more bandwidth than InfiniBand "), article_link("https://developer.nvidia.com/blog/upgrading-multi-gpu-interconnectivity-with-the-third-generation-nvidia-nvswitch/")

    text("Within a node: NVLink connects GPUs directly, bypass CPU")
    image("https://developer-blogs.nvidia.com/wp-content/uploads/2022/08/NVLink-generations-1.png", width=800)

    text("Across nodes: NVSwitch connects GPUs directly, bypass Ethernet")
    image("https://developer-blogs.nvidia.com/wp-content/uploads/2022/08/NVLink-all-to-all-connectivity-1.png", width=800)

    text("H100: 18 NVLink 4.0 links => 900GB/sec")

    text("Let's check what our hardware setup is. "), article_link("https://guide.ncloud-docs.com/docs/en/server-baremetal-a100-check-vpc")

    if torch.cuda.is_available():
        system_text(["nvidia-smi", "topo", "-m"])
        text("Note GPUs are connected via NV18, also connected to NICs (for PCIe)")


def collective_operations():
    text("Collective operations are the conceptual primitives used for distributed programming "), article_link("https://en.wikipedia.org/wiki/Collective_operation")

    text("- Collective means that specify communication pattern across many (e.g., 256) nodes")
    text("- These are classic in the parallel programming literature from the 1980s")
    text("- For SIMD (Single Instruction, Multiple Data) parallelism")
    text("- Better/faster abstraction than managing point-to-point communication yourself")

    text("Terminology:")
    text("- **Rank**: a device (e.g., GPU)")
    text("- **World size**: number of devices")

    text("## Broadcast"), image("https://pytorch.org/tutorials/_images/broadcast.png", width=400)

    text("## Scatter"), image("https://pytorch.org/tutorials/_images/scatter.png", width=400)

    text("## Gather"), image("https://pytorch.org/tutorials/_images/gather.png", width=400)

    text("## Reduce"), image("https://pytorch.org/tutorials/_images/reduce.png", width=400)

    text("## All-gather"), image("https://pytorch.org/tutorials/_images/all_gather.png", width=400)

    text("## Reduce-scatter"), image("https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/_images/reducescatter.png", width=400)

    text("## All-reduce = reduce-scatter + all-gather"), image("https://pytorch.org/tutorials/_images/all_reduce.png", width=400)

    text("Way to remember:")
    text("- Reduce: performs some associative/commutative operation (sum, min, max)")
    text("- Broadcast/scatter is inverse of gather")
    text("- All: means destination is all devices")


def torch_distributed():
    text("## PyTorch distributed library (`torch.distributed`) ")
    link("[Documentation]", url="https://pytorch.org/docs/stable/distributed.html")

    text("- Provides clean interface for collective operations (e.g., `all_reduce`)")
    text("- Backends: gloo (CPU), nccl (GPU)")
    text("- Also supports higher-level abstractions (e.g., `FullyShardedDataParallel`)")

    text("## NVIDIA Collective Communication Library (NCCL)")

    text("NCCL translates collective operations into low-level packets that are sent between GPUs. "), link(title="[talk]", url="https://www.nvidia.com/en-us/on-demand/session/gtcspring21-s31880/")
    text("- Detect toplogy of hardware (e.g., number of nodes, switches, NVLink/PCIe)")
    text("- Optimize the path between ranks; ring (good bandwidth), tree (good latency)")
    image("https://developer-blogs.nvidia.com/wp-content/uploads/2019/02/DBtree.png", width=400)
    text("- Launches CUDA kernels to send/receive data")

    # Walk through examples
    link(title="[stdout]", url="var/lecture_08_stdout.txt")
    spawn(collective_operations_main, world_size=4)


def collective_operations_main(rank: int, world_size: int):
    """Try out some collective operations."""
    # Note: this function is running asynchronously for each process (world_size)

    setup(rank, world_size)

    # All-reduce
    dist.barrier()  # Waits for all processes to get to this point

    tensor = torch.tensor([0., 1, 2, 3], device=get_device(rank)) + rank  # Both input and output

    print(f"Rank {rank} [before all-reduce]: {tensor}")
    dist.all_reduce(tensor=tensor, op=dist.ReduceOp.SUM, async_op=False)  # Modifies tensor in place
    print(f"Rank {rank} [after all-reduce]: {tensor}")

    # Reduce-scatter
    dist.barrier()

    input = torch.arange(world_size, dtype=torch.float32, device=get_device(rank)) + rank  # Input
    output = torch.empty(1, device=get_device(rank))  # Allocate output

    print(f"Rank {rank} [before reduce-scatter]: input = {input}, output = {output}")
    dist.reduce_scatter_tensor(output=output, input=input, op=dist.ReduceOp.SUM, async_op=False)
    print(f"Rank {rank} [after reduce-scatter]: input = {input}, output = {output}")

    # All-gather
    dist.barrier()

    input = output  # Input is the output of reduce-scatter
    output = torch.empty(world_size, device=get_device(rank))  # Allocate output

    print(f"Rank {rank} [before all-gather]: input = {input}, output = {output}")
    dist.all_gather_into_tensor(output_tensor=output, input_tensor=input, async_op=False)
    print(f"Rank {rank} [after all-gather]: input = {input}, output = {output}")

    text("Recall that all-reduce = reduce-scatter + all-gather!")

    cleanup()


def benchmarking():
    text("## Benchmarking"), link("https://github.com/stas00/ml-engineering/blob/master/network/benchmarks/all_reduce_bench.py")

    text("Let's see how fast commmunication happens (will restrict to just one node).")

    text("### All-reduce")
    spawn(all_reduce, world_size=2, num_elements=1024**2)
    spawn(all_reduce, world_size=4, num_elements=1024**2)

    text("### Reduce-scatter")
    spawn(reduce_scatter, world_size=2, num_elements=1024**2)
    spawn(reduce_scatter, world_size=4, num_elements=1024**2)

    link(title="Reference on reasoning about operations", url="https://github.com/NVIDIA/nccl-tests/blob/master/doc/PERFORMANCE.md#allreduce")


def all_reduce(rank: int, world_size: int, num_elements: int):
    setup(rank, world_size)

    # Create tensor
    tensor = torch.randn(num_elements, device=get_device(rank))

    # Warmup
    dist.all_reduce(tensor=tensor, op=dist.ReduceOp.SUM, async_op=False)
    if torch.cuda.is_available():
        torch.cuda.synchronize()  # Wait for CUDA kerels to finish
        dist.barrier()            # Wait for all the processes to get here

    # All reduce
    start_time = time.time()
    dist.all_reduce(tensor=tensor, op=dist.ReduceOp.SUM, async_op=False)
    if torch.cuda.is_available():
        torch.cuda.synchronize()  # Wait for CUDA kerels to finish
        dist.barrier()            # Wait for all the processes to get here
    end_time = time.time()

    duration = end_time - start_time
    print(f"[all_reduce] Rank {rank}: all_reduce(world_size={world_size}, num_elements={num_elements}) took {round(duration * 1000)} ms")

    # Estimate the bandwidth
    size_bytes = tensor.element_size() * tensor.numel()
    sent_bytes = size_bytes * 2 * (world_size - 1)  # 2x because of send and receive
    total_duration = world_size * duration
    bandwidth = sent_bytes / total_duration
    print(f"[all_reduce] Rank {rank}: all_reduce estimated bandwidth = {round(bandwidth / 1024**3)} GB/sec")

    cleanup()


def reduce_scatter(rank: int, world_size: int, num_elements: int):
    setup(rank, world_size)

    # Create tensor
    input = torch.randn(world_size, num_elements, device=get_device(rank))
    output = torch.empty(num_elements, device=get_device(rank))

    # Warmup
    dist.reduce_scatter_tensor(output=output, input=input, op=dist.ReduceOp.SUM, async_op=False)
    if torch.cuda.is_available():
        torch.cuda.synchronize()  # Wait for CUDA kerels to finish
        dist.barrier()            # Wait for all the processes to get here

    # All reduce
    start_time = time.time()
    dist.reduce_scatter_tensor(output=output, input=input, op=dist.ReduceOp.SUM, async_op=False)
    if torch.cuda.is_available():
        torch.cuda.synchronize()  # Wait for CUDA kerels to finish
        dist.barrier()            # Wait for all the processes to get here
    end_time = time.time()

    duration = end_time - start_time
    print(f"[reduce_scatter] Rank {rank}: reduce_scatter(world_size={world_size}, num_elements={num_elements}) took {round(duration * 1000)} ms")

    # Estimate the bandwidth
    data_bytes = output.element_size() * output.numel()  # How much data in the output
    sent_bytes = data_bytes * (world_size - 1)  # How much needs to be sent
    total_duration = world_size * duration  # Total time for transmission
    bandwidth = sent_bytes / total_duration
    print(f"[reduce_scatter] Rank {rank}: reduce_scatter estimated bandwidth = {round(bandwidth / 1024**3)} GB/sec")

    cleanup()


def data_parallelism():
    text("## Distributed Data Parallel (DDP)")
    image("images/data-parallelism.png", width=300)
    text("Sharding strategy: each rank gets a slice of the data")

    # Generate data
    batch_size = 128
    num_dim = 1024
    data = torch.randn(batch_size, num_dim)

    spawn(data_parallelism_main, world_size=4, data=data, num_layers=4, num_steps=1)

    text("Notes:")
    text("- Losses are different across nodes (computed on local data)")
    text("- Gradients are same, and therefore parameters are the same")


def data_parallelism_main(rank: int, world_size: int, data: torch.Tensor, num_layers: int, num_steps: int):
    setup(rank, world_size)

    # Get the slice of data for this rank
    batch_size = data.size(0) // world_size  # @inspect batch_size
    num_dim = data.size(1)  # @inspect num_dim
    start_index = rank * batch_size  # @inspect start_index
    end_index = start_index + batch_size  # @inspect end_index
    data = data[start_index:end_index].to(get_device(rank))

    # Create MLP: # gelu(gelu(x @ params[0]) @ params[1]) ...
    params = [get_init_params(num_dim, num_dim, rank) for i in range(num_layers)]
    optimizer = torch.optim.AdamW(params, lr=1e-3)

    for step in range(num_steps):
        # Forward pass
        x = data
        for param in params:
            x = x @ param
            x = F.gelu(x)
        loss = x.square().mean()  # Loss function is average squared magnitude

        # Backward pass
        loss.backward()

        # Sync gradients across workers (NEW!)
        if torch.cuda.is_available():
            for param in params:
                dist.all_reduce(tensor=param.grad, op=dist.ReduceOp.AVG, async_op=False)

        # Update parameters
        optimizer.step()

        print(f"[ddp] Rank {rank}: step = {step}, loss = {loss.item()}, params = {[summarize_tensor(params[i]) for i in range(num_layers)]}")

    cleanup()


def tensor_parallelism():
    image("images/tensor-parallelism.png", width=300)
    text("Sharding strategy: each rank gets part of each layer, transfer all data/activations")

    # Create data
    batch_size = 128
    num_dim = 1024
    data = torch.randn(batch_size, num_dim)

    spawn(tensor_parallelism_main, world_size=4, data=data, num_layers=4)


def tensor_parallelism_main(rank: int, world_size: int, data: torch.Tensor, num_layers: int):
    setup(rank, world_size)

    # Note: no sharding of the data
    data = data.to(get_device(rank))
    batch_size = data.size(0)  # @inspect batch_size
    num_dim = data.size(1)  # @inspect num_dim
    sharded_num_dim = num_dim // world_size  # Shard `num_dim`  @inspect sharded_num_dim

    # Create model (each rank gets 1/world_size of the parameters)
    params = [get_init_params(num_dim, sharded_num_dim, rank) for i in range(num_layers)]

    # Forward pass
    x = data
    for i in range(num_layers):
        # Compute activations (batch_size x sharded_num_dim)
        x = x @ params[i]  # Note: this is only on a slice of the parameters
        x = F.gelu(x)

        # Allocate memory for activations (world_size x batch_size x sharded_num_dim)
        activations = [torch.empty(batch_size, sharded_num_dim, device=get_device(rank)) for _ in range(world_size)]

        # Send via all gather
        dist.all_gather(tensor_list=activations, tensor=x, async_op=False)

        # Just concatenate them to get (batch_size x num_dim)
        x = torch.cat(activations, dim=1)

    print(f"Rank {rank}: forward pass produced activations {summarize_tensor(x)}")

    # Backward pass: homework exercise

    cleanup()


def pipeline_parallelism():
    image("images/pipeline-parallelism.png", width=300)
    text("Sharding strategy: each rank gets subset of layers, transfer all data/activations")

    # Create data
    batch_size = 128
    num_dim = 1024
    data = torch.randn(batch_size, num_dim)

    spawn(pipeline_parallelism_main, world_size=2, data=data, num_layers=4, num_micro_batches=4)


def pipeline_parallelism_main(rank: int, world_size: int, data: torch.Tensor, num_layers: int, num_micro_batches: int):
    setup(rank, world_size)

    # All the data
    data = data.to(get_device(rank))
    batch_size = data.size(0)  # @inspect batch_size
    num_dim = data.size(1)  # @inspect num_dim

    # Split up layers
    num_layers_per_rank = int_divide(num_layers, world_size)  # @inspect num_layers_per_rank
    micro_batch_size = int_divide(batch_size, num_micro_batches)  # @inspect micro_batch_size

    # Each rank gets a subset of layers
    params = [get_init_params(num_dim, num_dim, rank) for i in range(num_layers_per_rank)]

    # Forward pass

    # Break up into micro batches to minimize the bubble
    if rank == 0:
        micro_batches = data.chunk(chunks=num_micro_batches, dim=0)  # The data
    else:
        micro_batches = [torch.empty(micro_batch_size, num_dim, device=get_device(rank)) for _ in range(num_micro_batches)]

    for x in micro_batches:
        # Get from previous rank
        if rank - 1 >= 0:
            dist.recv(tensor=x, src=rank - 1)

        # Do the compute
        for param in params:
            x = x @ param
            x = F.gelu(x)

        print(f"[pipeline] Rank {rank}: forward pass produced {summarize_tensor(x)}")

        # Send to the next rank
        if rank + 1 < world_size:
            dist.send(tensor=x, dst=rank + 1)

    text("Not handled: overlapping communication/computation to eliminate pipeline bubbles")

    cleanup()

############################################################

def setup(rank: int, world_size: int):
    # This is where master lives (rank 0)
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "15623"

    if torch.cuda.is_available():
        dist.init_process_group("nccl", rank=rank, world_size=world_size)
    else:
        dist.init_process_group("gloo", rank=rank, world_size=world_size)


def cleanup():
    torch.distributed.destroy_process_group()


def benchmark(description: str, run: Callable, num_warmups: int = 1, num_trials: int = 3):
    """Benchmark `func` by running it `num_trials`, and return all the times."""
    # Warmup: first times might be slower due to compilation, things not cached.
    # Since we will run the kernel multiple times, the timing that matters is steady state.
    for _ in range(num_warmups):
        run()
    torch.cuda.synchronize()  # Wait for CUDA threads to finish (important!)

    # Time it for real now!
    times: List[float] = []
    for trial in range(num_trials):  # Do it multiple times to capture variance
        start_time = time.time()

        run()  # Actually perform computation
        torch.cuda.synchronize()  # Wait for CUDA threads to finish (important!)

        end_time = time.time()
        times.append((end_time - start_time) * 1000)

    mean_time = mean(times)
    text(f"{description}: {list(map(round1, sorted(times)))} (mean {round1(mean_time)} ms)", pop_stack=True)


if __name__ == "__main__":
    main()
