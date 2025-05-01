from sympy import symbols, oo
from execute_util import text, link, image
from lecture_util import article_link
from references import Reference, llama3, gqa, mla, longformer, sparse_transformer, mistral_7b

# Define symbols corresponding to the shape of the Transformer model
B, S, T, D, F, N, K, H, L, V = symbols("B S T D F N K H L V", positive=True)
c = symbols("c", positive=True)  # Just a constant that helps with taking limits
memory_bandwidth = symbols("memory_bandwidth", positive=True)

scaling_book_transformers = Reference(title="[Scaling book chapter on Transformers]", url="https://jax-ml.github.io/scaling-book/transformers/")
scaling_book_inference = Reference(title="[Scaling book chapter on Transformers]", url="https://jax-ml.github.io/scaling-book/inference/")

def main():
    text("**Inference**: given a **fixed model**, generate responses given prompts")

    text("### Understanding the inference workload")
    motivation()
    review_transformer()
    review_of_arithmetic_intensity()
    arithmetic_intensity_of_inference()
    throughput_and_latency()

    text("### Taking shortcuts (lossy)")
    reduce_kv_cache_size()
    alternatives_to_the_transformer()
    quantization()
    model_pruning()

    text("Game: reduce inference complexity without hurting accuracy")

    text("From scratch recipe:")
    text("1. Define faster model architecture")
    text("2. Train faster model")

    text("Distillation recipe:")
    text("1. Define faster model architecture")
    text("2. Initialize weights using original model (different architecture)")
    text("3. Repair faster model (distillation)")

    text("### Use shortcuts but double check (lossless)")
    speculative_sampling()

    text("### Handling dynamic workloads")
    text("Batching over sequences in live traffic is tricky because:")
    text("1. Requests arrive at different times (waiting for batch is bad for early requests)")
    text("2. Sequences have shared prefixes (e.g., system prompts, generating multiple samples)")
    text("3. Sequences have different lengths (padding is inefficient)")

    continuous_batching()
    paged_attention()

    text("### Summary")
    text("- Inference is important (actual use, evaluation, reinforcement learning)")
    text("- Different characteristics compared to training (memory-bound, dynamic)")
    text("- Techniques: new architectures, quantization, pruning/distillation, speculative decoding")
    text("- Ideas from systems (speculative execution, paging)")
    text("- New architectures have huge potential for improvement")


def motivation():
    text("Inference shows up in many places:")
    text("- Actual use (chatbots, code completion, batch data processing)")
    text("- Model evaluation (e.g., on instruction following)")
    text("- Test-time compute (thinking requires more inference)")
    text("- Training via reinforcement learning (sample generation, then score)")

    text("Why **efficiency** matters: training is one-time cost, inference is repeated many times")
    image("images/openai-100b-tokens.png", width=400); link(title=" [tweet]", url="https://x.com/sama/status/1756089361609981993")
    image("images/cursor-1b-lines.png", width=400); link(title=" [tweet]", url="https://x.com/amanrsanger/status/1916968123535880684")

    text("Metrics:")
    text("- Time-to-first-token (TTFT): matters for interactive applications")
    text("- Latency: matters for interactive applications")
    text("- Throughput (tokens per second): useful for batch processing applications")

    text("Key considerations in efficiency:")
    text("- Training (supervised): you see all tokens, can parallelize over sequence (matmul in Transformer)")
    text("- Inference: you have to generate sequentially, can't parallelize, so harder to saturate memory")

    text("Companies doing inference (big deal for anyone who has a product or platform):")
    text("- Providers serving closed models (OpenAI, Anthropic, Google, etc.)")
    text("- Providers serving open-weight models (Together, Fireworks, DeepInfra, etc.)")

    text("Open-source packages:")
    text("- vLLM (Berkeley) "), link(title="[talk]", url="https://www.youtube.com/watch?v=8BaEwoTk8XI")
    text("- Tensor-RT (NVIDIA) "), article_link("https://nvidia.github.io/TensorRT-LLM/overview.html")
    text("- TGI (Hugging Face) "), article_link("https://huggingface.co/docs/text-generation-inference/en/index")


def review_transformer():
    link(scaling_book_transformers)
    image("https://jax-ml.github.io/scaling-book/assets/img/transformer-diagram.png", width=600)
    text("Simplifications (following conventions): F = 4*D, D = N*H, N = K*G, S = T")
    text("FLOPs for a feedforward pass: 6 * (B*T) * (num_params + O(T))")


def review_of_arithmetic_intensity():
    text("Setup: multiply X (B x D) and W (D x F) matrix")
    text("Intuition: B is batch size, D is hidden dimension, F is up-projection dimension in MLP")

    text("Let's do FLOPs and memory read/write accounting for the matrix multiplication (X * W).")
    flops = 0
    bytes_transferred = 0

    text("Steps:")
    text("1. Read X (B x D) from HBM")
    bytes_transferred += 2 * B * D
    text("2. Read W (D x F) from HBM")
    bytes_transferred += 2 * D * F
    text("3. Compute Y = X (B x D) * W (D x F)")
    flops += 2 * B * D * F
    text("4. Write Y (B x F) to HBM")
    bytes_transferred += 2 * B * F

    text("Let's take stock of the accounting results.")
    assert flops == 2 * B * D * F
    assert bytes_transferred == 2 * B * D + 2 * D * F + 2 * B * F
    text("Recall that arithmetic intensity is how much compute we did per byte transferred. We want this to be high.")
    intensity = (flops / bytes_transferred).simplify()  # @inspect intensity

    text("Assuming B is much less than D and F, then we can simplify:")
    intensity = intensity.subs(D, c*B).subs(F, c*B).limit(c, oo).simplify()  # @inspect intensity
    assert intensity == B

    text("Accelerator intensity of H100 (FLOP/s / memory bandwidth) ≈ 295 FLOPs/byte")
    accelerator_intensity = 989e12 / 3.35e12  # @inspect accelerator_intensity
    assert round(accelerator_intensity) == 295

    text("If computation intensity > accelerator intensity, **compute-bound** (good)")
    text("If computation intensity < accelerator intensity, **memory-bound** (bad)")
    text("Conclusion: compute-bound iff B > 295")

    text("Extreme case (B = 1, corresponding to matrix-vector product):")
    text("- Arithmetic intensity: 1")
    text("- Memory-bound (read D x F matrix, perform only 2 D F FLOPs)")
    text("- This is basically what happens with generation...")


def arithmetic_intensity_of_inference():
    link(scaling_book_inference)

    image("https://jax-ml.github.io/scaling-book/assets/img/naive-inference-1400.webp", width=600)
    text("Naive inference: to generate each token, feed history into Transformer")
    text("Time complexity: generating T tokens takes O(T^3) time (one feedforward pass is O(T^2))")
    text("Observation: a lot of the work can be shared across prefixes")

    text("Solution: store **KV cache** in HBM")
    image("https://jax-ml.github.io/scaling-book/assets/img/cached-inference-1400.webp", width=600)
    text("KV cache: for every sequence (B), token (S), layer (L), head (K), store H-dimensional vector")

    text("Two stages of inference:")
    text("1. **Prefill**: Given a prompt (parallelizable like in training)")
    text("2. **Generation**: generate new response tokens (sequential)")

    text("Let's compute the FLOPs and memory read/write for both the MLP and attention layers.")
    text("S is the number of tokens we're conditioning on, T is the number of tokens we're generating.")
    text("Later, we'll specialize to prefill (T = S) and generation (T = 1).")

    text("### MLP layers (only looking at the matrix multiplications)")
    flops = 0
    bytes_transferred = 0
    text("Steps:")
    text("1. Read X (B x T x D) from HBM")
    bytes_transferred += 2*B*T*D
    text("2. Read Wup (D x F), Wgate (D x F), Wdown (F x D) from HBM")
    bytes_transferred += 3 * 2*D*F
    text("3. Compute U = X (B x T x D) @ Wup (D x F)")
    flops += 2*B*T*D*F
    text("4. Write U (B x T x F) to HBM")
    bytes_transferred += 2*B*T*F
    text("5. Compute G = X (B x T x F) * Wgate (D x F) with GeLU activation")
    flops += 2*B*T*D*F
    text("6. Write G (B x T x F) to HBM")
    bytes_transferred += 2*B*T*F
    text("7. Compute Y = GeLU(G)*U (B x T x F) @ Wdown (F x D)")
    flops += 2*B*T*D*F
    text("8. Write Y (B x T x D) to HBM")
    bytes_transferred += 2*B*T*D

    text("Let's take stock of the accounting results.")
    assert flops == 6*B*T*D*F
    assert bytes_transferred == 4*B*T*D + 4*B*T*F + 6*D*F
    intensity = (flops / bytes_transferred).simplify()  # @inspect intensity
    text("Assume that B*T is much smaller than D and F.")
    intensity = intensity.subs(D, c*B*T).subs(F, c*B*T).limit(c, oo).simplify()  # @inspect intensity
    assert intensity == B*T

    text("For the two stages:")
    text("1. Prefill: easy to make compute-bound (good) by making B T large enough")
    text("2. Generation:")
    text("- Generating one token at a time (T = 1)")
    text("- B is number of concurrent requests, hard to make large enough!")

    text("### Attention layers (focusing on the matrix multiplications with FlashAttention)")
    flops = 0
    bytes_transferred = 0
    text("Steps:")
    text("1. Read Q (B x T x D), K (B x S x D), V (B x S x D) from HBM")
    bytes_transferred += 2*B*T*D + 2 * 2*B*S*D
    text("2. Compute A = Q (B x T x D) @ K (B x S x D)")
    flops += 2*B*S*T*D
    text("3. Compute Y = A (B x S x T x K x G) @ V (B x S x K x H)")
    flops += 2*B*S*T*D
    text("4. Write Y (B x T x D) to HBM")
    bytes_transferred += 2*B*T*D

    assert flops == 4*B*S*T*D
    assert bytes_transferred == 4*B*S*D + 4*B*T*D
    intensity = (flops / bytes_transferred).simplify()  # @inspect intensity
    assert intensity == S*T / (S + T)

    text("For the two stages:")
    text("1. Prefill: T = S")
    prefill_intensity = intensity.subs(T, S).simplify()  # @inspect prefill_intensity
    assert prefill_intensity == S/2  # Good!
    text("2. Generation: T = 1")
    generate_intensity = intensity.subs(T, 1).simplify()  # @inspect generate_intensity
    assert generate_intensity < 1  # Bad!

    text("Unlike MLPs, no dependence on B, so batching doesn't help!")
    text("Why?")
    text("- In MLP layers, every sequence hits the same MLP weights (Wup, Wgate, Wdown don't depend on B)")
    text("- In attention layers, every sequence has its own vectors KV cache (Q, K, V all depend on B)")

    text("Summary")
    text("- Prefill is compute-bound, generation is memory-bound")
    text("- MLP intensity is B (requires concurrent requests), attention intensity is 1 (impossible to improve)")


def compute_transformer_stats(config):  # @inspect config
    """Return symbols corresponding to various statistics of a Transformer."""
    text("The memory, throughput, and latency depends on the shape of the Transformer. "), text(" "), link("")

    text("Compute the number of parameters in the Transformer:")
    num_params = 2*V*D + D*F*3*L + (2*D*N*H + 2*D*K*H)*L
    text("To store parameters, just use bf16 (training requires fp32)")
    parameter_size = num_params * 2  # 2 for bf16
    
    text("Also don't need gradients and optimizer states since we're not training")
    text("But we do have to store the KV cache (selective activations) for each sequence (of length S):")
    kv_cache_size = S * (K*H) * L * 2 * 2  # 2 for key + value, 2 for bf16

    text("Total memory usage:")
    memory = B * kv_cache_size + parameter_size
    text("Latency is determined by memory IO (read all parameters and KV cache for each step)")
    latency = memory / memory_bandwidth
    text("Throughput is the inverse of latency, but generate B tokens in parallel")
    throughput = B / latency

    # Substitute
    num_params = num_params.subs(config).simplify()  # @inspect num_params
    memory = memory.subs(config).simplify()  # @inspect memory
    latency = latency.subs(config).simplify()  # @inspect latency
    throughput = throughput.subs(config).simplify()  # @inspect throughput

    return num_params, memory, latency, throughput

def llama2_13b_config(args={}):
    return {S: 1024, D: 5120, F: 13824, N: 40, K: 40, H: 128, L: 40, V: 32000, memory_bandwidth: 3.35e12, **args}

def throughput_and_latency():
    text("So we have shown that inference is memory-bound.")
    text("Let us now compute the theoretical maximum latency and throughput of a single request.")
    text("Assumption: can overlap compute and communication perfectly and ignore various overhead.")
    
    text("Instantiate latency and throughput for Llama 2 13B on an H100:")
    config = llama2_13b_config()
    num_params, memory, latency, throughput = compute_transformer_stats(config)
    assert num_params == 13_015_449_600  # Sanity check

    text("If we use a batch size of 1:")
    bs1_memory = memory.subs(B, 1).simplify()   # @inspect bs1_memory
    bs1_latency = latency.subs(B, 1).simplify()   # @inspect bs1_latency
    bs1_throughput = throughput.subs(B, 1).simplify()   # @inspect bs1_throughput

    text("If we use a batch size of 64:")
    bs64_memory = memory.subs(B, 64).simplify()   # @inspect bs64_memory
    bs64_latency = latency.subs(B, 64).simplify()   # @inspect bs64_latency
    bs64_throughput = throughput.subs(B, 64).simplify()   # @inspect bs64_throughput

    text("If we use a batch size of 256:")
    bs256_memory = memory.subs(B, 256).simplify()   # @inspect bs256_memory
    bs256_latency = latency.subs(B, 256).simplify()   # @inspect bs256_latency
    bs256_throughput = throughput.subs(B, 256).simplify()   # @inspect bs256_throughput
    text("Doesn't fit into memory, but throughput gains are diminishing too...")

    text("**Tradeoff** between latency and throughput:")
    text("1. Smaller batch sizes yields better latency but worse throughput")
    text("2. Larger batch sizes yields better throughput but worse latency")

    text("Easy parallelism: if you launch M copies of the model, latency is the same, throughput increases by M!")
    text("Harder parallelism: shard the model and the KV cache "), link(title="[Jax book chapter]", url="https://jax-ml.github.io/scaling-book/inference/")
    text("Note: time-to-first-token (TTFT) is essentially a function of prefill")


def reduce_kv_cache_size():
    text("Recall that memory is the bottleneck for inference")
    text("Key: reduce the size of the KV cache")
    text("Try to make sure we don't lose too much accuracy")

    text("### Grouped-query attention (GQA) "), link(gqa)
    
    image("https://jax-ml.github.io/scaling-book/assets/img/gmqa.png", width=600)
    text("Key: have N query heads, but only K key and value heads, each interacting with N/K query heads")
    text("Multi-headed attention (MHA): K=N")
    text("Multi-query attention (MQA): K=1")
    text("Group-query attention (GQA): K is somewhere in between")

    text("Latency/throughput improvements:")
    image("images/gqa-speed.png", width=400); text(" "); link(gqa)
    text("Reduce the KV cache by a factor of N/K")
    config = llama2_13b_config({K: 40, B: 64})  # Original Llama 2 13B
    k40_num_params, k40_memory, k40_latency, k40_throughput = compute_transformer_stats(config)  # @inspect k40_latency, @inspect k40_throughput

    config = llama2_13b_config({K: 8, B: 64})  # Use 1:5 ratio
    k8_num_params, k8_memory, k8_latency, k8_throughput = compute_transformer_stats(config)  # @inspect k8_latency, @inspect k8_throughput

    text("This also means we can use a larger batch size:")
    config = llama2_13b_config({K: 8, B: 256})  # Increase batch size
    k8_bs_num_params, k8_bs_memory, k8_bs_latency, k8_bs_throughput = compute_transformer_stats(config)  # @inspect k8_bs_latency, @inspect k8_bs_throughput

    text("Check that accuracy isn't hurt:")
    image("images/gqa-accuracy.png", width=800); text(" "); link(gqa)

    text("### Multi-head latent attention (MLA) "), link(mla)
    image("images/mla-schema.png", width=800)
    text("Key idea: project down each key and value vector from N*H to C")
    text("DeepSeek v2: go from N*H = 16384 to C = 512")
    text("Wrinkle: MLA is not compatible with RoPE, so need to add additional 64 dimensions for RoPE, so 512 + 64 = 576 total dimensions")
    text("Latency/throughput improvements follow similarly from the KV cache reduction")

    text("Check accuracy: MLA is better than GQA "); link(mla)
    image("images/mla-accuracy.png", width=800)

    text("### Cross-layer attention (CLA) "), link("https://arxiv.org/abs/2405.12981")
    image("images/cla-diagram.png", width=500)
    text("Key idea: share key-value across **layers** (GQA just shares across heads)")
    text("Improve the pareto frontier of accuracy and KV cache size (latency and throughput)")
    image("images/cla-results.png", width=700)

    text("### Local attention "), link(longformer), link(sparse_transformer), link(mistral_7b)
    image("images/longformer-attention.png", width=800)
    text("Intuition: local context is most important, so most of the time don't need all the context")
    text("Effective context scales linearly with the number of layers")
    text("KV cache is independent of sequence length!")

    text("Problem: this can still hurt accuracy")
    text("Solution: interleave local attention with global attention (hybrid layers)")
    text("Character.ai uses 1 global layer every 6 layers (in addition to CLA) "), article_link("https://research.character.ai/optimizing-inference/")
    image("https://research.character.ai/content/images/2024/06/figure1-2-1.png", width=800)

    text("Summary:")
    text("- Goal: reduce the KV cache size (since inference is memory-limited) without hurting accuracy")
    text("- Lower-dimensional KV cache (GQA, MLA, shared KV cache)")
    text("- Local attention on some of the layers")


def alternatives_to_the_transformer():
    text("We have shown that tweaking the architecture of the Transformer, we can improve latency and throughput.")
    text("But can we improve things even more if we go beyond the Transformer?")
    text("Two directions: state-space models and diffusion models")

    text("## State-space models")
    link(title="[presentation from CS229S]", url="https://docs.google.com/presentation/d/1wrQO4uzwWr73SGj7aFxeVR9Cz0PY-mzJipn12enM39k/edit#slide=id.p")
    text("- Key idea: use ideas from signal processing to model long-context sequences in a sub-quadratic time")
    text("- S4: recurrent , leverages FFT, solved synthetic long-context task (XPath)"), link("https://arxiv.org/abs/2111.00396")
    text("- Weaknesses: bad at solving associative recall tasks (that Transformers do well)")
    text("- Mamba: selective (input-dependent) state space model"), link("https://arxiv.org/abs/2312.00752")
    text("- BASED: simple idea of using linear attention + local attention "), link("https://arxiv.org/abs/2402.18668")
    text("- Jamba: Transformer-Mamba hybrid from AI21 "), link("https://arxiv.org/abs/2403.19887")

    text("### Diffusion models")
    text("- Popular in computer vision, but harder to get working for text "), link("https://arxiv.org/abs/2205.14217")
    text("Key: can generate every token in parallel (not autoregressively), refine multiple time steps")
    text("Start with random noise (over entire sequence), iteratively refine it")
    text("Results from Inception Labs "), article_link("https://www.inceptionlabs.ai/news")
    text("https://x.com/i/status/1894847919624462794")
    image("https://framerusercontent.com/images/K2zvhtaTsz5ehDFoWx6KQHOqCyk.jpg", width=600)


def quantization():
    text("Key idea: reduce the precision of numbers")
    text("Less memory means higher latency/throughput (since inference is memory-limited)")
    text("Also requires fewer FLOPs")
    text("Of course we have to worry about accuracy...")

    article_link("https://www.baseten.co/blog/fp8-efficient-model-inference-with-8-bit-floating-point-numbers/")
    image("https://www.datocms-assets.com/104802/1709770809-twitter-post-20.png", width=400)
    text("- fp32 (4 bytes): needed for parameters and optimizer states during training")
    text("- bf16 (2 bytes): default for inference")
    text("- fp8 (1 byte) [-240, 240] for e4m3 on H100s: can train if you dare "), link("https://arxiv.org/pdf/2310.18313")
    text("- int8 (1 byte) [-128, 127]: less accurate but cheaper than fp8, but inference only"), link("https://arxiv.org/pdf/2303.17951")
    text("- int4 (0.5 bytes) [-8, 7]: cheaper, less accurate"), link("https://arxiv.org/pdf/2303.17951")

    text("Quantization-aware training (QAT): train with quantization, but doesn't scale up")
    text("Post-training quantization (PTQ): run on sample data to determine scale and zero point for each layer or tensor")
    link(title="[Overview of approaches]", url="https://apxml.com/posts/llm-quantization-techniques-explained")

    text("### LLM.int8() "), link("https://arxiv.org/abs/2208.07339")
    article_link("https://huggingface.co/blog/hf-bitsandbytes-integration")
    image("https://huggingface.co/blog/assets/96_hf_bitsandbytes_integration/quant-freeze.png", width=500)
    text("Problem: outliers (which appear in larger networks) screw everything up")
    text("Solution: extract outliers and process them in fp16")
    image("https://huggingface.co/blog/assets/96_hf_bitsandbytes_integration/Mixed-int8.gif", width=600)
    text("It works well (but is 15-23% slower than fp16):")
    image("images/llm-int8-bloom.png", width=500)

    text("### Activation-aware quantization")
    text("Key observation: 0.1-1% of weights are salient, if don't quantize")
    link("https://arxiv.org/abs/2306.00978")
    text("- 4x lower memory, 3.2x speedup over FP16")
    image("images/awq-schema.png", width=800)


def model_pruning():
    link("https://arxiv.org/abs/2407.14679")
    image("images/pruning-kd-loop.png", width=600)
    text("Algorithm")
    text("1. Identify important {layer, head, hidden dimension} on calibration dataset (1024 samples)")
    text("2. Remove unimportant layers to get a smaller model")
    text("3. Distill the original model onto pruned model (fine-tuned)")

    text("Results:")
    image("images/pruning-kd.png", width=600)


def speculative_sampling():
    text("Recall the two stages of inference:")
    text("- Prefill: given a sequence, encode in parallel, can compute probability (compute-bound)")
    text("- Generation: generate one token at a time (memory-bound)")
    text("In other words, checking is easier than generation")

    text("Speculative sampling "); link("https://arxiv.org/abs/2211.17192"); link("https://arxiv.org/abs/2302.01318")
    text("- Use a cheaper **draft model** p to guess a few (e.g., 4) tokens")
    text("- Evaluate with target model q (process tokens in parallel), and accept if it looks good")
    link("[Speculative sampling video]", url="https://storage.googleapis.com/gweb-research2023-media/media/SpeculativeDecoding-1-Illustration.mp4")
    article_link("https://research.google/blog/looking-back-at-speculative-decoding/")

    image("images/speculative-sampling-algorithm.png", width=600)
    text("This is modified rejection sampling that guarantees generating at least one candidate (rejection sampling will keep looping)")
    text("Guarantees to be an **exact sample** from the target model!")

    text("Proof by example: assume two vocabulary elements {A, B}")
    text("- Target [q(A), q(B)]")
    text("- Draft [p(A), p(B)]")
    text("- Assume p(A) > q(A) [draft model oversamples A].")
    text("- Extra [0, 1]")
    text("Compute the probabilities:")
    text("- P[sampling A] = p(A) * (q(A) / p(A)) = q(A)")
    text("- P[sampling B] = p(B) * 1 + p(A) * (1 - q(A) / p(A)) = q(B)")

    image("images/speculative-sampling-results.png", width=600)
    image("images/speculative-sampling-stats.png", width=600)

    text("In practice:")
    text("- Target model has 70B parameters, draft model has 8B parameters")
    text("- Target model has 8B parameters, draft model has 1B parameters")
    text("- Try to make draft model as close to target (distillation)")

    text("Extensions:")
    text("- Medusa: draft model generates multiple tokens in parallel"), link("https://arxiv.org/abs/2401.10774")
    text("- EAGLE: draft model takes high-level features from target model"), link("https://arxiv.org/pdf/2401.15077")
    image("images/medusa-eagle.png", width=400)

    text("Summary:")
    text("- Exact sampling from target model (thanks to math)!")
    text("- Exploits asymmetry between checking and generation")
    text("- Lots of room for innovation on the draft model (involves training)")


def continuous_batching():
    link(title="Orca: A Distributed Serving System for Transformer-Based Generative Models", url="https://www.usenix.org/system/files/osdi22-yu.pdf"), link(title="[talk]", url="https://www.youtube.com/watch?v=Ob9PPLxETYU")

    text("Problem:")
    text("- Training: get a dense block of tokens (batch size x sequence length)")
    text("- Inference: requests arrive and finish at different times, so you have a ragged array")
    image("https://images.ctfassets.net/xjan103pcp94/1LJioEsEdQQpDCxYNWirU6/82b9fbfc5b78b10c1d4508b60e72fdcf/cb_02_diagram-static-batching.png", width=400)

    text("Solution: iteration-level scheduling")
    text("- Decode step by step")
    text("- Add new requests to the batch as they arrive (so don't have to wait until generation completes)")

    text("Problem:")
    text("- Batching only works when all sequences have the same dimensionality (right?)")
    text("- But each request might have a different length")

    text("Solution: selective batching")
    text("- Training: when all sequences of the same length, operate on a B x S x H tensor")
    text("- But we might have different lengths: [3, H], [9, H], [5, H], etc.")
    text("- Attention computation: process each sequence separately")
    text("- Non-attention computation: concatenate all the sequences together to [3 + 9 + 5, H]")


def paged_attention():
    text("Paper that introduced vLLM in addition to PagedAttention "), link("https://arxiv.org/pdf/2309.06180.pdf")

    text("Previous status quo:")
    text("- Request comes in")
    text("- Allocate section of KV cache for prompt and response (up to a max length)")
    image("images/paged-attention-fragmentation.png", width=800)
    text("Problem: fragmentation (what happens to your hard drive)")
    text("- But this is wasteful since we might generate much fewer tokens (internal fragmentation)!")
    text("- Might be extra unused space between sections (external fragmentation)!")

    text("Solution: PagedAttention (remember operating systems)")
    text("- Divide the KV cache of a sequence into non-contiguous **blocks**")
    image("images/paged-attention-blocks.png", width=400)

    text("Two requests share the KV caches:")
    image("images/paged-attention-logical.png", width=800)

    text("In general, multiples types of sharing KV caches across sequences")
    image("images/paged-attention-sharing.png", width=600)
    text("- System prompt")
    text("- Sampling multiple responses per prompt (e.g., for program synthesis)")
    text("- Decoding (beam search)")

    text("Solution: share prefixes, copy-on-write at the block level")
    image("images/paged-attention-parallel.png", width=400)

    text("Other vLLM optimizations:")
    text("- Kernel to fusing block read and attention (reduce kernel launch overhead)")
    text("- Use latest kernels (FlashAttention, FlashDecoding)")
    text("- Use CUDA graphs to avoid kernel launch overhead")

    text("Summary: use ideas from operating systems (paging) to make use of memory for dynamic workloads")


if __name__ == "__main__":
    main()
