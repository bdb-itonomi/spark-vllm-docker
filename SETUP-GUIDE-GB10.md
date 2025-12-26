# Running vLLM with Devstral-Small-2-24B on NVIDIA GB10

This document describes the process of setting up vLLM to serve the Devstral-Small-2-24B model on an NVIDIA GB10 (Grace Blackwell) system, including problems encountered and solutions.

## System Overview

| Component | Details |
|-----------|---------|
| **Host** | <your-gb10-host> |
| **Hardware** | NVIDIA GB10 (Grace Blackwell integrated CPU/GPU) |
| **Memory** | 119GB unified memory |
| **OS** | Linux aarch64 (ARM64), Ubuntu |
| **CUDA** | 13.0, Driver 580.95.05 |
| **Docker** | v28.5.1 |

## Model Details

| Property | Value |
|----------|-------|
| **Location** | `$HOME/models/devstral-small-2-24b/safetensors/` |
| **Architecture** | Mistral3ForConditionalGeneration (PixtralForConditionalGeneration) |
| **Type** | Multimodal (text + vision via Pixtral) |
| **Quantization** | FP8 |
| **Size** | ~26GB safetensors |
| **Default Context** | 393,216 tokens |

---

## Original Plan

### Step 1: Clone spark-vllm-docker Repository
```bash
cd ~/dev
git clone https://github.com/eugr/spark-vllm-docker.git
```

### Step 2: Build vLLM Docker Image
```bash
cd spark-vllm-docker
./build-and-copy.sh --use-wheels
```

### Step 3: Run vLLM with Local Model
```bash
docker run \
  --privileged --gpus all \
  -it --rm \
  --network host --ipc=host \
  -v $HOME/models/devstral-small-2-24b/safetensors:/models/devstral-small-2-24b \
  vllm-node \
  bash -c "vllm serve /models/devstral-small-2-24b \
    --port 8000 --host 0.0.0.0 \
    --gpu-memory-utilization 0.7 \
    --trust-remote-code"
```

---

## Problems Encountered and Solutions

### Problem 1: Docker Build Network Timeout

**Symptom:**
```
× Failed to download `nvidia-cusparse==12.6.2.49`
├─▶ Failed to extract archive
├─▶ I/O operation failed during extraction
╰─▶ Failed to download distribution due to network timeout. Try increasing
    UV_HTTP_TIMEOUT (current value: 30s).
```

**Cause:** The default UV_HTTP_TIMEOUT of 30 seconds was insufficient for downloading large NVIDIA packages over the network.

**Unsuccessful Attempt:**
- First build attempt with default timeout failed

**Solution:**
```bash
UV_HTTP_TIMEOUT=300 ./build-and-copy.sh --use-wheels
```

Setting the timeout to 300 seconds allowed the build to complete successfully.

---

### Problem 2: KV Cache Memory Insufficient for Default Context Length

**Symptom:**
```
ValueError: To serve at least one request with the models's max seq len (393216),
60.00 GiB KV cache is needed, which is larger than the available KV cache memory
(56.13 GiB). Based on the available memory, the estimated maximum model length is
367824.
```

**Cause:** The model's default maximum sequence length of 393,216 tokens requires 60GB of KV cache memory, but after loading the 24GB model, only 56GB remained available.

**Analysis:**
- Model memory: 24.1 GiB
- Total GPU memory at 70% utilization: ~83 GiB
- Available for KV cache: 56.13 GiB
- Required for 393K context: 60 GiB

**Unsuccessful Attempt:**
```bash
docker run ... vllm serve /models/devstral-small-2-24b \
  --gpu-memory-utilization 0.7 \
  --trust-remote-code
```

**Solution:**
Added `--max-model-len 65536` to limit context to 64K tokens, and increased GPU memory utilization to 0.85:
```bash
docker run ... vllm serve /models/devstral-small-2-24b \
  --gpu-memory-utilization 0.85 \
  --max-model-len 65536 \
  --trust-remote-code
```

This resulted in 74.43 GiB available for KV cache, supporting up to 487,744 tokens with 7.44x concurrency for 65K requests.

---

### Problem 3: CUDA Graph Capture Error on GB10

**Symptom:**
```
torch.AcceleratorError: CUDA error: operation not permitted when stream is capturing
Search for `cudaErrorStreamCaptureUnsupported' in
https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html

... during capture_model() / _capture_cudagraphs()
```

**Cause:** The NVIDIA GB10 GPU (CUDA capability 12.1) has compatibility issues with CUDA graph capture in the current PyTorch/vLLM stack. PyTorch reported supporting capabilities 8.0-12.0, but GB10 is 12.1.

**Technical Details:**
- The error occurred during `torch.cuda.graph()` context manager
- Specifically in the `CUDAGraphRunner.__call__` method
- The `empty_strided_cuda` operation was not permitted during stream capture

**Unsuccessful Attempt:**
```bash
docker run ... vllm serve /models/devstral-small-2-24b \
  --gpu-memory-utilization 0.85 \
  --max-model-len 65536 \
  --trust-remote-code
```

The model loaded successfully (24.1 GiB) but crashed during CUDA graph warmup/capture phase.

**Solution:**
Added `--enforce-eager` to disable CUDA graphs and torch.compile optimizations:
```bash
docker run ... vllm serve /models/devstral-small-2-24b \
  --gpu-memory-utilization 0.85 \
  --max-model-len 65536 \
  --enforce-eager \
  --trust-remote-code
```

This forces eager execution mode, bypassing the CUDA graph capture that was failing. Trade-off: slightly lower inference performance, but full functionality.

---

## Final Working Configuration

```bash
docker run \
  --privileged \
  --gpus all \
  -d \
  --name vllm-devstral \
  --network host \
  --ipc=host \
  -v $HOME/models/devstral-small-2-24b/safetensors:/models/devstral-small-2-24b \
  vllm-node \
  vllm serve /models/devstral-small-2-24b \
    --port 8000 \
    --host 0.0.0.0 \
    --gpu-memory-utilization 0.85 \
    --max-model-len 65536 \
    --enforce-eager \
    --trust-remote-code
```

### Configuration Flags Explained

| Flag | Value | Purpose |
|------|-------|---------|
| `--privileged` | - | Full hardware access for GPU |
| `--gpus all` | - | Enable GPU passthrough |
| `--network host` | - | Required for NCCL networking |
| `--ipc=host` | - | Shared memory for performance |
| `-v ...` | model path | Mount local safetensors into container |
| `--gpu-memory-utilization` | 0.85 | Use 85% of GPU memory |
| `--max-model-len` | 65536 | Limit context to 64K tokens |
| `--enforce-eager` | - | Disable CUDA graphs (GB10 compatibility) |
| `--trust-remote-code` | - | Allow custom model code |

---

## Performance Characteristics

| Metric | Value |
|--------|-------|
| **Model Memory** | 24.1 GiB |
| **Model Load Time** | ~140 seconds |
| **Available KV Cache** | 74.82 GiB |
| **KV Cache Tokens** | 490,368 |
| **Max Concurrent 64K Requests** | 7.44x |
| **Attention Backend** | FLASH_ATTN |

---

## API Verification

### List Models
```bash
curl http://<your-gb10-host>:8000/v1/models
```

### Text Completion
```bash
curl -X POST http://<your-gb10-host>:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "/models/devstral-small-2-24b", "prompt": "Hello!", "max_tokens": 50}'
```

### Chat Completion
```bash
curl -X POST http://<your-gb10-host>:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "/models/devstral-small-2-24b", "messages": [{"role": "user", "content": "What is 2+2?"}]}'
```

---

## Container Management

### View Logs
```bash
docker logs -f vllm-devstral
```

### Stop Server
```bash
docker stop vllm-devstral
```

### Remove Container
```bash
docker rm vllm-devstral
```

### Restart Server
```bash
docker start vllm-devstral
```

---

## Lessons Learned

1. **GB10 is cutting-edge hardware** - CUDA capability 12.1 is newer than what current PyTorch (supporting 8.0-12.0) fully supports, leading to CUDA graph issues.

2. **Eager mode is the workaround** - Until PyTorch and vLLM add full support for compute capability 12.1, `--enforce-eager` is necessary on GB10.

3. **Memory planning is critical** - Large context models need careful consideration of model size vs KV cache requirements. The 24GB model with 393K context needs 60GB+ KV cache.

4. **Network timeouts need adjustment** - When building Docker images that download large ML packages, increase UV_HTTP_TIMEOUT for reliability.

5. **Unified memory architecture** - GB10's unified CPU/GPU memory (119GB) is advantageous for large models, but still requires proper memory utilization settings.

---

## Future Improvements

- Monitor PyTorch releases for native GB10/compute capability 12.1 support
- Test with CUDA graph support once available
- Consider `--load-format fastsafetensors` for faster loading (requires memory headroom)
- Benchmark throughput with different `--max-model-len` settings
