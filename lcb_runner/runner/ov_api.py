# Copyright (C) 2023-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import argparse
import sys
from typing import Optional

import openvino_genai as ov_genai
from openvino import get_version



from openvino_genai import (
    AggregationMode,
    CacheEvictionConfig,
    ContinuousBatchingPipeline,
    GenerationConfig,
    SchedulerConfig,
    KVCrushConfig,
    KVCrushAnchorPointMode,
    SparseAttentionConfig,
    SparseAttentionMode,
)

def get_scheduler_config(num_kv_blocks: Optional[int], max_num_batched_tokens: Optional[int] = 1024) -> SchedulerConfig:
    scheduler_config = SchedulerConfig()
    if num_kv_blocks is not None:
        scheduler_config.num_kv_blocks = num_kv_blocks
        # scheduler_config.max_num_batched_tokens = 32 * num_kv_blocks
        scheduler_config.max_num_batched_tokens = max_num_batched_tokens
    scheduler_config.dynamic_split_fuse = True
    scheduler_config.max_num_seqs = 16
    scheduler_config.use_cache_eviction = False
    print("num_kv_blocks:", scheduler_config.num_kv_blocks
          , ", max_num_batched_tokens:", scheduler_config.max_num_batched_tokens)
    return scheduler_config



# ========== 本地 Python API ==========
# TODO: 实现本地模型调用接口
def ov_api(prompt, model, temperature=0.5, max_new_tokens=1024, prompt_file=None, num_warmup  = 0, num_iter = 1, num_kv_blocks = 2048, max_num_batched_tokens=1024, benchmark = False):
    """
    本地 Python API 接口 - 请手动实现
    
    参数:
        prompt: 输入的提示词
        model: 模型名称
        temperature: 温度参数
        max_new_tokens: 最大生成 token 数
        num_kv_blocks: KV 块数量
        benchmark: 是否进行基准测试
    
    返回:
        str: 模型生成的回复
    """

    if prompt is not None and prompt_file is not None:
        raise RuntimeError("Cannot specify both --prompt and --prompt_file options simultaneously!")
    else:
        if prompt_file is not None:
            with open(prompt_file, "r", encoding="utf-8") as f:
                prompt = [f.read()]
        else:
            prompt = ["The Sky is blue because"] if prompt is None else [prompt]
    if len(prompt) == 0:
        raise RuntimeError("Prompt is empty!")

    print(f"openvino runtime version: {get_version()}, genai version: {ov_genai.__version__}")

    # Perf metrics is stored in DecodedResults.
    # In order to get DecodedResults instead of a string input should be a list.

    config = ov_genai.GenerationConfig()
    config.max_new_tokens = max_new_tokens
    config.apply_chat_template = False
    config.ignore_eos = False
    
    if benchmark == True:
        config.ignore_eos = True
    else:
        num_warmup = 0
        num_iter = 1

    print("Model path:", model.model_path)
    print("Device:", model.device)
    # print("Enable Cache Eviction:", model.enable_cache_eviction)
    # print("Enable KV Crush:", model.enable_kvcrush)
    # print("Enable Sparse Attention:", model.enable_sparse_attention)
    print("Number of KV Blocks:", num_kv_blocks)
    print("Max new tokens:", config.max_new_tokens)
    print("Ignore EOS:", config.ignore_eos)

    ov_config = {}
    ov_config["KV_CACHE_PRECISION"] = "i8"
    ov_config["KEY_CACHE_QUANT_MODE"] = "BY_TOKEN"

    scheduler_config = get_scheduler_config(num_kv_blocks, max_num_batched_tokens=max_num_batched_tokens)
    if model.enable_cache_eviction :
        config_size = "large"
        # config_size = "small"
        if config_size == "small":
            start_size = 16
            recent_size = 16
            max_cache_size = 64
        else:
            start_size = 32
            recent_size = 32
            max_cache_size = 512

        budget = 2
        if model.enable_kvcrush == False:
            eviction_config = CacheEvictionConfig(
                start_size=start_size, 
                recent_size=recent_size, 
                max_cache_size=max_cache_size,
                aggregation_mode=AggregationMode.NORM_SUM,
                snapkv_window_size=8,
            )
        else:
            eviction_config = CacheEvictionConfig(
                start_size=start_size, 
                recent_size=recent_size, 
                max_cache_size=max_cache_size,
                aggregation_mode=AggregationMode.NORM_SUM,
                apply_rotation=False,
                snapkv_window_size=8,
                kvcrush_config=KVCrushConfig(budget=budget, anchor_point_mode=KVCrushAnchorPointMode.MEAN)
            )
            print("Eviction config: KV Crush")
            
            print("budget:", budget)

        print("start_size:", start_size
              , ", recent_size:", recent_size
              , ", max_cache_size:", max_cache_size)

        scheduler_config.use_cache_eviction = True
        scheduler_config.cache_eviction_config = eviction_config
        print("Eviction is ON")
    else:
        print("Eviction is OFF")


    if model.enable_sparse_attention :
        sparse_method = "TRISHAPE"  # "TRISHAPE" or "XATTENTION"
        sparse_method = "XATTENTION"  # "TRISHAPE" or "XATTENTION"
        if sparse_method == "TRISHAPE":
            scheduler_config.sparse_attention_config=SparseAttentionConfig(
                mode=SparseAttentionMode.TRISHAPE
            )
            print("Sparse Attention Mode: TRISHAPE")
        else:
            scheduler_config.sparse_attention_config=SparseAttentionConfig(
                mode=SparseAttentionMode.XATTENTION
            )
            print("Sparse Attention Mode: XATTENTION")

        scheduler_config.use_sparse_attention = True
        print("Sparse Attention: ON")
    else:
        print("Sparse Attention: OFF")


    # model_cb = ContinuousBatchingPipeline(args.model, scheduler_config, args.device)

    if model.device == "NPU":
        pipe = ov_genai.LLMPipeline(model.model_path, model.device)
    else:
        # scheduler_config = ov_genai.SchedulerConfig()
        # scheduler_config.enable_prefix_caching = False
        # scheduler_config.max_num_batched_tokens = sys.maxsize
        # scheduler_config.max_num_batched_tokens = 32 * args.num_kv_blocks
        pipe = ov_genai.LLMPipeline(model.model_path, model.device, scheduler_config=scheduler_config, **ov_config)

    input_data = pipe.get_tokenizer().encode(prompt)
    prompt_token_size = input_data.input_ids.get_shape()[1]
    print(f"Prompt token size: {prompt_token_size}")

    for _ in range(num_warmup):
        pipe.generate(prompt, generation_config=config)

    res = pipe.generate(prompt, generation_config=config)
    perf_metrics = res.perf_metrics
    text = res.texts[0]
    for _ in range(num_iter - 1):
        res = pipe.generate(prompt, generation_config=config)
        perf_metrics += res.perf_metrics

    print(f"Output token size: {res.perf_metrics.get_num_generated_tokens()}")
    # print(f"Load time: {perf_metrics.get_load_time():.2f} ms")
    # print(
    #     f"Generate time: {perf_metrics.get_generate_duration().mean:.2f} ± {perf_metrics.get_generate_duration().std:.2f} ms"
    # )
    # print(
    #     f"Tokenization time: {perf_metrics.get_tokenization_duration().mean:.2f} ± {perf_metrics.get_tokenization_duration().std:.2f} ms"
    # )
    # print(
    #     f"Detokenization time: {perf_metrics.get_detokenization_duration().mean:.2f} ± {perf_metrics.get_detokenization_duration().std:.2f} ms"
    # )
    print(f"TTFT: {perf_metrics.get_ttft().mean:.2f} ± {perf_metrics.get_ttft().std:.2f} ms")
    print(f"TPOT: {perf_metrics.get_tpot().mean:.2f} ± {perf_metrics.get_tpot().std:.2f} ms")
    print(f"Throughput : {perf_metrics.get_throughput().mean:.2f} ± {perf_metrics.get_throughput().std:.2f} tokens/s")

    if benchmark == True:
        open("benchmark_result.txt", "a", encoding="utf-8").write(f"{model.enable_sparse_attention}, {model.enable_cache_eviction}, {model.enable_kvcrush}, {prompt_token_size}, {res.perf_metrics.get_num_generated_tokens()}, {max_num_batched_tokens}, {perf_metrics.get_ttft().mean:.2f}, {perf_metrics.get_tpot().mean:.2f}\n")
    # print(f"Generated text: \n{text}")
    return text

def main():
    parser = argparse.ArgumentParser(description="OpenVINO GenAI API")
    parser.add_argument("--model_path", type=str, default="/home/vpp/repo/openvino.genai/samples/python/text_generation/qwen2.5", help="Path to model")
    parser.add_argument("--device", type=str, default="GPU", help="Device to use")
    parser.add_argument("--eviction", action="store_true", default=False, help="Enable cache eviction")
    parser.add_argument("--kvcrush", action="store_true", default=False, help="Enable KV Crush")
    parser.add_argument("--sparse", action="store_true", default=False, help="Enable sparse attention")
    parser.add_argument("--prompt_file", type=str, default="./30k.txt", help="Prompt file")
    parser.add_argument("--num_warmup", type=int, default=1, help="Number of warmup iterations")
    parser.add_argument("--num_iter", type=int, default=2, help="Number of iterations")
    parser.add_argument("--max_new_tokens", type=int, default=1024, help="Maximum number of new tokens to generate")
    parser.add_argument("--num_kv_blocks", type=int, default=2048, help="Number of KV blocks")
    parser.add_argument("--max_num_batched_tokens", type=int, default=1024, help="Maximum number of batched tokens")
    parser.add_argument("--prompt", type=str, default="The Sky is blue because", help="Prompt text")
    args = parser.parse_args()

    model = lambda: None
    model.model_path = args.model_path
    model.device = args.device
    model.enable_cache_eviction = args.eviction
    model.enable_kvcrush = args.kvcrush
    model.enable_sparse_attention = args.sparse

    out = ov_api(None, model, prompt_file=args.prompt_file, num_warmup=args.num_warmup, num_iter=args.num_iter, max_new_tokens=args.max_new_tokens, num_kv_blocks=args.num_kv_blocks, max_num_batched_tokens=args.max_num_batched_tokens, benchmark=True)
    # out = ov_api(args.prompt, model)
    # print(out)

    # prompt = "The Sky is blue because"
    # model = lambda: None
    # model.model_path = "/home/vpp/repo/openvino.genai/samples/python/text_generation/qwen2.5"
    # model.device = "GPU"
    # model.enable_cache_eviction = True
    # model.enable_kvcrush = True
    # model.enable_sparse_attention = False
    # out = ov_api(None, model, prompt_file = "./30k.txt", num_warmup = 1, num_iter = 2)
    # out = ov_api(prompt, model)
    # print(out)


if __name__ == "__main__":
    main()