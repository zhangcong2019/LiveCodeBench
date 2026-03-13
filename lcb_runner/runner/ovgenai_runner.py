from lcb_runner.runner.base_runner import BaseRunner


class OVGenAIRunner(BaseRunner):
    """
    OpenVINO GenAI Runner
    
    使用 Intel OpenVINO GenAI 进行本地推理
    """
    
    def __init__(self, args, model):
        super().__init__(args, model)
        
        # OV GenAI 配置 (hardcode)
        self.ov_model_path = '/home/vpp/repo/openvino.genai/samples/python/text_generation/qwen2.5'
        self.ov_device = 'GPU'
        self.ov_num_kv_blocks = 2048
        self.ov_max_num_batched_tokens = 1024
        
        # 从 model 名称判断是否启用特定功能
        model_name_lower = model.model_name.lower()
        self.ov_enable_cache_eviction = 'eviction' in model_name_lower
        self.ov_enable_kvcrush = 'kvcrush' in model_name_lower
        self.ov_enable_sparse_attention = 'sparsity' in model_name_lower
        
        print(f"OVGenAI Model: {self.ov_model_path}")
        print(f"OVGenAI Device: {self.ov_device}")
        print(f"KV Blocks: {self.ov_num_kv_blocks}")
    
    def _run_single(self, prompt: str | list[dict[str, str]]) -> list[str]:
        """
        调用 OV GenAI API 生成回复
        
        参数:
            prompt: 输入的提示词 (str 或 messages 列表)
        
        返回:
            list[str]: 模型生成的 n 个回复
        """
        # 处理 messages 格式 (ChatML 等)
        if isinstance(prompt, list):
            # 将 messages 格式转换为单一字符串
            prompt_text = ""
            for msg in prompt:
                role = msg.get("role", "user")
                content = msg.get("content", "")
                prompt_text += f"{role}: {content}\n"
            prompt = prompt_text.strip()
        
        # 创建 mock model 对象供 ov_api 使用
        class OVModel:
            pass
        
        ov_model = OVModel()
        ov_model.model_path = self.ov_model_path
        ov_model.device = self.ov_device
        ov_model.enable_cache_eviction = self.ov_enable_cache_eviction
        ov_model.enable_kvcrush = self.ov_enable_kvcrush
        ov_model.enable_sparse_attention = self.ov_enable_sparse_attention
        
        # 导入并调用 ov_api
        from lcb_runner.runner.ov_api import ov_api
        
        result = ov_api(
            prompt=prompt,
            model=ov_model,
            temperature=self.args.temperature,
            max_new_tokens=self.args.max_tokens,
            num_kv_blocks=self.ov_num_kv_blocks,
            max_num_batched_tokens=self.ov_max_num_batched_tokens,
            benchmark=False
        )
        
        # 返回 n 个结果
        return [result for _ in range(self.args.n)]
