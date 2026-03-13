from lcb_runner.runner.base_runner import BaseRunner


class OVGenAIRunner(BaseRunner):
    """
    本地 Python API Runner
    
    TODO: 请手动实现本地模型调用逻辑
    """
    
    def __init__(self, args, model):
        super().__init__(args, model)
        # ========================================
        # 在这里初始化你的本地 API 连接
        # 例如:
        #   - 初始化 HTTP 客户端
        #   - 加载本地模型
        #   - 设置 API 地址和密钥
        # ========================================
        
        # raise NotImplementedError("OVGenAIRunner 尚未实现，请手动补充!")
    
    def _run_single(self, prompt: str | list[dict[str, str]]) -> list[str]:
        """
        调用本地 API 生成回复
        
        参数:
            prompt: 输入的提示词 (str 或 messages 列表)
        
        返回:
            list[str]: 模型生成的 n 个回复
        """
        # ========================================
        # TODO: 实现你的本地 API 调用逻辑
        # 
        # 示例:
        # import requests
        # response = requests.post(
        #     "http://localhost:8000/v1/chat/completions",
        #     json={
        #         "model": self.model.model_name,
        #         "messages": [{"role": "user", "content": prompt}],
        #         "n": self.args.n,
        #         "temperature": self.args.temperature,
        #         "max_tokens": self.args.max_tokens,
        #     }
        # )
        # results = [choice.message.content for choice in response.json()["choices"]]
        # return results
        # ========================================
        
        raise NotImplementedError("OVGenAIRunner._run_single() 尚未实现，请手动补充!")
        
        # # 这里是占位符，返回空字符串
        # return ["" for _ in range(self.args.n)]
