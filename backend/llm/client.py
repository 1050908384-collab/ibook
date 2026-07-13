import ollama
import httpx
from typing import Dict, Optional, Any
from backend.config import settings


class OllamaClient:
    def __init__(self, host: str = settings.OLLAMA_HOST):
        self.host = host
        self.client = ollama.Client(host=self.host)
    
    def generate(self, model: str, prompt: str, **kwargs) -> str:
        response = self.client.generate(
            model=model,
            prompt=prompt,
            **kwargs
        )
        return response.get("response", "")
    
    def chat(self, model: str, messages: list, **kwargs) -> str:
        response = self.client.chat(
            model=model,
            messages=messages,
            **kwargs
        )
        return response.get("message", {}).get("content", "")


class DeepSeekClient:
    def __init__(self, api_key: str = None, base_url: str = None):
        self.api_key = api_key or settings.DEEPSEEK_API_KEY
        self.base_url = base_url or settings.DEEPSEEK_BASE_URL
        self.client = httpx.Client()
    
    def generate(self, model: str, prompt: str, **kwargs) -> str:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        payload = {
            "model": model,
            "prompt": prompt,
            "max_tokens": kwargs.get("max_tokens", 2048),
            "temperature": kwargs.get("temperature", 0.7)
        }
        
        try:
            response = self.client.post(
                f"{self.base_url}/chat/completions",
                headers=headers,
                json=payload
            )
            response.raise_for_status()
            data = response.json()
            return data["choices"][0]["message"]["content"]
        except Exception as e:
            raise RuntimeError(f"DeepSeek API error: {str(e)}")
    
    def chat(self, model: str, messages: list, **kwargs) -> str:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        payload = {
            "model": model,
            "messages": messages,
            "max_tokens": kwargs.get("max_tokens", 2048),
            "temperature": kwargs.get("temperature", 0.7)
        }
        
        try:
            response = self.client.post(
                f"{self.base_url}/chat/completions",
                headers=headers,
                json=payload
            )
            response.raise_for_status()
            data = response.json()
            return data["choices"][0]["message"]["content"]
        except Exception as e:
            raise RuntimeError(f"DeepSeek API error: {str(e)}")


class LLMClient:
    _instance: Optional['LLMClient'] = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._init_clients()
        return cls._instance
    
    def _init_clients(self):
        self.ollama_client = OllamaClient()
        self.deepseek_client = DeepSeekClient()
    
    def get_client(self, provider: str = None):
        provider = provider or settings.LLM_PROVIDER
        if provider == "ollama":
            return self.ollama_client
        elif provider == "deepseek":
            return self.deepseek_client
        else:
            raise ValueError(f"Unknown LLM provider: {provider}")
    
    def generate(self, model: str, prompt: str, provider: str = None, **kwargs) -> str:
        client = self.get_client(provider)
        return client.generate(model, prompt, **kwargs)
    
    def chat(self, model: str, messages: list, provider: str = None, **kwargs) -> str:
        client = self.get_client(provider)
        return client.chat(model, messages, **kwargs)