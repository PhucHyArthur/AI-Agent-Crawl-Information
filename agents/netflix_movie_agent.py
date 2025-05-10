from smolagents import CodeAgent, TransformersModel, FinalAnswerTool
from tools.web_crawler import fetch_netflix_movie_info
import torch

def build_movie_agent():
    model = TransformersModel(
        model_id="Qwen/Qwen2.5-Coder-3B-Instruct",
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        max_new_tokens=2000
    )

    agent = CodeAgent(
        model=model,
        tools=[
            fetch_netflix_movie_info,
            FinalAnswerTool()
        ],
        additional_authorized_imports=["requests", "bs4"],
        verbosity_level=2,
        name="netflix_movie_agent"
    )
    return agent