from __future__ import annotations

from openai import OpenAI

from src.config import load_config


def create_client(config_path: str | None = None) -> tuple[OpenAI, str, dict]:
    config = load_config(config_path)
    llm = config["llm"]
    client = OpenAI(api_key=llm["api_key"], base_url=llm["base_url"])
    return client, llm["model"], config
