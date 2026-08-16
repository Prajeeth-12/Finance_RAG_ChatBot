import os
from dotenv import load_dotenv
from openai import OpenAI

# Load environment variables from .env
load_dotenv()

NVIDIA_API_KEY = os.getenv("NVIDIA_API_KEY")
NVIDIA_BASE_URL = "https://integrate.api.nvidia.com/v1"

if not NVIDIA_API_KEY:
    raise ValueError(
        "NVIDIA_API_KEY not found. Please set it in your environment or .env file."
    )

client = OpenAI(
    api_key=NVIDIA_API_KEY,
    base_url=NVIDIA_BASE_URL
)

EMBEDDING_MODEL = "nvidia/nv-embedqa-e5-v5"
LLM_MODEL = "meta/llama-3.1-8b-instruct"


def get_embedding(text: str, input_type: str = "passage", model: str = EMBEDDING_MODEL) -> list:
    """
    Generate embedding for text using NVIDIA NIM.
    input_type should be 'passage' for indexing documents, or 'query' for search queries.
    """
    response = client.embeddings.create(
        input=[text],
        model=model,
        encoding_format="float",
        extra_body={"input_type": input_type, "truncate": "NONE"}
    )
    return response.data[0].embedding


def get_embeddings_batch(texts: list, input_type: str = "passage", model: str = EMBEDDING_MODEL) -> list:
    """
    Generate embeddings for a batch of texts using NVIDIA NIM.
    """
    response = client.embeddings.create(
        input=texts,
        model=model,
        encoding_format="float",
        extra_body={"input_type": input_type, "truncate": "NONE"}
    )
    return [item.embedding for item in response.data]


def get_llm_completion(prompt: str, model: str = LLM_MODEL, temperature: float = 0.0) -> str:
    """
    Generate completion using NVIDIA NIM LLM.
    """
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "user", "content": prompt}
        ],
        temperature=temperature,
        top_p=0.7,
        max_tokens=1024
    )
    return response.choices[0].message.content
