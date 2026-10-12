"""
ollama_client.py
----------------
Handles all communication with the local Ollama instance:
- Health check and model availability
- Local text embeddings using nomic-embed-text
- Grounded LLM chat inference using llama3.2:3b
"""

import requests
from typing import List, Dict, Any, Optional

OLLAMA_BASE_URL = "http://localhost:11434"
DEFAULT_CHAT_MODEL = "llama3.2:3b"
DEFAULT_EMBED_MODEL = "nomic-embed-text"


def check_ollama_status(base_url: str = OLLAMA_BASE_URL) -> Dict[str, Any]:
    """
    Check if the local Ollama instance is reachable and verify required models.
    Returns a status dict:
      - connected (bool)
      - models (list of str)
      - chat_model_ready (bool)
      - embed_model_ready (bool)
      - error (str or None)
    """
    try:
        response = requests.get(f"{base_url}/api/tags", timeout=4)
        if response.status_code == 200:
            data = response.json()
            models_installed = [m.get("name", "") for m in data.get("models", [])]
            
            # Check for chat model (e.g., 'llama3.2:3b' or 'llama3.2:latest')
            chat_ready = any(
                "llama3.2" in m for m in models_installed
            )
            # Check for embed model (e.g., 'nomic-embed-text:latest')
            embed_ready = any(
                "nomic-embed-text" in m for m in models_installed
            )
            
            return {
                "connected": True,
                "models": models_installed,
                "chat_model_ready": chat_ready,
                "embed_model_ready": embed_ready,
                "error": None
            }
        else:
            return {
                "connected": False,
                "models": [],
                "chat_model_ready": False,
                "embed_model_ready": False,
                "error": f"Ollama returned HTTP status {response.status_code}"
            }
    except requests.exceptions.ConnectionError:
        return {
            "connected": False,
            "models": [],
            "chat_model_ready": False,
            "embed_model_ready": False,
            "error": "Ollama is not running. Start Ollama and try again."
        }
    except Exception as e:
        return {
            "connected": False,
            "models": [],
            "chat_model_ready": False,
            "embed_model_ready": False,
            "error": str(e)
        }


def get_embeddings(
    texts: List[str],
    model: str = DEFAULT_EMBED_MODEL,
    base_url: str = OLLAMA_BASE_URL,
    batch_size: int = 25
) -> List[List[float]]:
    """
    Generate embeddings for a list of text strings using Ollama's /api/embed endpoint.
    Processes in batches for stability.
    """
    if not texts:
        return []

    all_embeddings: List[List[float]] = []

    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        payload = {
            "model": model,
            "input": batch
        }
        try:
            resp = requests.post(f"{base_url}/api/embed", json=payload, timeout=60)
            if resp.status_code == 200:
                data = resp.json()
                embeddings = data.get("embeddings", [])
                all_embeddings.extend(embeddings)
            else:
                # Fallback to single text /api/embeddings endpoint if /api/embed returned an error
                fallback_embeddings = []
                for t in batch:
                    single_resp = requests.post(
                        f"{base_url}/api/embeddings",
                        json={"model": model, "prompt": t},
                        timeout=30
                    )
                    if single_resp.status_code == 200:
                        fallback_embeddings.append(single_resp.json().get("embedding", []))
                    else:
                        raise RuntimeError(f"Embedding failed with status {single_resp.status_code}: {single_resp.text}")
                all_embeddings.extend(fallback_embeddings)
        except requests.exceptions.ConnectionError:
            raise ConnectionError("Ollama is not running. Start Ollama and try again.")
        except Exception as e:
            raise RuntimeError(f"Failed to generate embeddings: {e}")

    return all_embeddings


def get_single_embedding(
    text: str,
    model: str = DEFAULT_EMBED_MODEL,
    base_url: str = OLLAMA_BASE_URL
) -> List[float]:
    """Generate embedding for a single string."""
    results = get_embeddings([text], model=model, base_url=base_url)
    if results:
        return results[0]
    return []


def generate_chat_response(
    messages: List[Dict[str, str]],
    model: str = DEFAULT_CHAT_MODEL,
    base_url: str = OLLAMA_BASE_URL,
    temperature: float = 0.1,
    timeout: int = 90
) -> str:
    """
    Send messages to Ollama /api/chat and return the generated text response.
    """
    payload = {
        "model": model,
        "messages": messages,
        "stream": False,
        "options": {
            "temperature": temperature
        }
    }
    try:
        resp = requests.post(f"{base_url}/api/chat", json=payload, timeout=timeout)
        if resp.status_code == 200:
            data = resp.json()
            return data.get("message", {}).get("content", "").strip()
        else:
            raise RuntimeError(f"Ollama chat error (HTTP {resp.status_code}): {resp.text}")
    except requests.exceptions.ConnectionError:
        raise ConnectionError("Ollama is not running. Start Ollama and try again.")
    except Exception as e:
        raise RuntimeError(f"Chat generation error: {e}")
