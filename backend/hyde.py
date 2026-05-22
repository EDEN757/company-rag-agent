import concurrent.futures
import logging
import re
import numpy as np
from openai import OpenAI
from embed import embed
from config import OLLAMA_HOST, LLM_MODEL

log = logging.getLogger(__name__)
_client: OpenAI | None = None
_pool = concurrent.futures.ThreadPoolExecutor(max_workers=2, thread_name_prefix="hyde-embed")

_PROMPT = (
    "/no_think Write one short passage (1-2 sentences) from a company document, email, or chat "
    "that would directly answer this question:\n\n{query}\n\n"
    "Write only the passage, no preamble."
)


def init_hyde():
    global _client
    _client = OpenAI(base_url=f"{OLLAMA_HOST}/v1", api_key="ollama")


def hyde_embed(query: str) -> list[float]:
    """Return a query embedding averaged with a hypothetical-document embedding.

    Uses a tiny num_ctx (512) so the LLM call is fast (~3-5 s on GPU).
    Falls back to the raw query embedding on any error.
    Only affects the dense/vector branch — BM25 still uses the original query text.
    """
    if _client is None:
        return embed(query)
    try:
        # Embed the raw query immediately — runs concurrently while the LLM generates
        # the hypothetical document (LLM call is the bottleneck at ~3-5s).
        q_future = _pool.submit(embed, query)

        resp = _client.chat.completions.create(
            model=LLM_MODEL,
            messages=[{"role": "user", "content": _PROMPT.format(query=query)}],
            max_tokens=80,
            temperature=0.7,
            extra_body={"options": {"num_ctx": 512, "think": False}},
        )
        raw = resp.choices[0].message.content or ""
        hypothetical = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()
        log.debug(f"HyDE hypothetical: {hypothetical[:120]}")
        if not hypothetical:
            log.warning("HyDE: empty hypothetical (model only emitted think tokens) — using raw query embedding.")
            return q_future.result(timeout=30.0)

        h_vec = np.array(embed(hypothetical), dtype=np.float32)
        q_vec = np.array(q_future.result(timeout=30.0), dtype=np.float32)

        avg = (q_vec + h_vec) / 2.0
        norm = np.linalg.norm(avg)
        if norm > 0:
            avg /= norm
        return avg.tolist()
    except Exception as e:
        log.warning(f"HyDE failed ({e}) — using raw query embedding.")
        return embed(query)
