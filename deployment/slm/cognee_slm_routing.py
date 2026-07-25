"""Route cognee's entity extraction to a per-entity-type LoRA adapter served by
LoRAX/vLLM on the T4.

Scaffold only — see epic https://github.com/pajoma/cognee/issues/8 and
integration sub-issue https://github.com/pajoma/cognee/issues/13.

Mechanism: cognee resolves LLM settings through the `llm_config` ContextVar
(see cognee/infrastructure/llm/config.py:get_llm_context_config). Any LLM call
made inside an async context where that ContextVar is set uses those settings.
Entity extraction is NOT wrapped in a pipeline_stage, so we set the ContextVar
directly around the call.

Each adapter is selected by putting its name in `llm_model` — the custom/openai
provider passes that straight through as the OpenAI `model` field, which is how
both LoRAX and vLLM pick the adapter.
"""

import os
from contextlib import contextmanager

from cognee.context_global_variables import llm_config as llm_config_ctx
from cognee.infrastructure.llm.config import get_llm_context_config

# Base endpoint of the multi-LoRA server (LoRAX :8080, vLLM :8000).
SLM_ENDPOINT = os.environ.get("SLM_ENDPOINT", "http://localhost:8080/v1")
# Local server typically needs no real key; set SLM_API_KEY if yours does.
SLM_API_KEY = os.environ.get("SLM_API_KEY", "YOUR_API_KEY_HERE")


@contextmanager
def use_adapter(adapter_name: str):
    """Route every LLM call in this block to `adapter_name` on the SLM server.

    Example:
        with use_adapter("drug"):
            entities = await extractor.extract_entities(text)
    """
    base = get_llm_context_config()
    routed = base.model_copy(
        update={
            "llm_provider": "custom",       # OpenAI-compatible passthrough
            "llm_model": adapter_name,      # <- selects the LoRA adapter
            "llm_endpoint": SLM_ENDPOINT,
            "llm_api_key": SLM_API_KEY,
        }
    )
    token = llm_config_ctx.set(routed)
    try:
        yield
    finally:
        llm_config_ctx.reset(token)


# --- usage sketch -----------------------------------------------------------
# Pick the adapter by the entity type you're extracting for a given dataset/chunk.
async def extract_for_type(extractor, text: str, entity_type: str):
    adapter = {
        "DRUG": "drug",
        "GENE": "gene",
        "DOSAGE": "dosage",
    }[entity_type]
    with use_adapter(adapter):
        return await extractor.extract_entities(text)


# Notes:
#  - Keep heavy tasks (graph extraction, completion) OFF this context — they stay
#    on the big model. Only wrap the narrow, structured entity-extraction calls.
#  - Set STRUCTURED_OUTPUT_FRAMEWORK=litellm_native (JSON fallback + self-correct)
#    rather than the default instructor, since local models trip instructor's
#    schema/tool-call modes (see ollama/generic adapters).
#  - Better long-term: close the routing gap in issue #13 so entity extraction is
#    wrapped in its own pipeline_stage and driven by llm_<stage>_* env vars
#    instead of this manual ContextVar wrapper.
