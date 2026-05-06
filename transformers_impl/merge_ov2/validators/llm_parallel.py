import torch

from transformers import AutoModelForCausalLM, AutoTokenizer, logging

from ..utils import cosine_similarity


logger = logging.get_logger(__name__)


def run(model, llm_path: str, sample_text: str):
    device = next(model.model.language_model.parameters()).device
    dtype = next(model.model.language_model.parameters()).dtype

    orig = AutoModelForCausalLM.from_pretrained(llm_path, torch_dtype=dtype, low_cpu_mem_usage=True).to(device).eval()
    tok = AutoTokenizer.from_pretrained(llm_path, use_fast=True)
    inputs = tok(sample_text, return_tensors="pt").to(device)

    with torch.no_grad():
        h = model.model.language_model(
            input_ids=inputs["input_ids"], attention_mask=inputs.get("attention_mask")
        ).last_hidden_state
        merged_logits = model.lm_head(h)
        orig_logits = orig(**inputs).logits

    diff = (merged_logits - orig_logits).abs().mean().item()
    sim = cosine_similarity(merged_logits.flatten(0, 1).cpu(), orig_logits.flatten(0, 1).cpu())
    logger.info(f"LLM diff={diff:.6f} cos={sim:.6f}")
    if not (sim > 0.99 and diff < 5e-2):
        raise ValueError(f"LLM mismatch (diff={diff}, cos={sim})")
    logger.info("LLM consistency OK")

    del orig
    torch.cuda.empty_cache()
