import logging
import numpy as np
import torch

logger = logging.getLogger(__name__)


def llama_cpp_python(processor):
    """
    Adapter function for llama-cpp-python.
    
    Args:
        processor: A GrammarConstrainedLogitsProcessor instance
        
    Returns:
        A function that can be used as a logits processor with llama-cpp-python
    """
    reinit_attempts = 0
    reinit_max = 3
    accumulated_tokens = []
    
    def _force_eos(scores):
        eos_token = processor.grammar_constraint.tokenizer.eos_token_id
        logger.warning(f"Forcing EOS token: {eos_token}")
        mask = torch.full_like(scores, fill_value=-float("inf"))
        if scores.dim() == 2:
            mask[:, eos_token] = 0
        else:
            mask[eos_token] = 0
        return mask
    
    def adapter_func(input_ids, scores):
        nonlocal reinit_attempts, accumulated_tokens
        
        # Normalize input_ids to a list of token sequences
        if np.isscalar(input_ids):
            input_ids = [int(input_ids)]
        elif isinstance(input_ids, np.ndarray):
            input_ids = input_ids.tolist()
        elif isinstance(input_ids, list):
            input_ids = [int(i) if isinstance(i, np.generic) else i for i in input_ids]
        elif isinstance(input_ids, np.generic):
            input_ids = [int(input_ids)]
        
        # Ensure we have a batch (list of token lists)
        if input_ids and isinstance(input_ids[0], int):
            input_ids = [input_ids]
            
        # Convert scores to a torch.Tensor if needed
        if not isinstance(scores, torch.Tensor):
            scores = torch.tensor(scores)
            
        # Ensure scores is 2D: [batch, vocab_size]
        if scores.dim() == 1:
            scores = scores.unsqueeze(0)
            
        # Track tokens for debugging
        if len(input_ids[0]) > len(accumulated_tokens):
            new_token = input_ids[0][-1]
            accumulated_tokens.append(new_token)
            try:
                token_text = processor.grammar_constraint.tokenizer.decode([new_token])
                logger.debug(f"Added token: {new_token} ({token_text})")
            except Exception:
                logger.debug(f"Added token: {new_token} (cannot decode)")
                
        # Check for consistency: if the length of our input token sequence
        # does not match what the grammar expects, then reinitialize
        current_length = len(input_ids[0])
        if hasattr(processor.grammar_constraint, "last_size") and processor.grammar_constraint.last_size is not None:
            expected_length = processor.grammar_constraint.last_size + 1
            if current_length != expected_length:
                logger.warning(f"Length mismatch: current={current_length}, expected={expected_length}. Reinitializing.")
                processor.reset()
                reinit_attempts = 0
                
        try:
            processed_scores = processor.process_logits(input_ids, scores)
            reinit_attempts = 0
        except ValueError as e:
            error_msg = str(e)
            if "All stacks are empty" in error_msg:
                # Try to recover by reinitializing the grammar constraint
                if reinit_attempts < reinit_max:
                    logger.warning(f"Grammar constraint error: {error_msg}. Attempt {reinit_attempts+1}/{reinit_max} to recover.")
                    processor.reset()
                    reinit_attempts += 1
                    try:
                        processed_scores = processor.process_logits(input_ids, scores)
                    except ValueError as e2:
                        logger.error(f"Recovery failed: {str(e2)}")
                        processed_scores = _force_eos(scores)
                else:
                    # If reinitialization has already been attempted enough times,
                    # treat the output as complete and force EOS
                    logger.error(f"Max retries ({reinit_max}) exceeded. Current text: {processor.grammar_constraint.tokenizer.decode(accumulated_tokens)}")
                    processed_scores = _force_eos(scores)
            else:
                logger.error(f"Unexpected error: {error_msg}")
                raise e
                
        # Remove the batch dimension if present
        if processed_scores.dim() == 2 and processed_scores.size(0) == 1:
            processed_scores = processed_scores.squeeze(0)
        return processed_scores.detach().cpu().numpy()
    
    return adapter_func
