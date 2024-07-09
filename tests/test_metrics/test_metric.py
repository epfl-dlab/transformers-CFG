import math
from unittest import TestCase
import torch
from transformers_cfg.metrics import ConstrainedDecodingMetric
from transformers_cfg.metrics.metrics import ConstrainedDecodingMetricOutput


class TestConstrainedDecodingMetric(TestCase):
    def setUp(self):
        """Prepare resources for the tests."""
        self.metric = ConstrainedDecodingMetric()
        # mock output from a huggingface model
        # vocab size = 2, batch size = 1, sequence length = 2
        # we assume that the logits are the same for all two tokens
        self.logits = (torch.tensor([[0.1, 0.1], [0.1, 0.1]]),)
        # we assume the constrained decoding rejects the second token in step 1
        # and the first token in step 2
        self.scores = (torch.tensor([[0.1, -math.inf], [-math.inf, 0.1]]),)
        # we assume the final output tokens are [0, 1], i.e. token index 0 is selected in step 1
        # and token index 1 is selected in step 2, which is coherent with the constrained scores above
        self.sequences = torch.tensor([[0, 1]]).T

        # Output structure mimicking a huggingface model generation output
        self.hf_output = {
            "scores": self.scores,
            "logits": self.logits,
            "sequences": self.sequences,
        }

    def test_output_class(self):
        """Test the computation from model output."""
        result = self.metric.compute_from_model_output(self.hf_output)

        # Check if the result is an instance of the expected data class
        self.assertIsInstance(result, ConstrainedDecodingMetricOutput)

        # Check if the result contains expected keys
        self.assertTrue(hasattr(result, "original_token_probs"))
        self.assertTrue(hasattr(result, "renormalised_token_probs"))
        self.assertTrue(hasattr(result, "total_rejection_prob_gain"))
        self.assertTrue(hasattr(result, "total_rejection_entropy_gain"))

        # Check if metadata contains expected items
        self.assertIn("num_steps", result.metadata)
        self.assertIn("num_batches", result.metadata)

    def test_compute_result(self):
        """Test the computation from model output."""
        result = self.metric.compute_from_model_output(self.hf_output)

        # Exact equality checks
        # The output sequence is [0, 1]
        # we know that the token 0 at step 1 has an equal logit, so the probability is 0.5
        # same for token 1 at step 2
        self.assertTrue(
            torch.equal(result.original_token_probs, torch.tensor([[0.5, 0.5]]))
        )
        # as the constrained decoding rejects the second token in step 1 and the first token in step 2
        # the probability of the selected token is 1.0 in both steps
        self.assertTrue(
            torch.equal(result.renormalised_token_probs, torch.tensor([[1.0, 1.0]]))
        )
        # the rejection probability gain is 0.5 for both steps
        self.assertTrue(
            torch.equal(result.total_rejection_prob_gain, torch.tensor([[0.5, 0.5]]))
        )
        # the rejection entropy gain is simply -log_2(0.5) = 1.0 for both steps
        self.assertTrue(
            torch.equal(result.total_rejection_entropy_gain, torch.tensor([[1.0, 1.0]]))
        )
