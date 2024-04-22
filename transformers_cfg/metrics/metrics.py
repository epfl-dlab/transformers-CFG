import datasets
from dataclasses import dataclass, field
import torch
import pandas as pd

_DESCRIPTION = "TODO"

_KWARGS_DESCRIPTION = "TODO"

_CITATION = ""

# https://github.com/huggingface/datasets/blob/main/metrics/accuracy/accuracy.py


@dataclass
class ConstrainedDecodingMetricOutput:
    logits: torch.Tensor
    scores: torch.Tensor
    sequences: torch.Tensor

    original_token_probs: torch.Tensor
    renormalised_token_probs: torch.Tensor
    total_rejection_prob_gain: torch.Tensor
    total_rejection_entropy_gain: torch.Tensor
    metadata: dict = field(default_factory=dict)
    df: dict = field(init=False, repr=False)  # Dictionary to store DataFrames

    def __post_init__(self):
        # Assuming original_token_probs is representative of tensor dimensions for steps and batches
        self.num_steps, self.num_batches = self.original_token_probs.shape
        # Store these dimensions in metadata for easy access
        self.metadata["num_steps"] = self.num_steps
        self.metadata["num_batches"] = self.num_batches
        self.df = self._to_df()

    def _to_df(self):
        """Converts tensors to a dictionary of pandas DataFrames for easier analysis, preserving original tensor shape."""
        step_labels = [f"Step {i+1}" for i in range(self.num_steps)]
        batch_labels = [f"Batch {i+1}" for i in range(self.num_batches)]

        # Convert tensors to DataFrames with labeled rows and columns
        df_dict = {
            "original_token_probs": pd.DataFrame(
                self.original_token_probs.numpy(),
                index=step_labels,
                columns=batch_labels,
            ),
            "renormalised_token_probs": pd.DataFrame(
                self.renormalised_token_probs.numpy(),
                index=step_labels,
                columns=batch_labels,
            ),
            "total_rejection_prob_gain": pd.DataFrame(
                self.total_rejection_prob_gain.numpy(),
                index=step_labels,
                columns=batch_labels,
            ),
            "total_rejection_entropy_gain": pd.DataFrame(
                self.total_rejection_entropy_gain.numpy(),
                index=step_labels,
                columns=batch_labels,
            ),
            "sequences": pd.DataFrame(
                self.sequences.numpy(), index=step_labels, columns=batch_labels
            ),
        }
        return df_dict

    def __repr__(self):
        # Collecting all attribute names except for the tensors and metadata contents
        attributes_info = ", ".join(
            f"{k}" for k in self.__dict__.keys() if k != "metadata"
        )

        # Creating a string for metadata with more detail
        meta_info = ", ".join(f"{k}: {v}" for k, v in self.metadata.items())

        # Combining everything into one representation string
        return f"SequenceLikelihoodResult({attributes_info}, metadata: {{{meta_info}}})"

    def to_csv(self, directory):
        """Saves DataFrames to CSV files."""
        for key, df in self.df.items():
            df.to_csv(f"{directory}/{key}.csv", index=False)

    @classmethod
    def from_csv(cls, directory):
        """Loads DataFrames from CSV files and recreates the SequenceLikelihoodResult object using simplified logic."""

        keys = [
            "original_token_probs",
            "renormalised_token_probs",
            "total_rejection_prob_gain",
            "total_rejection_entropy_gain",
            "sequences",
        ]
        tensors_dict = {}

        # Helper function to load CSV and convert to tensor
        def load_tensor(filename):
            df = pd.read_csv(f"{directory}/{filename}.csv")
            return torch.tensor(df.values)

        # Load and convert all tensors
        for key in keys:
            tensors_dict[key] = load_tensor(key)

        # Assume metadata is stored correctly or can be inferred
        num_steps, num_batches = tensors_dict["original_token_probs"].shape[:2]
        metadata = {"num_steps": num_steps, "num_batches": num_batches}

        return cls(**tensors_dict, metadata=metadata)


class RejectProbDropFromConstraint(datasets.Metric):
    def _info(self):
        return datasets.MetricInfo(
            description=_DESCRIPTION,
            citation=_CITATION,
            inputs_description=_KWARGS_DESCRIPTION,
            features=datasets.Features(
                {
                    "scores": datasets.Sequence(
                        datasets.Sequence(datasets.Value("float"))
                    ),
                    # the scores we expect is Tuple[torch.Tensor]
                    "logits": datasets.Sequence(
                        datasets.Sequence(datasets.Value("float"))
                    ),
                }
            ),
            reference_urls=[],
        )

    def compute_from_output(self, hf_output):
        scores = hf_output["scores"]
        logits = hf_output["logits"]

        return super().compute(scores=scores, logits=logits)

    def _compute(self, scores, logits):
        """
        The input scores and logits are 3D tensor of shape (n_steps, n_batch, n_tokens)
        """
        scores = torch.tensor(scores)
        logits = torch.tensor(logits)
        probs = torch.nn.functional.softmax(logits, dim=-1)
        accept_mask = scores != float("-inf")
        accept_prob = (probs * accept_mask).sum(dim=-1)
        reject_prob = 1 - accept_prob
        return reject_prob


class ConstrainedDecodingMetric(datasets.Metric):
    def _info(self):
        return datasets.MetricInfo(
            description=_DESCRIPTION,
            citation=_CITATION,
            inputs_description=_KWARGS_DESCRIPTION,
            features=datasets.Features(
                {
                    "scores": datasets.Sequence(
                        datasets.Sequence(datasets.Value("float"))
                    ),
                    "logits": datasets.Sequence(
                        datasets.Sequence(datasets.Value("float"))
                    ),
                    "sequences": datasets.Sequence(datasets.Value("int32")),
                }
            ),
            reference_urls=[],
        )

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.underlying_metric = RejectProbDropFromConstraint()

    def compute_from_model_output(self, hf_output) -> ConstrainedDecodingMetricOutput:
        scores = hf_output["scores"]
        logits = hf_output["logits"]
        sequences = hf_output["sequences"]

        # the sequences by default contains also the input_ids, we only want the generated tokens, i.e. last n_steps tokens
        n_generated_tokens = len(logits)
        # get the last n_generated_tokens of the sequences
        sequences = sequences[:, -n_generated_tokens:]

        # N.B. the sequences needs to be transposed because the first dimension of the sequences is the batch size
        # and the first dimension of the scores and logits is the number of steps
        # so we need to transpose the sequences to match the shape of the scores and logits
        return super().compute(scores=scores, logits=logits, sequences=sequences.T)

    def _compute(self, scores, logits, sequences) -> ConstrainedDecodingMetricOutput:
        """
        Args:
            scores: shape (n_steps, n_batch, n_vocab)
            logits:  shape (n_steps, n_batch, n_vocab)
            sequences:  shape (n_steps, n_batch)

        N.B.
        The Output From HF has the following shape:
        scores: (n_steps, n_batch, n_vocab)
        logits: (n_steps, n_batch, n_vocab)
        sequences: (n_batch, n_steps) # here we have to transpose the sequences

        The reason why we don't use the original sequence is because datasets.Metric requires the input's first dimension
        to be the same, batch size.

        Returns:
            original_token_probs: shape (n_steps, n_batch), same as the input sequences

        """

        # check the shape of the three inputs
        assert len(scores) == len(logits) == len(sequences), (
            f"The first dimension of the three inputs "
            f"must be the same, but got len(scores) "
            f"= {len(scores)}, len(logits) = "
            f"{len(logits)}, len(sequences) = "
            f'{len(sequences)}"'
            f"This is probably because the sequences "
            f"contains the input_ids, we only want the "
            f"generated tokens,"
            f"i.e. last n_steps tokens"
        )

        scores = torch.tensor(scores)
        logits = torch.tensor(logits)

        total_reject_prob_gain = self.underlying_metric.compute(
            scores=scores, logits=logits
        )
        # compute the information total_reject_entropy_gain of (1 - total_reject_prob_gain, total_reject_prob_gain)
        total_reject_entropy_gain = -total_reject_prob_gain * torch.log2(
            total_reject_prob_gain
        ) - (1 - total_reject_prob_gain) * torch.log2(1 - total_reject_prob_gain)

        sequences = torch.tensor(sequences)
        # original logit scores
        original_probs = torch.nn.functional.softmax(logits, dim=-1)
        renormalised_probs = torch.nn.functional.softmax(scores, dim=-1)
        # get the original_probs of the sequence
        original_token_probs = torch.gather(
            original_probs, 2, sequences.unsqueeze(2)
        ).squeeze(2)
        # get the scores of the sequence
        renormalised_token_probs = torch.gather(
            renormalised_probs, 2, sequences.unsqueeze(2)
        ).squeeze(2)

        # compute the likelihood of the original sequence

        # Create an instance of the data class
        result = ConstrainedDecodingMetricOutput(
            logits=logits,
            scores=scores,
            sequences=sequences,
            original_token_probs=original_token_probs,
            renormalised_token_probs=renormalised_token_probs,
            total_rejection_prob_gain=total_reject_prob_gain,
            total_rejection_entropy_gain=total_reject_entropy_gain,
            metadata={"Placeholder": "xx"},
        )

        return result


if __name__ == "__main__":
    scores = (
        torch.tensor([[0.1, 0.2, 0.3, 0.4], [0.1, 0.2, 0.3, 0.4]]),
        torch.tensor([[0.1, 0.2, 0.3, 0.4], [0.1, 0.2, 0.3, 0.4]]),
    )
    logits = (
        torch.tensor([[0.1, 0.2, 0.3, 0.4], [0.1, 0.2, 0.3, 0.4]]),
        torch.tensor([[0.1, 0.2, 0.3, 0.4], [0.1, 0.2, 0.3, 0.4]]),
    )
    # scores = (torch.tensor([[2., 2., 2., 2.]]),)
    # set scores[1] to -inf and renormalize
    scores[0][0][1] = float("-inf")
    # logits = (torch.tensor([[2., 2., 2., 2.]]),)

    sequences = torch.tensor([[0, 1], [2, 3]]).T
    # accept_prob_df = get_constrained_decoding_probability_df(scores, logits)
    # accept_prob = accept_prob_df.iloc[0, 0]
    # assert accept_prob == 0.75, f"accept_prob = {accept_prob}, expected 0.75"

    metric = ConstrainedDecodingMetric()

    result = metric.compute(scores=scores, logits=logits, sequences=sequences.T)

    df_dict = result._to_df()

    ############################################

    from transformers import AutoTokenizer, AutoModelForCausalLM

    model_id = "gpt2"

    # Load model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_id)
    model.generation_config.pad_token_id = model.generation_config.eos_token_id

    bad_words_ids = [
        tokenizer.encode(" http"),
        tokenizer.encode(" shopping"),
    ]

    # Generate
    input_ids = tokenizer(
        [
            "This is a valid json string for http request:",
            "This is a valid json string for shopping cart:",
        ],
        add_special_tokens=False,
        return_tensors="pt",
        padding=True,
    )["input_ids"]
    output = model.generate(
        input_ids,
        max_length=30,
        bad_words_ids=bad_words_ids,
        repetition_penalty=1,
        num_return_sequences=1,
        return_dict_in_generate=True,
        output_scores=True,
        output_logits=True,
    )

    # decode output
    generations = tokenizer.batch_decode(output["sequences"], skip_special_tokens=True)
    print(generations)

    result = metric.compute_from_model_output(output)

    # result.to_csv("results")
