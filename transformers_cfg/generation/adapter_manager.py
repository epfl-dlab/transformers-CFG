"""Adapter loading and invocation for grammar-constrained generation."""

import importlib
import logging

logger = logging.getLogger(__name__)


class AdapterManager:
    """
    Manages loading and invocation of generation adapters.

    Adapters allow integration with different generation frameworks beyond
    the default transformers library. This manager handles dynamic loading
    of adapter modules and provides a unified interface for adapter invocation.

    Attributes:
        adapter_name: Name of the adapter to use
        adapter_func: Loaded adapter function (None if using default transformers)

    Supported adapters:
        - "transformers": Default transformers behavior (no adapter)
        - "llama-cpp-python" / "llama_cpp_python": llama.cpp integration
        - Custom adapters: Can be added in transformers_cfg/adapters/

    Example:
        >>> manager = AdapterManager("llama-cpp-python", logits_processor)
        >>> result = manager.invoke_adapter(input_ids, scores)
    """

    def __init__(self, adapter_name: str, processor):
        """
        Initialize adapter manager.

        Args:
            adapter_name: Name of the adapter to load
            processor: LogitsProcessor instance to pass to the adapter

        Raises:
            Warning: If adapter loading fails (falls back to default behavior)
        """
        # Normalize adapter name (create alias for llama-cpp-python)
        if adapter_name == "llama-cpp-python":
            adapter_name = "llama_cpp_python"

        self.adapter_name = adapter_name
        self.adapter_func = None

        # Load adapter if not using default transformers behavior
        if adapter_name != "transformers":
            self._load_adapter(adapter_name, processor)

    def _load_adapter(self, adapter_name: str, processor):
        """
        Dynamically load adapter module and function.

        Args:
            adapter_name: Name of the adapter module to load
            processor: LogitsProcessor to pass to the adapter

        Raises:
            Warning: If loading fails (logs warning and continues)
        """
        try:
            # Import the adapter module
            adapter_module = importlib.import_module(
                f"transformers_cfg.adapters.{adapter_name}"
            )
            # Get the adapter function with the same name as the module
            adapter_func = getattr(adapter_module, adapter_name)
            # Create the adapter function with this processor
            self.adapter_func = adapter_func(processor)
        except (ImportError, AttributeError) as e:
            logger.warning(
                f"Failed to load adapter '{adapter_name}': {str(e)}. "
                f"Falling back to default transformers behavior."
            )
            self.adapter_func = None

    def has_adapter(self) -> bool:
        """
        Check if an adapter is loaded.

        Returns:
            True if an adapter function is loaded, False otherwise
        """
        return self.adapter_func is not None

    def invoke_adapter(self, input_ids, scores):
        """
        Invoke the loaded adapter function.

        Args:
            input_ids: Input token IDs
            scores: Logits scores

        Returns:
            Processed logits from the adapter

        Raises:
            ValueError: If no adapter is loaded
        """
        if self.adapter_func is None:
            raise ValueError(
                f"No adapter loaded. Cannot invoke adapter for '{self.adapter_name}'."
            )

        return self.adapter_func(input_ids, scores)
