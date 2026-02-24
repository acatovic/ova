import logging

try:
    import torch  # type: ignore
except Exception:  # pragma: no cover - torch may be absent in MLX installs
    torch = None


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
)


logger = logging.getLogger("ova")


def get_device():
    if torch is None:
        logger.warning("Torch not installed. Falling back to CPU.")
        return "cpu"

    if torch.cuda.is_available():
        device = "cuda:0"
        logger.info(f"CUDA available. Using {device}")
        return device

    logger.warning("CUDA not available. Falling back to CPU (this will be slower)")
    return "cpu"
