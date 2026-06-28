import os
import logging
from dataclasses import dataclass
from pathlib import Path
from dotenv import load_dotenv

# Load .env file variables into os.environ
load_dotenv()

# Hugging Face authentication.
# The HF libraries (sentence-transformers, transformers, huggingface_hub) all read the
# token from the HF_TOKEN environment variable. This project historically stored it under
# a custom name (HUGGINGFACE_HUB_KEY) and only passed it to the tokenizer load in
# fed_narrative_rag._chunk_document -- so the SentenceTransformer model load in _embed
# (which runs on every query via find_fed_narrative) stayed unauthenticated and emitted
# the "set a HF_TOKEN" rate-limit warning. Bridge whatever name is set onto HF_TOKEN once,
# here, so EVERY HF Hub access is authenticated. The token value is never logged.
_hf_token = (
    os.getenv("HF_TOKEN")
    or os.getenv("HUGGINGFACE_HUB_KEY")
    or os.getenv("HUGGINGFACEHUB_API_TOKEN")
)
if _hf_token:
    os.environ["HF_TOKEN"] = _hf_token

LOGGING_LEVELS = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
LOG_FILE_NAME = "sector_rotation_agent.log"

@dataclass(frozen=True)
class AppConfig:
    # -------------- MODEL CONFIGURATION ----------------------------------------
    # ModelLocations enum
        # These will determine which model services are invoked below
    
    model_location = "cloud_only"    # <- valid Values: {"local_only", "mixed", "cloud_only"}


    """
    Another alternative
    model_roles = {
        "generative": ("anthropic",    "claude-sonnet-4-5-20250929"),  # hypotheses, summary
        "judging":    ("ollama-local", "gemma4:12b"),                  # critic, decompose
    }
    """

    # Setting this up for a couple different options to make it easy to switch.
    # Mistral is cheaper and performs well on reasoning tasks and I can run locally for testing
    # Claude Sonnet is more knowledgeable about the world as of mid-2026
 
    # Local Ollama only: the context window (num_ctx). Cloud providers manage their own
    # context and ignore this. gemma4:12b needs more than Ollama's small default, or long
    # generations (e.g. the executive summary) run to the context wall and return empty.
    ## --- Anthropic (cloud) ---
    #cloud_model_service = "anthropic"
    #cloud_model = "claude-sonnet-4-5-20250929"
    ## --- Ollama (local) ---
    local_model_service = "ollama-local"
    local_model = "gemma4:12b"
    ## --- Hugging Face (cloud) ---
    #cloud_model_service = "huggingface"
    #cloud_model = "Qwen/Qwen2.5-7B-Instruct"
    ## --- OpenRouter (cloud) ---
    openrouter_url = "https://openrouter.ai/api/v1"
    cloud_model_service = "open_router"
    cloud_model = "anthropic/claude-opus-4.8"
    
    # Global completion ceiling applied to EVERY model call (Ollama: num_predict). It is a
    # max, not a target -- well-behaved models stop well short, and the JSON critic/decompose
    # calls use a fraction of it. Do NOT lower it back toward ~1000: the ToT hypothesis
    # fan-out returns a JSON ARRAY of candidate regimes WITH rationale strings and needs the
    # headroom, or it truncates mid-array (finish=length) and generate_hypotheses can't parse
    # it. (A too-small cap is also what silently dropped the executive summary on local.)
    model_num_ctx = 16384
    default_max_tokens = 4096
    default_temperature = 0.7  # consider a higher temperature for more diverse outputs
    
    ##  ----- Embeddings  -----------
    embedding_model = "all-MiniLM-L6-v2"
    embedding_model_max_tokens = int(256)
    # ^-- Note, if this is changed, re-seeding is required for historical_analogs and fed_narrative_rag 
    # Text embedding model. MUST be identical on the seed and query sides (see SYMMETRY
    # above). all-MiniLM-L6-v2 is a fast, 384-dim sentence-transformers default that fits
    # the 8 GB local-GPU target; swap deliberately, and re-seed if you do.
    # TODO: upgrade to BAAI/bge-small-en-v1.5
    #           - Requires "Represent this sentence for searching relevant passages: " prepended to the query
    #       could also use local ollama nomic-embed-text
    
    
    # -------------- LOGGING CONFIGURATION -------------------------------------
    logging_level: str = os.getenv("LOGGING_LEVEL", "INFO").upper()
    if logging_level not in LOGGING_LEVELS:
        raise RuntimeError(f"Invalid LOG_LEVEL: {logging_level}")

    log_file_path = os.getenv("LOG_FILE_PATH")
    log_file_dir = os.getenv("LOG_FOLDER")
    if log_file_path == "": # use current directory
        if log_file_dir == "":
            log_file = Path(__file__).resolve().parent.parent.parent / LOG_FILE_NAME  # src/sector_rotation_agent/main.py, logs should be peer of src
        else:
            log_file = Path(__file__).resolve().parent.parent.parent / Path(log_file_dir) / LOG_FILE_NAME  # type: ignore
    else:  # use the given directory
        log_file = Path(log_file_path) / LOG_FILE_NAME# pyright: ignore[reportArgumentType]
    
    #make sure the target log folder & file exist
    log_file.parent.mkdir(parents=True, exist_ok=True)
    log_file.touch(exist_ok=True)

    
    log_level = getattr(logging, logging_level, None)
    if not isinstance(log_level, int):
        raise RuntimeError(f"Invalid LOG_LEVEL: {logging_level}")

    logging.basicConfig(
        filename=log_file,
        level=log_level,
        format='%(asctime)s | %(levelname)s | %(filename)s:%(funcName)s:%(lineno)d | %(message)s'
        )

    # App Settings
    #db_url: str = os.getenv("DATABASE_URL", "sqlite:///default.db")

##  ----------------------------------------------------------------
# Create a single global configuration object
settings = AppConfig()

# Quiet noisy third-party DEBUG logging. At LOGGING_LEVEL=DEBUG the markdown-pdf export
# (its markdown-it parser) and the HTTP/PDF stacks emit hundreds of per-token / per-request
# lines that otherwise drown the app's own DEBUG output in the combined log. None of these
# carry app logic (our logs live under the sector_rotation_agent.* namespace), so cap the
# known-chatty libraries at WARNING.
for _noisy in ("markdown_it", "markdown_pdf", "httpx", "httpcore", "urllib3", "PIL", "fontTools"):
    logging.getLogger(_noisy).setLevel(logging.WARNING)

# Confirm configuration once at import, now that logging.basicConfig has run. Logs the
# operational essentials only -- never secrets/API keys: model service/name, locality,
# embedding model, log level, and where logs are written.
logging.getLogger(__name__).info(
    "Config loaded: cloud_service=%s, cloud_model=%s, local_service=%s, local_model=%s, embedding=%s, level=%s, log_file=%s",
    settings.cloud_model_service, settings.cloud_model, settings.local_model_service,
    settings.local_model, settings.embedding_model, settings.logging_level, settings.log_file,
)