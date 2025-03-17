import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,  # Change to DEBUG for more details
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    handlers=[
        logging.FileHandler("handoff_eval.log"),  # Log to file
        logging.StreamHandler(),  # Log to console
    ],
)

# Create a logger for the package
logger = logging.getLogger("handoff_eval")
# Silence httpx logs (set to WARNING level to avoid excessive output)
logging.getLogger("httpx").setLevel(logging.WARNING)
