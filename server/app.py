"""
Misinformation Patrol Environment - Server Application
Entry point for the OpenEnv deployment on Hugging Face Spaces
"""

import sys
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Add parent directory to path to access inference.py
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from inference import create_demo
except ImportError:
    logger.warning("inference.py not found in parent directory")
    create_demo = None


def main():
    """Main entry point for the server"""
    logger.info("Starting Misinformation Patrol Environment Server")
    
    if create_demo is None:
        logger.error("Failed to import create_demo from inference.py")
        raise ImportError("inference.py must define create_demo() function")
    
    # Create and launch the demo
    demo = create_demo()
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True,
    )


if __name__ == "__main__":
    main()
