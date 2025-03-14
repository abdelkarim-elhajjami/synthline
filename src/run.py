import os
import sys
import argparse
from gui import GUI
from logger import Logger

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Synthline - Synthetic Data Generation Tool")
    
    parser.add_argument(
        "--deepseek-key", 
        default=os.environ.get("DEEPSEEK_API_KEY", "your_key_here"),
        help="DeepSeek API key (default: environment variable DEEPSEEK_API_KEY)"
    )
    
    parser.add_argument(
        "--openai-key", 
        default=os.environ.get("OPENAI_API_KEY", "your_key_here"),
        help="OpenAI API key (default: environment variable OPENAI_API_KEY)"
    )
    
    return parser.parse_args()


def main():
    """Main entry point for the application."""
    args = parse_args()
    
    logger = Logger()
    
    try:
        gui = GUI(args.deepseek_key, args.openai_key, logger=logger)
        gui.run()
        
    except Exception as e:
        print(f"Error starting Synthline: {e}")
        logger.log_error(str(e), "main")
        sys.exit(1)

if __name__ == "__main__":
    main()