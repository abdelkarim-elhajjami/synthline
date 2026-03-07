"""Main entry point for the Synthline application."""

from gui import GUI

if __name__ == "__main__":
    DEEPSEEK_API_KEY = "your_key_here"
    OPENAI_API_KEY = "your_key_here"
    
    gui = GUI(DEEPSEEK_API_KEY, OPENAI_API_KEY)
    gui.run()