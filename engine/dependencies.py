from typing import Optional

from core.fm import FM
from core.generator import Generator
from core.llm import LLMClient
from core.output import Output
from core.promptline import Promptline
from utils.ctx import SystemContext
from utils.logger import Logger
from config import settings

class Dependencies:
    """Dependency Injection Container."""
    
    def __init__(self):
        self._logger: Optional[Logger] = None
        self._features = None
        self._llm_client: Optional[LLMClient] = None
        self._promptline: Optional[Promptline] = None
        self._output: Optional[Output] = None
        self._generator: Optional[Generator] = None
        self._system_ctx: Optional[SystemContext] = None
        
    @property
    def logger(self) -> Logger:
        if not self._logger:
            self._logger = Logger()
        return self._logger
        
    @property
    def features(self):
        if not self._features:
            self._features = FM().features
        return self._features
        
    @property
    def llm_client(self) -> LLMClient:
        if not self._llm_client:
            self._llm_client = LLMClient(
                logger=self.logger,
                openai_key=settings.OPENAI_API_KEY,
                openrouter_key=settings.OPENROUTER_API_KEY,
                ollama_base_url=settings.OLLAMA_BASE_URL
            )
        return self._llm_client
        
    @property
    def promptline(self) -> Promptline:
        if not self._promptline:
            self._promptline = Promptline(
                llm_client=self.llm_client, 
                logger=self.logger
            )
        return self._promptline
        
    @property
    def output(self) -> Output:
        if not self._output:
            self._output = Output(logger=self.logger)
        return self._output
        
    @property
    def generator(self) -> Generator:
        if not self._generator:
            self._generator = Generator(
                llm=self.llm_client,
                promptline=self.promptline,
                logger=self.logger
            )
        return self._generator

    @property
    def system_ctx(self) -> SystemContext:
        if not self._system_ctx:
            self._system_ctx = SystemContext()
        return self._system_ctx

dependencies = Dependencies()

def get_dependencies() -> Dependencies:
    return dependencies