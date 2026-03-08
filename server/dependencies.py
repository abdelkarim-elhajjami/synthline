from typing import Optional
from pathlib import Path
import yaml

from synthline.core.align_scorer import AlignScorer
from synthline.core.align_verifier import AlignVerifier
from synthline.core.fm_parser import FM
from synthline.core.generator import Generator
from synthline.core.llm import LLMClient
from synthline.core.pace import PACE
from synthline.core.promptline import Promptline
from utils.ctx import SystemContext
from synthline.utils.logger import Logger
from settings import settings

from fastapi import Header

_global_system_ctx = SystemContext()

class Dependencies:
    """Dependency Injection Container (Session Scoped)."""
    def __init__(self, session_id: Optional[str] = None):
        self.session_id = session_id
        self._logger: Optional[Logger] = None
        self._fm: Optional[FM] = None
        self._glossary = None
        self._features = None
        self._llm_client: Optional[LLMClient] = None
        self._promptline: Optional[Promptline] = None
        self._generator: Optional[Generator] = None
        self._pace: Optional[PACE] = None
        self._align_scorer: Optional[AlignScorer] = None
        self._align_verifier: Optional[AlignVerifier] = None
        self._system_ctx = _global_system_ctx
        
    @property
    def logger(self) -> Logger:
        if not self._logger:
            self._logger = Logger(debug_mode=settings.LOG_LEVEL.lower() == "debug")
        return self._logger
        
    @property
    def features(self):
        if not self._features:
            self._features = self.fm.to_dict()
        return self._features

    @property
    def fm(self) -> FM:
        if not self._fm:
            fm_path = self._resolve_fm_path()
            if not fm_path.exists():
                global_path = self._resolve_global_fm_path()
                if not global_path.exists():
                    raise FileNotFoundError(
                        f"No active feature model found for session {self.session_id}. "
                        "Upload an fm.xml via /api/features/upload."
                    )
                fm_path = global_path
            self._fm = FM(str(fm_path))
        return self._fm

    def _resolve_global_fm_path(self) -> Path:
        configured = settings.FM_XML_PATH or "config/uploaded_fm.xml"
        path = Path(configured)
        if path.is_absolute():
            return path
        cwd = Path.cwd() / path
        module = Path(__file__).resolve().parent / path
        return cwd if cwd.exists() else module

    def _resolve_fm_path(self) -> Path:
        if self.session_id:
            return Path.cwd() / "sessions" / self.session_id / "fm.xml"
        
        return self._resolve_global_fm_path()

    def _resolve_global_glossary_path(self) -> Path:
        configured = settings.GLOSSARY_PATH or "config/glossary.yaml"
        path = Path(configured)
        if path.is_absolute():
            return path
        cwd = Path.cwd() / path
        module = Path(__file__).resolve().parent / path
        return cwd if cwd.exists() else module

    def _resolve_glossary_path(self) -> Path:
        if self.session_id:
            return Path.cwd() / "sessions" / self.session_id / "glossary.yaml"
        return self._resolve_global_glossary_path()

    def reload_feature_model(self) -> None:
        self._fm = None
        self._features = None
        self._promptline = None
        self._generator = None

    def update_feature_model(self, xml_content: bytes) -> dict:
        if not xml_content:
            raise ValueError("Empty FM XML payload.")

        target_path = self._resolve_fm_path()
        target_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = target_path.with_name(f".{target_path.name}.upload.tmp")

        try:
            tmp_path.write_bytes(xml_content)
            FM(str(tmp_path))
            tmp_path.replace(target_path)
        finally:
            if tmp_path.exists():
                tmp_path.unlink()

        self.reload_feature_model()
        return self.features

    def update_glossary(self, yaml_content: bytes) -> dict:
        if not yaml_content:
            raise ValueError("Empty glossary payload.")

        target_path = self._resolve_glossary_path()
        replaced = target_path.exists()
        target_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = target_path.with_name(f".{target_path.name}.upload.tmp")

        try:
            tmp_path.write_bytes(yaml_content)
            data = yaml.safe_load(tmp_path.read_text(encoding="utf-8")) or {}
            if not isinstance(data, dict):
                raise ValueError("Glossary YAML must be a key/value mapping.")
            cleaned = {str(key): str(value) for key, value in data.items() if value is not None}
            tmp_path.write_text(yaml.safe_dump(cleaned, sort_keys=False), encoding="utf-8")
            tmp_path.replace(target_path)
        finally:
            if tmp_path.exists():
                tmp_path.unlink()

        self._glossary = None
        self._promptline = None
        self._generator = None
        return {"entries": len(cleaned), "replaced": replaced}
        
    @property
    def llm_client(self) -> LLMClient:
        if not self._llm_client:
            self._llm_client = LLMClient(
                logger=self.logger,
                openai_key=settings.OPENAI_API_KEY,
                openrouter_key=settings.OPENROUTER_API_KEY,
                ollama_base_url=settings.OLLAMA_BASE_URL,
                hf_token=settings.HF_TOKEN
            )
        return self._llm_client
        
    @property
    def promptline(self) -> Promptline:
        if not self._promptline:
            self._promptline = Promptline(fm=self.fm, glossary=self.glossary)
        return self._promptline

    @property
    def pace(self) -> PACE:
        if not self._pace:
            self._pace = PACE(llm_client=self.llm_client, logger=self.logger)
        return self._pace

    @property
    def glossary(self) -> dict:
        if self._glossary is None:
            glossary_path = self._resolve_glossary_path()
            if not glossary_path.exists() and self.session_id:
                global_path = self._resolve_global_glossary_path()
                glossary_path = global_path if global_path.exists() else glossary_path

            if not glossary_path.exists():
                self._glossary = {}
                return self._glossary

            try:
                data = yaml.safe_load(glossary_path.read_text(encoding="utf-8")) or {}
                if isinstance(data, dict):
                    self._glossary = {
                        str(key): str(value) for key, value in data.items() if value is not None
                    }
                else:
                    self._glossary = {}
            except Exception as exc:
                self.logger.log_error(
                    f"Failed to load glossary file {glossary_path}: {exc}",
                    "startup",
                )
                self._glossary = {}
        return self._glossary
        
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
    def align_scorer(self) -> AlignScorer:
        if not self._align_scorer:
            self._align_scorer = AlignScorer(
                logger=self.logger,
                model_name=AlignScorer.DEFAULT_MODEL_NAME,
            )
        return self._align_scorer

    @property
    def align_verifier(self) -> AlignVerifier:
        if not self._align_verifier:
            self._align_verifier = AlignVerifier(
                align_scorer=self.align_scorer,
                logger=self.logger,
            )
        return self._align_verifier

    @property
    def system_ctx(self) -> SystemContext:
        return self._system_ctx

dependencies = Dependencies()

def get_dependencies(x_session_id: Optional[str] = Header(None, alias="X-Session-ID")) -> Dependencies:
    """Factory for request-scoped dependencies."""
    if x_session_id:
        return Dependencies(session_id=x_session_id)
    return dependencies
