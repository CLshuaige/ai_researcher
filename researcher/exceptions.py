class ResearcherError(Exception):
    """Base exception for AI Researcher system"""
    pass


class ConfigurationError(ResearcherError):
    """Configuration-related errors"""
    pass


class AgentError(ResearcherError):
    """Agent execution errors"""
    pass


class DebateError(ResearcherError):
    """Debate process errors"""
    pass


class WorkflowError(ResearcherError):
    """Workflow execution errors"""
    pass


class LLMError(ResearcherError):
    """LLM API errors"""
    pass


class FileOperationError(ResearcherError):
    """File I/O errors"""
    pass
