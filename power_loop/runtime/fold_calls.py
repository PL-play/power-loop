"""History-fold calls use the same task-managed model-call path."""
from .model_calls import ModelCalls


class FoldCalls(ModelCalls):
    def __init__(self, *, session_id, bus=None, cancel_token=None):
        super().__init__(session_id=session_id, bus=bus, cancel_token=cancel_token, purpose='fold')
