from google.protobuf.message import Message
from google.protobuf.struct_pb2 import Struct

REVIEW_DECISION_UNSPECIFIED: int
REVIEW_DECISION_APPROVE: int
REVIEW_DECISION_REJECT: int

class CreateWorkflowRequest(Message):
    case_id: str
    def __init__(self, *, case_id: str = ...) -> None: ...

class GetRunRequest(Message):
    run_id: str
    def __init__(self, *, run_id: str = ...) -> None: ...

class SubmitReviewRequest(Message):
    review_id: str
    decision: int
    reviewer_id: str
    reason: str
    bank_account_verified: bool
    def __init__(
        self,
        *,
        review_id: str = ...,
        decision: int = ...,
        reviewer_id: str = ...,
        reason: str = ...,
        bank_account_verified: bool = ...,
    ) -> None: ...

class ResumeWorkflowRequest(Message):
    run_id: str
    def __init__(self, *, run_id: str = ...) -> None: ...

class WorkflowSnapshot(Message):
    run_id: str
    case_id: str
    status: str
    snapshot: Struct
    def __init__(
        self,
        *,
        run_id: str = ...,
        case_id: str = ...,
        status: str = ...,
        snapshot: Struct | None = ...,
    ) -> None: ...

class ReviewSnapshot(Message):
    review_id: str
    run_id: str
    status: str
    snapshot: Struct
    def __init__(
        self,
        *,
        review_id: str = ...,
        run_id: str = ...,
        status: str = ...,
        snapshot: Struct | None = ...,
    ) -> None: ...

class ListReviewsResponse(Message):
    reviews: list[ReviewSnapshot]
    def __init__(self, *, reviews: list[ReviewSnapshot] = ...) -> None: ...

class HealthResponse(Message):
    status: str
    def __init__(self, *, status: str = ...) -> None: ...
