from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any

import grpc
import pytest
from fastapi.testclient import TestClient
from google.protobuf import empty_pb2, json_format

from enterprise_eval.events import InMemoryEventPublisher
from enterprise_eval.vendor_payment_service import VendorPaymentWorkflowService
from services.api.app import create_app
from services.grpc.generated import vendor_payment_pb2 as pb2
from services.grpc.generated import vendor_payment_pb2_grpc as pb2_grpc
from services.grpc.server import GrpcServerHandle, create_grpc_server


@dataclass(frozen=True)
class GrpcHarness:
    core_service: VendorPaymentWorkflowService
    handle: GrpcServerHandle
    channel: grpc.Channel
    stub: pb2_grpc.VendorPaymentWorkflowServiceStub


@pytest.fixture
def grpc_harness() -> Iterator[GrpcHarness]:
    core_service = VendorPaymentWorkflowService(
        publisher=InMemoryEventPublisher()
    )
    handle = create_grpc_server(core_service, port=0)
    handle.server.start()
    channel = grpc.insecure_channel(handle.address)
    grpc.channel_ready_future(channel).result(timeout=5.0)
    harness = GrpcHarness(
        core_service=core_service,
        handle=handle,
        channel=channel,
        stub=pb2_grpc.VendorPaymentWorkflowServiceStub(channel),
    )
    try:
        yield harness
    finally:
        channel.close()
        handle.server.stop(grace=0).wait(timeout=5.0)


def _snapshot(message: pb2.WorkflowSnapshot) -> dict[str, Any]:
    return json_format.MessageToDict(
        message.snapshot,
        preserving_proto_field_name=True,
    )


def test_health_rpc(grpc_harness: GrpcHarness) -> None:
    response = grpc_harness.stub.Health(empty_pb2.Empty(), timeout=5.0)
    assert response.status == "ok"


def test_create_workflow_routes_changed_account_to_review(
    grpc_harness: GrpcHarness,
) -> None:
    response = grpc_harness.stub.CreateWorkflow(
        pb2.CreateWorkflowRequest(
            case_id="bank_account_change_review"
        ),
        timeout=5.0,
    )
    snapshot = _snapshot(response)

    assert response.status == "review_required"
    assert snapshot["state"]["human_review_pending"] is True
    assert snapshot["state"]["payment_approved"] is False
    assert snapshot["human_review"]["status"] == "pending"


def test_approval_and_resume_complete_payment(
    grpc_harness: GrpcHarness,
) -> None:
    created = grpc_harness.stub.CreateWorkflow(
        pb2.CreateWorkflowRequest(
            case_id="bank_account_change_review"
        ),
        timeout=5.0,
    )
    created_snapshot = _snapshot(created)
    review_id = created_snapshot["human_review"]["review_id"]

    reviewed = grpc_harness.stub.SubmitReview(
        pb2.SubmitReviewRequest(
            review_id=review_id,
            decision=pb2.REVIEW_DECISION_APPROVE,
            reviewer_id="finance-reviewer-01",
            reason="Verified through independent callback.",
            bank_account_verified=True,
        ),
        timeout=5.0,
    )
    final = grpc_harness.stub.ResumeWorkflow(
        pb2.ResumeWorkflowRequest(run_id=created.run_id),
        timeout=5.0,
    )
    snapshot = _snapshot(final)

    assert reviewed.status == "review_completed"
    assert final.status == "completed"
    assert snapshot["state"]["payment_approved"] is True
    assert snapshot["evaluation"]["task_success"] is True
    assert snapshot["evaluation"]["policy_violation"] is False


def test_rejected_review_remains_unpaid(
    grpc_harness: GrpcHarness,
) -> None:
    created = grpc_harness.stub.CreateWorkflow(
        pb2.CreateWorkflowRequest(
            case_id="bank_account_change_review"
        ),
        timeout=5.0,
    )
    created_snapshot = _snapshot(created)
    review_id = created_snapshot["human_review"]["review_id"]

    grpc_harness.stub.SubmitReview(
        pb2.SubmitReviewRequest(
            review_id=review_id,
            decision=pb2.REVIEW_DECISION_REJECT,
            reviewer_id="finance-reviewer-02",
            reason="Independent verification failed.",
        ),
        timeout=5.0,
    )
    final = grpc_harness.stub.ResumeWorkflow(
        pb2.ResumeWorkflowRequest(run_id=created.run_id),
        timeout=5.0,
    )
    snapshot = _snapshot(final)

    assert snapshot["state"]["payment_approved"] is False
    assert snapshot["evaluation"]["task_success"] is False
    assert snapshot["evaluation"]["failure_type"] == (
        "human_review_rejected"
    )


def test_list_reviews_returns_typed_records(
    grpc_harness: GrpcHarness,
) -> None:
    created = grpc_harness.stub.CreateWorkflow(
        pb2.CreateWorkflowRequest(
            case_id="bank_account_change_review"
        ),
        timeout=5.0,
    )
    response = grpc_harness.stub.ListReviews(
        empty_pb2.Empty(),
        timeout=5.0,
    )

    assert len(response.reviews) == 1
    assert response.reviews[0].run_id == created.run_id
    assert response.reviews[0].status == "pending"


def test_unknown_run_maps_to_not_found(
    grpc_harness: GrpcHarness,
) -> None:
    with pytest.raises(grpc.RpcError) as raised:
        grpc_harness.stub.GetRun(
            pb2.GetRunRequest(run_id="not-a-real-run"),
            timeout=5.0,
        )
    assert raised.value.code() is grpc.StatusCode.NOT_FOUND


def test_invalid_review_decision_maps_to_invalid_argument(
    grpc_harness: GrpcHarness,
) -> None:
    created = grpc_harness.stub.CreateWorkflow(
        pb2.CreateWorkflowRequest(
            case_id="bank_account_change_review"
        ),
        timeout=5.0,
    )
    snapshot = _snapshot(created)

    with pytest.raises(grpc.RpcError) as raised:
        grpc_harness.stub.SubmitReview(
            pb2.SubmitReviewRequest(
                review_id=snapshot["human_review"]["review_id"],
                decision=pb2.REVIEW_DECISION_UNSPECIFIED,
                reviewer_id="finance-reviewer-01",
                reason="Missing explicit decision.",
            ),
            timeout=5.0,
        )
    assert raised.value.code() is grpc.StatusCode.INVALID_ARGUMENT


def test_rest_and_grpc_share_the_same_business_service(
    grpc_harness: GrpcHarness,
) -> None:
    created = grpc_harness.stub.CreateWorkflow(
        pb2.CreateWorkflowRequest(case_id="valid_payment"),
        timeout=5.0,
    )
    rest_client = TestClient(create_app(grpc_harness.core_service))
    response = rest_client.get(f"/runs/{created.run_id}")

    assert response.status_code == 200
    assert response.json()["run_id"] == created.run_id
    assert response.json()["status"] == "completed"
