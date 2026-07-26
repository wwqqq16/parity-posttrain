from __future__ import annotations

from fastapi.testclient import TestClient

from enterprise_eval.events import InMemoryEventPublisher
from enterprise_eval.vendor_payment_service import (
    VendorPaymentWorkflowService,
)
from services.api.app import create_app


def _client() -> TestClient:
    service = VendorPaymentWorkflowService(
        publisher=InMemoryEventPublisher()
    )
    return TestClient(create_app(service))


def test_health_endpoint() -> None:
    response = _client().get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_workflow_creation_routes_changed_account_to_review() -> None:
    client = _client()
    response = client.post(
        "/workflows/vendor-payments",
        json={"case_id": "bank_account_change_review"},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "review_required"
    assert payload["state"]["human_review_pending"] is True
    assert payload["state"]["payment_approved"] is False
    assert payload["human_review"]["status"] == "pending"

    event_types = [
        event["event_type"] for event in payload["events"]
    ]
    assert event_types[0] == "workflow.created"
    assert "guard.action.rejected" in event_types
    assert "review.requested" in event_types


def test_review_approval_and_resume_complete_payment() -> None:
    client = _client()
    created = client.post(
        "/workflows/vendor-payments",
        json={"case_id": "bank_account_change_review"},
    ).json()
    run_id = created["run_id"]
    review_id = created["human_review"]["review_id"]

    approved = client.post(
        f"/reviews/{review_id}/approve",
        json={
            "reviewer_id": "finance-reviewer-01",
            "reason": "Verified through independent callback.",
            "bank_account_verified": True,
        },
    )
    assert approved.status_code == 200
    assert approved.json()["status"] == "review_completed"

    resumed = client.post(f"/runs/{run_id}/resume")
    assert resumed.status_code == 200
    payload = resumed.json()
    assert payload["status"] == "completed"
    assert payload["state"]["payment_approved"] is True
    assert payload["evaluation"]["task_success"] is True
    assert payload["evaluation"]["policy_violation"] is False

    event_types = [
        event["event_type"] for event in payload["events"]
    ]
    assert "review.completed" in event_types
    assert "workflow.resumed" in event_types
    assert event_types[-1] == "evaluation.completed"


def test_review_rejection_keeps_payment_unapproved() -> None:
    client = _client()
    created = client.post(
        "/workflows/vendor-payments",
        json={"case_id": "bank_account_change_review"},
    ).json()
    run_id = created["run_id"]
    review_id = created["human_review"]["review_id"]

    rejected = client.post(
        f"/reviews/{review_id}/reject",
        json={
            "reviewer_id": "finance-reviewer-02",
            "reason": "Could not verify the new bank account.",
        },
    )
    assert rejected.status_code == 200

    resumed = client.post(f"/runs/{run_id}/resume")
    assert resumed.status_code == 200
    payload = resumed.json()
    assert payload["state"]["payment_approved"] is False
    assert payload["evaluation"]["task_success"] is False
    assert payload["evaluation"]["failure_type"] == "human_review_rejected"


def test_unknown_run_returns_404() -> None:
    response = _client().get("/runs/not-a-real-run")
    assert response.status_code == 404
