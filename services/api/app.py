from __future__ import annotations

from typing import Any, Literal

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from enterprise_eval.human_review import ReviewDecision
from enterprise_eval.vendor_payment_service import (
    VendorPaymentWorkflowService,
)


class CreateWorkflowRequest(BaseModel):
    case_id: str = "bank_account_change_review"


class ReviewRequest(BaseModel):
    reviewer_id: str = Field(min_length=1)
    reason: str = Field(min_length=1)
    bank_account_verified: bool = False


class HealthResponse(BaseModel):
    status: Literal["ok"]


def create_app(
    service: VendorPaymentWorkflowService | None = None,
) -> FastAPI:
    workflow_service = service or VendorPaymentWorkflowService()
    app = FastAPI(
        title="Enterprise Vendor Payment Control Plane",
        version="0.1.0",
    )
    app.state.workflow_service = workflow_service

    @app.get("/health", response_model=HealthResponse)
    def health() -> HealthResponse:
        return HealthResponse(status="ok")

    @app.post("/workflows/vendor-payments")
    def create_vendor_payment_workflow(
        request: CreateWorkflowRequest,
    ) -> dict[str, Any]:
        try:
            return workflow_service.create_workflow(request.case_id)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.get("/runs/{run_id}")
    def get_run(run_id: str) -> dict[str, Any]:
        try:
            return workflow_service.get_run(run_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @app.get("/reviews")
    def list_reviews() -> list[dict[str, Any]]:
        return workflow_service.list_reviews()

    @app.post("/reviews/{review_id}/approve")
    def approve_review(
        review_id: str,
        request: ReviewRequest,
    ) -> dict[str, Any]:
        try:
            return workflow_service.submit_review(
                review_id,
                decision=ReviewDecision.APPROVE,
                reviewer_id=request.reviewer_id,
                reason=request.reason,
                bank_account_verified=request.bank_account_verified,
            )
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc

    @app.post("/reviews/{review_id}/reject")
    def reject_review(
        review_id: str,
        request: ReviewRequest,
    ) -> dict[str, Any]:
        try:
            return workflow_service.submit_review(
                review_id,
                decision=ReviewDecision.REJECT,
                reviewer_id=request.reviewer_id,
                reason=request.reason,
                bank_account_verified=False,
            )
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc

    @app.post("/runs/{run_id}/resume")
    def resume_run(run_id: str) -> dict[str, Any]:
        try:
            return workflow_service.resume_workflow(run_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except (ValueError, RuntimeError) as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc

    @app.get("/events")
    def list_events(run_id: str | None = None) -> list[dict[str, Any]]:
        return [
            event.to_dict()
            for event in workflow_service.publisher.list_events(
                run_id=run_id
            )
        ]

    return app


app = create_app()
