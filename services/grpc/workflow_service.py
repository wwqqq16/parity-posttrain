from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any, NoReturn

import grpc
from google.protobuf import empty_pb2, json_format, struct_pb2

from enterprise_eval.human_review import ReviewDecision
from enterprise_eval.vendor_payment_service import VendorPaymentWorkflowService
from services.grpc.generated import vendor_payment_pb2 as pb2
from services.grpc.generated import vendor_payment_pb2_grpc as pb2_grpc


class VendorPaymentWorkflowGrpcServicer(
    pb2_grpc.VendorPaymentWorkflowServiceServicer
):
    """Typed gRPC adapter over the existing workflow business service."""

    def __init__(self, service: VendorPaymentWorkflowService) -> None:
        self.service = service

    def Health(
        self,
        request: empty_pb2.Empty,
        context: grpc.ServicerContext,
    ) -> pb2.HealthResponse:
        del request, context
        return pb2.HealthResponse(status="ok")

    def CreateWorkflow(
        self,
        request: pb2.CreateWorkflowRequest,
        context: grpc.ServicerContext,
    ) -> pb2.WorkflowSnapshot:
        if not request.case_id.strip():
            self._abort(
                context,
                grpc.StatusCode.INVALID_ARGUMENT,
                "case_id must not be empty.",
            )
        try:
            snapshot = self.service.create_workflow(request.case_id)
        except ValueError as exc:
            self._abort(context, grpc.StatusCode.INVALID_ARGUMENT, self._message(exc))
        return self._workflow_snapshot(snapshot)

    def GetRun(
        self,
        request: pb2.GetRunRequest,
        context: grpc.ServicerContext,
    ) -> pb2.WorkflowSnapshot:
        if not request.run_id.strip():
            self._abort(
                context,
                grpc.StatusCode.INVALID_ARGUMENT,
                "run_id must not be empty.",
            )
        try:
            snapshot = self.service.get_run(request.run_id)
        except KeyError as exc:
            self._abort(context, grpc.StatusCode.NOT_FOUND, self._message(exc))
        return self._workflow_snapshot(snapshot)

    def ListReviews(
        self,
        request: empty_pb2.Empty,
        context: grpc.ServicerContext,
    ) -> pb2.ListReviewsResponse:
        del request, context
        return pb2.ListReviewsResponse(
            reviews=[
                self._review_snapshot(review)
                for review in self.service.list_reviews()
            ]
        )

    def SubmitReview(
        self,
        request: pb2.SubmitReviewRequest,
        context: grpc.ServicerContext,
    ) -> pb2.WorkflowSnapshot:
        decision = self._review_decision(request.decision, context)
        if not request.review_id.strip():
            self._abort(
                context,
                grpc.StatusCode.INVALID_ARGUMENT,
                "review_id must not be empty.",
            )
        if not request.reviewer_id.strip():
            self._abort(
                context,
                grpc.StatusCode.INVALID_ARGUMENT,
                "reviewer_id must not be empty.",
            )
        if not request.reason.strip():
            self._abort(
                context,
                grpc.StatusCode.INVALID_ARGUMENT,
                "reason must not be empty.",
            )
        if decision is ReviewDecision.REJECT and request.bank_account_verified:
            self._abort(
                context,
                grpc.StatusCode.INVALID_ARGUMENT,
                "Rejected reviews cannot mark the bank account as verified.",
            )
        try:
            snapshot = self.service.submit_review(
                request.review_id,
                decision=decision,
                reviewer_id=request.reviewer_id,
                reason=request.reason,
                bank_account_verified=request.bank_account_verified,
            )
        except KeyError as exc:
            self._abort(context, grpc.StatusCode.NOT_FOUND, self._message(exc))
        except ValueError as exc:
            self._abort(
                context,
                grpc.StatusCode.FAILED_PRECONDITION,
                self._message(exc),
            )
        return self._workflow_snapshot(snapshot)

    def ResumeWorkflow(
        self,
        request: pb2.ResumeWorkflowRequest,
        context: grpc.ServicerContext,
    ) -> pb2.WorkflowSnapshot:
        if not request.run_id.strip():
            self._abort(
                context,
                grpc.StatusCode.INVALID_ARGUMENT,
                "run_id must not be empty.",
            )
        try:
            snapshot = self.service.resume_workflow(request.run_id)
        except KeyError as exc:
            self._abort(context, grpc.StatusCode.NOT_FOUND, self._message(exc))
        except (ValueError, RuntimeError) as exc:
            self._abort(
                context,
                grpc.StatusCode.FAILED_PRECONDITION,
                self._message(exc),
            )
        return self._workflow_snapshot(snapshot)

    @staticmethod
    def _review_decision(
        value: int,
        context: grpc.ServicerContext,
    ) -> ReviewDecision:
        if value == pb2.REVIEW_DECISION_APPROVE:
            return ReviewDecision.APPROVE
        if value == pb2.REVIEW_DECISION_REJECT:
            return ReviewDecision.REJECT
        VendorPaymentWorkflowGrpcServicer._abort(
            context,
            grpc.StatusCode.INVALID_ARGUMENT,
            "decision must be APPROVE or REJECT.",
        )

    @staticmethod
    def _workflow_snapshot(
        snapshot: Mapping[str, Any],
    ) -> pb2.WorkflowSnapshot:
        return pb2.WorkflowSnapshot(
            run_id=str(snapshot["run_id"]),
            case_id=str(snapshot["case_id"]),
            status=str(snapshot["status"]),
            snapshot=VendorPaymentWorkflowGrpcServicer._to_struct(snapshot),
        )

    @staticmethod
    def _review_snapshot(
        snapshot: Mapping[str, Any],
    ) -> pb2.ReviewSnapshot:
        return pb2.ReviewSnapshot(
            review_id=str(snapshot["review_id"]),
            run_id=str(snapshot["run_id"]),
            status=str(snapshot["status"]),
            snapshot=VendorPaymentWorkflowGrpcServicer._to_struct(snapshot),
        )

    @staticmethod
    def _to_struct(value: Mapping[str, Any]) -> struct_pb2.Struct:
        normalized = json.loads(json.dumps(dict(value)))
        message = struct_pb2.Struct()
        json_format.ParseDict(normalized, message)
        return message

    @staticmethod
    def _message(exc: BaseException) -> str:
        if isinstance(exc, KeyError) and exc.args:
            return str(exc.args[0])
        return str(exc)

    @staticmethod
    def _abort(
        context: grpc.ServicerContext,
        code: grpc.StatusCode,
        details: str,
    ) -> NoReturn:
        context.abort(code, details)
        raise AssertionError("gRPC context.abort() unexpectedly returned")
