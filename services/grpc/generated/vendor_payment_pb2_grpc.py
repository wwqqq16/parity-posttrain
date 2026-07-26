# Generated gRPC service code. DO NOT EDIT.
from __future__ import annotations

from typing import Any

import grpc
from google.protobuf import empty_pb2

from services.grpc.generated import vendor_payment_pb2 as vendor__payment__pb2


class VendorPaymentWorkflowServiceStub:
    def __init__(self, channel: grpc.Channel) -> None:
        self.Health = channel.unary_unary(
            "/enterprise.agent.workflow.v1.VendorPaymentWorkflowService/Health",
            request_serializer=empty_pb2.Empty.SerializeToString,
            response_deserializer=vendor__payment__pb2.HealthResponse.FromString,
        )
        self.CreateWorkflow = channel.unary_unary(
            "/enterprise.agent.workflow.v1.VendorPaymentWorkflowService/CreateWorkflow",
            request_serializer=vendor__payment__pb2.CreateWorkflowRequest.SerializeToString,
            response_deserializer=vendor__payment__pb2.WorkflowSnapshot.FromString,
        )
        self.GetRun = channel.unary_unary(
            "/enterprise.agent.workflow.v1.VendorPaymentWorkflowService/GetRun",
            request_serializer=vendor__payment__pb2.GetRunRequest.SerializeToString,
            response_deserializer=vendor__payment__pb2.WorkflowSnapshot.FromString,
        )
        self.ListReviews = channel.unary_unary(
            "/enterprise.agent.workflow.v1.VendorPaymentWorkflowService/ListReviews",
            request_serializer=empty_pb2.Empty.SerializeToString,
            response_deserializer=vendor__payment__pb2.ListReviewsResponse.FromString,
        )
        self.SubmitReview = channel.unary_unary(
            "/enterprise.agent.workflow.v1.VendorPaymentWorkflowService/SubmitReview",
            request_serializer=vendor__payment__pb2.SubmitReviewRequest.SerializeToString,
            response_deserializer=vendor__payment__pb2.WorkflowSnapshot.FromString,
        )
        self.ResumeWorkflow = channel.unary_unary(
            "/enterprise.agent.workflow.v1.VendorPaymentWorkflowService/ResumeWorkflow",
            request_serializer=vendor__payment__pb2.ResumeWorkflowRequest.SerializeToString,
            response_deserializer=vendor__payment__pb2.WorkflowSnapshot.FromString,
        )


class VendorPaymentWorkflowServiceServicer:
    def Health(self, request: Any, context: grpc.ServicerContext) -> Any:
        context.abort(grpc.StatusCode.UNIMPLEMENTED, "Method not implemented.")

    def CreateWorkflow(self, request: Any, context: grpc.ServicerContext) -> Any:
        context.abort(grpc.StatusCode.UNIMPLEMENTED, "Method not implemented.")

    def GetRun(self, request: Any, context: grpc.ServicerContext) -> Any:
        context.abort(grpc.StatusCode.UNIMPLEMENTED, "Method not implemented.")

    def ListReviews(self, request: Any, context: grpc.ServicerContext) -> Any:
        context.abort(grpc.StatusCode.UNIMPLEMENTED, "Method not implemented.")

    def SubmitReview(self, request: Any, context: grpc.ServicerContext) -> Any:
        context.abort(grpc.StatusCode.UNIMPLEMENTED, "Method not implemented.")

    def ResumeWorkflow(self, request: Any, context: grpc.ServicerContext) -> Any:
        context.abort(grpc.StatusCode.UNIMPLEMENTED, "Method not implemented.")


def add_VendorPaymentWorkflowServiceServicer_to_server(
    servicer: VendorPaymentWorkflowServiceServicer,
    server: grpc.Server,
) -> None:
    rpc_method_handlers = {
        "Health": grpc.unary_unary_rpc_method_handler(
            servicer.Health,
            request_deserializer=empty_pb2.Empty.FromString,
            response_serializer=vendor__payment__pb2.HealthResponse.SerializeToString,
        ),
        "CreateWorkflow": grpc.unary_unary_rpc_method_handler(
            servicer.CreateWorkflow,
            request_deserializer=vendor__payment__pb2.CreateWorkflowRequest.FromString,
            response_serializer=vendor__payment__pb2.WorkflowSnapshot.SerializeToString,
        ),
        "GetRun": grpc.unary_unary_rpc_method_handler(
            servicer.GetRun,
            request_deserializer=vendor__payment__pb2.GetRunRequest.FromString,
            response_serializer=vendor__payment__pb2.WorkflowSnapshot.SerializeToString,
        ),
        "ListReviews": grpc.unary_unary_rpc_method_handler(
            servicer.ListReviews,
            request_deserializer=empty_pb2.Empty.FromString,
            response_serializer=vendor__payment__pb2.ListReviewsResponse.SerializeToString,
        ),
        "SubmitReview": grpc.unary_unary_rpc_method_handler(
            servicer.SubmitReview,
            request_deserializer=vendor__payment__pb2.SubmitReviewRequest.FromString,
            response_serializer=vendor__payment__pb2.WorkflowSnapshot.SerializeToString,
        ),
        "ResumeWorkflow": grpc.unary_unary_rpc_method_handler(
            servicer.ResumeWorkflow,
            request_deserializer=vendor__payment__pb2.ResumeWorkflowRequest.FromString,
            response_serializer=vendor__payment__pb2.WorkflowSnapshot.SerializeToString,
        ),
    }
    generic_handler = grpc.method_handlers_generic_handler(
        "enterprise.agent.workflow.v1.VendorPaymentWorkflowService",
        rpc_method_handlers,
    )
    server.add_generic_rpc_handlers((generic_handler,))
