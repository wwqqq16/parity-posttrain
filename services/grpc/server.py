from __future__ import annotations

from concurrent import futures
from dataclasses import dataclass

import grpc

from enterprise_eval.vendor_payment_service import VendorPaymentWorkflowService
from services.grpc.generated import vendor_payment_pb2_grpc as pb2_grpc
from services.grpc.workflow_service import VendorPaymentWorkflowGrpcServicer


@dataclass(frozen=True)
class GrpcServerHandle:
    server: grpc.Server
    host: str
    port: int

    @property
    def address(self) -> str:
        return f"{self.host}:{self.port}"


def create_grpc_server(
    service: VendorPaymentWorkflowService,
    *,
    host: str = "127.0.0.1",
    port: int = 50051,
    max_workers: int = 8,
) -> GrpcServerHandle:
    if max_workers <= 0:
        raise ValueError("max_workers must be greater than zero.")
    if not 0 <= port <= 65535:
        raise ValueError("port must be between 0 and 65535.")

    server = grpc.server(
        futures.ThreadPoolExecutor(max_workers=max_workers)
    )
    pb2_grpc.add_VendorPaymentWorkflowServiceServicer_to_server(
        VendorPaymentWorkflowGrpcServicer(service),
        server,
    )
    bound_port = server.add_insecure_port(f"{host}:{port}")
    if bound_port == 0:
        raise RuntimeError(f"Could not bind gRPC server to {host}:{port}.")
    return GrpcServerHandle(server=server, host=host, port=bound_port)
