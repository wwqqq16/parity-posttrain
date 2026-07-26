from __future__ import annotations

import os

from enterprise_eval.event_transport import build_event_publisher_from_env
from enterprise_eval.vendor_payment_service import VendorPaymentWorkflowService
from services.grpc.server import create_grpc_server


def main() -> None:
    host = os.environ.get("GRPC_HOST", "127.0.0.1")
    port = int(os.environ.get("GRPC_PORT", "50051"))
    max_workers = int(os.environ.get("GRPC_MAX_WORKERS", "8"))

    service = VendorPaymentWorkflowService(
        publisher=build_event_publisher_from_env()
    )
    handle = create_grpc_server(
        service,
        host=host,
        port=port,
        max_workers=max_workers,
    )
    handle.server.start()
    print(f"gRPC workflow service listening on {handle.address}")
    try:
        handle.server.wait_for_termination()
    except KeyboardInterrupt:
        handle.server.stop(grace=2.0).wait(timeout=3.0)


if __name__ == "__main__":
    main()
