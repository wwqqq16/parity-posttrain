from __future__ import annotations

import grpc
from google.protobuf import json_format

from enterprise_eval.events import InMemoryEventPublisher
from enterprise_eval.vendor_payment_service import VendorPaymentWorkflowService
from services.grpc.generated import vendor_payment_pb2 as pb2
from services.grpc.generated import vendor_payment_pb2_grpc as pb2_grpc
from services.grpc.server import create_grpc_server


def main() -> None:
    core_service = VendorPaymentWorkflowService(
        publisher=InMemoryEventPublisher()
    )
    handle = create_grpc_server(core_service, port=0)
    handle.server.start()
    channel = grpc.insecure_channel(handle.address)

    try:
        grpc.channel_ready_future(channel).result(timeout=5.0)
        stub = pb2_grpc.VendorPaymentWorkflowServiceStub(channel)

        created = stub.CreateWorkflow(
            pb2.CreateWorkflowRequest(
                case_id="bank_account_change_review"
            ),
            timeout=5.0,
        )
        created_snapshot = json_format.MessageToDict(
            created.snapshot,
            preserving_proto_field_name=True,
        )
        review_id = str(created_snapshot["human_review"]["review_id"])

        reviewed = stub.SubmitReview(
            pb2.SubmitReviewRequest(
                review_id=review_id,
                decision=pb2.REVIEW_DECISION_APPROVE,
                reviewer_id="finance-reviewer-01",
                reason="Verified through an independent callback.",
                bank_account_verified=True,
            ),
            timeout=5.0,
        )
        final = stub.ResumeWorkflow(
            pb2.ResumeWorkflowRequest(run_id=created.run_id),
            timeout=5.0,
        )
        final_snapshot = json_format.MessageToDict(
            final.snapshot,
            preserving_proto_field_name=True,
        )

        print("GRPC WORKFLOW DEMO")
        print("=" * 52)
        print("Server address:", handle.address)
        print("Created status:", created.status)
        print("Review status:", reviewed.status)
        print("Final status:", final.status)
        print(
            "Payment approved:",
            final_snapshot["state"]["payment_approved"],
        )
        print(
            "Task success:",
            final_snapshot["evaluation"]["task_success"],
        )
        print(
            "Policy violation:",
            final_snapshot["evaluation"]["policy_violation"],
        )
        print(
            "Shared business-service events:",
            len(final_snapshot["events"]),
        )
    finally:
        channel.close()
        handle.server.stop(grace=0).wait(timeout=5.0)


if __name__ == "__main__":
    main()
