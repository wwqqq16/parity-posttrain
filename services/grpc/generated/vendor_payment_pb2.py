# Generated protocol buffer code. DO NOT EDIT.
# ruff: noqa: E501
from google.protobuf import descriptor_pool as _descriptor_pool
from google.protobuf import symbol_database as _symbol_database
from google.protobuf.internal import builder as _builder
from google.protobuf import empty_pb2 as google_dot_protobuf_dot_empty__pb2  # noqa: F401
from google.protobuf import struct_pb2 as google_dot_protobuf_dot_struct__pb2  # noqa: F401

_sym_db = _symbol_database.Default()

DESCRIPTOR = _descriptor_pool.Default().AddSerializedFile(b'\n1enterprise/agent/workflow/v1/vendor_payment.proto\x12\x1centerprise.agent.workflow.v1\x1a\x1bgoogle/protobuf/empty.proto\x1a\x1cgoogle/protobuf/struct.proto"(\n\x15CreateWorkflowRequest\x12\x0f\n\x07case_id\x18\x01 \x01(\t"\x1f\n\rGetRunRequest\x12\x0e\n\x06run_id\x18\x01 \x01(\t"\xac\x01\n\x13SubmitReviewRequest\x12\x11\n\treview_id\x18\x01 \x01(\t\x12>\n\x08decision\x18\x02 \x01(\x0e2,.enterprise.agent.workflow.v1.ReviewDecision\x12\x13\n\x0breviewer_id\x18\x03 \x01(\t\x12\x0e\n\x06reason\x18\x04 \x01(\t\x12\x1d\n\x15bank_account_verified\x18\x05 \x01(\x08"\'\n\x15ResumeWorkflowRequest\x12\x0e\n\x06run_id\x18\x01 \x01(\t"n\n\x10WorkflowSnapshot\x12\x0e\n\x06run_id\x18\x01 \x01(\t\x12\x0f\n\x07case_id\x18\x02 \x01(\t\x12\x0e\n\x06status\x18\x03 \x01(\t\x12)\n\x08snapshot\x18\x04 \x01(\x0b2\x17.google.protobuf.Struct"n\n\x0eReviewSnapshot\x12\x11\n\treview_id\x18\x01 \x01(\t\x12\x0e\n\x06run_id\x18\x02 \x01(\t\x12\x0e\n\x06status\x18\x03 \x01(\t\x12)\n\x08snapshot\x18\x04 \x01(\x0b2\x17.google.protobuf.Struct"T\n\x13ListReviewsResponse\x12=\n\x07reviews\x18\x01 \x03(\x0b2,.enterprise.agent.workflow.v1.ReviewSnapshot" \n\x0eHealthResponse\x12\x0e\n\x06status\x18\x01 \x01(\t*j\n\x0eReviewDecision\x12\x1f\n\x1bREVIEW_DECISION_UNSPECIFIED\x10\x00\x12\x1b\n\x17REVIEW_DECISION_APPROVE\x10\x01\x12\x1a\n\x16REVIEW_DECISION_REJECT\x10\x022\x90\x05\n\x1cVendorPaymentWorkflowService\x12N\n\x06Health\x12\x16.google.protobuf.Empty\x1a,.enterprise.agent.workflow.v1.HealthResponse\x12u\n\x0eCreateWorkflow\x123.enterprise.agent.workflow.v1.CreateWorkflowRequest\x1a..enterprise.agent.workflow.v1.WorkflowSnapshot\x12e\n\x06GetRun\x12+.enterprise.agent.workflow.v1.GetRunRequest\x1a..enterprise.agent.workflow.v1.WorkflowSnapshot\x12X\n\x0bListReviews\x12\x16.google.protobuf.Empty\x1a1.enterprise.agent.workflow.v1.ListReviewsResponse\x12q\n\x0cSubmitReview\x121.enterprise.agent.workflow.v1.SubmitReviewRequest\x1a..enterprise.agent.workflow.v1.WorkflowSnapshot\x12u\n\x0eResumeWorkflow\x123.enterprise.agent.workflow.v1.ResumeWorkflowRequest\x1a..enterprise.agent.workflow.v1.WorkflowSnapshotb\x06proto3')

_globals = globals()
_builder.BuildMessageAndEnumDescriptors(DESCRIPTOR, _globals)
_builder.BuildTopDescriptorsAndMessages(
    DESCRIPTOR,
    "services.grpc.generated.vendor_payment_pb2",
    _globals,
)
