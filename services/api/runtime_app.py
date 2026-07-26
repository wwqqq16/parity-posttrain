from __future__ import annotations

from enterprise_eval.event_transport import (
    build_event_publisher_from_env,
)
from enterprise_eval.vendor_payment_service import (
    VendorPaymentWorkflowService,
)
from services.api.app import create_app

service = VendorPaymentWorkflowService(
    publisher=build_event_publisher_from_env()
)
app = create_app(service)
