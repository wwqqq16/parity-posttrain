from google.protobuf import struct_pb2 as _struct_pb2
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from collections.abc import Mapping as _Mapping
from typing import ClassVar as _ClassVar, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor

class ResetRequest(_message.Message):
    __slots__ = ("case_id", "architecture", "episode_id")
    CASE_ID_FIELD_NUMBER: _ClassVar[int]
    ARCHITECTURE_FIELD_NUMBER: _ClassVar[int]
    EPISODE_ID_FIELD_NUMBER: _ClassVar[int]
    case_id: str
    architecture: str
    episode_id: str
    def __init__(self, case_id: _Optional[str] = ..., architecture: _Optional[str] = ..., episode_id: _Optional[str] = ...) -> None: ...

class ResetResponse(_message.Message):
    __slots__ = ("episode_id", "case_id", "architecture", "observation", "state_fingerprint", "info")
    EPISODE_ID_FIELD_NUMBER: _ClassVar[int]
    CASE_ID_FIELD_NUMBER: _ClassVar[int]
    ARCHITECTURE_FIELD_NUMBER: _ClassVar[int]
    OBSERVATION_FIELD_NUMBER: _ClassVar[int]
    STATE_FINGERPRINT_FIELD_NUMBER: _ClassVar[int]
    INFO_FIELD_NUMBER: _ClassVar[int]
    episode_id: str
    case_id: str
    architecture: str
    observation: str
    state_fingerprint: str
    info: _struct_pb2.Struct
    def __init__(self, episode_id: _Optional[str] = ..., case_id: _Optional[str] = ..., architecture: _Optional[str] = ..., observation: _Optional[str] = ..., state_fingerprint: _Optional[str] = ..., info: _Optional[_Union[_struct_pb2.Struct, _Mapping]] = ...) -> None: ...

class StepRequest(_message.Message):
    __slots__ = ("episode_id", "request_id", "action_type", "arguments")
    EPISODE_ID_FIELD_NUMBER: _ClassVar[int]
    REQUEST_ID_FIELD_NUMBER: _ClassVar[int]
    ACTION_TYPE_FIELD_NUMBER: _ClassVar[int]
    ARGUMENTS_FIELD_NUMBER: _ClassVar[int]
    episode_id: str
    request_id: str
    action_type: str
    arguments: _struct_pb2.Struct
    def __init__(self, episode_id: _Optional[str] = ..., request_id: _Optional[str] = ..., action_type: _Optional[str] = ..., arguments: _Optional[_Union[_struct_pb2.Struct, _Mapping]] = ...) -> None: ...

class StepResponse(_message.Message):
    __slots__ = ("episode_id", "request_id", "action_type", "observation", "reward", "terminated", "truncated", "state_fingerprint", "info")
    EPISODE_ID_FIELD_NUMBER: _ClassVar[int]
    REQUEST_ID_FIELD_NUMBER: _ClassVar[int]
    ACTION_TYPE_FIELD_NUMBER: _ClassVar[int]
    OBSERVATION_FIELD_NUMBER: _ClassVar[int]
    REWARD_FIELD_NUMBER: _ClassVar[int]
    TERMINATED_FIELD_NUMBER: _ClassVar[int]
    TRUNCATED_FIELD_NUMBER: _ClassVar[int]
    STATE_FINGERPRINT_FIELD_NUMBER: _ClassVar[int]
    INFO_FIELD_NUMBER: _ClassVar[int]
    episode_id: str
    request_id: str
    action_type: str
    observation: str
    reward: float
    terminated: bool
    truncated: bool
    state_fingerprint: str
    info: _struct_pb2.Struct
    def __init__(self, episode_id: _Optional[str] = ..., request_id: _Optional[str] = ..., action_type: _Optional[str] = ..., observation: _Optional[str] = ..., reward: _Optional[float] = ..., terminated: _Optional[bool] = ..., truncated: _Optional[bool] = ..., state_fingerprint: _Optional[str] = ..., info: _Optional[_Union[_struct_pb2.Struct, _Mapping]] = ...) -> None: ...
