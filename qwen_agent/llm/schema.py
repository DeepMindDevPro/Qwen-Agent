from typing import List, Literal, Optional, Tuple, Union

from pydantic import BaseModel, field_validator, model_validator

'''
基于 Pydantic v2 构建的聊天消息模型系统，用于处理多模态消息交互。
它提供了结构化的消息表示，支持文本、图片、文件等多种内容类型，并针对函数调用进行了专门设计
'''

DEFAULT_SYSTEM_MESSAGE = ''

ROLE = 'role'
CONTENT = 'content'
REASONING_CONTENT = 'reasoning_content'
NAME = 'name'

SYSTEM = 'system'
USER = 'user'
ASSISTANT = 'assistant'
FUNCTION = 'function'

FILE = 'file'
IMAGE = 'image'
AUDIO = 'audio'
VIDEO = 'video'

# 基础模型字典定义
class BaseModelCompatibleDict(BaseModel):
    # 使模型可以像字典一样操作
    def __getitem__(self, item):
        return getattr(self, item)

    def __setitem__(self, key, value):
        setattr(self, key, value)

    # 序列化时自动排除None值
    def model_dump(self, **kwargs):
        if 'exclude_none' not in kwargs:
            kwargs['exclude_none'] = True
        return super().model_dump(**kwargs)

    def model_dump_json(self, **kwargs):
        if 'exclude_none' not in kwargs:
            kwargs['exclude_none'] = True
        return super().model_dump_json(**kwargs)

    def get(self, key, default=None):
        try:
            value = getattr(self, key)
            if value:
                return value
            else:
                return default
        except AttributeError:
            return default

    def __str__(self):
        return f'{self.model_dump()}'

'''
  FunctionCall 类专门用于表示对外部函数的调用，包含函数名和参数字符串，支持与 AI 模型的函数调用能力集成.
'''
class FunctionCall(BaseModelCompatibleDict):
    name: str
    arguments: str

    def __init__(self, name: str, arguments: str):
        super().__init__(name=name, arguments=arguments)

    def __repr__(self):
        return f'FunctionCall({self.model_dump()})'

'''
通过 ContentItem 类实现对多种内容类型的支持，包括文本、图片、文件、音频和视频.
每个 ContentItem 实例只能包含一种类型的内容，确保数据的一致性
'''
class ContentItem(BaseModelCompatibleDict):
    text: Optional[str] = None
    image: Optional[str] = None
    file: Optional[str] = None
    audio: Optional[Union[str, dict]] = None
    video: Optional[Union[str, list]] = None

    def __init__(self,
                 text: Optional[str] = None,
                 image: Optional[str] = None,
                 file: Optional[str] = None,
                 audio: Optional[Union[str, dict]] = None,
                 video: Optional[Union[str, list]] = None):
        super().__init__(text=text, image=image, file=file, audio=audio, video=video)

    @model_validator(mode='after')
    def check_exclusivity(self):
        # 验证只能有一个字段被设置
        provided_fields = 0
        if self.text is not None:
            provided_fields += 1
        if self.image:
            provided_fields += 1
        if self.file:
            provided_fields += 1
        if self.audio:
            provided_fields += 1
        if self.video:
            provided_fields += 1

        if provided_fields != 1:
            raise ValueError("Exactly one of 'text', 'image', 'file', 'audio', or 'video' must be provided.")
        return self

    def __repr__(self):
        return f'ContentItem({self.model_dump()})'

   # 提供便捷访问类型和值的属性
   # Literal 类型确保 t 只能是预定义的五种内容类型之一（text, image, file, audio, video）
   # str 表示 v 的类型为字符串。但实际代码中，v 可能是字符串、字典或列表（取决于内容类型），这是代码的一个潜在问题（见下文分析）
    def get_type_and_value(self) -> Tuple[Literal['text', 'image', 'file', 'audio', 'video'], str]:
        (t, v), = self.model_dump().items()
        assert t in ('text', 'image', 'file', 'audio', 'video')
        return t, v

    @property
    def type(self) -> Literal['text', 'image', 'file', 'audio', 'video']:
        t, v = self.get_type_and_value()
        return t

    @property
    def value(self) -> str:
        t, v = self.get_type_and_value()
        return v


# Message类是一个用于表示多模态对话系统中消息结构的核心组件
class Message(BaseModelCompatibleDict):
    role: str
    content: Union[str, List[ContentItem]]
    reasoning_content: Optional[Union[str, List[ContentItem]]] = None
    name: Optional[str] = None
    function_call: Optional[FunctionCall] = None
    extra: Optional[dict] = None

    def __init__(self,
                 role: str,
                 content: Union[str, List[ContentItem]],
                 # 推理内容
                 reasoning_content: Optional[Union[str, List[ContentItem]]] = None,
                 name: Optional[str] = None,
                 function_call: Optional[FunctionCall] = None,
                 extra: Optional[dict] = None,
                 **kwargs):
        if content is None:
            content = ''
        super().__init__(role=role,
                         content=content,
                         reasoning_content=reasoning_content,
                         name=name,
                         function_call=function_call,
                         extra=extra)

    def __repr__(self):
        return f'Message({self.model_dump()})'

    @field_validator('role')
    def role_checker(cls, value: str) -> str:
        if value not in [USER, ASSISTANT, SYSTEM, FUNCTION]:
            raise ValueError(f'{value} must be one of {",".join([USER, ASSISTANT, SYSTEM, FUNCTION])}')
        return value
