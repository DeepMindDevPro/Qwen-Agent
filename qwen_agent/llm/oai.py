import copy
import logging
import os
from pprint import pformat
from typing import Dict, Iterator, List, Optional

import openai

if openai.__version__.startswith('0.'):
    from openai.error import OpenAIError  # noqa
else:
    from openai import OpenAIError

from qwen_agent.llm.base import ModelServiceError, register_llm
from qwen_agent.llm.function_calling import BaseFnCallModel
from qwen_agent.llm.schema import ASSISTANT, Message
from qwen_agent.log import logger

'''
TextChatAtOAI 类是一个用于与 OpenAI 模型进行文本对话的类，它继承自 BaseFnCallModel，支持流式和非流式的对话方式，
并且能够处理 OpenAI API 的不同版本。该类通过装饰器 @register_llm('oai') 注册到 LLM_REGISTRY 中，方便后续根据模型类型调用
'''

# 使用 register_llm 装饰器将该类注册到 LLM_REGISTRY 中，模型类型为 'oai'
@register_llm('oai')
class TextChatAtOAI(BaseFnCallModel):
    """
    用于与 OpenAI 模型进行文本对话的类。
    支持流式和非流式的对话方式，并且能够处理 OpenAI API 的不同版本。
    """
    def __init__(self, cfg: Optional[Dict] = None):
        """
        初始化 TextChatAtOAI 类。
        参数:
        cfg (Optional[Dict]): 配置字典，包含 API 相关的配置信息，如 api_base、api_key 等。
        """
        # 调用父类的初始化方法
        super().__init__(cfg)
        # 设置默认模型为 'gpt-4o-mini'
        self.model = self.model or 'gpt-4o-mini'
        # 如果 cfg 为 None，则初始化为空字典
        cfg = cfg or {}

        # 获取 API 基础地址，优先从 cfg 中获取，依次尝试 'api_base'、'base_url'、'model_server'
        api_base = cfg.get('api_base')
        api_base = api_base or cfg.get('base_url')
        api_base = api_base or cfg.get('model_server')
        # 去除首尾空格
        api_base = (api_base or '').strip()

        # 获取 API 密钥，优先从 cfg 中获取，若没有则从环境变量 'OPENAI_API_KEY' 中获取
        api_key = cfg.get('api_key')
        api_key = api_key or os.getenv('OPENAI_API_KEY')
        # 去除首尾空格，若为空则设置为 'EMPTY'
        api_key = (api_key or 'EMPTY').strip()

        # 根据 OpenAI API 版本进行不同的初始化
        if openai.__version__.startswith('0.'):
            # 旧版本 API
            if api_base:
                # 设置 API 基础地址
                openai.api_base = api_base
            if api_key:
                # 设置 API 密钥
                openai.api_key = api_key
            # 初始化完成请求和聊天完成请求的方法
            self._complete_create = openai.Completion.create
            self._chat_complete_create = openai.ChatCompletion.create
        else:
            # 新版本 API
            api_kwargs = {}
            if api_base:
                # 将 API 基础地址添加到 api_kwargs 中
                api_kwargs['base_url'] = api_base
            if api_key:
                # 将 API 密钥添加到 api_kwargs 中
                api_kwargs['api_key'] = api_key

            def _chat_complete_create(*args, **kwargs):
                """
                封装的聊天完成请求方法，处理新版本 API 不允许的参数。

                参数:
                *args: 位置参数
                **kwargs: 关键字参数

                返回:
                openai.ChatCompletion.create 的返回结果
                """
                # 新版本 API 不允许的参数，需要通过 extra_body 传递
                extra_params = ['top_k', 'repetition_penalty']
                if any((k in kwargs) for k in extra_params):
                    # 复制 extra_body 字典
                    kwargs['extra_body'] = copy.deepcopy(kwargs.get('extra_body', {}))
                    for k in extra_params:
                        if k in kwargs:
                            # 将不允许的参数移动到 extra_body 中
                            kwargs['extra_body'][k] = kwargs.pop(k)
                if 'request_timeout' in kwargs:
                    # 将 request_timeout 重命名为 timeout
                    kwargs['timeout'] = kwargs.pop('request_timeout')

                # 创建 OpenAI 客户端
                client = openai.OpenAI(**api_kwargs)
                return client.chat.completions.create(*args, **kwargs)

            def _complete_create(*args, **kwargs):
                """
                封装的完成请求方法，处理新版本 API 不允许的参数。

                参数:
                *args: 位置参数
                **kwargs: 关键字参数

                返回:
                openai.Completions.create 的返回结果
                """
                # 新版本 API 不允许的参数，需要通过 extra_body 传递
                extra_params = ['top_k', 'repetition_penalty']
                if any((k in kwargs) for k in extra_params):
                    # 复制 extra_body 字典
                    kwargs['extra_body'] = copy.deepcopy(kwargs.get('extra_body', {}))
                    for k in extra_params:
                        if k in kwargs:
                            # 将不允许的参数移动到 extra_body 中
                            kwargs['extra_body'][k] = kwargs.pop(k)
                if 'request_timeout' in kwargs:
                    # 将 request_timeout 重命名为 timeout
                    kwargs['timeout'] = kwargs.pop('request_timeout')

                # 创建 OpenAI 客户端
                client = openai.OpenAI(**api_kwargs)
                return client.completions.create(*args, **kwargs)

            # 初始化完成请求和聊天完成请求的方法
            self._complete_create = _complete_create
            self._chat_complete_create = _chat_complete_create

    def _chat_stream(
        self,
        messages: List[Message],
        delta_stream: bool,
        generate_cfg: dict,
    ) -> Iterator[List[Message]]:
        """
        流式聊天方法，支持增量流式和全量流式。

        参数:
        messages (List[Message]): 输入的消息列表
        delta_stream (bool): 是否使用增量流式
        generate_cfg (dict): 生成配置字典

        返回:
        Iterator[List[Message]]: 流式响应的消息列表迭代器
        """
        # 将消息列表转换为字典列表
        messages = self.convert_messages_to_dicts(messages)
        try:
            # 发起流式聊天请求
            response = self._chat_complete_create(model=self.model, messages=messages, stream=True, **generate_cfg)
            if delta_stream:
                # 增量流式处理
                for chunk in response:
                    if chunk.choices:
                        if hasattr(chunk.choices[0].delta, 'reasoning_content') and chunk.choices[0].delta.reasoning_content:
                            # 处理推理内容
                            yield [
                                Message(role=ASSISTANT,
                                        content='',
                                        reasoning_content=chunk.choices[0].delta.reasoning_content)
                            ]
                        if hasattr(chunk.choices[0].delta, 'content') and chunk.choices[0].delta.content:
                            # 处理内容
                            yield [Message(role=ASSISTANT, content=chunk.choices[0].delta.content)]
            else:
                # 全量流式处理
                full_response = ''
                full_reasoning_content = ''
                for chunk in response:
                    if chunk.choices:
                        if hasattr(chunk.choices[0].delta, 'reasoning_content') and chunk.choices[0].delta.reasoning_content:
                            # 累加推理内容
                            full_reasoning_content += chunk.choices[0].delta.reasoning_content
                        if hasattr(chunk.choices[0].delta, 'content') and chunk.choices[0].delta.content:
                            # 累加内容
                            full_response += chunk.choices[0].delta.content
                        # 生成全量响应消息
                        yield [Message(role=ASSISTANT, content=full_response, reasoning_content=full_reasoning_content)]
        except OpenAIError as ex:
            # 捕获 OpenAI 错误并抛出 ModelServiceError
            raise ModelServiceError(exception=ex)

    def _chat_no_stream(
        self,
        messages: List[Message],
        generate_cfg: dict,
    ) -> List[Message]:
        """
        非流式聊天方法。

        参数:
        messages (List[Message]): 输入的消息列表
        generate_cfg (dict): 生成配置字典

        返回:
        List[Message]: 响应的消息列表
        """
        # 将消息列表转换为字典列表
        messages = self.convert_messages_to_dicts(messages)
        try:
            # 发起非流式聊天请求
            response = self._chat_complete_create(model=self.model, messages=messages, stream=False, **generate_cfg)
            if hasattr(response.choices[0].message, 'reasoning_content'):
                # 处理包含推理内容的响应
                return [
                    Message(role=ASSISTANT,
                            content=response.choices[0].message.content,
                            reasoning_content=response.choices[0].message.reasoning_content)
                ]
            else:
                # 处理不包含推理内容的响应
                return [Message(role=ASSISTANT, content=response.choices[0].message.content)]
        except OpenAIError as ex:
            # 捕获 OpenAI 错误并抛出 ModelServiceError
            raise ModelServiceError(exception=ex)

    @staticmethod
    def convert_messages_to_dicts(messages: List[Message]) -> List[dict]:
        """
        将消息列表转换为字典列表。

        参数:
        messages (List[Message]): 输入的消息列表

        返回:
        List[dict]: 转换后的字典列表
        """
        # 将消息对象转换为字典
        messages = [msg.model_dump() for msg in messages]

        if logger.isEnabledFor(logging.DEBUG):
            # 记录调试信息
            logger.debug(f'LLM Input:\n{pformat(messages, indent=2)}')
        return messages