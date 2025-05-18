import copy
import json
import os
import random
import time
from abc import ABC, abstractmethod
from pprint import pformat
from typing import Any, Dict, Iterator, List, Literal, Optional, Tuple, Union

from qwen_agent.llm.schema import ASSISTANT, DEFAULT_SYSTEM_MESSAGE, SYSTEM, USER, FUNCTION, Message
from qwen_agent.log import logger
from qwen_agent.settings import DEFAULT_MAX_INPUT_TOKENS
from qwen_agent.utils.tokenization_qwen import tokenizer
from qwen_agent.utils.utils import (extract_text_from_message, format_as_multimodal_message, format_as_text_message,
                                    has_chinese_messages, json_dumps_compact, merge_generate_cfgs, print_traceback)

# 用于存储已注册的 LLM 模型类型及其对应的类
LLM_REGISTRY = {}


# 定义一个装饰器，用于将模型类型注册到 LLM_REGISTRY 中
def register_llm(model_type):
    def decorator(cls):
        # 将模型类型和对应的类添加到 LLM_REGISTRY 中
        LLM_REGISTRY[model_type] = cls
        return cls

    return decorator

#ModelServiceError 类是一个自定义的异常类，继承自 Python 内置的 Exception 类
class ModelServiceError(Exception):
    '''
    exception：一个可选的异常对象，类型为 Exception。如果提供了该参数，说明错误是由另一个异常引发的。
    code：一个可选的字符串，表示错误代码。错误代码可以用于标识不同类型的错误，方便后续的错误处理和调试。
    message：一个可选的字符串，表示错误消息。错误消息通常包含对错误的详细描述，帮助开发者理解错误的原因。
    extra：一个可选的字典，用于存储额外的错误信息。这些信息可以是与错误相关的上下文信息，例如请求的参数、模型的配置等。
    '''
    def __init__(self,
                 exception: Optional[Exception] = None,
                 code: Optional[str] = None,
                 message: Optional[str] = None,
                 extra: Optional[dict] = None):

        if exception is not None:
            # 如果传入了异常对象，使用该异常初始化父类
            super().__init__(exception)
        else:
            # 否则，使用错误代码和消息初始化父类
            super().__init__(f'\nError code: {code}. Error message: {message}')
        self.exception = exception
        self.code = code
        self.message = message
        self.extra = extra


# 定义 LLM 的基类，继承自 ABC 类，表示这是一个抽象基类
# BaseChatModel类是一个抽象基类（Abstract Base Class, ABC），用于定义大语言模型（LLM）的基本接口和行为
class BaseChatModel(ABC):
    """The base class of LLM"""

    '''
    support_multimodal_input：这是一个只读属性，用于判断模型是否原生支持多模态输入。多模态输入可能包括图像、音频、视频等多种形式。
    默认返回 False，表示模型不支持多模态输入。在实际使用中，具体的 LLM 类可以重写该属性，以表明其支持多模态输入的能力
    '''
    @property #属性
    def support_multimodal_input(self) -> bool:
        # 判断模型是否原生支持多模态输入，这会影响输入的预处理方式
        return False

    '''
    support_multimodal_output：同样是只读属性，用于判断模型是否能生成除文本之外的多模态输出。
    默认返回 False，表示模型仅能生成文本输出。具体的 LLM 类可以根据自身情况重写该属性
    '''
    @property
    def support_multimodal_output(self) -> bool:
        # 判断模型是否能生成除文本之外的多模态输出，这会影响输出的后处理方式
        return False

    '''
    support_audio_input：只读属性，用于判断模型是否支持音频输入。默认返回 False，具体的 LLM 类可以重写该属性.
    '''
    @property
    def support_audio_input(self) -> bool:
        return False

    def __init__(self, cfg: Optional[Dict] = None):
        # 如果没有传入配置，使用空字典
        cfg = cfg or {}
        # 获取模型名称并去除首尾空格
        self.model = cfg.get('model', '').strip()
        # 深拷贝生成配置
        generate_cfg = copy.deepcopy(cfg.get('generate_cfg', {}))
        # 获取缓存目录
        cache_dir = cfg.get('cache_dir', generate_cfg.pop('cache_dir', None))
        # 获取最大重试次数
        self.max_retries = generate_cfg.pop('max_retries', 0)
        # 保存生成配置
        self.generate_cfg = generate_cfg
        # 获取模型类型
        self.model_type = cfg.get('model_type', '')
        # 如果模型类型包含 'dashscope'，设置增量输出为 True
        # 这是为阿里云平台定制的
        if 'dashscope' in self.model_type:
            self.generate_cfg['incremental_output'] = True

        # 创建缓存对象
        if cache_dir:
            try:
                import diskcache
            except ImportError:
                # 打印导入错误信息
                print_traceback(is_error=False)
                logger.warning('Caching disabled because diskcache is not installed. Please `pip install diskcache`.')
                cache_dir = None
        if cache_dir:
            # 创建缓存目录
            os.makedirs(cache_dir, exist_ok=True)
            # 初始化缓存对象
            self.cache = diskcache.Cache(directory=cache_dir)
        else:
            self.cache = None

    '''
    该方法接收一个字符串类型的提示信息 prompt 作为输入，调用 chat 方法与大语言模型进行交互，然后对响应进行一系列的验证，
    确保响应符合预期，最后返回响应的文本内容
    '''
    def quick_chat(self, prompt: str) -> str:
        # 调用 chat 方法进行聊天
        '''
        messages=[Message(role=USER, content=prompt)]：构造一个消息列表，其中包含一个用户消息。Message 类用于封装消息信息，
        role=USER 表示这是用户的消息，content=prompt 表示消息的内容为用户输入的提示信息.
        '''
        '''
        *_, responses：使用解包操作获取 chat 方法返回的最后一个元素作为 responses。*_ 表示忽略前面的元素，只关注最后一个元素
        '''
        *_, responses = self.chat(messages=[Message(role=USER, content=prompt)])
        # 确保响应只有一个
        assert len(responses) == 1
        # 确保没有函数调用
        assert not responses[0].function_call
        # 确保响应内容是字符串
        # 使用 assert 语句进行断言，确保响应的内容是字符串类型。isinstance 函数用于检查对象是否为指定类型，
        # 如果响应内容不是字符串类型，则会抛出 AssertionError 异常。
        assert isinstance(responses[0].content, str)
        return responses[0].content


   # 方法注释如下
   #messages：该参数用于接收输入的消息列表，messages 是 “消息” 的复数形式，明确表示这是一个包含多个消息的列表
   #functions：用于传递函数调用的信息，在支持函数调用的大语言模型中，开发者可以定义一些函数，让模型根据需要调用这些函数来完成特定的任务。
    # functions 是 “函数” 的复数形式，清晰地表明这是一个包含多个函数定义的列表

   #stream：表示是否使用流式生成。在大语言模型的交互中，流式生成可以让模型逐步输出结果，而不是等待整个结果生成后再返回，这样可以提高用户体验。
    # stream 是 “流” 的意思，使用 stream 作为参数名能够准确地表达这个功能

    #delta_stream：用于控制是否以增量方式流式传输响应。当 delta_stream 为 True 时，模型会逐块返回响应；当为 False 时，会在每次迭代中返回完整的响应。
    # delta 有 “增量” 的意思，delta_stream 这个参数名明确表示了该参数与增量流式传输的关系

    #extra_generate_cfg：表示额外的生成配置参数。在使用大语言模型时，除了基本的配置外，开发者可能还需要传递一些额外的参数来控制模型的生成过程，
    # 如最大生成长度、温度等。extra 表示 “额外的”，generate_cfg 表示 “生成配置”，使用 extra_generate_cfg 作为参数名能够准确地表达这个含义

    '''
    返回值类型使用了 Union 类型注解，表示该方法可能返回多种不同类型的值。具体来说，可能返回一个消息列表（List[Message]）、一个字典列表（List[Dict]）、
    一个消息列表的迭代器（Iterator[List[Message]]）或一个字典列表的迭代器（Iterator[List[Dict]]）。这种类型注解能够让调用者清楚地知道该方法可能返回的结果类型，
    提高了代码的可读性和可维护性
    '''

    def chat(
            self,
            messages: List[Union[Message, Dict]],
            functions: Optional[List[Dict]] = None,
            stream: bool = True,
            delta_stream: bool = False,
            extra_generate_cfg: Optional[Dict] = None,
    ) -> Union[List[Message], List[Dict], Iterator[List[Message]], Iterator[List[Dict]]]:
        """LLM chat interface.
        Args:
            messages: Inputted messages.
            functions: Inputted functions for function calling. OpenAI format supported.
            stream: Whether to use streaming generation.
            delta_stream: Whether to stream the response incrementally.
              (1) When False (recommended): Stream the full response every iteration.
              (2) When True: Stream the chunked response, i.e, delta responses.
            extra_generate_cfg: Extra LLM generation hyper-paramters.
        Returns:
            the generated message list response by llm.
        """
        # 深拷贝输入消息
        messages = copy.deepcopy(messages)
        # 初始化返回消息类型为 'dict'
        _return_message_type = 'dict'
        # 定义输入消息列表
        new_messages = []
        # 统一输入消息类型为 List[Message]
        for msg in messages:
            #字典类型
            if isinstance(msg, dict):
                new_messages.append(Message(**msg))
            else:
                #Message类型
                new_messages.append(msg)
                _return_message_type = 'message'
        messages = new_messages

        # 校验消息
        if not messages:
            # 如果消息列表为空，抛出 ValueError 异常
            raise ValueError("Messages can not be empty.")

        # 缓存查找
        # 这段代码通过缓存机制，避免了重复的模型调用，提高了系统的响应速度和效率
        if self.cache is not None:
            # 生成缓存键
            cache_key = dict(messages=messages, functions=functions, extra_generate_cfg=extra_generate_cfg)
            # dict -> json str
            # 将字典转换为字符串：json_dumps_compact 是一个自定义函数，用于将字典转换为紧凑的 JSON 字符串。sort_keys=True 表示对字典的键进行排序，
            # 确保相同的输入会生成相同的缓存键
            cache_key: str = json_dumps_compact(cache_key, sort_keys=True)
            # 从缓存中获取值
            cache_value: str = self.cache.get(cache_key)
            if cache_value:
                #转成python列表
                cache_value: List[dict] = json.loads(cache_value)
                if _return_message_type == 'message':
                    #则将列表中的每个字典转换为 Message 对象
                    cache_value: List[Message] = [Message(**m) for m in cache_value]
                if stream:
                    #如果 stream 为 True，表示需要使用流式输出。使用 iter([cache_value]) 将缓存值列表转换为迭代器，以便按流式方式返回结果
                    cache_value: Iterator[List[Union[Message, dict]]] = iter([cache_value])
                return cache_value

        if stream and delta_stream:
            # 提示不推荐使用 delta_stream=True
            logger.warning(
                'Support for `delta_stream=True` is deprecated. '
                'Please use `stream=True and delta_stream=False` or `stream=False` instead. '
                'Using `delta_stream=True` makes it difficult to implement advanced postprocessing and retry mechanisms.'
            )

        # 合并生成配置
        generate_cfg = merge_generate_cfgs(base_generate_cfg=self.generate_cfg, new_generate_cfg=extra_generate_cfg)
        if 'seed' not in generate_cfg:
            # 如果配置中没有种子，随机生成一个
            generate_cfg['seed'] = random.randint(a=0, b=2 ** 30)
        if 'lang' in generate_cfg:
            # 如果配置中有语言信息，取出并删除
            # Literal['en', 'zh'] 是类型注解，表示 lang 只能是 'en'（英语）或 'zh'（中文）
            lang: Literal['en', 'zh'] = generate_cfg.pop('lang')
        else:
            # 根据消息内容判断语言
            lang: Literal['en', 'zh'] = 'zh' if has_chinese_messages(messages) else 'en'
        if not stream and 'incremental_output' in generate_cfg:
            # 如果不使用流式输出，删除增量输出配置
            generate_cfg.pop('incremental_output')

        if DEFAULT_SYSTEM_MESSAGE and messages[0].role != SYSTEM:
            # 如果有默认系统消息且第一条消息不是系统消息，添加系统消息
            messages = [Message(role=SYSTEM, content=DEFAULT_SYSTEM_MESSAGE)] + messages

        # 粗略截断输入消息，避免超过最大输入令牌数
        max_input_tokens = generate_cfg.pop('max_input_tokens', DEFAULT_MAX_INPUT_TOKENS)
        if max_input_tokens > 0:
            messages = _truncate_input_messages_roughly(
                messages=messages,
                max_tokens=max_input_tokens,
            )

        if functions:
            # 如果有函数列表，开启函数调用模式
            fncall_mode = True
        else:
            fncall_mode = False
        if 'function_choice' in generate_cfg:
            # 检查 function_choice 参数是否合法
            fn_choice = generate_cfg['function_choice']
            valid_fn_choices = [f.get('name', f.get('name_for_model', None)) for f in (functions or [])]
            valid_fn_choices = ['auto', 'none'] + [f for f in valid_fn_choices if f]
            if fn_choice not in valid_fn_choices:
                raise ValueError(f'The value of function_choice must be one of the following: {valid_fn_choices}. '
                                 f'But function_choice="{fn_choice}" is received.')
            if fn_choice == 'none':
                fncall_mode = False

        # 预处理消息
        messages = self._preprocess_messages(messages, lang=lang, generate_cfg=generate_cfg, functions=functions)
        if not self.support_multimodal_input:
            # 如果不支持多模态输入，将消息转换为文本消息
            messages = [format_as_text_message(msg, add_upload_info=False) for msg in messages]

        if not fncall_mode:
            # 如果不开启函数调用模式，删除相关配置
            for k in ['parallel_function_calls', 'function_choice', 'thought_in_content']:
                if k in generate_cfg:
                    del generate_cfg[k]

        def _call_model_service():
            if fncall_mode:
                # 如果开启函数调用模式，调用 _chat_with_functions 方法
                return self._chat_with_functions(
                    messages=messages,
                    functions=functions,
                    stream=stream,
                    delta_stream=delta_stream,
                    generate_cfg=generate_cfg,
                    lang=lang,
                )
            else:
                # 否则，根据消息情况调用相应方法
                if messages[-1].role == ASSISTANT:
                    assert not delta_stream, 'Continuation mode does not currently support `delta_stream`'
                    return self._continue_assistant_response(messages, generate_cfg=generate_cfg, stream=stream)
                else:
                    return self._chat(
                        messages,
                        stream=stream,
                        delta_stream=delta_stream,
                        generate_cfg=generate_cfg,
                    )

        if stream and delta_stream:
            # 如果使用流式增量输出，不进行重试
            output = _call_model_service()
        elif stream and (not delta_stream):
            # 如果使用流式全量输出，进行重试
            output = retry_model_service_iterator(_call_model_service, max_retries=self.max_retries)
        else:
            # 如果不使用流式输出，进行重试
            output = retry_model_service(_call_model_service, max_retries=self.max_retries)

        if isinstance(output, list):
            # 如果输出是列表，说明不是流式输出
            assert not stream
            # 打印 LLM 输出信息
            logger.debug(f'LLM Output:\n{pformat([_.model_dump() for _ in output], indent=2)}')
            # 后处理消息
            output = self._postprocess_messages(output, fncall_mode=fncall_mode, generate_cfg=generate_cfg)
            if not self.support_multimodal_output:
                # 如果不支持多模态输出，将消息转换为文本消息
                output = _format_as_text_messages(messages=output)
            if self.cache:
                # 如果有缓存，将输出存入缓存
                self.cache.set(cache_key, json_dumps_compact(output))
            # 将消息转换为目标类型
            return self._convert_messages_to_target_type(output, _return_message_type)
        else:
            # 如果输出是迭代器，说明是流式输出
            assert stream
            if delta_stream:
                # 如果使用流式增量输出，避免在处理停止词时出错
                generate_cfg = copy.deepcopy(generate_cfg)
                assert 'skip_stopword_postproc' not in generate_cfg
                generate_cfg['skip_stopword_postproc'] = True
            # 后处理消息迭代器
            output = self._postprocess_messages_iterator(output, fncall_mode=fncall_mode, generate_cfg=generate_cfg)

            def _format_and_cache() -> Iterator[List[Message]]:
                o = []
                for o in output:
                    if o:
                        if not self.support_multimodal_output:
                            o = _format_as_text_messages(messages=o)
                        yield o
                if o and (self.cache is not None):
                    # 如果有缓存，将最后一次输出存入缓存
                    self.cache.set(cache_key, json_dumps_compact(o))

            # 将消息迭代器转换为目标类型
            return self._convert_messages_iterator_to_target_type(_format_and_cache(), _return_message_type)

    def _chat(
            self,
            messages: List[Union[Message, Dict]],
            stream: bool,
            delta_stream: bool,
            generate_cfg: dict,
    ) -> Union[List[Message], Iterator[List[Message]]]:
        if stream:
            # 如果使用流式输出，调用 _chat_stream 方法
            return self._chat_stream(messages, delta_stream=delta_stream, generate_cfg=generate_cfg)
        else:
            # 否则，调用 _chat_no_stream 方法
            return self._chat_no_stream(messages, generate_cfg=generate_cfg)

    @abstractmethod
    def _chat_with_functions(
            self,
            messages: List[Union[Message, Dict]],
            functions: List[Dict],
            stream: bool,
            delta_stream: bool,
            generate_cfg: dict,
            lang: Literal['en', 'zh'],
    ) -> Union[List[Message], Iterator[List[Message]]]:
        # 抽象方法，用于处理带函数调用的聊天，子类必须实现
        raise NotImplementedError

    def _continue_assistant_response(
            self,
            messages: List[Message],
            generate_cfg: dict,
            stream: bool,
    ) -> Iterator[List[Message]]:
        # 抽象方法，用于继续生成助手的响应，子类必须实现
        raise NotImplementedError

    @abstractmethod
    def _chat_stream(
            self,
            messages: List[Message],
            delta_stream: bool,
            generate_cfg: dict,
    ) -> Iterator[List[Message]]:
        # 抽象方法，用于流式聊天，子类必须实现
        raise NotImplementedError

    @abstractmethod
    def _chat_no_stream(
            self,
            messages: List[Message],
            generate_cfg: dict,
    ) -> List[Message]:
        # 抽象方法，用于非流式聊天，子类必须实现
        raise NotImplementedError

    def _preprocess_messages(
            self,
            messages: List[Message],
            lang: Literal['en', 'zh'],
            generate_cfg: dict,
            functions: Optional[List[Dict]] = None,
    ) -> List[Message]:
        add_multimodel_upload_info = False
        if functions or (not self.support_multimodal_input):
            # 如果有函数列表或不支持多模态输入，添加多模态上传信息
            add_multimodel_upload_info = True
        add_audio_upload_info = False
        if functions or (not self.support_audio_input):
            # 如果有函数列表或不支持音频输入，添加音频上传信息
            add_audio_upload_info = True
        # 格式化消息为多模态消息
        messages = [
            format_as_multimodal_message(msg,
                                         add_upload_info=True,
                                         add_multimodel_upload_info=add_multimodel_upload_info,
                                         add_audio_upload_info=add_audio_upload_info,
                                         lang=lang) for msg in messages
        ]
        return messages

    def _postprocess_messages(
            self,
            messages: List[Message],
            fncall_mode: bool,
            generate_cfg: dict,
    ) -> List[Message]:
        # 格式化消息为多模态消息，不添加上传信息
        messages = [
            format_as_multimodal_message(msg,
                                         add_upload_info=False,
                                         add_multimodel_upload_info=False,
                                         add_audio_upload_info=False) for msg in messages
        ]
        if not generate_cfg.get('skip_stopword_postproc', False):
            # 如果不跳过停止词后处理，处理停止词
            stop = generate_cfg.get('stop', [])
            messages = _postprocess_stop_words(messages, stop=stop)
        return messages

    def _postprocess_messages_iterator(
            self,
            messages: Iterator[List[Message]],
            fncall_mode: bool,
            generate_cfg: dict,
    ) -> Iterator[List[Message]]:
        pre_msg = []
        for pre_msg in messages:
            # 对每个消息列表进行后处理
            yield self._postprocess_messages(pre_msg, fncall_mode=fncall_mode, generate_cfg=generate_cfg)
        # 打印 LLM 输出信息
        logger.debug(f'LLM Output:\n{pformat([_.model_dump() for _ in pre_msg], indent=2)}')

    def _convert_messages_to_target_type(self, messages: List[Message],
                                         target_type: str) -> Union[List[Message], List[Dict]]:
        if target_type == 'message':
            # 如果目标类型是 'message'，将字典转换为 Message 对象
            return [Message(**x) if isinstance(x, dict) else x for x in messages]
        elif target_type == 'dict':
            # 如果目标类型是 'dict'，将 Message 对象转换为字典
            return [x.model_dump() if not isinstance(x, dict) else x for x in messages]
        else:
            # 如果目标类型不合法，抛出 NotImplementedError 异常
            raise NotImplementedError

    def _convert_messages_iterator_to_target_type(
            self, messages_iter: Iterator[List[Message]],
            target_type: str) -> Union[Iterator[List[Message]], Iterator[List[Dict]]]:
        for messages in messages_iter:
            # 对每个消息列表进行类型转换
            yield self._convert_messages_to_target_type(messages, target_type)

    def quick_chat_oai(self, messages: List[dict], tools: Optional[list] = None) -> dict:
        """
        This is a temporary OpenAI-compatible interface that is encapsulated and may change at any time.
        It is mainly used for temporary interfaces and should not be overly dependent.
        - Only supports full streaming
        - The message is in dict format
        - Only supports text LLM
        """

        def _convert_to_qwen_agent_messages(messages):
            new_messages = []
            for msg in messages:
                if msg['role'] in ['system', 'user']:
                    new_messages.append(msg)
                elif msg['role'] == 'tool':
                    # 将 'tool' 角色转换为 'function' 角色
                    msg['role'] = 'function'
                    new_messages.append(msg)
                elif msg['role'] == 'assistant':
                    if msg['content']:
                        new_messages.append({'role': 'assistant', 'content': msg['content']})
                    if msg.get('tool_calls'):
                        for tool in msg.get('tool_calls'):
                            new_messages.append({
                                'role': 'assistant',
                                'content': '',
                                'function_call': {
                                    'name': tool['function']['name'],
                                    'arguments': tool['function']['arguments']
                                }
                            })
            return new_messages

        def _convert_to_oai_message(data):
            message = {'role': 'assistant', 'content': '', 'reasoning_content': '', 'tool_calls': []}

            for item in data:
                if item.get('reasoning_content'):
                    message['reasoning_content'] += item['reasoning_content']

                if item.get('content'):
                    message['content'] += item['content']

                if 'function_call' in item:
                    tool_call = {
                        'id': f"{len(message['tool_calls']) + 1}",
                        'type': 'function',
                        'function': {
                            'name': item['function_call']['name'],
                            'arguments': item['function_call']['arguments']
                        }
                    }
                    message['tool_calls'].append(tool_call)
            # 伪造令牌使用信息
            response = {
                'choices': [{
                    'message': message
                }],
                'usage': {
                    'prompt_tokens': 0,
                    'completion_tokens': 0,
                    'total_tokens': 0
                }
            }
            return response

        if tools:
            functions = [tool['function'] for tool in tools]
        else:
            functions = None
        for rsp in self.chat(
                messages=_convert_to_qwen_agent_messages(messages),
                functions=functions,
                stream=True,
        ):
            # 对每个响应进行转换并生成
            yield _convert_to_oai_message(rsp)


# 将消息转换为文本消息
def _format_as_text_messages(messages: List[Message]) -> List[Message]:
    for msg in messages:
        if isinstance(msg.content, list):
            for item in msg.content:
                assert item.type == 'text'
        else:
            assert isinstance(msg.content, str)
    messages = [format_as_text_message(msg, add_upload_info=False) for msg in messages]
    return messages


# 处理停止词，确保消息在停止词之前停止
def _postprocess_stop_words(messages: List[Message], stop: List[str]) -> List[Message]:
    messages = copy.deepcopy(messages)

    # Make sure it stops before stop words.
    trunc_messages = []
    for msg in messages:
        truncated = False
        trunc_content = []
        for i, item in enumerate(msg.content):
            item_type, item_text = item.get_type_and_value()
            if item_type == 'text':
                truncated, item.text = _truncate_at_stop_word(text=item_text, stop=stop)
            trunc_content.append(item)
            if truncated:
                break
        msg.content = trunc_content
        trunc_messages.append(msg)
        if truncated:
            break
    messages = trunc_messages

    # It may ends with partial stopword 'Observation' when the full stopword is 'Observation:'.
    # The following post-processing step removes partial stop words.
    partial_stop = []
    for s in stop:
        s = tokenizer.tokenize(s)[:-1]
        if s:
            s = tokenizer.convert_tokens_to_string(s)
            partial_stop.append(s)
    partial_stop = sorted(set(partial_stop))
    last_msg = messages[-1].content
    for i in range(len(last_msg) - 1, -1, -1):
        item_type, item_text = last_msg[i].get_type_and_value()
        if item_type == 'text':
            for s in partial_stop:
                if item_text.endswith(s):
                    last_msg[i].text = item_text[:-len(s)]
            break

    return messages


# 在文本中查找停止词并截断文本
def _truncate_at_stop_word(text: str, stop: List[str]):
    truncated = False
    for s in stop:
        k = text.find(s)
        if k >= 0:
            truncated = True
            text = text[:k]
    return truncated, text


# 粗略截断输入消息，避免超过最大输入令牌数
def _truncate_input_messages_roughly(messages: List[Message], max_tokens: int) -> List[Message]:
    if len([m for m in messages if m.role == SYSTEM]) >= 2:
        # 如果系统消息数量超过 1 条，抛出 ModelServiceError 异常
        raise ModelServiceError(
            code='400',
            message='The input messages must contain no more than one system message. '
                    ' And the system message, if exists, must be the first message.',
        )

    turns = []
    for m in messages:
        if m.role == SYSTEM:
            continue
        elif m.role == USER:
            turns.append([m])
        else:
            if turns:
                turns[-1].append(m)
            else:
                # 如果消息不以用户消息开头，抛出 ModelServiceError 异常
                raise ModelServiceError(
                    code='400',
                    message='The input messages (excluding the system message) must start with a user message.',
                )

    def _count_tokens(msg: Message) -> int:
        # 计算消息的令牌数
        return tokenizer.count_tokens(extract_text_from_message(msg, add_upload_info=True))

    def _truncate_message(msg: Message, max_tokens: int, keep_both_sides: bool = False):
        if isinstance(msg.content, str):
            # 截断字符串类型的消息内容
            content = tokenizer.truncate(msg.content, max_token=max_tokens, keep_both_sides=keep_both_sides)
        else:
            text = []
            for item in msg.content:
                if not item.text:
                    return None
                text.append(item.text)
            text = '\n'.join(text)
            # 截断列表类型的消息内容
            content = tokenizer.truncate(text, max_token=max_tokens, keep_both_sides=keep_both_sides)
        return Message(role=msg.role, content=content)

    if messages and messages[0].role == SYSTEM:
        # 如果第一条消息是系统消息，计算可用令牌数
        sys_msg = messages[0]
        available_token = max_tokens - _count_tokens(sys_msg)
    else:
        sys_msg = None
        available_token = max_tokens

    token_cnt = 0
    new_messages = []
    for i in range(len(messages) - 1, -1, -1):
        if messages[i].role == SYSTEM:
            continue
        cur_token_cnt = _count_tokens(messages[i])
        if cur_token_cnt <= available_token:
            # 如果当前消息的令牌数小于等于可用令牌数，添加到新消息列表中
            new_messages = [messages[i]] + new_messages
            available_token -= cur_token_cnt
        else:
            if (messages[i].role == USER) and (i != len(messages) - 1):
                # 如果是用户消息且不是最后一条消息，截断消息
                _msg = _truncate_message(messages[i], max_tokens=available_token)
                if _msg:
                    new_messages = [_msg] + new_messages
                break
            elif messages[i].role == FUNCTION:
                # 如果是函数消息，截断消息并保留两边内容
                _msg = _truncate_message(messages[i], max_tokens=available_token, keep_both_sides=True)
                if _msg:
                    new_messages = [_msg] + new_messages
                else:
                    break
            else:
                # 计算总令牌数
                token_cnt = (max_tokens - available_token) + cur_token_cnt
                break

    if sys_msg is not None:
        # 如果有系统消息，添加到新消息列表开头
        new_messages = [sys_msg] + new_messages

    if (sys_msg is not None and len(new_messages) < 2) or (sys_msg is None and len(new_messages) < 1):
        # 如果新消息列表长度不足，抛出 ModelServiceError 异常
        raise ModelServiceError(
            code='400',
            message=f'The input messages exceed the maximum context length ({max_tokens} tokens) after '
                    f'keeping only the system message (if exists) and the latest one user message (around {token_cnt} tokens). '
                    'To configure the context limit, please specifiy "max_input_tokens" in the model generate_cfg. '
                    f'Example: generate_cfg = {{..., "max_input_tokens": {(token_cnt // 100 + 1) * 100}}}',
        )
    return new_messages


# 重试函数调用，处理模型服务错误
def retry_model_service(
        fn,
        max_retries: int = 10,
) -> Any:
    """Retry a function"""

    num_retries, delay = 0, 1.0
    while True:
        try:
            return fn()

        except ModelServiceError as e:
            num_retries, delay = _raise_or_delay(e, num_retries, delay, max_retries)


# 重试迭代器调用，处理模型服务错误
def retry_model_service_iterator(
        it_fn,
        max_retries: int = 10,
) -> Iterator:
    """Retry an iterator"""

    num_retries, delay = 0, 1.0
    while True:
        try:
            for rsp in it_fn():
                yield rsp
            break

        except ModelServiceError as e:
            num_retries, delay = _raise_or_delay(e, num_retries, delay, max_retries)


# 处理重试逻辑，使用指数退避算法
def _raise_or_delay(
        e: ModelServiceError,
        num_retries: int,
        delay: float,
        max_retries: int = 10,
        max_delay: float = 300.0,
        exponential_base: float = 2.0,
) -> Tuple[int, float]:
    """Retry with exponential backoff"""

    if max_retries <= 0:  # no retry
        raise e

    # Bad request, e.g., incorrect config or input
    if e.code == '400':
        raise e

    # If harmful input or output detected, let it fail
    if e.code == 'DataInspectionFailed':
        raise e
    if 'inappropriate content' in str(e):
        raise e

    # Retry is meaningless if the input is too long
    if 'maximum context length' in str(e):
        raise e

    logger.warning('ModelServiceError - ' + str(e).strip('\n'))

    if num_retries >= max_retries:
        raise ModelServiceError(exception=Exception(f'Maximum number of retries ({max_retries}) exceeded.'))

    num_retries += 1
    jitter = 1.0 + random.random()
    delay = min(delay * exponential_base, max_delay) * jitter
    time.sleep(delay)
    return num_retries, delay


# 移除文本中的思考标签
def _rm_think(text: str) -> str:
    if '</think>' in text:
        return text.split('</think>')[-1].lstrip('\n')
    return text