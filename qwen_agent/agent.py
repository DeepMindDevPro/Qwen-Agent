import copy
import json
import traceback
from abc import ABC, abstractmethod
from typing import Dict, Iterator, List, Optional, Tuple, Union

from qwen_agent.llm import get_chat_model
from qwen_agent.llm.base import BaseChatModel
from qwen_agent.llm.schema import CONTENT, DEFAULT_SYSTEM_MESSAGE, ROLE, SYSTEM, ContentItem, Message
from qwen_agent.log import logger
from qwen_agent.tools import TOOL_REGISTRY, BaseTool, MCPManager
from qwen_agent.tools.base import ToolServiceError
from qwen_agent.tools.simple_doc_parser import DocParserError
from qwen_agent.utils.utils import has_chinese_messages, merge_generate_cfgs

# 定义一个抽象基类 Agent，作为所有代理类的基类
class Agent(ABC):
    """A base class for Agent.

    An agent can receive messages and provide response by LLM or Tools.
    Different agents have distinct workflows for processing messages and generating responses in the `_run` method.
    """

    # 构造器
    def __init__(self,
                 #类型: Optional 表示该参数可以为 None。List[Union[str, Dict, BaseTool]] 表示这是一个列表，列表中的元素可以是字符串、字典或者 BaseTool 类型的对象
                 #作用：该参数用于指定可以使用的工具列表。字符串可能是工具的名称，字典可能是工具的配置信息，BaseTool 类型的对象则是具体的工具实例。
                 #     例如，['code_interpreter', {'name': 'code_interpreter', 'timeout': 10}, CodeInterpreter()]
                 # 默认值：None，表示不使用任何工具。
                 function_list: Optional[List[Union[str, Dict, BaseTool]]] = None,
                 #类型：同样是 Optional 类型，列表元素可以是字典或者 BaseChatModel 类型的对象
                 #作用：用于指定语言模型（LLM）的配置或实例。字典可以包含模型的配置信息，如 {'model': '', 'api_key': '', 'model_server': ''}；
                 #     BaseChatModel 类型的对象则是已经初始化好的语言模型实例
                 #默认值：None，表示不指定语言模型
                 llm: Optional[Union[dict, BaseChatModel]] = None,
                 #类型: Optional 类型的字符串
                 #作用：指定与语言模型进行聊天时的系统消息。系统消息可以为模型提供一些上下文信息或指令
                 #默认值：DEFAULT_SYSTEM_MESSAGE，这是一个预定义的默认系统消息
                 system_message: Optional[str] = DEFAULT_SYSTEM_MESSAGE,
                 # 类型：Optional 类型的字符串。
                 # 作用: 指定该代理的名称。在多代理系统中，名称可以用于标识不同的代理。
                 # 默认值：None，表示不指定代理名称
                 name: Optional[str] = None,
                 # 类型:Optional 类型的字符串。
                 # 作用: 提供该代理的描述信息。描述信息可以用于多代理场景，帮助其他代理了解该代理的功能和特点
                 # 默认值: None, 表示不提供代理描述.
                 description: Optional[str] = None,
                 #类型:可变关键字参数
                 #作用: 允许传递额外的关键字参数，这些参数可以在子类中根据需要进行处理。
                 #     例如，在某些子类中可能会使用 files 参数来指定文件列表
                 **kwargs):
        """Initialization the agent.
        Args:
            function_list: One list of tool name, tool configuration or Tool object,
              such as 'code_interpreter', {'name': 'code_interpreter', 'timeout': 10}, or CodeInterpreter().
            llm: The LLM model configuration or LLM model object.
              Set the configuration as {'model': '', 'api_key': '', 'model_server': ''}.
            system_message: The specified system message for LLM chat.
            name: The name of this agent.
            description: The description of this agent, which will be used for multi_agent.
        """
        # 如果传入的 llm 是字典类型，调用 get_chat_model 函数实例化 LLM 对象
        if isinstance(llm, dict):
            self.llm = get_chat_model(llm)
        else:
            self.llm = llm
        # 初始化额外的生成配置
        self.extra_generate_cfg: dict = {}

        # 初始化工具映射字典
        self.function_map = {}
        # 如果传入了工具列表，初始化工具
        if function_list:
            for tool in function_list:
                self._init_tool(tool)

        # 设置系统消息
        self.system_message = system_message
        # 设置代理名称
        self.name = name
        # 设置代理描述
        self.description = description

    def run_nonstream(self, messages: List[Union[Dict, Message]], **kwargs) -> Union[List[Message], List[Dict]]:
        """Same as self.run, but with stream=False,
        meaning it returns the complete response directly
        instead of streaming the response incrementally."""
        # 调用 run 方法获取响应，并返回最后一次响应
        *_, last_responses = self.run(messages, **kwargs)
        return last_responses

    def run(self, messages: List[Union[Dict, Message]],
            **kwargs) -> Union[Iterator[List[Message]], Iterator[List[Dict]]]:
        """Return one response generator based on the received messages.

        This method performs a uniform type conversion for the inputted messages,
        and calls the _run method to generate a reply.

        Args:
            messages: A list of messages.

        Yields:
            The response generator.
        """
        # 深拷贝消息列表，避免修改原始数据
        messages = copy.deepcopy(messages)
        # 初始化返回消息类型为字典
        _return_message_type = 'dict'
        new_messages = []
        # 如果消息列表为空，返回消息类型为 Message 对象
        if not messages:
            _return_message_type = 'message'
        # 遍历消息列表，将字典类型的消息转换为 Message 对象
        for msg in messages:
            if isinstance(msg, dict):
                new_messages.append(Message(**msg))
            else:
                new_messages.append(msg)
                _return_message_type = 'message'

        # 如果未指定语言，根据消息内容判断语言
        if 'lang' not in kwargs:
            if has_chinese_messages(new_messages):
                kwargs['lang'] = 'zh'
            else:
                kwargs['lang'] = 'en'

        # 如果存在系统消息，将其添加到消息列表中
        if self.system_message:
            if not new_messages or new_messages[0][ROLE] != SYSTEM:
                # 在消息列表开头插入系统消息
                new_messages.insert(0, Message(role=SYSTEM, content=self.system_message))
            else:
                # 如果消息列表中已经有系统消息，将其与当前系统消息合并
                if isinstance(new_messages[0][CONTENT], str):
                    new_messages[0][CONTENT] = self.system_message + '\n\n' + new_messages[0][CONTENT]
                else:
                    assert isinstance(new_messages[0][CONTENT], list)
                    assert new_messages[0][CONTENT][0].text
                    new_messages[0][CONTENT] = [ContentItem(text=self.system_message + '\n\n')
                                               ] + new_messages[0][CONTENT]  # noqa

        # 调用 _run 方法生成响应
        for rsp in self._run(messages=new_messages, **kwargs):
            # 为没有名称的响应消息添加代理名称
            for i in range(len(rsp)):
                if not rsp[i].name and self.name:
                    rsp[i].name = self.name
            # 根据返回消息类型进行处理
            if _return_message_type == 'message':
                yield [Message(**x) if isinstance(x, dict) else x for x in rsp]
            else:
                yield [x.model_dump() if not isinstance(x, dict) else x for x in rsp]

    @abstractmethod
    def _run(self, messages: List[Message], lang: str = 'en', **kwargs) -> Iterator[List[Message]]:
        """Return one response generator based on the received messages.

        The workflow for an agent to generate a reply.
        Each agent subclass needs to implement this method.

        Args:
            messages: A list of messages.
            lang: Language, which will be used to select the language of the prompt
              during the agent's execution process.

        Yields:
            The response generator.
        """
        # 抽象方法，子类必须实现
        raise NotImplementedError

    def _call_llm(
        self,
        messages: List[Message],
        functions: Optional[List[Dict]] = None,
        stream: bool = True,
        extra_generate_cfg: Optional[dict] = None,
    ) -> Iterator[List[Message]]:
        """The interface of calling LLM for the agent.

        We prepend the system_message of this agent to the messages, and call LLM.

        Args:
            messages: A list of messages.
            functions: The list of functions provided to LLM.
            stream: LLM streaming output or non-streaming output.
              For consistency, we default to using streaming output across all agents.

        Yields:
            The response generator of LLM.
        """
        # 调用 LLM 的 chat 方法生成响应
        return self.llm.chat(messages=messages,
                             functions=functions,
                             stream=stream,
                             extra_generate_cfg=merge_generate_cfgs(
                                 base_generate_cfg=self.extra_generate_cfg,
                                 new_generate_cfg=extra_generate_cfg,
                             ))

    def _call_tool(self, tool_name: str, tool_args: Union[str, dict] = '{}', **kwargs) -> Union[str, List[ContentItem]]:
        """The interface of calling tools for the agent.

        Args:
            tool_name: The name of one tool.
            tool_args: Model generated or user given tool parameters.

        Returns:
            The output of tools.
        """
        # 如果工具名称不在工具映射字典中，返回错误信息
        if tool_name not in self.function_map:
            return f'Tool {tool_name} does not exists.'
        # 获取工具对象
        tool = self.function_map[tool_name]
        try:
            # 调用工具的 call 方法执行工具
            tool_result = tool.call(tool_args, **kwargs)
        except (ToolServiceError, DocParserError) as ex:
            # 捕获工具服务错误和文档解析错误，重新抛出异常
            raise ex
        except Exception as ex:
            # 捕获其他异常，记录错误信息并返回错误信息
            exception_type = type(ex).__name__
            exception_message = str(ex)
            traceback_info = ''.join(traceback.format_tb(ex.__traceback__))
            error_message = f'An error occurred when calling tool `{tool_name}`:\n' \
                            f'{exception_type}: {exception_message}\n' \
                            f'Traceback:\n{traceback_info}'
            logger.warning(error_message)
            return error_message

        # 根据工具结果的类型返回不同格式的结果
        if isinstance(tool_result, str):
            return tool_result
        elif isinstance(tool_result, list) and all(isinstance(item, ContentItem) for item in tool_result):
            return tool_result  # multimodal tool results
        else:
            return json.dumps(tool_result, ensure_ascii=False, indent=4)

    def _init_tool(self, tool: Union[str, Dict, BaseTool]):
        """
        初始化工具并将其添加到工具映射字典中。

        Args:
            tool (Union[str, Dict, BaseTool]): 工具的名称、配置字典或工具对象。
        """
        # 如果 tool 是 BaseTool 类型的实例
        if isinstance(tool, BaseTool):
            # 获取工具的名称
            tool_name = tool.name
            # 检查工具名称是否已存在于 function_map 中
            if tool_name in self.function_map:
                # 若存在，记录警告信息，提示将使用最新的工具
                logger.warning(f'Repeatedly adding tool {tool_name}, will use the newest tool in function list')
            # 将工具添加到 function_map 中，键为工具名称，值为工具对象
            self.function_map[tool_name] = tool
        # 如果 tool 是字典类型，并且包含 'mcpServers' 键
        elif isinstance(tool, dict) and 'mcpServers' in tool:
            # 使用 MCPManager 初始化配置，获取工具列表
            tools = MCPManager().initConfig(tool)
            # 遍历工具列表
            for tool in tools:
                # 获取工具的名称
                tool_name = tool.name
                # 检查工具名称是否已存在于 function_map 中
                if tool_name in self.function_map:
                    # 若存在，记录警告信息，提示将使用最新的工具
                    logger.warning(f'Repeatedly adding tool {tool_name}, will use the newest tool in function list')
                # 将工具添加到 function_map 中，键为工具名称，值为工具对象
                self.function_map[tool_name] = tool
        # 其他情况
        else:
            # 如果 tool 是字典类型
            if isinstance(tool, dict):
                # 从字典中获取工具名称
                tool_name = tool['name']
                # 将字典作为工具配置
                tool_cfg = tool
            # 如果 tool 不是字典类型
            else:
                # 将 tool 作为工具名称
                tool_name = tool
                # 工具配置设为 None
                tool_cfg = None
            # 检查工具名称是否未在 TOOL_REGISTRY 中注册
            if tool_name not in TOOL_REGISTRY:
                # 若未注册，抛出 ValueError 异常，提示工具未注册
                raise ValueError(f'Tool {tool_name} is not registered.')
            # 检查工具名称是否已存在于 function_map 中
            if tool_name in self.function_map:
                # 若存在，记录警告信息，提示将使用最新的工具
                logger.warning(f'Repeatedly adding tool {tool_name}, will use the newest tool in function list')
            # 根据工具名称从 TOOL_REGISTRY 中获取工具类，并使用工具配置初始化工具对象，添加到 function_map 中
            self.function_map[tool_name] = TOOL_REGISTRY[tool_name](tool_cfg)

    def _detect_tool(self, message: Message) -> Tuple[bool, str, str, str]:
        """A built-in tool call detection for func_call format message.

        Args:
            message: one message generated by LLM.

        Returns:
            Need to call tool or not, tool name, tool args, text replies.
        """
        func_name = None
        func_args = None

        # 如果消息中包含 function_call 属性
        if message.function_call:
            # 获取 function_call 对象
            func_call = message.function_call
            # 获取工具名称
            func_name = func_call.name
            # 获取工具参数
            func_args = func_call.arguments
        # 获取消息内容
        text = message.content
        if not text:
            text = ''

        # 返回是否需要调用工具、工具名称、工具参数和文本回复
        return (func_name is not None), func_name, func_args, text

# 定义一个最基本的代理类，继承自 Agent 类
class BasicAgent(Agent):

    def _run(self, messages: List[Message], lang: str = 'en', **kwargs) -> Iterator[List[Message]]:
        """
        根据接收到的消息返回一个响应生成器。

        Args:
            messages (List[Message]): 消息列表。
            lang (str, optional): 语言，默认为 'en'。
            **kwargs: 其他关键字参数。

        Yields:
            Iterator[List[Message]]: 响应生成器。
        """
        # 创建一个额外的生成配置字典，包含语言信息
        extra_generate_cfg = {'lang': lang}
        # 检查 kwargs 中是否包含 'seed' 关键字参数
        if kwargs.get('seed') is not None:
            # 如果包含，将 'seed' 添加到额外的生成配置中
            extra_generate_cfg['seed'] = kwargs['seed']
        # 调用 _call_llm 方法，传入消息和额外的生成配置，返回响应生成器
        return self._call_llm(messages, extra_generate_cfg=extra_generate_cfg)