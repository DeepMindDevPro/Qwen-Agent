import ast #Python 的抽象语法树模块，用于安全地将字符串转换为 Python 对象
import os
from typing import List, Literal

'''

定义了一系列与大语言模型（LLM）、智能体（agents）、工具（tools）以及检索增强生成（RAG）相关的默认设置。这些设置可以通过环境变量进行配置，以适应不同的应用场景

'''

# 如果输入消息的令牌数超过这个限制，LLM 会对其进行截断
# Settings for LLMs
DEFAULT_MAX_INPUT_TOKENS: int = int(os.getenv(
    'QWEN_AGENT_DEFAULT_MAX_INPUT_TOKENS', 58000))  # The LLM will truncate the input messages if they exceed this limit

# 定义了每次运行时智能体可以调用 LLM 的最大次数
# Settings for agents
MAX_LLM_CALL_PER_RUN: int = int(os.getenv('QWEN_AGENT_MAX_LLM_CALL_PER_RUN', 20))

# 定义了工具的默认工作空间
# Settings for tools
DEFAULT_WORKSPACE: str = os.getenv('QWEN_AGENT_DEFAULT_WORKSPACE', 'workspace')

# Settings for RAG
# 定义了 RAG 过程中为参考材料预留的最大令牌数
DEFAULT_MAX_REF_TOKEN: int = int(os.getenv('QWEN_AGENT_DEFAULT_MAX_REF_TOKEN',
                                           20000))  # The window size reserved for RAG materials

# 定义了 RAG 过程中每个文档块的最大令牌数
DEFAULT_PARSER_PAGE_SIZE: int = int(os.getenv('QWEN_AGENT_DEFAULT_PARSER_PAGE_SIZE',
                                              500))  # Max tokens per chunk when doing RAG

# 定义了 RAG 过程中默认的关键词生成策略
DEFAULT_RAG_KEYGEN_STRATEGY: Literal['None', 'GenKeyword', 'SplitQueryThenGenKeyword', 'GenKeywordWithKnowledge',
                                     'SplitQueryThenGenKeywordWithKnowledge'] = os.getenv(
                                         'QWEN_AGENT_DEFAULT_RAG_KEYGEN_STRATEGY', 'GenKeyword')

# 定义了 RAG 过程中默认的搜索器列表
DEFAULT_RAG_SEARCHERS: List[str] = ast.literal_eval(
    os.getenv('QWEN_AGENT_DEFAULT_RAG_SEARCHERS',
              "['keyword_search', 'front_page_search']"))  # Sub-searchers for hybrid retrieval
