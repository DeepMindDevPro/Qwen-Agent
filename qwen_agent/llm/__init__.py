import copy
from typing import Union

# 从不同模块导入所需的类和变量
from .azure import TextChatAtAzure
from .base import LLM_REGISTRY, BaseChatModel, ModelServiceError
from .oai import TextChatAtOAI
from .openvino import OpenVINO
from .qwen_dashscope import QwenChatAtDS
from .qwenaudio_dashscope import QwenAudioChatAtDS
from .qwenomni_oai import QwenOmniChatAtOAI
from .qwenvl_dashscope import QwenVLChatAtDS
from .qwenvl_oai import QwenVLChatAtOAI

#参数类型: Union[dict, str] => 该参数的类型可以是字典（dict）或者字符串（str）
#默认值：'qwen-plus' => 如果调用函数时没有提供 cfg 参数，那么默认使用 'qwen-plus' 作为模型配置
#作用: 当cfg为字符串时：
#     它被视为模型的名称。函数内部会将其转换为一个包含模型名称的字典，例如 {'model': 'qwen-plus'}
#当cfg为字典时: 它应该包含模型的详细配置信息, 如下见详细配置

#返回值: 类型：BaseChatModel
#函数返回一个继承自 BaseChatModel 的对象，这个对象代表一个具体的聊天模型实例，可以用于处理用户的聊天请求

def get_chat_model(cfg: Union[dict, str] = 'qwen-plus') -> BaseChatModel:
    """
    实例化大语言模型（LLM）对象的接口。

    参数:
        cfg: LLM 的配置信息，可以是字典或者字符串。
            如果是字符串，会将其作为模型名称处理；如果是字典，需要包含模型的相关配置信息。
            配置示例如下：
            cfg = {
                # 使用 DashScope 提供的模型服务
                'model': 'qwen-max',
                'model_server': 'dashscope',

                # 使用与 OpenAI API 兼容的自定义模型服务
                # 'model': 'Qwen',
                # 'model_server': 'http://127.0.0.1:7905/v1',

                # （可选）LLM 生成时的超参数
                'generate_cfg': {
                    'top_p': 0.8,
                    'max_input_tokens': 6500,
                    'max_retries': 10,
                }
            }

    返回:
        一个 LLM 对象，继承自 BaseChatModel 类。
    """
    # 如果 cfg 是字符串类型，将其转换为字典，以模型名称作为键
    if isinstance(cfg, str):
        cfg = {'model': cfg}

    # 如果配置中包含 'model_type' 字段
    if 'model_type' in cfg:
        # 获取模型类型
        model_type = cfg['model_type']
        # 检查该模型类型是否已在注册列表中
        if model_type in LLM_REGISTRY:
            # 对于 'oai' 和 'qwenvl_oai' 类型的模型
            if model_type in ('oai', 'qwenvl_oai'):
                # 如果模型服务器配置为 'dashscope'
                if cfg.get('model_server', '').strip() == 'dashscope':
                    # 深拷贝配置，避免修改原始配置
                    cfg = copy.deepcopy(cfg)
                    # 将模型服务器地址修改为 DashScope 的兼容模式地址
                    cfg['model_server'] = 'https://dashscope.aliyuncs.com/compatible-mode/v1'
            # 根据模型类型从注册列表中获取对应的类并实例化
            return LLM_REGISTRY[model_type](cfg)
        else:
            # 如果模型类型未注册，抛出 ValueError 异常
            raise ValueError(f'请从 {str(LLM_REGISTRY.keys())} 中选择有效的 model_type')

    # 如果配置中没有 'model_type' 字段，则根据 model 和 model_server 推断模型类型

    # 如果配置中包含 'azure_endpoint' 字段
    if 'azure_endpoint' in cfg:
        # 推断模型类型为 'azure'
        model_type = 'azure'
        # 将推断出的模型类型添加到配置中
        cfg['model_type'] = model_type
        # 根据模型类型从注册列表中获取对应的类并实例化
        return LLM_REGISTRY[model_type](cfg)

    # 如果配置中包含 'model_server' 字段
    if 'model_server' in cfg:
        # 检查模型服务器地址是否以 'http' 开头
        if cfg['model_server'].strip().startswith('http'):
            # 推断模型类型为 'oai'
            model_type = 'oai'
            # 将推断出的模型类型添加到配置中
            cfg['model_type'] = model_type
            # 根据模型类型从注册列表中获取对应的类并实例化
            llm_registry = LLM_REGISTRY[model_type](cfg)
            print('llm_registry:' + llm_registry)
            return llm_registry

    # 从配置中获取模型名称，如果没有则为空字符串
    model = cfg.get('model', '')

    # 如果模型名称中包含 '-vl'（区分大小写）
    if '-vl' in model.lower():
        # 推断模型类型为 'qwenvl_dashscope'
        model_type = 'qwenvl_dashscope'
        # 将推断出的模型类型添加到配置中
        cfg['model_type'] = model_type
        # 根据模型类型从注册列表中获取对应的类并实例化
        return LLM_REGISTRY[model_type](cfg)

    # 如果模型名称中包含 '-audio'（区分大小写）
    if '-audio' in model.lower():
        # 推断模型类型为 'qwenaudio_dashscope'
        model_type = 'qwenaudio_dashscope'
        # 将推断出的模型类型添加到配置中
        cfg['model_type'] = model_type
        # 根据模型类型从注册列表中获取对应的类并实例化
        return LLM_REGISTRY[model_type](cfg)

    # 如果模型名称中包含 'qwen'（区分大小写）
    if 'qwen' in model.lower():
        # 推断模型类型为 'qwen_dashscope'
        model_type = 'qwen_dashscope'
        # 将推断出的模型类型添加到配置中
        cfg['model_type'] = model_type
        # 根据模型类型从注册列表中获取对应的类并实例化
        return LLM_REGISTRY[model_type](cfg)

    # 如果以上条件都不满足，说明配置无效，抛出 ValueError 异常
    raise ValueError(f'无效的模型配置: {cfg}')

# 定义该模块对外暴露的类和函数
__all__ = [
    'BaseChatModel',
    'QwenChatAtDS',
    'TextChatAtOAI',
    'TextChatAtAzure',
    'QwenVLChatAtDS',
    'QwenVLChatAtOAI',
    'QwenAudioChatAtDS',
    'QwenOmniChatAtOAI',
    'OpenVINO',
    'get_chat_model',
    'ModelServiceError',
]