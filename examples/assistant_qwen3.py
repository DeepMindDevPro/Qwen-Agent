"""An agent implemented by assistant with qwen3"""
import os  # noqa

from qwen_agent.agents import Assistant
from qwen_agent.gui import WebUI

'''
助手设计使用模块化:
1.程序配置与初始化
2.工具调用系统集成
3.Web用户界面
'''
def init_agent_service():
    '''
    初始化并配置智能助手服务
    返回配置好的Assistant实例
    '''
    # LLM参数
    llm_cfg = {
        'model': 'qwen3:8b', # 指定使用的模型名称，需与ollama服务中的名称一致
        'model_type': 'oai',  # 使用OpenAI兼容模式，便于与ollama服务对接
        'model_server': 'http://localhost:11434/v1',  # Ollama服务的API地址
        'api_key': 'EMPTY', # 本地服务无需API密钥
        'generate_cfg': {
            'extra_body': {
                # 基础参数
                'temperature': 0.7, # 控制生成文本的随机性，值越低越确定 # 控制随机性 (0.0-2.0)
                'max_tokens': 2048, # 最大生成长度
                'stop': ['<|endoftext|>'], # 指定停止生成的标记
                # #高级参数
                'top_p': 0.9,  # 核采样概率阈值
                'top_k': 40,  # 采样时考虑的最高概率token数量
                'presence_penalty': 0.0,  # 惩罚新话题的引入 (-2.0-2.0)
                'frequency_penalty': 0.0,  # 惩罚重复tokens (-2.0-2.0)
                'repeat_penalty': 1.1,  # 重复惩罚因子 (Ollama特有参数)
                'mirostat_mode': 0,  # Mirostat算法模式 (0=关闭, 1=v1, 2=v2)
                'mirostat_tau': 5.0,  # Mirostat目标熵
                'mirostat_eta': 0.1,  # Mirostat学习率
                # 'stream': False,  # 是否流式输出
                # 'logprobs': None,  # 是否返回log概率
                # 'echo': False,  # 是否回显输入
            }
        }
    }

    #配置助手可以使用的工具列表
    tools = [
        {
            'mcpServers':
             {  # MCP服务器工具配置
                'time': {
                    'command': 'uvx', # 执行命令
                    'args': ['mcp-server-time', '--local-timezone=Asia/Shanghai'] # 带时区参数的时间获取命令
                },
                'fetch': {
                    'command': 'uvx', # 执行命令
                    'args': ['mcp-server-fetch'] #网络内容获取命令
                }
            }
        },
        'code_interpreter', # 内置的代码解释器工具，可执行代码并返回结果
    ]

    # 初始化助手实例
    bot = Assistant(
        # 传入LLM配置
        llm=llm_cfg,
        # 传入工具列表
        function_list=tools,
        # 助手名称
        name='小葛的千问助手',
        description="我是千问助手")
    return bot


def app_gui():
    """
    初始化并启动Web用户界面
    """
    bot = init_agent_service()
    # 配置聊天界面
    chatbot_config = {
        'prompt.suggestions': [
            # 示例提示：询问当前时间
            'What time is it?',
            # 示例提示：网页内容提取与可视化
            'https://github.com/orgs/QwenLM/repositories Extract markdown content of this page, then draw a bar chart to display the number of stars.'
        ],
        # 新增界面配置
        'ui_config': {
            'theme': 'dark',  # 支持明暗主题切换
            'max_history_length': 50,  # 历史消息最大数量
            'show_code_preview': True,  # 代码块预览功能
            'enable_file_upload': True,  # 支持文件上传
            'custom_styles': {  # 自定义样式
                'primary_color': '#165DFF',
                'font_family': 'Inter, sans-serif'
            }
        },
        # 新增快捷键配置
        'keyboard_shortcuts': {
            'send_message': 'Ctrl+Enter',
            'new_chat': 'Ctrl+N',
            'clear_history': 'Ctrl+Shift+H'
        }
    }
    # 创建并运行Web界面
    WebUI(
        # 传入助手实例
        bot,
        # 传入界面配置
        chatbot_config=chatbot_config
    ).run(server_port=8090)


if __name__ == '__main__':
    # 程序入口点，启动Web界面
    app_gui()
