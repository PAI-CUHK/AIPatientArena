import argparse

def get_args():
    parser = argparse.ArgumentParser(description="Run the benchmark with specified configurations.")
    parser.add_argument('--doctor_module', type=str, default='expert_mdp', help='Module name where the doctor class is implemented.')
    parser.add_argument('--doctor_class', type=str, required=True, help='Doctor class name to use for the benchmark.')
    parser.add_argument('--data_dir', type=str, required=True, help='Directory containing the development data files.')

    parser.add_argument('--output_filename', type=str, default="out.jsonl")
    parser.add_argument("--max_questions", type=int, default=15)

    parser.add_argument('--log_filename', type=str, default='log.log', help='Filename for logging general benchmark info.')

    parser.add_argument('--patient_model_name', type=str, default='gpt5', help='Patient model name')
    parser.add_argument('--doctor_model_name', type=str, default='gpt5', help='Doctor model name')

    # 新增参数
    parser.add_argument('--port', type=int, default=11434, help='Ollama 服务监听端口')
    parser.add_argument('--gpuid', type=str, default='0', help='指定 GPU ID，多个GPU用逗号分隔，如 "0,1"')
    parser.add_argument('--start_id', type=int, default=0, help='开始处理的样本索引')
    parser.add_argument('--end_id', type=int, default=0, help='最终处理的样本索引')

    # vLLM 相关
    parser.add_argument('--vllm_model', type=str, default="/FreedomIntelligence/HuatuoGPT-o1-72B",
                        help='vLLM 模型路径或 HuggingFace 模型名')
    parser.add_argument('--vllm_port', type=int, default=8000, help='vLLM 服务监听端口')
    
    # 模型类型选择
    parser.add_argument('--model_type', type=str, default='gpt', choices=['gpt', 'vllm'],
                        help='模型类型：gpt (使用API) 或 vllm (使用本地vLLM)')

    args = parser.parse_args()
    return args
