# 主要用于各种大模型api调用的配置的应用
import yaml

from enums.model_enums import ModelType


class Config:
    def __init__(self, config_file):
        # 读取配置文件
        try:
            with open(config_file, 'r') as f:
                self.config = yaml.load(f, Loader=yaml.FullLoader)
        except FileNotFoundError as e:
            raise FileNotFoundError(f"配置文件 {config_file} 不存在") from e
        except yaml.YAMLError as e:
            raise yaml.YAMLError(f"配置文件解析错误: {e}") from e

    # 通过模型名称，获取配置信息
    def model_config(self, model_type: ModelType):
        if model_type is ModelType.DEEPSEEK:
            return self.config['deepSeek']
        if model_type is ModelType.QWEN:
            return self.config['qwen']
        return None

    # 获取向量数据库配置信息
    def chroma_config(self):
        return self.config['chroma']


if __name__ == '__main__':
    config = Config('../conf/config.yml')
    print(ModelType.DEEPSEEK)
    data = config.model_config(ModelType.DEEPSEEK)
    print(data.get("modelName"))
