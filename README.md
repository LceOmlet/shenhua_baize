 请先查看下面的目录结构，再查看具体安排

conda activate shenhua_baize

```
project-root/
├── app.py
├── configs
│   └── settings.yaml
├── README.md
├── requirements.txt
├── src
│   ├── __init__.py
│   ├── processing
│   │   ├── __init__.py
│   │   ├── audio_processor.py
│   │   ├── image_processor.py
│   │   └── text_processor.py
│   ├── schemas.py
│   └── utils
│       ├── __init__.py
│       ├── file_utils.py
│       └── model_utils.py
├── temp
│   └── .gitkeep
└── tests
    └── .gitkeep
```

创建分支

```
git checkout -b visual          #A
git checkout -b audio main  #B
git checkout -b filesystem    #C
git checkout -b interface      #D
```

可随意创建辅助文件

**同学A（视觉模型）**

```

https://ollama.com/bsahane/Qwen2.5-VL-7B-Instruct

project-root/
├── models/vision/               ← 视觉模型文件存放位置
├── src/processing/image_processor.py  ← 图像处理核心逻辑
├── configs/settings.yaml         → 需关注 vision_model_path 等配置项
└── src/schemas.py                → 结果字段定义需与模型输出对齐
```

**同学B（音频模型）**


需要会用huggingface
hf模型自带一个部署教程，虽然性能低一点，但是肯定能用：
Qwen/Qwen2-Audio-7B-Instruct

si-pbc/hertz-dev

其实minicpm也能当视觉模型，但是先当咱们不知道吧

```
https://ollama.com/library/minicpm-v
project-root/
├── models/speech/               ← 语音模型文件存放位置
├── src/processing/audio_processor.py  ← 音频处理核心逻辑
├── configs/settings.yaml         → 需关注 speech_model_config 等配置项
└── src/schemas.py                → 结果字段需包含时间戳等音频特征
```

**同学C（文件处理）**

```
project-root/
├── src/utils/file_utils.py       ← 文件上传/存储/清理实现
├── temp/                        → 临时文件存储策略
├── app.py                       → 需对接文件上传回调接口 (第14-18行)
└── configs/settings.yaml         → 管理 max_file_size 等限制参数
```

**同学D（界面实现）**

```
project-root/
├── app.py                       ← 主界面逻辑 (第5-28行)
├── src/schemas.py                → 需确保界面展示字段匹配数据结构
├── src/utils/model_utils.py      → 模型加载进度显示集成
└── temp/                        → 文件预览功能涉及临时文件访问
```

**路径对照表**：

| 同学 | 主要开发文件       | 配置文件                  | 相关数据接口             | 存储位置      |
| ---- | ------------------ | ------------------------- | ------------------------ | ------------- |
| A    | image_processor.py | settings.yaml (vision段)  | schemas.ExtractionResult | models/vision |
| B    | audio_processor.py | settings.yaml (speech段)  | schemas.ExtractionResult | models/speech |
| C    | file_utils.py      | settings.yaml (storage段) | app.py 文件上传回调      | temp/         |
| D    | app.py             | 无需配置修改              | schemas.py 数据模型      | 无            |

每位同学只需在指定路径范围内开发，通过 schemas.py 定义的数据接口进行跨模块交互，最终在 app.py 通过统一界面集成。
