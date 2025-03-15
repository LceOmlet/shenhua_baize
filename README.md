
# **项目介绍：shenhua_baize**

## **项目概述**
`shenhua_baize` 是一个多模态处理项目，旨在通过集成视觉、音频处理以及文件管理功能，提供一个高效、便捷的多媒体分析平台。项目支持多种模型和工具，能够处理图像、音频等多种数据类型，并通过统一的界面或命令行工具进行操作。

## **项目结构**
项目采用模块化设计，主要分为以下几个部分：

1. **视觉模块（Vision）**  
   - **主要文件**：`src/processing/image_processor.py`  
   - **配置文件**：`configs/settings.yaml`（`vision_model_path` 等配置项）  
   - **功能描述**：负责图像的处理与分析，支持加载视觉模型（如 Qwen2.5-VL-7B-Instruct），并根据模型输出定义结果字段。

2. **音频模块（Audio）**  
   - **主要文件**：`src/processing/audio_processor.py`  
   - **配置文件**：`configs/settings.yaml`（`speech_model_config` 等配置项）  
   - **功能描述**：负责音频的处理与分析，支持加载音频模型（如 Qwen2-Audio-7B-Instruct），并提取音频特征（如时间戳）。

3. **文件处理模块（Filesystem）**  
   - **主要文件**：`src/utils/file_utils.py`  
   - **临时文件存储位置**：`temp/`  
   - **配置文件**：`configs/settings.yaml`（`max_file_size` 等限制参数）  
   - **功能描述**：实现文件的上传、存储和清理功能，并对接文件上传回调接口。

4. **界面模块（Interface）**  
   - **主要文件**：`app.py`  
   - **功能描述**：负责项目的主界面逻辑，集成视觉、音频模块的输出，并展示结果。

## **分支说明**
项目包含多个分支，分别针对不同的应用场景和开发需求：

- **`cmd_app` 分支**  
  该分支专注于命令行应用的开发。通过命令行界面，用户可以直接调用项目的功能模块（如图像处理、音频处理等），并获取处理结果。命令行应用适合在服务器环境或自动化脚本中使用。

- **`ZTL_app` 分支**  
  该分支基于 Streamlit 框架开发，提供了一个图形化用户界面（GUI）。用户可以通过浏览器访问该应用，上传文件并查看处理结果。Streamlit 应用适合在交互式环境中使用，便于非技术用户操作。

## **开发指南**
1. **环境准备**  
   使用以下命令激活项目环境：
   ```bash
   conda activate shenhua_baize
   ```
   安装所需的依赖包：
   ```bash
   pip install -r requirements.txt
   ```

2. **运行模块**  
   - 图像处理模块：
     ```bash
     CUDA_VISIBLE_DEVICES=5 python -m src.processing.image_processor
     ```
   - 音频处理模块：
     ```bash
     CUDA_VISIBLE_DEVICES=5 python -m src.processing.audio_processor
     ```

3. **开发流程**  
   - 根据任务分工，每位开发者只需在指定路径范围内开发。
   - 通过 `src/schemas.py` 定义的数据接口进行跨模块交互。
   - 最终在 `app.py` 中集成所有模块的功能。

## **项目优势**
- **模块化设计**：易于扩展和维护，支持多种模型和工具。
- **多分支支持**：同时提供命令行和图形化界面，满足不同用户需求。
- **灵活配置**：通过 `settings.yaml` 配置文件，方便调整项目参数。

## **未来展望**
- 持续优化视觉和音频处理算法，提升性能。
- 增加更多模型支持，拓展应用范围。
- 改进用户界面，提升用户体验。

---

希望这份项目介绍能够帮助你更好地理解 `shenhua_baize` 的功能和结构。
