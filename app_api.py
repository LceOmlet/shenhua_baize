# project-root/app.py
import os
import json
import argparse
from datetime import datetime
from pathlib import Path
from src.utils.file_utils import unified_process

def format_result(result, process_time: float) -> str:
    """格式化输出结果"""
    output = {
        "文件路径": result.original_data,
        "内容类型": result.content_type,
        "处理耗时": f"{process_time:.2f}秒",
        "置信度": f"{result.confidence:.2%}",
        "提取结果": result.extracted_fields
    }
    return json.dumps(output, indent=2, ensure_ascii=False, default=str)

def process_single_file(file_path: str):
    """处理单个文件并打印结果"""
    try:
        path = Path(file_path)
        if not path.exists():
            print(f"错误：文件不存在 - {file_path}")
            return

        start_time = datetime.now()
        result = unified_process(path)
        process_time = (datetime.now() - start_time).total_seconds()

        print("\n" + "="*50)
        print(f"处理结果：{path.name}")
        print(format_result(result, process_time))
        print("="*50 + "\n")

    except Exception as e:
        print(f"处理过程中发生错误：{str(e)}")

def interactive_mode():
    """交互式处理模式"""
    print("欢迎使用智能文档处理系统")
    print("输入文件路径或输入 'q' 退出\n")
    
    while True:
        file_path = input("请输入文件路径：").strip()
        
        if file_path.lower() in ['q', 'quit', 'exit']:
            print("程序已退出")
            break
            
        process_single_file(file_path)

def batch_mode(file_paths: list):
    """批量处理模式"""
    print(f"开始批量处理 {len(file_paths)} 个文件...\n")
    
    total_results = []
    for idx, path in enumerate(file_paths, 1):
        print(f"正在处理文件 ({idx}/{len(file_paths)})：{path}")
        start_time = datetime.now()
        result = unified_process(path)
        process_time = (datetime.now() - start_time).total_seconds()
        
        result_dict = result.model_dump()
        result_dict["processing_time"] = process_time
        total_results.append(result_dict)
    
    # 保存批量处理结果
    output_path = Path("processing_results.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(total_results, f, indent=2, ensure_ascii=False)
    print(f"\n批量处理完成，结果已保存至：{output_path.absolute()}")

def main():
    # 命令行参数解析
    parser = argparse.ArgumentParser(
        description="智能文档处理系统 - 支持音频/图片/文本文件分析",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        "-f", "--files", 
        nargs="+",
        help="指定一个或多个文件路径（支持通配符）\n示例：python app.py -f audio.mp3 image.jpg"
    )
    
    args = parser.parse_args()
    
    # 运行模式判断
    if args.files:
        # 处理通配符扩展
        expanded_files = []
        for pattern in args.files:
            expanded_files.extend(Path().glob(pattern))
        
        # 验证文件存在
        valid_files = [str(p) for p in expanded_files if p.exists()]
        if not valid_files:
            print("错误：未找到匹配的有效文件")
            return
            
        batch_mode(valid_files)
    else:
        interactive_mode()

if __name__ == "__main__":
    main()
