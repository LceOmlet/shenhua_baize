from ..schemas import order_fields



def build_prompt(text=None) -> str:
    if text is not None:
        text = f"""请从以下文本内容中提取结构化信息：
【文本内容】
{text}
"""
    else:
        text = ""
    """构建结构化提取提示语"""
    fields_desc = "\n".join(
        [f"- {k}: {v}" for k, v in order_fields.items()])
    
    return f"""
{text}
【提取字段】
{fields_desc}

【输出要求】
1. 必须返回标准的JSON格式，以 {{ 开始，以 }} 结束
2. 所有字段必须使用双引号包裹
3. 缺失字段保留为空字符串
4. 严格遵循字段格式
5. 金额单位：人民币元
6. 不要包含任何其他说明文字，只返回JSON对象
"""