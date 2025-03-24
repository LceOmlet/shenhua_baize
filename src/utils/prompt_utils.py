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
1. 返回纯净JSON，无额外字符
2. 缺失字段保留为空白
3. 严格遵循字段格式
4. 金额单位：人民币元
"""