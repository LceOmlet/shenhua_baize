from pydantic import BaseModel


order_fields = {
            "客户名称": "string",
            "发车联系人": "string",
            "发车联系人手机号": "string (11位数字)",
            "始发地": "省+市+区+详细地址",
            "收车联系人": "string",
            "收车联系人手机号": "string (11位数字)",
            "目的地": "省+市+区+详细地址",
            "装车时间": "datetime (yyyy-MM-dd)",
            "发车时间": "datetime (yyyy-MM-dd)",
            "订单总金额": "float (万元)",
            "汽车品牌": "string",
            "车辆系列": "string",
            "车值": "float (万元)",
            "车辆数量": "integer",
            "VIN码": "string (多个用英文逗号分隔)",
            "承运类型": "enum (嗨拉自营/竞价撮合)",
            "购买保险": "enum (是/否)",
            "保险类型": "enum (尊享服务/全程无忧)",
            "车辆类型": "enum (轿车/suv/皮卡)",
            "车辆产地": "enum (国产/进口)",
            "板车要求": "string",
            "验车要求": "string",
            "备注说明": "string",
            "时效类型": "enum (常规单/提车紧急单)",
            "运力方式": "enum (抢单/派单)",
            "车辆性质": "enum (新车/二手车)",
            "订单金额是否含税": "enum (含税/不含税/待确认)"
        }

class ExtractionResult(BaseModel):
    content_type: str  # image/audio/text
    original_data: str
    extracted_fields: dict
    confidence: float
