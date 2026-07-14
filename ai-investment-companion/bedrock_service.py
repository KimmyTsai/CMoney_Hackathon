"""
bedrock_service.py
------------------
AI 投資樹洞 - AWS Bedrock 串接模組
使用 Converse API 進行「組合層」一次性診斷（一個組合一次呼叫）
"""

import os
import json
import boto3
from botocore.exceptions import ClientError, NoCredentialsError

# 可用環境變數覆蓋，不用改程式碼：
#   export BEDROCK_REGION=us-west-2
#   export BEDROCK_MODEL_ID=$(aws bedrock list-foundation-models --by-provider anthropic ... 查到的 ID)
BEDROCK_REGION = os.environ.get("BEDROCK_REGION", "us-west-2")
MODEL_ID = os.environ.get("BEDROCK_MODEL_ID", "REPLACE_WITH_ACTUAL_MODEL_ID")
MAX_TOKENS = 1200

SYSTEM_PROMPT = (
    "你是 CMoney「AI 投資樹洞」，一位溫暖、誠實、專業的投資陪伴教練，服務定期定額存股族。"
    "使用者向你敞開了真實持股，你收到的是完整的組合層診斷數據（含逐檔明細）。"
    "請以繁體中文寫一份 250-350 字的組合陪伴診斷，必須做到："
    "1) 先肯定現金流體質：成本殖利率、連續配息年數、年股息現金流；"
    "2) 誠實區分「帳面損益」與「含息總報酬」，並對照 0050 同期表現，說明 2025 高股息落後大盤是風格輪動而非選錯；"
    "3) 用產業配置與最大單一權重點出集中度，用成本買點分位講出他的進場習慣；"
    "4) 若某檔社群多空比極端偏多但績效落後大盤，提醒同溫層效應（對照同學會數據）；"
    "5) 若系統警示中有帳面重虧且持續加碼的訊號，溫和點出「越攤越平」的思考；"
    "6) 結尾用最近的除息日或填息追蹤設計一個回訪鉤子，並以一句「開啟持股防護罩」的引導收尾。"
    "【法遵限制·極重要】絕對不能出現任何「買進/賣出/加碼/減碼」等具體投資建議，"
    "一律用「值得想想」「可以留意」的觀察框架。口吻像懂投資的朋友，分 2-3 段，不用條列。"
)


class BedrockService:
    """AWS Bedrock 服務封裝（Converse API）"""

    def __init__(self, region: str = BEDROCK_REGION, model_id: str = MODEL_ID):
        self.region = region
        self.model_id = model_id
        self._client = None

    @property
    def client(self):
        if self._client is None:
            self._client = boto3.client("bedrock-runtime", region_name=self.region)
        return self._client

    def diagnose_portfolio(self, context: dict) -> dict:
        """
        組合層診斷：一個組合一次呼叫。
        回傳 {"ok": bool, "text": str, "error": str|None}
        失敗時 ok=False —— 前端必須醒目顯示錯誤，不可假裝成功。
        """
        user_text = "我的持股組合診斷數據如下：" + json.dumps(context, ensure_ascii=False, default=str)
        try:
            resp = self.client.converse(
                modelId=self.model_id,
                system=[{"text": SYSTEM_PROMPT}],
                messages=[{"role": "user", "content": [{"text": user_text}]}],
                inferenceConfig={"maxTokens": MAX_TOKENS, "temperature": 0.4},
            )
            text = resp["output"]["message"]["content"][0]["text"]
            return {"ok": True, "text": text, "error": None}
        except NoCredentialsError:
            return {"ok": False, "text": "", "error":
                    "找不到 AWS credentials。請先執行 aws configure 填入主辦方提供的金鑰。"}
        except ClientError as e:
            code = e.response["Error"]["Code"]
            hints = {
                "AccessDeniedException": "模型未啟用或權限不足：到 Bedrock console 的 Model access 確認，或洽工作人員。",
                "ValidationException": "MODEL_ID 可能錯誤：用 aws bedrock list-foundation-models --by-provider anthropic 查實際 ID（注意 us./apac. 前綴）。",
                "ResourceNotFoundException": "此 region 沒有這個模型，確認 BEDROCK_REGION 是否為主辦方指定區域。",
            }
            return {"ok": False, "text": "", "error": f"Bedrock 錯誤 [{code}]：{hints.get(code, str(e))}"}
        except Exception as e:
            return {"ok": False, "text": "", "error": f"未預期錯誤：{e}"}


_service_instance = None


def get_bedrock_service() -> BedrockService:
    global _service_instance
    if _service_instance is None:
        _service_instance = BedrockService()
    return _service_instance
