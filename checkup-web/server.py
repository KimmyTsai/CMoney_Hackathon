# AI 健診後端：FastAPI → Amazon Bedrock (Claude)
# 安裝: pip install fastapi uvicorn boto3
# 執行: uvicorn server:app --port 8000
import boto3, json
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

REGION = "us-west-2"   # 改成主辦方指定 region
MODEL_ID = "REPLACE_ME"  # 用 aws bedrock list-foundation-models --by-provider anthropic 查實際 ID

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
client = boto3.client("bedrock-runtime", region_name=REGION)

SYSTEM = ("你是 CMoney「存股健診」AI，服務定期定額存股族。根據診斷數據以繁體中文寫 220-300 字健診報告："
          "先肯定現金流體質（成本殖利率、連配年數、年股息），再誠實區分帳面損益與含息總報酬、對照 0050，"
          "指出集中度與買點分位意涵，若社群多空比極端偏多但績效落後要提醒同溫層，"
          "結尾用即將到來的除息日或填息追蹤當回訪鉤子。禁止直接投資建議，用「值得想想／可以留意」框架。"
          "口吻像懂投資的朋友，不用條列，分 2-3 段。")

class Req(BaseModel):
    payload: dict

@app.post("/checkup")
def checkup(req: Req):
    resp = client.converse(
        modelId=MODEL_ID,
        system=[{"text": SYSTEM}],
        messages=[{"role": "user", "content": [{"text": "我的持股健診數據：" + json.dumps(req.payload, ensure_ascii=False)}]}],
        inferenceConfig={"maxTokens": 1000},
    )
    return {"text": resp["output"]["message"]["content"][0]["text"]}
