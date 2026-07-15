"""
ocr_service.py
--------------
AI 投資樹洞 - OCR 圖片解析模組
使用 EasyOCR 辨識看盤 App 庫存截圖
支援：結構化格式（代號/股數/均價）與表格式截圖
"""

import re
import numpy as np
from PIL import Image

_reader = None


def _get_reader():
    global _reader
    if _reader is None:
        import easyocr
        _reader = easyocr.Reader(["ch_tra", "en"], gpu=False)
    return _reader


def ocr_parse_image(uploaded_file) -> list:
    """
    解析看盤 App 截圖，回傳持股清單
    策略：
    1. 先用關鍵字模式（代號/股數/均價/成本）做語義解析
    2. 如果關鍵字模式失敗，退回啟發式行解析
    """
    try:
        reader = _get_reader()
        image = Image.open(uploaded_file)
        img_array = np.array(image)
        results = reader.readtext(img_array)

        if not results:
            return [], ""

        # 合併所有文字
        full_text = " ".join([r[1].strip() for r in results])

        # 策略 1：關鍵字語義解析
        parsed = _parse_by_keywords(full_text)
        if parsed:
            return parsed, full_text

        # 策略 2：逐行啟發式解析（針對表格式截圖）
        return _parse_by_lines(results), full_text

    except Exception as e:
        raise OCRError(f"OCR 解析失敗：{str(e)}")


def _parse_by_keywords(text: str) -> list:
    """
    關鍵字語義解析：尋找「代號」「股數」「均價/成本」等中文標籤
    適用於結構化的庫存資訊截圖
    """
    holdings = []

    # 正規化文字（全形→半形、移除空格）
    text = text.replace("：", ":").replace("，", ",")
    # OCR 常見問題：數字中間被插入空格（如 "2 000" → "2000"）
    text = re.sub(r"(\d)\s+(\d)", r"\1\2", text)
    # OCR 常見問題：字母 O/o 被誤判為數字 0（或反過來）
    # 在數字語境中（前後有數字），把 O/o 替換成 0
    text = re.sub(r"(\d)[Oo]", r"\g<1>0", text)
    text = re.sub(r"[Oo](\d)", r"0\1", text)
    # 連續的 O 在數字旁也修正（如 2OO0 → 2000）
    text = re.sub(r"(\d)[Oo]+(\d)", lambda m: m.group(1) + "0" * (m.end() - m.start() - 2) + m.group(2), text)
    # 最後再清理一次：如果還有殘留的 O 夾在數字中
    text = re.sub(r"(\d)([Oo]+)(\d)", lambda m: m.group(1) + "0" * len(m.group(2)) + m.group(3), text)

    # 模式 1：「代號：2330」或「代號:2330台積電」
    stock_patterns = [
        r"代[號号][:\s]*(\d{4,6})",
        r"股票[:\s]*(\d{4,6})",
        r"標的[:\s]*(\d{4,6})",
    ]
    stock_id = None
    for pat in stock_patterns:
        m = re.search(pat, text)
        if m:
            stock_id = m.group(1)
            break

    if not stock_id:
        return []

    # 模式 2：股數
    shares = 1000
    shares_patterns = [
        r"股[數数][:\s]*([\d,\s]+)\s*股",
        r"股[數数][:\s]*([\d,\s]+)",
        r"持有[:\s]*([\d,\s]+)\s*股",
        r"([\d,]+)\s*股",
    ]
    for pat in shares_patterns:
        m = re.search(pat, text)
        if m:
            # 移除空格和逗號再轉數字
            val_str = m.group(1).replace(",", "").replace(" ", "")
            if val_str.isdigit() and int(val_str) > 0:
                shares = int(val_str)
                break

    # 模式 3：成本/均價
    cost = 0.0
    cost_patterns = [
        r"[均平]?[價价][:\s]*([\d,.]+)\s*元?",
        r"成本[:\s]*([\d,.]+)\s*元?",
        r"買[進进][均平]?[價价]?[:\s]*([\d,.]+)",
        r"Cost[:\s]*([\d,.]+)",
    ]
    for pat in cost_patterns:
        m = re.search(pat, text, re.IGNORECASE)
        if m:
            cost = float(m.group(1).replace(",", ""))
            break

    holdings.append({
        "stock_id": stock_id,
        "cost": cost,
        "shares": shares,
    })

    # 檢查是否有多筆（同一張圖多檔持股）
    # 用 findall 找所有「代號：XXXX」
    all_stocks = re.findall(r"代[號号][:\s]*(\d{4,6})", text)
    if len(all_stocks) > 1:
        # 多筆模式：嘗試分段解析
        holdings = _parse_multi_stocks(text, all_stocks)

    return holdings


def _parse_multi_stocks(text: str, stock_ids: list) -> list:
    """解析同一張圖中的多筆持股"""
    holdings = []

    # 找到每個代號的位置，用位置切段
    positions = []
    for sid in stock_ids:
        idx = text.find(sid)
        if idx >= 0:
            positions.append((idx, sid))
    positions.sort()

    for i, (pos, sid) in enumerate(positions):
        # 取該代號到下一個代號之間的文字段
        end = positions[i + 1][0] if i + 1 < len(positions) else len(text)
        segment = text[pos:end]

        shares = 1000
        m = re.search(r"([\d,]+)\s*股", segment)
        if m:
            val = int(m.group(1).replace(",", ""))
            if val > 0:
                shares = val

        cost = 0.0
        # 找「均價/成本/價」後面的數字
        m = re.search(r"[價价成本均][:\s]*([\d,.]+)", segment)
        if m:
            cost = float(m.group(1).replace(",", ""))
        else:
            # 備用：找「X元」
            m = re.search(r"([\d,.]+)\s*元", segment)
            if m:
                cost = float(m.group(1).replace(",", ""))

        holdings.append({"stock_id": sid, "cost": cost, "shares": shares})

    # 去重
    seen = set()
    unique = []
    for h in holdings:
        if h["stock_id"] not in seen:
            seen.add(h["stock_id"])
            unique.append(h)
    return unique


def _parse_by_lines(results: list) -> list:
    """
    退回方案：逐行啟發式解析
    適用於表格式截圖（每行一檔股票）
    改進：排除明顯不是股票代號的數字（年份、金額等）
    """
    # 先收集所有文字行
    text_entries = []
    for bbox, text, confidence in results:
        y_center = (bbox[0][1] + bbox[2][1]) / 2
        x_center = (bbox[0][0] + bbox[2][0]) / 2
        text_entries.append({"text": text.strip(), "y": y_center, "x": x_center})

    text_entries.sort(key=lambda e: (e["y"], e["x"]))

    # 分行
    lines = []
    current_line = [text_entries[0]]
    for entry in text_entries[1:]:
        if abs(entry["y"] - current_line[-1]["y"]) < 20:
            current_line.append(entry)
        else:
            lines.append(current_line)
            current_line = [entry]
    lines.append(current_line)

    # 合併全文用於排除
    full_text = " ".join([e["text"] for e in text_entries])

    # 收集被「代號」「股票」等關鍵字標記的數字
    # 如果有關鍵字，優先用關鍵字模式（已在上面處理）
    # 這裡是純啟發式

    parsed = []
    # 排除清單：出現在「均價」「成本」「日期」附近的數字不是代號
    exclude_numbers = set()

    # 找金額相關數字
    for m in re.finditer(r"[價价成本均][:\s]*([\d,.]+)", full_text):
        val = m.group(1).replace(",", "").split(".")[0]
        if val.isdigit():
            exclude_numbers.add(val)

    # 找日期相關數字
    for m in re.finditer(r"(\d{4})[/\-]", full_text):
        exclude_numbers.add(m.group(1))

    for line in lines:
        line_text = " ".join([e["text"] for e in line])

        # 跳過明顯的標題行或日期行
        if any(kw in line_text for kw in ["日期", "均價", "成本", "買進"]):
            continue

        # 找 4~6 位數字
        stock_match = re.search(r"\b(\d{4,6})\b", line_text)
        if not stock_match:
            continue

        candidate = stock_match.group(1)

        # 排除：明顯不是股票代號的
        if candidate in exclude_numbers:
            continue
        if candidate.startswith("20") and len(candidate) == 4:
            # 可能是年份 2020~2029
            if 2020 <= int(candidate) <= 2029:
                continue

        # 提取同行其他數字
        remaining = line_text[stock_match.end():]
        numbers = re.findall(r"([\d,]+\.?\d*)", remaining)
        cleaned = []
        for n in numbers:
            try:
                cleaned.append(float(n.replace(",", "")))
            except ValueError:
                continue

        cost = 0.0
        shares = 1000
        for num in cleaned:
            if num >= 100 and num < 50000 and cost == 0:
                cost = num
            elif num >= 100 and shares == 1000 and num != cost:
                shares = int(num)

        parsed.append({"stock_id": candidate, "cost": cost, "shares": shares})

    # 去重
    seen = set()
    unique = []
    for h in parsed:
        if h["stock_id"] not in seen:
            seen.add(h["stock_id"])
            unique.append(h)
    return unique


class OCRError(Exception):
    """OCR 處理專用例外"""
    pass
