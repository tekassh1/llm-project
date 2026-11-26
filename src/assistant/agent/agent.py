"""Цикл агента. Требования FR-7, FR-8, FR-9 и FR-10."""

import json
import re
from collections.abc import Callable

from assistant.agent.tools import TOOL_SCHEMAS
from assistant.infra.erp_client import ErpUnavailable
from assistant.infra.grounding import code_type, find_codes, normalize_code
from assistant.infra.llm import LLMClient, ToolCall, tool_message
from assistant.infra.schemas import AgentResult, ToolStep

REFUSAL = "В базе знаний не найдено."
ERP_RETRIES = 3
CORRECTION = (
    "Проверка не пройдена: в ответе есть коды, цены или остатки, которых нет ни в вопросе, "
    "ни в результатах инструментов. Исправь ответ, опираясь только на проверенные данные, "
    "или ответь, что данных нет."
)

AGENT_SYSTEM = (
    "Ты ассистент диспетчера сервисной службы «ПолярТех». Отвечай по-русски, кратко. "
    "Сначала выясняй факты по документам через search_knowledge (или summarize_topic для обзорных вопросов), "
    "затем проверяй остатки и цену детали через ERP. "
    "Номер детали бери из вопроса или из результатов поиска по базе знаний, не выдумывай коды. "
    "Количество для заказа должно быть названо пользователем. "
    "Не придумывай цены и остатки: используй только то, что вернули инструменты. "
    "Если данных для ответа нет, ответь: «В базе знаний не найдено.»"
)


def _bare_numbers(text: str) -> set[int]:
    """Целые числа из текста за вычетом цифр, входящих в коды сущностей."""
    stripped = text
    for code in find_codes(text):
        stripped = stripped.replace(code, " ")
    return {int(number) for number in re.findall(r"\d+", stripped)}


def _quantity_named(question: str, qty: object) -> bool:
    if not isinstance(qty, int) or isinstance(qty, bool):
        return False
    return qty in _bare_numbers(question)


def _cache_key(name: str, args: dict) -> str:
    return json.dumps([name, args], ensure_ascii=False, sort_keys=True)


def _dedup(items: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for item in items:
        if item not in seen:
            seen.add(item)
            result.append(item)
    return result


def _call_with_retries(func: Callable[..., dict], args: dict) -> tuple[dict | None, str | None]:
    for attempt in range(ERP_RETRIES):
        try:
            return func(**args), None
        except ErpUnavailable:
            if attempt == ERP_RETRIES - 1:
                return None, "данные ERP получить не удалось после нескольких попыток"
    return None, "данные ERP получить не удалось после нескольких попыток"


def _answer_is_grounded(text: str, codes: set[str], numbers: set[int]) -> bool:
    return all(code in codes for code in find_codes(text)) and all(number in numbers for number in _bare_numbers(text))


def run_agent(
    question: str,
    llm: LLMClient,
    tools: dict[str, Callable[..., dict]],
    *,
    max_steps: int = 6,
    confirm: bool = False,
) -> AgentResult:
    """Ведёт диалог с моделью в цикле «рассуждение, вызов инструмента, наблюдение».

    Модель получает `TOOL_SCHEMAS` через `llm.chat(..., tools=...)`. Каждый вызов записывается
    в `steps` как `ToolStep`. Агент сам проверяет номер детали в аргументах (он должен встречаться
    в вопросе или в результатах прошлых инструментов), количество в заказе (должно быть названо
    пользователем) и повторные вызовы. При `ErpUnavailable` вызов повторяется до двух раз.
    Заказ создаётся только при `confirm=True`, иначе агент возвращает `needs_confirmation=True`
    и `pending_action`. Перед выдачей ответа коды, цены и остатки сверяются с вопросом
    и результатами инструментов.
    """
    messages: list[dict] = [
        {"role": "system", "content": AGENT_SYSTEM},
        {"role": "user", "content": question},
    ]
    steps: list[ToolStep] = []
    sources: list[str] = []
    cache: dict[str, dict] = {}
    grounding_codes: set[str] = set(find_codes(question))
    grounding_numbers: set[int] = _bare_numbers(question)
    steps_taken = 0
    corrected = False

    while steps_taken < max_steps:
        response = llm.chat(messages, tools=TOOL_SCHEMAS)
        messages.append(response.as_message())

        if not response.tool_calls:
            text = response.content.strip()
            if _answer_is_grounded(text, grounding_codes, grounding_numbers):
                return AgentResult(
                    answer=text,
                    steps=steps,
                    sources=_dedup(sources),
                    refused=text == REFUSAL,
                )
            if not corrected:
                corrected = True
                messages.append({"role": "user", "content": CORRECTION})
                continue
            return AgentResult(answer=REFUSAL, steps=steps, sources=_dedup(sources), refused=True)

        for call in response.tool_calls:
            if steps_taken >= max_steps:
                break
            step, pending = _run_call(call, tools, question, confirm, grounding_codes, cache)
            if pending is not None:
                return AgentResult(
                    answer="Для создания заказа требуется подтверждение.",
                    steps=steps,
                    sources=_dedup(sources),
                    needs_confirmation=True,
                    pending_action=pending,
                )
            steps.append(step)
            steps_taken += 1
            if step.result is not None:
                blob = json.dumps(step.result, ensure_ascii=False)
                grounding_codes.update(find_codes(blob))
                grounding_numbers.update(_bare_numbers(blob))
                if isinstance(step.result.get("sources"), list):
                    sources.extend(step.result["sources"])
            messages.append(tool_message(call, step.result if step.result is not None else {"error": step.error}))

    return AgentResult(
        answer="Работа прервана: достигнут лимит шагов, ответ подготовить не удалось.",
        steps=steps,
        sources=_dedup(sources),
    )


def _run_call(
    call: ToolCall,
    tools: dict[str, Callable[..., dict]],
    question: str,
    confirm: bool,
    grounding_codes: set[str],
    cache: dict[str, dict],
) -> tuple[ToolStep, dict | None]:
    """Выполняет один вызов инструмента. Возвращает шаг и, если нужно подтверждение, pending_action."""
    name, args = call.name, call.arguments

    if name not in tools:
        return ToolStep(tool=name, args=args, error=f"неизвестный инструмент: {name}"), None
    if "_raw" in args:
        return ToolStep(tool=name, args=args, error="не удалось разобрать аргументы вызова"), None

    part = args.get("part_id")
    if part is not None:
        code = normalize_code(part) if isinstance(part, str) else None
        if code is None or code_type(code) != "Part":
            return ToolStep(tool=name, args=args, error=f"некорректный номер детали: {part!r}"), None
        if code not in grounding_codes:
            hint = f"деталь {code} не встречалась в вопросе или результатах; сначала вызови search_knowledge"
            return ToolStep(tool=name, args=args, error=hint), None

    if name == "create_service_order" and not _quantity_named(question, args.get("qty")):
        hint = "количество не названо пользователем; уточни, сколько штук заказать"
        return ToolStep(tool=name, args=args, error=hint), None

    if name == "create_service_order" and not confirm:
        return ToolStep(tool=name, args=args), {"tool": name, "args": args}

    key = _cache_key(name, args)
    if key in cache:
        return ToolStep(tool=name, args=args, result=cache[key]), None

    result, error = _call_with_retries(tools[name], args)
    if result is not None:
        cache[key] = result
    return ToolStep(tool=name, args=args, result=result, error=error), None
