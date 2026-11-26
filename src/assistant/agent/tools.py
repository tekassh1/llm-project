"""Инструменты агента. Требование FR-6."""

from collections.abc import Callable

from assistant.graphrag.search import global_search, local_search
from assistant.infra.erp_client import ErpClient, ErpNotFound, ErpRejected
from assistant.infra.grounding import code_type, normalize_code
from assistant.infra.llm import LLMClient
from assistant.infra.schemas import KnowledgeBase

WAREHOUSES = ["Москва", "Санкт-Петербург", "Казань", "Новосибирск"]

TOOL_SCHEMAS: list[dict] = [
    {
        "type": "function",
        "function": {
            "name": "search_knowledge",
            "description": (
                "Найти факты по документам сервисной службы (граф знаний): что означает код ошибки, "
                "какая деталь стоит в модели, чем она заменена. Возвращает ответ, факты, источники и признак отказа."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "question": {
                        "type": "string",
                        "description": "Вопрос по документам, например с кодом ошибки и моделью.",
                    }
                },
                "required": ["question"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "summarize_topic",
            "description": (
                "Ответить на обзорный вопрос по сводкам сообществ графа знаний "
                "(например, какой узел чаще всего отказывает в линейке)."
            ),
            "parameters": {
                "type": "object",
                "properties": {"question": {"type": "string", "description": "Обзорный вопрос по документам."}},
                "required": ["question"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "check_stock",
            "description": "Проверить остаток детали на складе и узнать, снята ли она с производства.",
            "parameters": {
                "type": "object",
                "properties": {
                    "part_id": {"type": "string", "description": "Номер детали вида P-1042 или P-1042B."},
                    "warehouse": {"type": "string", "enum": WAREHOUSES, "description": "Склад из списка."},
                },
                "required": ["part_id", "warehouse"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_price",
            "description": "Узнать цену детали в рублях и признак discontinued (снята с производства).",
            "parameters": {
                "type": "object",
                "properties": {"part_id": {"type": "string", "description": "Номер детали вида P-1042 или P-1042B."}},
                "required": ["part_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "create_service_order",
            "description": "Создать заказ детали на склад. Выполняется только после подтверждения человеком.",
            "parameters": {
                "type": "object",
                "properties": {
                    "part_id": {"type": "string", "description": "Номер детали вида P-1042 или P-1042B."},
                    "warehouse": {"type": "string", "enum": WAREHOUSES, "description": "Склад из списка."},
                    "qty": {"type": "integer", "description": "Количество, названное пользователем."},
                    "model": {"type": "string", "description": "Модель оборудования, например AR-210."},
                },
                "required": ["part_id", "warehouse", "qty", "model"],
            },
        },
    },
]


def _check_part(part_id: object) -> str | None:
    code = normalize_code(part_id) if isinstance(part_id, str) else None
    if code is None or code_type(code) != "Part":
        return f"некорректный номер детали: {part_id!r}; ожидается код вида P-1042 или P-1042B"
    return None


def _check_warehouse(warehouse: object) -> str | None:
    if warehouse not in WAREHOUSES:
        return f"неизвестный склад: {warehouse!r}; выберите один из {', '.join(WAREHOUSES)}"
    return None


def _check_qty(qty: object) -> str | None:
    if not isinstance(qty, int) or isinstance(qty, bool) or qty <= 0:
        return f"некорректное количество: {qty!r}; ожидается целое больше нуля"
    return None


def build_tools(kb: KnowledgeBase, llm: LLMClient, erp: ErpClient) -> dict[str, Callable[..., dict]]:
    """Словарь «имя инструмента -> функция». Каждая функция возвращает словарь.

    Функция проверяет свои аргументы до обращения к ERP: склад из списка, номер детали вида P-1042
    или P-1042B, количество больше нуля. При ошибке возвращает {"error": "..."} и ничего не выполняет.
    `ErpNotFound` и `ErpRejected` превращаются в {"error": "..."} с разным текстом, чтобы исходы
    оставались различимыми. `ErpUnavailable` передаётся дальше: повторы делает агент (FR-8).
    """

    def search_knowledge(question: str) -> dict:
        answer = local_search(question, kb, llm)
        return {"answer": answer.text, "facts": answer.facts, "sources": answer.sources, "refused": answer.refused}

    def summarize_topic(question: str) -> dict:
        answer = global_search(question, kb, llm)
        return {"answer": answer.text, "sources": answer.sources, "refused": answer.refused}

    def check_stock(part_id: str, warehouse: str) -> dict:
        error = _check_part(part_id) or _check_warehouse(warehouse)
        if error:
            return {"error": error}
        try:
            return erp.get_stock(part_id, warehouse)
        except ErpNotFound as exc:
            return {"error": f"данных нет: деталь {part_id} не найдена в ERP ({exc})"}
        except ErpRejected as exc:
            return {"error": f"ERP отклонила запрос остатка: {exc}"}

    def get_price(part_id: str) -> dict:
        error = _check_part(part_id)
        if error:
            return {"error": error}
        try:
            return erp.get_price(part_id)
        except ErpNotFound as exc:
            return {"error": f"данных нет: деталь {part_id} не найдена в ERP ({exc})"}
        except ErpRejected as exc:
            return {"error": f"ERP отклонила запрос цены: {exc}"}

    def create_service_order(part_id: str, warehouse: str, qty: int, model: str) -> dict:
        error = _check_part(part_id) or _check_warehouse(warehouse) or _check_qty(qty)
        if error:
            return {"error": error}
        try:
            return erp.create_order(part_id, warehouse, qty, model)
        except ErpNotFound as exc:
            return {"error": f"данных нет: деталь {part_id} не найдена в ERP ({exc})"}
        except ErpRejected as exc:
            return {"error": f"ERP отклонила заказ: {exc}"}

    return {
        "search_knowledge": search_knowledge,
        "summarize_topic": summarize_topic,
        "check_stock": check_stock,
        "get_price": get_price,
        "create_service_order": create_service_order,
    }
