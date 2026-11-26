
from collections import Counter, defaultdict

import networkx as nx

from assistant.infra.llm import SEED, LLMClient
from assistant.infra.schemas import Community

MAX_SUMMARY_WORDS = 150
TOP_EDGES = 15  # сколько самых весомых связей показываем модели
TOP_ITEMS = 8  # сколько строк оставляем в рейтингах отказов, замен и лидеров
TOP_SOURCES = 12

SYSTEM_PROMPT = (
    "Ты пишешь сводку сообщества графа знаний сервисной службы «ПолярТех». "
    "Ответь одним связным абзацем на русском языке без заголовков и списков, не длиннее 150 слов. "
    "Скажи: какие сущности входят в сообщество, как они связаны между собой и что встречается чаще всего. "
    "Все числа уже посчитаны кодом и приведены в фактах: используй их дословно, ничего не считай, "
    "не округляй и не выдумывай. Не придумывай коды деталей, названия и связи, которых нет в фактах. "
    "Если данных мало, напиши об этом прямо."
)


def detect_communities(graph: nx.DiGraph) -> list[Community]:
    if graph.number_of_nodes() == 0:
        return []

    undirected = _undirected(graph)
    groups = nx.community.louvain_communities(undirected, weight="weight", resolution=1.0, seed=SEED)

    ordered = sorted((sorted(group) for group in groups), key=lambda nodes: (-len(nodes), nodes[0]))
    return [Community(community_id=number, nodes=nodes) for number, nodes in enumerate(ordered)]


def summarize_community(community: Community, graph: nx.DiGraph, llm: LLMClient) -> str:
    nodes = sorted({node for node in community.nodes if node in graph})
    if not nodes:
        return ""

    facts = _community_facts(community.community_id, nodes, graph)
    response = llm.chat(
        [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": facts},
        ],
        max_tokens=600,
    )

    summary = " ".join(response.content.split())
    if not summary:
        summary = _fallback_summary(nodes, graph)
    return " ".join(summary.split()[:MAX_SUMMARY_WORDS])


def _undirected(graph: nx.DiGraph) -> nx.Graph:
    undirected = nx.Graph()
    undirected.add_nodes_from(sorted(graph.nodes))
    edges = sorted(graph.edges(data=True), key=lambda item: (item[0], item[1]))
    for source, target, data in edges:
        if source == target:
            continue
        weight = _weight(data)
        if undirected.has_edge(source, target):
            undirected[source][target]["weight"] += weight
        else:
            undirected.add_edge(source, target, weight=weight)
    return undirected


def _weight(data: dict) -> float:
    try:
        return max(0.0, float(data.get("weight", 1)))
    except (TypeError, ValueError):
        return 1.0


def _number(value: float) -> str:
    return str(int(value)) if float(value).is_integer() else f"{value:.1f}"


def _name(graph: nx.DiGraph, node: str) -> str:
    return str(graph.nodes[node].get("name") or node.split(":", 1)[-1])


def _kind(graph: nx.DiGraph, node: str) -> str:
    return str(graph.nodes[node].get("type") or node.split(":", 1)[0])


def _community_facts(community_id: int, nodes: list[str], graph: nx.DiGraph) -> str:
    members = set(nodes)

    by_type: dict[str, list[str]] = defaultdict(list)
    for node in nodes:
        by_type[_kind(graph, node)].append(_name(graph, node))

    internal: list[tuple[str, str, dict]] = []
    outside = 0
    for source, target, data in graph.edges(data=True):
        if source in members and target in members:
            internal.append((source, target, data))
        elif source in members or target in members:
            outside += 1

    per_type: dict[str, list[float]] = defaultdict(list)
    failures: Counter[str] = Counter()
    degree: Counter[str] = Counter()
    replacements: list[str] = []
    for source, target, data in internal:
        weight = _weight(data)
        kind = str(data.get("type") or "RELATED")
        per_type[kind].append(weight)
        degree[source] += weight
        degree[target] += weight
        if kind == "FAILED":
            failures[_name(graph, target)] += weight
        if kind == "REPLACED_BY":
            replacements.append(f"{_name(graph, source)} -> {_name(graph, target)} (вес {_number(weight)})")

    lines = [f"Сообщество {community_id}. Узлов: {len(nodes)}. Связей внутри: {len(internal)}.", ""]

    lines.append("Сущности по типам:")
    for kind in sorted(by_type):
        names = sorted(by_type[kind])
        shown = ", ".join(names[:TOP_ITEMS])
        rest = f" и ещё {len(names) - TOP_ITEMS}" if len(names) > TOP_ITEMS else ""
        lines.append(f"- {kind} ({len(names)}): {shown}{rest}")
    lines.append("")

    lines.append("Связи по типам (число связей и суммарный вес):")
    if per_type:
        for kind in sorted(per_type, key=lambda item: (-sum(per_type[item]), item)):
            weights = per_type[kind]
            lines.append(f"- {kind}: связей {len(weights)}, суммарный вес {_number(sum(weights))}")
    else:
        lines.append("- связей внутри сообщества нет")
    lines.append("")

    lines.append("Самые весомые связи (вес = число чанков, где связь найдена):")
    top = sorted(internal, key=lambda item: (-_weight(item[2]), item[0], item[1]))[:TOP_EDGES]
    for source, target, data in top:
        kind = str(data.get("type") or "RELATED")
        lines.append(
            f"- {_name(graph, source)} ({_kind(graph, source)}) -{kind}-> "
            f"{_name(graph, target)} ({_kind(graph, target)}): вес {_number(_weight(data))}"
        )
    if not top:
        lines.append("- нет")
    lines.append("")

    lines.append("Что встречается чаще всего:")
    if failures:
        rating = "; ".join(f"{name} — {_number(weight)}" for name, weight in failures.most_common(TOP_ITEMS))
        lines.append(f"- отказы узлов (сумма весов FAILED): {rating}")
    else:
        lines.append("- отказов (FAILED) в сообществе нет")
    if replacements:
        lines.append(f"- замены деталей: {'; '.join(replacements[:TOP_ITEMS])}")
    leaders = "; ".join(f"{_name(graph, node)} — {_number(weight)}" for node, weight in degree.most_common(TOP_ITEMS))
    lines.append(f"- узлы по суммарному весу связей: {leaders or 'нет'}")
    lines.append(f"- связей с другими сообществами: {outside}")

    sources = _sources(graph, nodes, internal)
    if sources:
        shown = ", ".join(sources[:TOP_SOURCES])
        rest = f" и ещё {len(sources) - TOP_SOURCES}" if len(sources) > TOP_SOURCES else ""
        lines.append(f"Документы-источники: {shown}{rest}")

    return "\n".join(lines)


def _sources(graph: nx.DiGraph, nodes: list[str], internal: list[tuple[str, str, dict]]) -> list[str]:
    chunks: set[str] = set()
    for node in nodes:
        chunks.update(graph.nodes[node].get("sources") or [])
    for _, _, data in internal:
        chunks.update(data.get("sources") or [])
    return sorted({str(chunk).split("#", 1)[0] for chunk in chunks})


def _fallback_summary(nodes: list[str], graph: nx.DiGraph) -> str:
    counts = Counter(_kind(graph, node) for node in nodes)
    composition = ", ".join(f"{kind}: {count}" for kind, count in sorted(counts.items()))
    heaviest = max(graph.edges(nodes, data=True), key=lambda item: _weight(item[2]), default=None)
    text = f"В сообществе {len(nodes)} узлов ({composition})."
    if heaviest is not None:
        source, target, data = heaviest
        kind = str(data.get("type") or "RELATED")
        text += (
            f" Самая весомая связь: {_name(graph, source)} -{kind}-> {_name(graph, target)},"
            f" вес {_number(_weight(data))}."
        )
    return text