import json
from itertools import combinations, pairwise

import networkx as nx

from assistant.infra.grounding import code_type, find_codes, find_names
from assistant.infra.llm import LLMClient
from assistant.infra.schemas import MAP_SCHEMA, Answer, KnowledgeBase

MAX_FACTS_CHARS = 8000  # предельный объём списка фактов для модели
PATH_DEPTH = 3  # связей на пути между опорными сущностями
NEIGHBOUR_DEPTH = 2  # глубина окрестности при одной опорной сущности
REPLACEMENT_DEPTH = 5  # предельная длина цепочки замен — защита от циклов
MAX_PARTIAL_CHARS = 6000  # объём частичных ответов на шаге reduce
REFUSAL = "В базе знаний не найдено."

LOCAL_SYSTEM = (
    "Ты отвечаешь диспетчеру сервисной службы «ПолярТех» только по фактам графа знаний. "
    "Факты пронумерованы, каждый имеет вид «сущность -СВЯЗЬ-> сущность». "
    "Отвечай кратко, по-русски, опирайся только на эти факты. "
    "Направление связи читай слева направо: HAS_PART ведёт от модели к детали, "
    "INDICATES — от кода ошибки к узлу, REPLACED_BY — от снятой детали к её замене. "
    "Если в фактах есть цепочка REPLACED_BY, называй последнюю деталь цепочки: прежние сняты с производства. "
    "Не выдумывай коды, названия, цены и остатки. "
    "Если фактов для ответа не хватает, ответь: «В базе знаний не найдено.»"
)

MAP_SYSTEM = (
    "Ты отвечаешь на обзорный вопрос по одной сводке сообщества графа знаний. "
    'Верни JSON по схеме {"answer": строка, "score": целое от 0 до 10}. '
    "В answer дай короткий частичный ответ только по этой сводке; если сводка не про вопрос, напиши, что данных нет. "
    "score — насколько сводка помогла ответить: 0, если она не относится к вопросу, 10, если ответ полностью из неё. "
    "Числа бери из сводки дословно, ничего не считай и не выдумывай."
)

REDUCE_SYSTEM = (
    "Ты объединяешь частичные ответы по разным сообществам графа знаний в один итоговый ответ. "
    "Отвечай по-русски, кратко и связно, только по приведённым частичным ответам. "
    "Сохраняй коды, названия и числа из них дословно, ничего не добавляй от себя. "
    "Если частичные ответы противоречат друг другу, скажи об этом. Повторы объединяй."
)


def local_search(question: str, kb: KnowledgeBase, llm: LLMClient) -> Answer:
    graph = kb.graph
    anchors: list[str] = []
    for code in find_codes(question):
        node = f"{code_type(code)}:{code}"
        if node not in graph:
            return Answer(REFUSAL, refused=True)
        if node not in anchors:
            anchors.append(node)
    for kind, name in find_names(question, kb.vocabulary):
        node = f"{kind}:{name}"
        if node in graph and node not in anchors:
            anchors.append(node)
    if not anchors:
        return Answer(REFUSAL, refused=True)

    base, chains = _select_edges(graph, anchors)
    facts: list[str] = []
    used: list[dict] = []
    size = 0

    def take(edges: list[tuple[tuple[str, str], dict]], limit: int) -> None:
        nonlocal size
        for (source, target), data in edges:
            fact = _fact(graph, source, target, data)
            line = f"{len(facts) + 1}. {fact}"
            if size + len(line) + 1 > limit:
                break
            facts.append(fact)
            used.append(data)
            size += len(line) + 1

    take(list(chains.items()), MAX_FACTS_CHARS // 2)
    take([(key, data) for key, data in sorted(base.items(), key=_by_weight) if key not in chains], MAX_FACTS_CHARS)
    if not facts:
        return Answer(REFUSAL, refused=True)

    numbered = "\n".join(f"{number}. {fact}" for number, fact in enumerate(facts, start=1))
    response = llm.chat(
        [
            {"role": "system", "content": LOCAL_SYSTEM},
            {"role": "user", "content": f"Вопрос: {question}\n\nФакты:\n{numbered}"},
        ],
        max_tokens=512,
    )
    text = response.content.strip()
    if not text:
        return Answer(REFUSAL, refused=True)
    return Answer(text=text, sources=_documents(used), facts=facts)


def global_search(question: str, kb: KnowledgeBase, llm: LLMClient) -> Answer:
    partials: list[tuple[int, str, list[str]]] = []
    for community in kb.communities:
        summary = community.summary.strip()
        if not summary:
            continue
        prompt = f"Вопрос: {question}\n\nСводка сообщества {community.community_id}:\n{summary}"
        response = llm.chat(
            [
                {"role": "system", "content": MAP_SYSTEM},
                {"role": "user", "content": prompt},
            ],
            json_schema=MAP_SCHEMA,
            max_tokens=400,
        )
        parsed = _partial(response.content)
        if parsed is None:
            continue
        answer, score = parsed
        if score > 0 and answer:
            partials.append((score, answer, community.nodes))
    if not partials:
        return Answer(REFUSAL, refused=True)

    partials.sort(key=lambda item: -item[0])
    lines: list[str] = []
    size = 0
    for number, (score, answer, _) in enumerate(partials, start=1):
        line = f"{number}. (полезность {score}) {answer}"
        if size + len(line) + 1 > MAX_PARTIAL_CHARS:
            break
        lines.append(line)
        size += len(line) + 1

    response = llm.chat(
        [
            {"role": "system", "content": REDUCE_SYSTEM},
            {"role": "user", "content": f"Вопрос: {question}\n\nЧастичные ответы:\n" + "\n".join(lines)},
        ],
        max_tokens=512,
    )
    text = response.content.strip()
    if not text:
        return Answer(REFUSAL, refused=True)

    chunks: set[str] = set()
    for _, _, nodes in partials:
        for node in nodes:
            if node in kb.graph:
                chunks.update(kb.graph.nodes[node].get("sources") or [])
    return Answer(text=text, sources=_docs(chunks))


def _select_edges(
        graph: nx.DiGraph, anchors: list[str]
) -> tuple[dict[tuple[str, str], dict], dict[tuple[str, str], dict]]:
    undirected = graph.to_undirected(as_view=True)
    base: dict[tuple[str, str], dict] = {}
    chains: dict[tuple[str, str], dict] = {}

    if len(anchors) == 1:
        area = set(nx.single_source_shortest_path_length(undirected, anchors[0], cutoff=NEIGHBOUR_DEPTH))
        for source, target, data in graph.subgraph(area).edges(data=True):
            base[(source, target)] = data
        return base, chains

    on_paths: set[str] = set()
    for first, second in combinations(anchors, 2):
        for path in nx.all_simple_paths(undirected, first, second, cutoff=PATH_DEPTH):
            on_paths.update(path)
            for source, target in pairwise(path):
                for head, tail in ((source, target), (target, source)):
                    if graph.has_edge(head, tail):
                        base[(head, tail)] = graph[head][tail]
    for node in sorted(on_paths):
        if _kind(graph, node) == "Part":
            _replacements(graph, node, chains)
    return base, chains


def _replacements(graph: nx.DiGraph, part: str, chains: dict[tuple[str, str], dict]) -> None:
    frontier = {part}
    seen = {part}
    for _ in range(REPLACEMENT_DEPTH):
        following: set[str] = set()
        for node in frontier:
            for target in graph.successors(node):
                if graph[node][target].get("type") != "REPLACED_BY":
                    continue
                chains[(node, target)] = graph[node][target]
                if target not in seen:
                    seen.add(target)
                    following.add(target)
        if not following:
            break
        frontier = following


def _by_weight(item: tuple[tuple[str, str], dict]) -> tuple[float, str, str]:
    (source, target), data = item
    try:
        weight = float(data.get("weight", 1))
    except (TypeError, ValueError):
        weight = 1.0
    return (-weight, source, target)


def _kind(graph: nx.DiGraph, node: str) -> str:
    return str(graph.nodes[node].get("type") or node.split(":", 1)[0])


def _name(graph: nx.DiGraph, node: str) -> str:
    return str(graph.nodes[node].get("name") or node.split(":", 1)[-1])


def _fact(graph: nx.DiGraph, source: str, target: str, data: dict) -> str:
    kind = str(data.get("type") or "RELATED")
    return f"{_name(graph, source)} -{kind}-> {_name(graph, target)}"


def _documents(edges: list[dict]) -> list[str]:
    chunks: set[str] = set()
    for data in edges:
        chunks.update(data.get("sources") or [])
    return _docs(chunks)


def _docs(chunks: set[str]) -> list[str]:
    return sorted({str(chunk).split("#", 1)[0] for chunk in chunks})


def _partial(content: str) -> tuple[str, int] | None:
    text = content.strip()
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end <= start:
        return None
    try:
        data = json.loads(text[start : end + 1])
    except json.JSONDecodeError:
        return None
    if not isinstance(data, dict):
        return None
    answer = data.get("answer")
    if not isinstance(answer, str) or not answer.strip():
        return None
    try:
        score = int(float(data.get("score", 0)))
    except (TypeError, ValueError):
        return None
    return answer.strip(), max(0, min(10, score))