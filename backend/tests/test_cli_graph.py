from app.cli import graph_levels


def _t(key, deps):
    return {"key": key, "depends_on": deps, "status": "pending", "title": key, "action": "x"}


def test_layers_follow_dependencies_and_group_parallel_tasks():
    layers = graph_levels([_t("A", []), _t("B", ["A"]), _t("C", ["A"]), _t("D", ["B", "C"])])
    assert [[t["key"] for t in layer] for layer in layers] == [["A"], ["B", "C"], ["D"]]


def test_unknown_dependency_does_not_crash():
    assert [[t["key"] for t in layer] for layer in graph_levels([_t("A", ["ghost"])])] == [["A"]]
