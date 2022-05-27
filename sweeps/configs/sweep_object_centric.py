from sweeps import sweep_utils as sw


def get_sweep():
    general = [
        sw.sweep("seed", range(10)),
        sw.sweep("model", ["monet", "slot-attention", "genesis", "space"]),
        sw.sweep("allow_resume", ["True"]),
        sw.sweep("num_workers", ["1"]),
    ]  # List[List[Dict]]

    dataset = sw.sweep(
        "dataset",
        ["clevr", "multidsprites", "objects_room", "shapestacks", "tetrominoes"],
    )
    for item in dataset:
        if item["dataset"] == "clevr":
            item["+dataset.variant"] = "6"
    dataset = [dataset]  # List[List[Dict]]

    return sw.product(general + dataset)
