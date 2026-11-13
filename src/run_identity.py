"""Stable experiment names shared by single runs, grids and sweeps."""


def model_tag(model, jk_mode="cat", head="mlp"):
    if model == "jknet" and jk_mode != "cat":
        return f"jknet-{jk_mode}"
    if model == "attentivefp" and head == "linear":
        return "attentivefp-linear"
    return model


def run_name(dataset, model, readout, seed, sp=None, split_idx=None,
             jk_mode="cat", head="mlp"):
    tag = model_tag(model, jk_mode, head)
    name = f"{dataset.upper()}_{tag}_{readout}"
    if sp is not None:
        name += f"_sp{sp:.2f}_split{split_idx}"
    return name + f"_seed{seed}"
