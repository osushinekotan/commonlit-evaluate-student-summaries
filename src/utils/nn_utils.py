def calc_steps(
    iters_per_epoch: int,
    max_epochs: int,
    gradient_accumulation_steps: int | None,
) -> tuple[int, int]:
    num_training_steps = (
        iters_per_epoch if gradient_accumulation_steps is None else iters_per_epoch // gradient_accumulation_steps
    ) * max_epochs
    return num_training_steps
