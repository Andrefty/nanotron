import torch
from datasets import load_from_disk
from nanotron import distributed as dist
from nanotron.doremi.dataloader import DistributedSamplerForDoReMi
from nanotron.doremi.doremi_context import DoReMiContext
from nanotron.parallel import ParallelContext
from tqdm import tqdm

if __name__ == "__main__":
    DP_SIZE = 4
    # # domain_weights = torch.tensor(
    # #     [
    # #         0.34356916553540745,
    # #         # 0.16838812972610234,
    # #         # 0.24711766854236725,
    # #         # 0.0679225638705455,
    # #         # 0.059079828519653675,
    # #         # 0.043720261601881555,
    # #         # 0.01653850841342608,
    # #         # 0.00604146633842096,
    # #         # 0.04342813428189645,
    # #         # 0.0041942731702987,
    # #     ]
    # # )
    # domain_weights = torch.tensor([0.6, 0.4])

    # dataset1 = load_dataset("stas/c4-en-10k", split="train[:100]")
    # datasets = [dataset1 for _ in range(len(domain_weights))]

    DATASET_PATH = "/fsx/phuc/project_data/doremi/datasets/the_pile_splitted/tokenized_data"
    DOMAIN_KEYS = [
        "Github",
        "FreeLaw",
        "OpenWebText2",
        "PubMed Abstracts",
        "DM Mathematics",
        "OpenSubtitles",
        "HackerNews",
        "NIH ExPorter",
        "PubMed Central",
        "Enron Emails",
    ]
    # TOKENIZED_DATASETS = {f"{domain_name}": f"{DATASET_PATH}/{domain_name}" for domain_name in DOMAIN_KEYS}
    TOKENIZED_DATASETS = [f"{DATASET_PATH}/{domain_name}" for domain_name in DOMAIN_KEYS]
    domain_weights = torch.tensor(
        [
            0.34356916553540745,
            0.16838812972610234,
            0.24711766854236725,
            0.0679225638705455,
            0.059079828519653675,
            0.043720261601881555,
            0.01653850841342608,
            0.00604146633842096,
            0.04342813428189645,
            0.0041942731702987,
        ]
    )

    datasets = []
    for dataset_path in tqdm(TOKENIZED_DATASETS, desc="Loading tokenized dataset from disk"):
        d = load_from_disk(dataset_path)
        datasets.append(d)

    parallel_context = ParallelContext(
        data_parallel_size=DP_SIZE,
        pipeline_parallel_size=1,
        tensor_parallel_size=1,
    )

    global_batch_size = 32
    num_microbatches = 2
    batch_size = global_batch_size // (num_microbatches * DP_SIZE)

    dp_size = dist.get_world_size(parallel_context.dp_pg)
    dp_rank = dist.get_rank(parallel_context.dp_pg)
    domain_keys = [f"domain {i}" for i in range(len(datasets))]
    doremi_context = DoReMiContext(domain_weights, domain_keys, is_proxy=False)

    sampler = DistributedSamplerForDoReMi(
        datasets,
        batch_size=batch_size,
        num_microbatches=num_microbatches,
        num_replicas=dp_size,
        rank=dp_rank,
        doremi_context=doremi_context,
        parallel_context=parallel_context,
    )

    # microbatch_idx = 0
    # yielded_idxs = []
    # for idxs in sampler:
    #     # NOTE: check that the indicies are not repeated
    #     assert not set(idxs).intersection(
    #         yielded_idxs
    #     ), f"microbatch_idx: {microbatch_idx}, yielded_idxs: {yielded_idxs}, idxs: {idxs}"

    #     microbatch_idx += 1
    #     yielded_idxs.extend(idxs)

    iter_sampler = iter(sampler)
    epoch = 0
    yieled_idxs = []
    while True:
        # idxs = (next(sampler) for _ in range(8))

        idxs = []
        for _ in range(num_microbatches):
            idxs.extend(next(iter_sampler))

        # NOTE: check not repeating idxs
        assert not set(idxs).intersection(yieled_idxs), f"epoch: {epoch}"

        if epoch % 1000 == 0:
            print(f"rank: {dist.get_rank(parallel_context.dp_pg)}, epoch: {epoch} \n \n")

        epoch += 1
        yieled_idxs.extend(idxs)