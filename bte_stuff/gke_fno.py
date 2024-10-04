# SPDX-FileCopyrightText: Copyright (c) 2023 - 2024 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import modulus.sym
from modulus.sym.hydra import to_absolute_path, instantiate_arch, ModulusConfig
from modulus.sym.key import Key

from modulus.sym.solver import Solver
from modulus.sym.domain import Domain
from modulus.sym.domain.constraint import SupervisedGridConstraint
from modulus.sym.domain.validator import GridValidator
from modulus.sym.dataset import HDF5GridDataset

from modulus.sym.utils.io.plotter import GridValidatorPlotter

@modulus.sym.main(config_path="conf", config_name="config_FNO")
def run(cfg: ModulusConfig) -> None:
    # [keys]
    # Load training/test data
    input_keys = [Key("temperature", scale=(shift_temperature, scale_temperature))]
    output_keys = [
        Key("qx", scale=(shift_qx, scale_qx)),
        Key("qy", scale=(shift_qy, scale_qy)),
    ]

    # Dataset paths (Ensure the dataset is prepared for GKE)
    train_path = to_absolute_path("datasets/GKE_train.hdf5")
    test_path = to_absolute_path("datasets/GKE_test.hdf5")

    # [datasets]
    # Make datasets
    train_dataset = HDF5GridDataset(
        train_path, invar_keys=["temperature"], outvar_keys=["qx", "qy"], n_examples=1000
    )
    test_dataset = HDF5GridDataset(
        test_path, invar_keys=["temperature"], outvar_keys=["qx", "qy"], n_examples=100
    )
    # [datasets]

    # [init-model]
    # Initialize the FNO model with decoder network
    decoder_net = instantiate_arch(
        cfg=cfg.arch.decoder,
        output_keys=output_keys,
    )
    fno = instantiate_arch(
        cfg=cfg.arch.fno,
        input_keys=input_keys,
        decoder_net=decoder_net,
    )
    nodes = [fno.make_node("fno")]
    # [init-model]

    # [constraint]
    # Make the domain and add the constraints (Supervised learning using the dataset)
    domain = Domain()

    # Add data-driven constraint (Supervised learning)
    supervised = SupervisedGridConstraint(
        nodes=nodes,
        dataset=train_dataset,
        batch_size=cfg.batch_size.grid,
        num_workers=4,  # Number of parallel data loaders
    )
    domain.add_constraint(supervised, "supervised")
    # [constraint]

    # [validator]
    # Add validator to validate the model's performance on unseen data
    val = GridValidator(
        nodes,
        dataset=test_dataset,
        batch_size=cfg.batch_size.validation,
        plotter=GridValidatorPlotter(n_examples=5),
    )
    domain.add_validator(val, "test")
    # [validator]

    # Initialize the solver
    slv = Solver(cfg, domain)

    # Start the solver
    slv.solve()


if __name__ == "__main__":
    run()