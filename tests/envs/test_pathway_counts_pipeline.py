# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Local fixtures for the true-counts FASTQ cache boundary."""

import pytest
from pathway_analysis_env.bench import run_true_counts_pipeline as pipeline


@pytest.mark.parametrize("mate", [1, 2])
@pytest.mark.parametrize("cached", [False, True])
def test_incomplete_pair_is_never_used_as_single_end(
    tmp_path, monkeypatch, mate, cached
):
    sample = pipeline.Sample("sample", "SRR1", "control")
    partial = tmp_path / f"SRR1_{mate}.fastq"
    if cached:
        partial.write_text("@read\nACGT\n+\n!!!!\n")

    calls = []

    def fake_run(cmd):
        calls.append(cmd)
        partial.write_text("@read\nACGT\n+\n!!!!\n")

    monkeypatch.setattr(pipeline, "_run", fake_run)
    with pytest.raises(RuntimeError, match="Incomplete paired FASTQ"):
        pipeline._fasterq_dump_one(sample, out_dir=tmp_path, threads_per_job=1)
    assert len(calls) == (0 if cached else 1)
    assert partial.exists()  # Preserve the cache for diagnosis/recovery.


@pytest.mark.parametrize("paired", [False, True])
@pytest.mark.parametrize("cached", [False, True])
def test_complete_fastqs_keep_their_layout(tmp_path, monkeypatch, paired, cached):
    sample = pipeline.Sample("sample", "SRR1", "control")
    names = ["SRR1_1.fastq", "SRR1_2.fastq"] if paired else ["SRR1.fastq"]
    paths = [tmp_path / name for name in names]
    calls = []

    def create_files():
        for path in paths:
            path.write_text("@read\nACGT\n+\n!!!!\n")

    def fake_run(cmd):
        calls.append(cmd)
        create_files()

    if cached:
        create_files()
    monkeypatch.setattr(pipeline, "_run", fake_run)
    result = pipeline._fasterq_dump_one(sample, out_dir=tmp_path, threads_per_job=2)
    assert result == ("sample", "SRR1", (paths[0], paths[1] if paired else None))
    assert len(calls) == (0 if cached else 1)


def test_single_fastq_does_not_mask_incomplete_pair(tmp_path, monkeypatch):
    for name in ("SRR1.fastq", "SRR1_1.fastq"):
        (tmp_path / name).write_text("@read\nACGT\n+\n!!!!\n")

    def unexpected_run(cmd):
        pytest.fail(
            "Incomplete cached outputs should fail before invoking fasterq-dump"
        )

    monkeypatch.setattr(pipeline, "_run", unexpected_run)
    with pytest.raises(RuntimeError, match="Incomplete paired FASTQ"):
        pipeline._fasterq_dump_one(
            pipeline.Sample("sample", "SRR1", "control"),
            out_dir=tmp_path,
            threads_per_job=1,
        )
