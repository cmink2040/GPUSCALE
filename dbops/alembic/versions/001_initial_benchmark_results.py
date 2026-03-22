"""Initial benchmark_results table.

Revision ID: 001
Revises: None
Create Date: 2026-03-22
"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "001"
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "benchmark_results",
        # Identity
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True, server_default=sa.text("gen_random_uuid()")),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
        # GPU
        sa.Column("gpu_name", sa.Text(), nullable=False),
        sa.Column("gpu_vram_gb", sa.Numeric(), nullable=False),
        sa.Column("gpu_count", sa.Integer(), nullable=False),
        # Environment
        sa.Column("provider", sa.Text(), nullable=False),
        sa.Column("engine", sa.Text(), nullable=False),
        # Model
        sa.Column("model_name", sa.Text(), nullable=False),
        sa.Column("quantization", sa.Text(), nullable=False),
        # Workload
        sa.Column("workload_version", sa.Text(), nullable=False),
        sa.Column("workload_config", postgresql.JSONB(), nullable=True),
        # Performance metrics
        sa.Column("tokens_per_sec", sa.Numeric(), nullable=False),
        sa.Column("time_to_first_token_ms", sa.Numeric(), nullable=False),
        sa.Column("prompt_eval_tokens_per_sec", sa.Numeric(), nullable=True),
        # Resource metrics
        sa.Column("peak_vram_mb", sa.Numeric(), nullable=True),
        sa.Column("avg_power_draw_w", sa.Numeric(), nullable=True),
        sa.Column("peak_power_draw_w", sa.Numeric(), nullable=True),
        sa.Column("avg_gpu_util_pct", sa.Numeric(), nullable=True),
        sa.Column("avg_gpu_temp_c", sa.Numeric(), nullable=True),
        sa.Column("total_wall_time_s", sa.Numeric(), nullable=True),
        # Versions
        sa.Column("engine_version", sa.Text(), nullable=True),
        # Host (local only)
        sa.Column("host_os", sa.Text(), nullable=True),
        sa.Column("host_kernel", sa.Text(), nullable=True),
        sa.Column("host_driver_version", sa.Text(), nullable=True),
        # Container
        sa.Column("container_image", sa.Text(), nullable=True),
        sa.Column("container_driver_version", sa.Text(), nullable=True),
        # Raw output
        sa.Column("raw_output", postgresql.JSONB(), nullable=True),
        # Moderation
        sa.Column("flagged", sa.Boolean(), nullable=False, server_default=sa.text("false")),
        # Check constraints
        sa.CheckConstraint("gpu_vram_gb > 0", name="ck_gpu_vram_positive"),
        sa.CheckConstraint("gpu_count >= 1 AND gpu_count <= 64", name="ck_gpu_count_range"),
        sa.CheckConstraint("provider IN ('local', 'vast.ai', 'runpod')", name="ck_provider_enum"),
        sa.CheckConstraint("engine IN ('llama.cpp', 'vllm')", name="ck_engine_enum"),
        sa.CheckConstraint("tokens_per_sec > 0", name="ck_tps_positive"),
        sa.CheckConstraint("time_to_first_token_ms >= 0", name="ck_ttft_non_negative"),
        sa.CheckConstraint("prompt_eval_tokens_per_sec IS NULL OR prompt_eval_tokens_per_sec >= 0", name="ck_prompt_eval_non_negative"),
        sa.CheckConstraint("peak_vram_mb IS NULL OR peak_vram_mb >= 0", name="ck_vram_non_negative"),
        sa.CheckConstraint("avg_power_draw_w IS NULL OR avg_power_draw_w >= 0", name="ck_avg_power_non_negative"),
        sa.CheckConstraint("peak_power_draw_w IS NULL OR peak_power_draw_w >= 0", name="ck_peak_power_non_negative"),
        sa.CheckConstraint("avg_gpu_util_pct IS NULL OR (avg_gpu_util_pct >= 0 AND avg_gpu_util_pct <= 100)", name="ck_gpu_util_range"),
        sa.CheckConstraint("avg_gpu_temp_c IS NULL OR (avg_gpu_temp_c >= 0 AND avg_gpu_temp_c <= 150)", name="ck_gpu_temp_range"),
        sa.CheckConstraint("total_wall_time_s IS NULL OR total_wall_time_s > 0", name="ck_wall_time_positive"),
        sa.CheckConstraint("peak_power_draw_w IS NULL OR avg_power_draw_w IS NULL OR peak_power_draw_w >= avg_power_draw_w", name="ck_peak_gte_avg_power"),
    )

    # Indexes
    op.create_index("ix_br_gpu_name", "benchmark_results", ["gpu_name"])
    op.create_index("ix_br_model_name", "benchmark_results", ["model_name"])
    op.create_index("ix_br_engine", "benchmark_results", ["engine"])
    op.create_index("ix_br_provider", "benchmark_results", ["provider"])
    op.create_index("ix_br_quantization", "benchmark_results", ["quantization"])
    op.create_index("ix_br_created_at", "benchmark_results", ["created_at"])
    op.create_index(
        "ix_br_flagged",
        "benchmark_results",
        ["flagged"],
        postgresql_where=sa.text("flagged = true"),
    )


def downgrade() -> None:
    op.drop_table("benchmark_results")
