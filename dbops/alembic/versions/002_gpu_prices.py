"""Add gpu_prices table.

Revision ID: 002
Revises: 001
Create Date: 2026-04-14
"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "002"
down_revision: Union[str, None] = "001"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "gpu_prices",
        # Identity
        sa.Column(
            "id",
            postgresql.UUID(as_uuid=True),
            primary_key=True,
            server_default=sa.text("gen_random_uuid()"),
        ),
        sa.Column(
            "collected_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.func.now(),
        ),
        # GPU identity
        sa.Column("gpu_name", sa.Text(), nullable=False),
        sa.Column("gpu_vram_gb", sa.Numeric(), nullable=False),
        # Source + price
        sa.Column("source", sa.String(32), nullable=False),
        sa.Column("unit", sa.String(16), nullable=False),
        sa.Column("price_usd", sa.Numeric(), nullable=False),
        # Optional listing details (mostly for ebay/amazon manual rows)
        sa.Column("listing_url", sa.Text(), nullable=True),
        sa.Column("seller", sa.Text(), nullable=True),
        sa.Column("notes", sa.Text(), nullable=True),
        sa.Column("raw_metadata", postgresql.JSONB(), nullable=True),
        # Constraints
        sa.CheckConstraint("price_usd > 0", name="ck_price_usd_positive"),
        sa.CheckConstraint("gpu_vram_gb > 0", name="ck_gp_gpu_vram_positive"),
        sa.CheckConstraint(
            "source IN ('ebay', 'amazon', 'vast', 'vast (community)', 'runpod')",
            name="ck_gp_source_enum",
        ),
        sa.CheckConstraint(
            "unit IN ('one_time', 'per_hour')",
            name="ck_gp_unit_enum",
        ),
        sa.CheckConstraint(
            "(source IN ('ebay', 'amazon') AND unit = 'one_time') "
            "OR (source IN ('vast', 'vast (community)', 'runpod') AND unit = 'per_hour')",
            name="ck_gp_source_unit_consistency",
        ),
    )

    # Indexes
    op.create_index("ix_gp_gpu_name", "gpu_prices", ["gpu_name"])
    op.create_index("ix_gp_source", "gpu_prices", ["source"])
    op.create_index("ix_gp_collected_at", "gpu_prices", ["collected_at"])
    op.create_index(
        "ix_gp_gpu_source_collected",
        "gpu_prices",
        ["gpu_name", "source", "collected_at"],
    )


def downgrade() -> None:
    op.drop_index("ix_gp_gpu_source_collected", table_name="gpu_prices")
    op.drop_index("ix_gp_collected_at", table_name="gpu_prices")
    op.drop_index("ix_gp_source", table_name="gpu_prices")
    op.drop_index("ix_gp_gpu_name", table_name="gpu_prices")
    op.drop_table("gpu_prices")
