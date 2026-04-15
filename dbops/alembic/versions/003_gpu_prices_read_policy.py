"""Enable RLS + public read policy on gpu_prices.

The table was created in 002 with Supabase's default RLS-on-without-policies,
which blocks the anon/publishable key from reading anything. The leaderboard
frontend (results-disp) reads the table via the publishable key, so we add a
SELECT-only policy that mirrors the one benchmark_results already has.

Revision ID: 003
Revises: 002
Create Date: 2026-04-15
"""

from typing import Sequence, Union

from alembic import op

revision: str = "003"
down_revision: Union[str, None] = "002"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.execute("ALTER TABLE gpu_prices ENABLE ROW LEVEL SECURITY")
    op.execute(
        "CREATE POLICY read_public ON gpu_prices FOR SELECT USING (true)"
    )


def downgrade() -> None:
    op.execute("DROP POLICY IF EXISTS read_public ON gpu_prices")
    # Leave RLS enabled on downgrade — turning it off would open writes on a
    # table that only the service role should mutate.
