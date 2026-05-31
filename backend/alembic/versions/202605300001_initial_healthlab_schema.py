"""initial healthlab schema

Revision ID: 202605300001
Revises:
Create Date: 2026-05-30 17:15:00
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

revision: str = "202605300001"
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "reports",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("patient_name", sa.String(length=120), nullable=False),
        sa.Column("age", sa.Integer(), nullable=False),
        sa.Column("gender", sa.String(length=32), nullable=True),
        sa.Column("phone", sa.String(length=32), nullable=True),
        sa.Column("email", sa.String(length=255), nullable=True),
        sa.Column("city", sa.String(length=80), nullable=True),
        sa.Column("lab_branch", sa.String(length=120), nullable=True),
        sa.Column("test_type", sa.String(length=120), nullable=False),
        sa.Column("doctor_name", sa.String(length=120), nullable=True),
        sa.Column("status", sa.String(length=32), nullable=False),
        sa.Column("priority", sa.String(length=24), nullable=False),
        sa.Column("sample_collected_at", sa.DateTime(), nullable=True),
        sa.Column("result_due_at", sa.DateTime(), nullable=True),
        sa.Column("notes", sa.Text(), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(),
            server_default=sa.text("CURRENT_TIMESTAMP"),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(),
            server_default=sa.text("CURRENT_TIMESTAMP"),
            nullable=False,
        ),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(op.f("ix_reports_city"), "reports", ["city"], unique=False)
    op.create_index(op.f("ix_reports_id"), "reports", ["id"], unique=False)
    op.create_index(op.f("ix_reports_lab_branch"), "reports", ["lab_branch"], unique=False)
    op.create_index(op.f("ix_reports_patient_name"), "reports", ["patient_name"], unique=False)
    op.create_index(op.f("ix_reports_priority"), "reports", ["priority"], unique=False)
    op.create_index(op.f("ix_reports_result_due_at"), "reports", ["result_due_at"], unique=False)
    op.create_index(op.f("ix_reports_status"), "reports", ["status"], unique=False)
    op.create_index(op.f("ix_reports_test_type"), "reports", ["test_type"], unique=False)


def downgrade() -> None:
    op.drop_index(op.f("ix_reports_test_type"), table_name="reports")
    op.drop_index(op.f("ix_reports_status"), table_name="reports")
    op.drop_index(op.f("ix_reports_result_due_at"), table_name="reports")
    op.drop_index(op.f("ix_reports_priority"), table_name="reports")
    op.drop_index(op.f("ix_reports_patient_name"), table_name="reports")
    op.drop_index(op.f("ix_reports_lab_branch"), table_name="reports")
    op.drop_index(op.f("ix_reports_id"), table_name="reports")
    op.drop_index(op.f("ix_reports_city"), table_name="reports")
    op.drop_table("reports")
