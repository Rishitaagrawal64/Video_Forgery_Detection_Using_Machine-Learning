from __future__ import annotations

import datetime as dt
import html
import io
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.platypus import (
    Image,
    KeepTogether,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

COLLEGE_NAME = "Swami Keshvanand Institute of Technology, Management & Gramothan"
DEPARTMENT_NAME = "Department of Computer Science & Engineering"
OUTPUT_DIR = Path("weekly_reports")

# If the same student commits under different names (e.g. from different laptops or
# the GitHub web editor), map every variant to ONE display name here.
# Keys are matched case-insensitively. Check the "Authors found" lines in the
# workflow log to see which name/email variants exist.
# Example: {"dekxmax": "Dekx Max", "dekx max": "Dekx Max", "dekxmax@gmail.com": "Dekx Max"}
AUTHOR_ALIASES: dict[str, str] = {}


def run_git(args: list[str]) -> str:
    return subprocess.check_output(
        ["git", *args], text=True, encoding="utf-8", errors="replace"
    ).strip()


def get_repo_info() -> tuple[str, str]:
    try:
        root = run_git(["rev-parse", "--show-toplevel"])
        repo_name = Path(root).name
    except (subprocess.CalledProcessError, OSError):
        repo_name = Path.cwd().name

    try:
        branch = run_git(["rev-parse", "--abbrev-ref", "HEAD"])
    except (subprocess.CalledProcessError, OSError):
        branch = "unknown"
    return repo_name, branch


_name_registry: dict[str, str] = {}   # lowercase name  -> display name
_email_registry: dict[str, str] = {}  # lowercase email -> display name


def canonical_author(name: str, email: str) -> str:
    """Merge one person's identities: AUTHOR_ALIASES first, then same email, then same name ignoring case/spacing."""
    name = " ".join(name.split())
    email_key = email.strip().lower()
    aliases = {k.strip().lower(): v for k, v in AUTHOR_ALIASES.items()}

    for key in (name.lower(), email_key):
        if key in aliases:
            display = aliases[key]
            _name_registry[name.lower()] = display
            _email_registry[email_key] = display
            return display

    display = _email_registry.get(email_key) or _name_registry.get(name.lower()) or name
    _name_registry.setdefault(name.lower(), display)
    _email_registry.setdefault(email_key, display)
    return display


def is_bot(author: str) -> bool:
    """Skip only real bot accounts (e.g. 'github-actions[bot]'), not names like 'Abbott'."""
    name = author.lower()
    return name.endswith("[bot]") or "github-actions" in name


def get_git_metrics(interval: str = "weekly"):
    today = dt.date.today()
    if interval == "weekly":
        since = (today - dt.timedelta(days=7)).isoformat()
        scope_title = f"Last 7 Days (since {since})"
        date_args = [f"--since={since}"]
    elif interval == "monthly":
        since = (today - dt.timedelta(days=30)).isoformat()
        scope_title = f"Last 30 Days (since {since})"
        date_args = [f"--since={since}"]
    elif interval == "final":
        scope_title = "Complete Project Lifecycle (all commits)"
        date_args = []
    else:
        raise ValueError("Interval must be weekly, monthly, or final.")

    log_format = "COMMIT|||%h|||%aN|||%aE|||%ad|||%s"  # %aN/%aE honour .mailmap
    raw = run_git([
        "log", "--no-merges", f"--pretty=format:{log_format}",
        "--date=short", "--numstat", *date_args,
    ])

    students = defaultdict(
        lambda: {"commits": 0, "added": 0, "deleted": 0, "active_days": set()}
    )
    timeline = defaultdict(lambda: defaultdict(int))
    student_logs = defaultdict(list)
    seen_identities: set[str] = set()
    current_author = None

    for raw_line in raw.splitlines():
        line = raw_line.strip()
        if not line:
            continue

        if line.startswith("COMMIT|||"):
            parts = line.split("|||", 5)
            if len(parts) < 6:
                current_author = None
                continue
            _, sha, raw_author, email, date_str, message = parts
            if is_bot(raw_author):
                current_author = None
                continue

            author = canonical_author(raw_author, email)
            seen_identities.add(f"{raw_author} <{email}> -> {author}")
            current_author = author
            data = students[author]
            data["commits"] += 1
            data["active_days"].add(date_str)
            student_logs[author].append((date_str, sha, message))

            try:
                commit_date = dt.datetime.strptime(date_str, "%Y-%m-%d").date()
                # Keys are chronologically sortable; pretty labels are made in make_chart().
                if interval == "weekly":
                    period_key = commit_date.isoformat()  # e.g. 2026-10-22
                elif interval == "monthly":
                    iso = commit_date.isocalendar()
                    period_key = f"{iso.year}-W{iso.week:02d}"
                else:
                    period_key = commit_date.strftime("%Y-%m")
                timeline[period_key][author] += 1
            except ValueError:
                pass
        elif current_author:
            parts = line.split()
            if len(parts) >= 3 and parts[0].isdigit() and parts[1].isdigit():
                students[current_author]["added"] += int(parts[0])
                students[current_author]["deleted"] += int(parts[1])

    for identity in sorted(seen_identities):
        print(f"Authors found: {identity}")

    return students, timeline, student_logs, scope_title


def pretty_label(period_key: str, interval: str) -> str:
    if interval == "weekly":
        try:
            return dt.date.fromisoformat(period_key).strftime("%a (%b %d)")
        except ValueError:
            return period_key
    return period_key


def make_chart(students, timeline, interval: str) -> io.BytesIO:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 3.5))
    authors = list(students.keys())
    periods = sorted(timeline.keys())  # chronological thanks to sortable keys
    labels = [pretty_label(p, interval) for p in periods]

    if periods and authors:
        for author in authors:
            values = [timeline[p].get(author, 0) for p in periods]
            ax1.plot(labels, values, marker="o", linewidth=1.8, label=author)
        ax1.set_title(f"Commit Timeline ({interval.title()})")
        ax1.set_ylabel("Commits")
        ax1.tick_params(axis="x", rotation=35)
        ax1.grid(True, linestyle="--", alpha=0.4)
        ax1.legend(fontsize=7)
    else:
        ax1.text(0.5, 0.5, "No commits in this period", ha="center", va="center")
        ax1.set_axis_off()

    if authors:
        net = [students[a]["added"] - students[a]["deleted"] for a in authors]
        ax2.bar(authors, net)
        ax2.set_title("Net Lines of Code")
        ax2.set_ylabel("Added minus deleted")
        ax2.tick_params(axis="x", rotation=25)
        ax2.grid(axis="y", linestyle="--", alpha=0.4)
    else:
        ax2.text(0.5, 0.5, "No code changes recorded", ha="center", va="center")
        ax2.set_axis_off()

    fig.tight_layout()
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", dpi=180, bbox_inches="tight")
    plt.close(fig)
    buffer.seek(0)
    return buffer


def generate_pdf(interval: str = "weekly") -> Path:
    repo_name, branch_name = get_repo_info()
    students, timeline, student_logs, scope_title = get_git_metrics(interval)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    date_stamp = dt.date.today().isoformat()
    label = {"weekly": "Weekly_Progress_Report_Form-3",
             "monthly": "Monthly_Progress_Report_Form-3",
             "final": "Final_Project_Report"}[interval]
    output_path = OUTPUT_DIR / f"{repo_name}_{label}_{date_stamp}.pdf"

    doc = SimpleDocTemplate(
        str(output_path), pagesize=letter,
        rightMargin=30, leftMargin=30, topMargin=28, bottomMargin=28,
    )
    styles = getSampleStyleSheet()
    college_style = ParagraphStyle(
        "College", parent=styles["Heading1"], fontSize=13, leading=16,
        alignment=1, textColor=colors.HexColor("#0F172A"), spaceAfter=2,
    )
    dept_style = ParagraphStyle(
        "Department", parent=styles["Normal"], fontSize=9, alignment=1,
        textColor=colors.HexColor("#475569"), spaceAfter=5,
    )
    section_style = ParagraphStyle(
        "Section", parent=styles["Heading2"], fontSize=10.5, leading=13,
        textColor=colors.HexColor("#0F172A"), spaceBefore=7, spaceAfter=4,
    )
    small_style = ParagraphStyle(
        "Small", parent=styles["Normal"], fontSize=7.5, leading=9,
    )
    centered_style = ParagraphStyle(
        "Centered", parent=styles["Normal"], fontSize=8, leading=10,
        alignment=1,
    )

    report_title = {
        "weekly": "Weekly Progress Report (Form-3)",
        "monthly": "Monthly Progress Report (Form-3)",
        "final": "Final Project Evaluation Report",
    }[interval]

    story = [
        Paragraph(f"<b>{html.escape(COLLEGE_NAME)}</b>", college_style),
        Paragraph(html.escape(DEPARTMENT_NAME), dept_style),
        Paragraph(f"<u><b>{html.escape(report_title)}</b></u>", styles["Heading2"]),
        Spacer(1, 5),
        Paragraph(
            f"<b>Repository:</b> {html.escape(repo_name)} &nbsp; | &nbsp; "
            f"<b>Branch:</b> {html.escape(branch_name)}",
            styles["Normal"],
        ),
        Paragraph(
            f"<b>Evaluation window:</b> {html.escape(scope_title)} &nbsp; | &nbsp; "
            f"<b>Generated:</b> {dt.date.today().strftime('%B %d, %Y')}",
            styles["Normal"],
        ),
        Spacer(1, 8),
        Paragraph("1. Individual Contribution Breakdown", section_style),
    ]

    total_commits = sum(row["commits"] for row in students.values())
    table_data = [[
        "Student Name", "Commits (%)", "Lines Added", "Lines Deleted",
        "Net LOC", "Active Days",
    ]]
    if students:
        for name, data in sorted(students.items()):
            pct = 100 * data["commits"] / total_commits if total_commits else 0
            table_data.append([
                Paragraph(html.escape(name), small_style),
                f'{data["commits"]} ({pct:.1f}%)',
                f'+{data["added"]:,}', f'-{data["deleted"]:,}',
                f'{data["added"] - data["deleted"]:,}',
                str(len(data["active_days"])),
            ])
    else:
        table_data.append(["No commits found for this period.", "-", "-", "-", "-", "-"])

    summary_table = Table(
        table_data, repeatRows=1,
        colWidths=[145, 75, 75, 75, 65, 70],
    )
    summary_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1E293B")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 7.5),
        ("ALIGN", (1, 1), (-1, -1), "CENTER"),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#CBD5E1")),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F8FAFC")]),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ]))
    story.extend([summary_table, Spacer(1, 8),
                  Paragraph("2. Visual Trends & Volume", section_style)])
    chart_buffer = make_chart(students, timeline, interval)
    story.append(Image(chart_buffer, width=520, height=165))
    story.extend([Spacer(1, 5),
                  Paragraph("3. Detailed Commit Logs & Mentor Evaluation", section_style)])

    if not student_logs:
        story.append(Paragraph("No commit logs found for this timeframe.", styles["Normal"]))
    else:
        for student_name, logs in sorted(student_logs.items()):
            section = [Paragraph(
                f"<b>Student:</b> {html.escape(student_name)} — {len(logs)} commit(s)",
                styles["Heading3"],
            )]
            log_data = [["Date", "Hash", "Commit Message", "Mentor Marks (/10)"]]
            for index, (date_value, sha, message) in enumerate(logs):
                log_data.append([
                    Paragraph(html.escape(date_value), centered_style),
                    Paragraph(html.escape(sha), centered_style),
                    Paragraph(html.escape(message or "(No commit message)"), small_style),
                    Paragraph("<b>_____ / 10</b>", centered_style) if index == 0 else "",
                ])
            log_table = Table(log_data, colWidths=[65, 55, 330, 85], repeatRows=1)
            style = [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#475569")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, -1), 7.5),
                ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#CBD5E1")),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("TOPPADDING", (0, 0), (-1, -1), 3),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
                ("BACKGROUND", (3, 1), (3, -1), colors.HexColor("#FEF3C7")),
            ]
            if len(log_data) > 2:
                style.append(("SPAN", (3, 1), (3, len(log_data) - 1)))
            log_table.setStyle(TableStyle(style))
            section.extend([log_table, Spacer(1, 6)])
            story.append(KeepTogether(section))

    story.extend([
        Spacer(1, 18),
        Table(
            [[
                Paragraph("<b>Name:</b> ______________________<br/><b>Designation:</b> Project Mentor<br/><br/><b>Signature:</b> __________________", styles["Normal"]),
                Paragraph("<b>Name:</b> ______________________<br/><b>Designation:</b> Lab Coordinator<br/><br/><b>Signature:</b> __________________", styles["Normal"]),
            ]],
            colWidths=[270, 270],
            style=TableStyle([("VALIGN", (0, 0), (-1, -1), "TOP")]),
        ),
    ])

    doc.build(story)
    print(f"[SUCCESS] Generated: {output_path}")
    print(f"Found {len(students)} contributor(s) and {total_commits} commits.")
    return output_path


if __name__ == "__main__":
    selected_interval = sys.argv[1].lower() if len(sys.argv) > 1 else "weekly"
    generate_pdf(selected_interval)