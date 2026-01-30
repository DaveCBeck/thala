"""Task management commands."""
import json
import sys

from ..queue_manager import TaskCategory, TaskPriority, TaskQueueManager, TaskStatus
from ..workflows import DEFAULT_WORKFLOW_TYPE, get_available_types


def cmd_add(args):
    """Add task to queue."""
    manager = TaskQueueManager()

    # Determine task type
    task_type = getattr(args, "type", None) or DEFAULT_WORKFLOW_TYPE

    # Validate task type
    available_types = get_available_types()
    if task_type not in available_types:
        print(f"Invalid task type: {task_type}")
        print(f"Valid: {', '.join(available_types)}")
        sys.exit(1)

    # Parse category
    try:
        # Try enum first, then as string
        categories = manager.get_categories()
        if args.category.upper() in [c.upper() for c in categories]:
            category = args.category.lower()
        elif args.category.upper() in [c.name for c in TaskCategory]:
            category = TaskCategory[args.category.upper()].value
        else:
            category = args.category.lower()  # Accept any string
    except Exception:
        category = args.category.lower()

    # Parse priority
    try:
        priority = TaskPriority[args.priority.upper()]
    except KeyError:
        print(f"Invalid priority: {args.priority}")
        print(f"Valid: {', '.join(p.name.lower() for p in TaskPriority)}")
        sys.exit(1)

    # Parse tags
    tags = None
    if args.tags:
        tags = [t.strip() for t in args.tags.split(",") if t.strip()]

    # Build task based on type
    if task_type == "lit_review_full":
        # Parse research questions (lit_review only)
        questions = None
        if args.questions:
            questions = [q.strip() for q in args.questions.split("|") if q.strip()]

        # Parse date range (lit_review only)
        date_range = None
        if args.from_year or args.to_year:
            date_range = (args.from_year or 2000, args.to_year or 2026)

        task_id = manager.add_task(
            task_type=task_type,
            topic=args.topic,
            category=category,
            priority=priority,
            research_questions=questions,
            quality=args.quality,
            language=args.language,
            date_range=date_range,
            notes=args.notes,
            tags=tags,
        )
        identifier = args.topic

    elif task_type == "web_research":
        task_id = manager.add_task(
            task_type=task_type,
            query=args.topic,  # Reuse positional arg as query
            category=category,
            priority=priority,
            quality=args.quality,
            language=args.language if args.language != "en" else None,
            notes=args.notes,
            tags=tags,
        )
        identifier = args.topic

    else:
        # Generic fallback
        task_id = manager.add_task(
            task_type=task_type,
            topic=args.topic,
            category=category,
            priority=priority,
            quality=args.quality,
            notes=args.notes,
            tags=tags,
        )
        identifier = args.topic

    print(f"Added task: {task_id}")
    print(f"  Type: {task_type}")
    print(f"  {'Query' if task_type == 'web_research' else 'Topic'}: {identifier[:60]}{'...' if len(identifier) > 60 else ''}")
    print(f"  Category: {category}")
    print(f"  Priority: {priority.name}")
    print(f"  Quality: {args.quality}")


def cmd_list(args):
    """List tasks in queue."""
    manager = TaskQueueManager()

    status_filter = None
    if args.status:
        try:
            status_filter = TaskStatus[args.status.upper()]
        except KeyError:
            print(f"Invalid status: {args.status}")
            print(f"Valid: {', '.join(s.name.lower() for s in TaskStatus)}")
            sys.exit(1)

    category_filter = None
    if args.category:
        category_filter = args.category.lower()

    tasks = manager.list_tasks(status=status_filter, category=category_filter)

    if not tasks:
        print("Queue is empty")
        return

    # Group by status
    by_status: dict[str, list] = {}
    for task in tasks:
        status = task["status"]
        if status not in by_status:
            by_status[status] = []
        by_status[status].append(task)

    # Print in status order
    status_order = ["in_progress", "pending", "paused", "completed", "failed"]
    for status in status_order:
        if status not in by_status:
            continue

        group = by_status[status]
        print(f"\n=== {status.upper()} ({len(group)}) ===")

        for task in group:
            try:
                priority = TaskPriority(task["priority"]).name
            except ValueError:
                priority = str(task["priority"])

            task_type = task.get("task_type", DEFAULT_WORKFLOW_TYPE)
            task_identifier = task.get("topic") or task.get("query", "unknown")
            type_badge = f"[{task_type}]" if task_type != DEFAULT_WORKFLOW_TYPE else ""

            print(f"  [{task['id'][:8]}] {type_badge} {task_identifier[:50]}...")
            print(f"           Category: {task['category']}, Priority: {priority}")

            if task.get("started_at"):
                print(f"           Started: {task['started_at'][:19]}")
            if task.get("current_phase"):
                print(f"           Phase: {task['current_phase']}")
            if task.get("error_message"):
                print(f"           Error: {task['error_message'][:50]}...")


def cmd_reorder(args):
    """Reorder/export queue for LLM editing."""
    manager = TaskQueueManager()

    pending = manager.list_tasks(status=TaskStatus.PENDING)

    if not pending:
        print("No pending tasks to reorder")
        return

    # Export current order in LLM-friendly format
    if args.export:
        data = {
            "instructions": "Reorder tasks by editing the 'order' list. Highest priority first. "
                          "You can also edit priority values (1=low, 2=normal, 3=high, 4=urgent).",
            "tasks": [
                {
                    "id": t["id"],
                    "topic": t["topic"],
                    "category": t["category"],
                    "priority": TaskPriority(t["priority"]).name,
                    "notes": t.get("notes"),
                }
                for t in pending
            ],
            "order": [t["id"] for t in pending],
        }
        print(json.dumps(data, indent=2))
        return

    # Import new order
    if args.input:
        with open(args.input, "r") as f:
            data = json.load(f)

        new_order = data.get("order", [])
        manager.reorder(new_order)
        print(f"Reordered {len(new_order)} tasks")
        return

    # No action specified - show current order
    print("Current task order (pending only):")
    for i, task in enumerate(pending, 1):
        priority = TaskPriority(task["priority"]).name
        print(f"  {i}. [{task['id'][:8]}] {task['topic'][:40]}... ({priority})")

    print("\nUse --export to export for editing, --input FILE to import new order")


def cmd_config(args):
    """Configure concurrency settings."""
    manager = TaskQueueManager()

    if args.mode:
        manager.set_concurrency(
            mode=args.mode,
            max_concurrent=args.max_concurrent or 1,
            stagger_hours=args.stagger_hours or 36.0,
        )
        print(f"Set concurrency mode: {args.mode}")
        if args.mode == "stagger_hours":
            print(f"  Stagger: {args.stagger_hours or 36.0} hours")
        else:
            print(f"  Max concurrent: {args.max_concurrent or 1}")
    else:
        # Show current config
        config = manager.get_concurrency_config()
        print("Current concurrency config:")
        print(f"  Mode: {config['mode']}")
        print(f"  Max concurrent: {config['max_concurrent']}")
        print(f"  Stagger hours: {config['stagger_hours']}")
