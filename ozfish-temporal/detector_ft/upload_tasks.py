"""Create one CVAT task per cvat_tasks/<name>/ (images.zip + pre-label annotations.xml) in a
`detector-ft` project with the single label `fish` (made if missing). Resumable: existing task
names are skipped. Login from env CVAT_USER / CVAT_PASS or asked for, never stored.

    python upload_tasks.py                 # all 12
    python upload_tasks.py test_000 test_001 train_000
"""
import argparse
import getpass
import os
from pathlib import Path

HERE = Path(__file__).resolve().parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("names", nargs="*", help="task folders to upload (default: all)")
    ap.add_argument("--tasks", default=str(HERE / "cvat_tasks"))
    ap.add_argument("--host", default="http://localhost:8080")
    ap.add_argument("--project", default="detector-ft")
    ap.add_argument("--replace", action="store_true",
                    help="delete same-named tasks in the project first (their labels are lost)")
    args = ap.parse_args()

    dirs = [Path(args.tasks) / n for n in args.names] or sorted(p for p in Path(args.tasks).iterdir() if p.is_dir())
    from cvat_sdk import make_client
    from cvat_sdk.core.proxies.tasks import ResourceType

    creds = (os.environ.get("CVAT_USER") or input("CVAT username: "),
             os.environ.get("CVAT_PASS") or getpass.getpass("CVAT password: "))
    with make_client(args.host, credentials=creds) as client:
        proj = next((p for p in client.projects.list() if p.name == args.project), None)
        if proj is None:
            proj = client.projects.create(spec={"name": args.project,
                                                "labels": [{"name": "fish", "type": "rectangle"}]})
            print(f"created project {proj.id}: {args.project}")
        done = {t.name for t in client.tasks.list()}
        if args.replace:  # delete this project's tasks with the same names, then recreate them
            for t in client.tasks.list():
                if t.project_id == proj.id and t.name in {d.name for d in dirs}:
                    t.remove()
                    done.discard(t.name)
                    print(f"deleted old task {t.id}: {t.name}")
        for d in dirs:
            if d.name in done:
                print(f"skip {d.name} (exists)")
                continue
            task = client.tasks.create_from_data(
                spec={"name": d.name, "project_id": proj.id},
                resource_type=ResourceType.LOCAL,
                resources=[str(d / "images.zip")],
                annotation_path=str(d / "annotations.xml"),
                annotation_format="CVAT 1.1",
            )
            print(f"task {task.id}: {d.name} ({task.size} frames)")


if __name__ == "__main__":
    main()
