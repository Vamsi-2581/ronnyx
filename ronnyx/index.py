import argparse
import asyncio
from pathlib import Path

from ronnyx import knowledge
from ronnyx.config import load_config


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build the Ronnyx knowledge base from your documents.",
    )
    parser.add_argument(
        "--init",
        action="store_true",
        help="Create the knowledge folder and GraphRAG settings, then exit.",
    )
    parser.add_argument(
        "--yes",
        action="store_true",
        help="Skip the cost confirmation before indexing.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = load_config()

    if not config.knowledge:
        print("[error] no 'knowledge' section in ronnyx.yaml")
        return 1

    root = Path(config.knowledge["root"])

    try:
        if args.init:
            knowledge.init_project(root, model=config.llm_model)
            print(
                f"Created {root}. Put your documents in {root / 'input'}, then run ronnyx-index."
            )
            return 0

        if not (root / "settings.yaml").exists():
            print(f"[error] {root} is not initialized. Run: ronnyx-index --init")
            return 1

        if not args.yes:
            print(
                "Indexing calls the LLM for every chunk of every document and can be expensive."
            )
            if input("Continue? [y/N] ").strip().lower() != "y":
                print("Aborted.")
                return 1

        ok = asyncio.run(knowledge.build_index(root, config.llm_api_key))
    except ImportError:
        print(f"[error] {knowledge.INSTALL_HINT}")
        return 1
    except ValueError as exc:
        print(f"[error] {exc}")
        return 1

    if not ok:
        print(
            f"[error] indexing failed. Details are in {root / 'logs' / 'indexing-engine.log'}"
        )
        return 1
    print("Index built. Restart ronnyx-serve to load it.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
