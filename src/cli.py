import sys
import os
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.rule import Rule

from ingest import clone_repository, build_project_tree
from embeddings import load_documents, chunk_documents, save_to_chroma
from analyzer import get_readme_content, generate_first_menu

console = Console()


def welcome():
    console.print(Panel("🐙 [bold]GitHub AI Guide[/bold]", expand=False))
    console.print()


def run():
    welcome()

    repo_url = Prompt.ask("[dim]Repository URL[/dim]")
    console.print()

    with Progress(SpinnerColumn(), TextColumn("{task.description}"), console=console) as progress:
        t = progress.add_task("Cloning repository...", total=None)
        repo_path = clone_repository(repo_url)
        file_list = build_project_tree(repo_path)

        progress.update(t, description="Indexing code...")
        docs = load_documents(repo_path, file_list)
        chunks = chunk_documents(docs)
        save_to_chroma(chunks)

        progress.update(t, description="Analyzing project...")
        readme = get_readme_content(repo_path, file_list)
        result = generate_first_menu(file_list, readme)

    console.print(Rule())
    console.print()
    console.print(f"[bold]📄 Summary[/bold]")
    console.print(f"[dim]{result.summary}[/dim]")
    console.print()
    console.print("[bold]Where do you want to start?[/bold]")
    console.print()

    for i, option in enumerate(result.options, 1):
        console.print(f"  [bold cyan]{i}[/bold cyan]. {option}")

    console.print()
    choice = Prompt.ask("[dim]Choose[/dim]", choices=["1", "2", "3"])
    selected = result.options[int(choice) - 1]

    console.print()
    console.print(f"[bold green]→[/bold green] {selected}")
    console.print()


if __name__ == "__main__":
    run()