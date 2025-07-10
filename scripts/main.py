"""Run the SkillWrapper algorithm with a Click-based CLI."""

from pathlib import Path

import click
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from skillwrapper.refactored.skill_wrapper import SkillWrapper


@click.group()
@click.argument("domain_yaml", type=click.Path(exists=True, path_type=Path))
@click.argument("env_yaml", type=click.Path(exists=True, path_type=Path))
@click.pass_context
def skillwrapper_cli(ctx: click.Context, domain_yaml: Path, env_yaml: Path) -> None:
    """Define a command-line interface for the SkillWrapper algorithm."""
    ctx.ensure_object(dict)  # Create ctx.obj if it doesn't exist
    ctx.obj["system"] = SkillWrapper(domain_yaml, env_yaml)
    ctx.obj["console"] = Console()


@skillwrapper_cli.command()
@click.pass_context
def interactive(ctx: click.Context) -> None:
    """Run SkillWrapper in an interactive loop with a menu-driven interface."""
    system: SkillWrapper = ctx.obj["system"]
    console: Console = ctx.obj["console"]

    header = Text("🎮 SkillWrapper Interactive Mode", style="bold green")
    info = Text(f"Domain: {system.domain.name} | Environment: {system.env.name}", style="cyan")
    header_content = Text.assemble(header, "\n", info)
    console.print(Panel(header_content, border_style="green"))

    menu_table = Table(title="Available Operations", border_style="cyan", title_style="bold cyan")
    menu_table.add_column("Option", style="bold", width=8)
    menu_table.add_column("Description", style="white")

    menu_items = [
        ("1", "Change environment"),
        ("2", "Propose and execute exploratory skill sequence"),
        ("3", "Invent predicates"),
        ("4", "Learn operators"),
        ("5", "Run complete loop"),
        ("6", "Save progress"),
        ("7", "Load progress"),
        ("8", "Show status"),
        ("9", "Quit"),
    ]

    for option, description in menu_items:
        menu_table.add_row(option, description)

    while True:
        console.print()
        console.print(menu_table)

        choice = click.prompt("\nSelect operation", type=click.IntRange(1, 9))

        try:
            if choice == 1:
                env_path = click.prompt(
                    "Enter environment YAML path",
                    type=click.Path(exists=True, path_type=Path),
                )
                system.change_environment(env_path)
                console.print("[green]✓[/green] Environment changed successfully")

            elif choice == 2:
                console.print(
                    "[yellow]🔍 Proposing and executing exploratory skill sequence...[/yellow]",
                )
                system.propose_and_execute_skills()
                console.print("[green]✓[/green] Skill proposal and execution completed")

            elif choice == 3:
                console.print("[yellow]🧠 Inventing predicates...[/yellow]")
                system.invent_predicates()
                console.print("[green]✓[/green] Predicate invention completed")

            elif choice == 4:
                console.print("[yellow]📚 Learning operators...[/yellow]")
                system.learn_operators()
                console.print("[green]✓[/green] Operator learning completed")

            elif choice == 5:
                console.print("[yellow]🔄 Running complete loop...[/yellow]")
                system.run_complete_loop()
                console.print("[green]✓[/green] Complete loop finished")

            elif choice == 6:
                console.print("[yellow]💾 Saving progress...[/yellow]")
                system.save_progress()
                console.print("[green]✓[/green] Progress saved successfully")

            elif choice == 7:
                console.print("[yellow]📂 Loading progress...[/yellow]")
                system.load_progress()
                console.print("[green]✓[/green] Progress loaded successfully")

            elif choice == 8:
                console.print("[yellow]📊 System Status:[/yellow]")
                system.print_status()

            elif choice == 9:
                if click.confirm("Save progress before exiting?", default=None):
                    console.print("[yellow]💾 Saving progress...[/yellow]")
                    system.save_progress()

                goodbye_panel = Panel(
                    Text("👋 Goodbye!", style="bold green", justify="center"),
                    border_style="green",
                )
                console.print(goodbye_panel)
                break

        except click.Abort:
            console.print("[yellow]⚠️ Operation canceled[/yellow]")

        except Exception as exc:
            error_panel = Panel(
                Text(f"❌ Error: {exc}", style="bold red"),
                title="Error",
                border_style="red",
            )
            console.print(error_panel)


if __name__ == "__main__":
    skillwrapper_cli()
