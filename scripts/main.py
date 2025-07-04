"""Run the SkillWrapper algorithm with a Click CLI."""

import logging
from pathlib import Path

import click

from skillwrapper.refactored.skill_wrapper import SkillWrapper

main_logger = logging.getLogger(__name__)


@click.group()
@click.argument("domain_yaml", type=click.Path(exists=True, path_type=Path))
@click.argument("env_yaml", type=click.Path(exists=True, path_type=Path))
@click.option(
    "--logs-dir",
    type=click.Path(path_type=Path),
    default=Path("logs"),
    help="Directory for log files",
)
@click.pass_context
def cli(ctx: click.Context, domain_yaml: Path, env_yaml: Path, logs_dir: Path) -> None:
    """Define a command-line interface group for the SkillWrapper algorithm."""
    ctx.ensure_object(dict)  # Create ctx.obj if it doesn't exist
    ctx.obj["system"] = SkillWrapper(domain_yaml, env_yaml)


@cli.command()
@click.pass_context
def interactive(ctx: click.Context) -> None:
    """Run SkillWrapper in interactive mode with a menu-driven interface."""
    system: SkillWrapper = ctx.obj["system"]

    click.echo(click.style("\n🎮 SkillWrapper Interactive Mode", fg="green", bold=True))
    click.echo(
        click.style(f"Domain: {system.domain.name} | Environment: {system.env.name}", fg="cyan"),
    )

    while True:
        click.echo("\n" + "=" * 50)
        click.echo(click.style("Available Operations:", fg="cyan", bold=True))
        click.echo("1. Change environment")
        click.echo("2. Propose and execute exploratory skill sequence")
        click.echo("3. Invent predicates")
        click.echo("4. Learn operators")
        click.echo("5. Run complete loop")
        click.echo("6. Save progress")
        click.echo("7. Load progress")
        click.echo("8. Show status")
        click.echo("9. Quit")

        choice = click.prompt("\nSelect operation", type=click.IntRange(1, 9))

        try:
            if choice == 1:
                env_path = click.prompt(
                    "Enter environment YAML path",
                    type=click.Path(exists=True, path_type=Path),
                )
                system.change_environment(env_path)

            elif choice == 2:
                system.propose_and_execute()

            elif choice == 3:
                system.invent_predicates()

            elif choice == 4:
                system.learn_operators()

            elif choice == 5:
                system.run_complete_loop()

            elif choice == 6:
                system.save_progress()

            elif choice == 7:
                system.load_progress()

            elif choice == 8:
                system.print_status()

            elif choice == 9:
                if click.confirm("Save progress before exiting?"):
                    system.save_progress()
                click.echo(click.style("👋 Goodbye!", fg="green"))
                break

        except click.Abort:
            click.echo("\nOperation cancelled")
        except Exception as exc:
            click.echo(click.style(f"\n❌ Error: {exc}", fg="red"))
            main_logger.exception("Error in interactive mode")


@cli.command()
@click.option("--iterations", "-n", default=1, help="Number of complete loops to run")
@click.option("--save-after-each", is_flag=True, help="Save checkpoint after each loop")
@click.pass_context
def run(ctx: click.Context, iterations: int, save_after_each: bool) -> None:
    """Run complete loops of the SkillWrapper algorithm automatically.

    :param ctx: Click context providing access to the SkillWrapper system instance
    :param iterations: Number of complete loops to run
    :param save_after_each: Should progress be saved after each loop?
    """
    system: SkillWrapper = ctx.obj["system"]

    for i in range(iterations):
        click.echo(click.style(f"\n🔄 Loop {i + 1}/{iterations}", fg="blue", bold=True))
        system.run_complete_loop()

        if save_after_each:
            system.save_progress()

        # Don't prompt for continuation after the last iteration
        if (i < iterations - 1) and not click.confirm("Continue to next loop?", default=True):
            break

    system.print_status()

    if click.confirm("Save final progress?"):
        system.save_progress()


@cli.command()
@click.pass_context
def propose(ctx: click.Context) -> None:
    """Propose and execute a single exploratory sequence of skills."""
    ctx.obj["system"].propose_and_execute()


@cli.command()
@click.pass_context
def learn(ctx: click.Context) -> None:
    """Invent predicates and learn updated operators based on the current dataset."""
    ctx.obj["system"].invent_predicates()
    ctx.obj["system"].learn_operators()


@cli.command()
@click.pass_context
def status(ctx: click.Context) -> None:
    """Show the current system status."""
    ctx.obj["system"].print_status()


if __name__ == "__main__":
    cli(obj={})
