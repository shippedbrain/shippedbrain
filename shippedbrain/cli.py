import click
import os
import shippedbrain

@click.group()
@click.version_option(version="0.1", help="Get current installed shippedbrain version in system")
@click.help_option(help="Get help on how to use shippedbrain")
def cli():
    """Shipped Brain cli - Upload your models to app.shippedbrain.com"""
    pass

@cli.command("upload")
@click.option("--run_id", "-r", help="The run_id of logged mlflow model", type=str, required=True)
@click.option("--model_name", "-m", help="The model name to display in app.shippedbrain.com", type=str, required=True)
@click.option("--flavor", "-f", help="The mlflow flow flavor of the model", default="pyfunc", type=str)
@click.help_option(help="Get help on how to use the 'upload' command")
def upload(run_id: str, model_name: str, flavor: str):
    """Deploy a model to app.shippedbrain.com - create a REST endpoint and get a hosted model web page

    Set the environment variable SHIPPED_BRAIN_EMAIL to skip de email prompt
    """

    email = os.getenv("SHIPPED_BRAIN_EMAIL")
    password = os.getenv("SHIPPED_BRAIN_PASSWORD")

    if email:
        click.echo("Using environment variable SHIPPED_BRAIN_EMAIL")
        click.echo(f"email: {email}")
    else:
        email = click.prompt(text="email")

    if password:
        click.echo("Using environment variable SHIPPED_BRAIN_PASSWORD")
    else:
        password = click.prompt(text="password", hide_input=True)

    shippedbrain.upload(email=email, password=password, run_id=run_id, model_name=model_name)

if __name__ == "__main__":
    cli()