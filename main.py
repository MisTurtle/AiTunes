import argparse
import aitunes.utils as utils
from aitunes.user_controls.gui import GUI
from aitunes.user_controls.headless import HeadlessActionPipeline

parser = argparse.ArgumentParser(description="AiTunes Job Submission Endpoint")
actions = HeadlessActionPipeline()

parser.add_argument("--headless", action="store_true", help="Run without GUI")
parser.add_argument("--experiment", "-E", type=str, help="Name of the experiment (Required with Headless)")
parser.add_argument("--scenario", "-S", type=str, help="Which scenario to load (Required with Headless)")
parser.add_argument("--model", "-M", type=str, help="A path to the model that needs to be loaded (No specifying it will use the latest available, if any) (Optional, default=none)")
parser.add_argument("--epochs", "-e", type=int, help="How many epochs to train the model (Required with Headless)")
parser.add_argument("--save_every", "-s", type=int, default=0, help="How often to create a checkpoint during training. 0 means no checkpoint is created until the end of training (Optional, default=0)")
parser.add_argument("--quiet", "-Q", action="store_true", help="Disable training progress updates (Optional, recommended for job submission)")
parser.add_argument("--evaluate", action="store_true", help="Run an evaluation of the model (Optional, default=false)")

if __name__ == "__main__":
    args = parser.parse_args()
    actions.quiet(args.quiet)

    if args.headless:
        if not actions.select_experiment(args.experiment):
            choices = "\n - ".join(actions.list_scripted_experiments())
            parser.error(f"Experiment {args.experiment} does not exist. Choices are: \n - {choices}")
        if not actions.select_scenario(args.scenario):
            choices = "\n - ".join(map(lambda x: x.identifier, actions.list_scenarios()))
            parser.error(f"Scenario {args.scenario} does not exist. Choices are: \n - {choices}")
        if args.model is not None and args.model.lower() != "none":
            if not actions.select_model(args.model):
                choices = "\n - ".join(actions.list_models())
                parser.error(f"Model {args.model} was not found. Choices are: \n - {choices}")
        elif len(actions.list_models()) > 0:
            actions.select_model(actions.list_models()[0])

        if args.evaluate:
            actions.evaluate()
        else:
            if args.epochs < 0:
                parser.error(f"Epoch number should be higher than 0, but {args.epochs} was provided.")
            if args.save_every < 0:
                parser.error(f"Checkpoint period should be higher than 0, but {args.save_every} was provided.")

            actions.train(args.epochs, args.save_every, False)
    else:
        GUI().mainloop()
