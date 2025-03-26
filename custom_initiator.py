from aitunes.user_controls.headless import HeadlessActionPipeline

actions = HeadlessActionPipeline()
print(actions.list_production_releases())
