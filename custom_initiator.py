from aitunes.user_controls.headless import HeadlessActionPipeline

actions = HeadlessActionPipeline()
actions.select_experiment('CIFAR10')
actions.select_scenario('cvae_dim32')
actions.select_model('assets\\Models\\cifar10\\cvae_dim32.pth')
actions.evaluate()
