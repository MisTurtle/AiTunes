# from aitunes.user_controls.gui import GUI

# g = GUI()
# g.mainloop()

from aitunes.user_controls.headless import HeadlessActionPipeline

actions = HeadlessActionPipeline()
actions.select_experiment('GtzanReconstruction')
actions.select_scenario('gtzan_cvae_v1.2-LOW32')
actions.select_model('history\\gtzan\\gtzan_cvae_v1.2-LOW32\\20250212_113617\\checkpoint_75.pth')
actions.train(100, 25, False)
