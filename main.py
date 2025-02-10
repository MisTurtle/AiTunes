from aitunes.user_controls.gui import GUI

g = GUI()
g.mainloop()

# from aitunes.user_controls.headless import HeadlessActionPipeline

# actions = HeadlessActionPipeline()
# actions.select_experiment('MnistReconstruction')

# for scenario in actions.list_scenarios():
#     actions.select_scenario(scenario)
#     actions.train(10, 5, False)
