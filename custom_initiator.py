from aitunes.user_controls.headless import HeadlessActionPipeline

actions = HeadlessActionPipeline()
actions.select_experiment('SINEWAVE')
actions.select_scenario(actions.list_scenarios()[-1])
# actions.train(2, 0, False)
actions.select_release_model()
actions.interactive_evaluation()


# TODO : Add docstring kind of everywhere if it's important
# --- Commit
# TODO : Once the above is done, train on SINEWAVE and GTZAN
# --- Commit
# TODO : Add some kind of hooks system to scenarios (returning values as a dict for example) to schedule actions every x epochs
# TODO : Implement the hooks system for codebook reset every once in a while
# --- Commit

# Faire un scénario pour les idées comme ça:
# Idée: Si on met deux ResNet CVAE bout à bout, alors on obtient une image au milieu qui n'est compréhensible que par le deuxième.
# Alors si on dessine une image et qu'on la passe au deuxième réseau, qu'est-ce qu'il se passe?
