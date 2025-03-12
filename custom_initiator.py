from aitunes.user_controls.headless import HeadlessActionPipeline

actions = HeadlessActionPipeline()

actions.select_experiment('CIFAR 10')
for s in actions.list_scenarios():
    print('Running for %s' % s)
    actions.select_scenario(s)
    actions.train(200, 20, False)

actions.select_experiment('SINEWAVE')
for s in actions.list_scenarios():
    print('Running for %s' % s)
    actions.select_scenario(s)
    actions.train(200, 20, False)
    
actions.select_experiment('GTZAN')
for s in actions.list_scenarios():
    print('Running for %s' % s)
    actions.select_scenario(s)
    actions.train(200, 20, False)


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
