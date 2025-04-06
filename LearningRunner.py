from DQN.DQN_Learning import DQN


DQN_eng = DQN(load_model_path="DQN_policy_model_2025_04_06_21_33_eps100")
# print("starting evaluation")
# DQN_eng.env.SA.problem.evaluate_tsp_files()
print("starting learning")
DQN_eng.run(episodes=2000)


#Learning episode 42/100
# Traceback (most recent call last):
#   File "c:/Users/piotr/OneDrive/Pulpit/MAGISTERKA/RL_agent_for_SA/TestRunner.py", line 12, in <module>
#     DQN_eng.run(100)
#   File "c:/Users/piotr/OneDrive/Pulpit/MAGISTERKA/RL_agent_for_SA/DQN/DQN_Learning.py", line 85, in run
#     state = self.env.reset()
#             ^^^^^^^^^^^^^^^^
#   File "c:/Users/piotr/OneDrive/Pulpit/MAGISTERKA/RL_agent_for_SA/DQN/DQN_SA.py", line 64, in reset
#     else:

#   File "c:/Users/piotr/OneDrive/Pulpit/MAGISTERKA/RL_agent_for_SA/SA.py", line 14, in __init__
#     self.problem = Problem.VRP()
#                    ^^^^^^^^^^^^^
#   File "c:/Users/piotr/OneDrive/Pulpit/MAGISTERKA/RL_agent_for_SA/Problem.py", line 36, in __init__
#     self.distances = [[problem.get_weight(i, j) for j in problem.get_nodes()] for i in problem.get_nodes()]
#                        ^^^^^^^^^^^^^^^^^^^^^^^^
# MemoryError
# PS C:/Users/piotr/OneDrive/Pulpit/MAGISTERKA/RL_agent_for_SA> 
#

# def CHECK_IF_NETWORK_IS_LEARNING_GIVEN_DATA():
#     DQN_engine = DQN()
#     DQN_engine.epsilon = 0
#     action_selction = DQN_engine.policy_net(torch.Tensor(DQN_engine.env.observation()))
#     print(action_selction)

#     for i in range(1000):
#         DQN_engine.FORCE_learnNetwork(None,None,None,None)
#         if i%100 == 0:
#             action_selction = DQN_engine.policy_net(torch.Tensor(DQN_engine.env.observation()))
#             print(action_selction)

#     DQN_engine.epsilon = 0
#     action_selction = DQN_engine.policy_net(torch.Tensor(DQN_engine.env.observation()))
#     print(action_selction)

