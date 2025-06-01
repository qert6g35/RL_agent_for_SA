from PPO.PPO_Learning import PPO

PPO_engine = PPO(
    save_agent_path="PPO_2025_05_27_20_44",
    load_agent_path="NN_Models/PPO/G6_with_offset_onFiew_first_steps/PPO_2025_05_27_20_44_updates14570",
    verssioning_offset=14570
    )
PPO_engine.run_learning()