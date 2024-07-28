from experiment_utils import plot_training_curves, evaluate_checkpoints

if __name__ == '__main__':
    # plot curves for using single encoder and two encoders
    print("################# Plotting Curves of Single vs Two Encoders #################")
    checkpoint_paths = [
            "manual_checkpoints/two_encoders/point_Spiral11x11/original_seed0",
            "manual_checkpoints/two_encoders/point_Spiral11x11/random_seed0",
            "manual_checkpoints/only_sa/fixed_action/point_Spiral11x11/1_million_steps",
            "manual_checkpoints/only_sa/sampled_action/point_Spiral11x11/1_million_steps",
            "manual_checkpoints/only_sa/sampled_action/point_Spiral11x11/1_million_steps_random_action"
        ]
    curve_labels = [
        "Two Encoders",
        "Two Encoders (Random Actor)",
        "Only SA (Fixed)",
        "Only SA (Sampled)",
        "Only SA (Random Actor)"
    ]
    saving_path = "experiment_results/experiment_03/single_vs_two_encoders/"
    vars = ["success_1000", "actor_loss", "critic_loss"]
    plot_training_curves(checkpoint_paths, curve_labels, saving_path, vars)

    # plot curves for using greedy agent vs parameterized agent
    print("################# Plotting Curves of Greedy vs Parameterized CRL #################")
    checkpoint_paths = [
            [
                "manual_checkpoints/two_encoders/point_Spiral11x11/original_seed0",
                "manual_checkpoints/two_encoders/point_Spiral11x11/original_seed21",
                "manual_checkpoints/two_encoders/point_Spiral11x11/original_seed42",
                "manual_checkpoints/two_encoders/point_Spiral11x11/original_seed97",
                "manual_checkpoints/two_encoders/point_Spiral11x11/original_seed1453"
            ],
            "manual_checkpoints/two_encoders/point_Spiral11x11/random_seed0",
            [
                "manual_checkpoints/two_encoders/point_Spiral11x11/greedy_randominit_25actions_seed42",
                "manual_checkpoints/two_encoders/point_Spiral11x11/greedy_randominit_25actions_seed21",
                "manual_checkpoints/two_encoders/point_Spiral11x11/greedy_randominit_25actions_seed0",
                "manual_checkpoints/two_encoders/point_Spiral11x11/greedy_randominit_25actions_seed97",
                "manual_checkpoints/two_encoders/point_Spiral11x11/greedy_randominit_25actions_seed1453"
            ],
            [
                "manual_checkpoints/two_encoders/point_Spiral11x11/greedy_randominit_9actions_seed97",
                "manual_checkpoints/two_encoders/point_Spiral11x11/greedy_randominit_9actions_seed0"
            ],
            [
                "manual_checkpoints/two_encoders/point_Spiral11x11/greedy_randominit_81actions_seed0",
                "manual_checkpoints/two_encoders/point_Spiral11x11/greedy_randominit_81actions_seed21",
                "manual_checkpoints/two_encoders/point_Spiral11x11/greedy_randominit_81actions_seed42",
                "manual_checkpoints/two_encoders/point_Spiral11x11/greedy_randominit_81actions_seed97"

            ]
        ]
    
    curve_labels = [
        #"Eysenbach et al.",
        "Parameterized Actor",
        "Random Actor",
        #"Mine",
        "95% Greedy Actor (25 actions)",
        "95% Greedy Actor (9 actions)",
        "95% Greedy Actor (81 actions)"
    ]

    saving_path = "experiment_results/experiment_03/param_vs_greedy/"

    #saving_path = "experiment_results/experiment_03/blog_figures/"

    vars = ["success_1000", "actor_loss", "critic_loss"]
    plot_training_curves(checkpoint_paths, curve_labels, saving_path, vars, max_steps=[500_000, None, None])

    print("################# Evaluating Greedily vs Parameterized Training #################")
        
    checkpoint_paths = [
            [
                "manual_checkpoints/two_encoders/point_Spiral11x11/original_seed0",
                "manual_checkpoints/two_encoders/point_Spiral11x11/original_seed21",
                "manual_checkpoints/two_encoders/point_Spiral11x11/original_seed42"
            ],
            [
                "manual_checkpoints/two_encoders/point_Spiral11x11/greedy_randominit_25actions_seed42",
                "manual_checkpoints/two_encoders/point_Spiral11x11/greedy_randominit_25actions_seed21",
                "manual_checkpoints/two_encoders/point_Spiral11x11/greedy_randominit_25actions_seed0"
            ],
            [
                "manual_checkpoints/two_encoders/point_Spiral11x11/greedy_randominit_9actions_seed0",
                "manual_checkpoints/two_encoders/point_Spiral11x11/greedy_randominit_9actions_seed97"
            ],
            [
                "manual_checkpoints/two_encoders/point_Spiral11x11/greedy_randominit_81actions_seed0"
            ],
            [
                "manual_checkpoints/two_encoders/point_Spiral11x11/random_seed0"
            ]
        ]
    
    labels = [
        #"Eysenbach et al.",
        #"Mine",
        "Parameterized",
        "Greedy (25 actions)",
        "Greedy (09 actions)",
        "Greedy (81 actions)",
        "Random"
    ]

    saving_path = "experiment_results/experiment_03/param_vs_greedy_eval/"

    evaluate_checkpoints(checkpoint_paths, labels)

    """
    Parameterized is evaluated across 3 seeds.
    Parameterized success rate mean across seeds: 0.7133333333333333
    Parameterized success rate std across seeds: 0.026246692913372727
    Parameterized average steps to reach goal first time: 6.355140186915888
    Greedy (25 actions) is evaluated across 3 seeds.
    Greedy (25 actions) success rate mean across seeds: 0.7400000000000001
    Greedy (25 actions) success rate std across seeds: 0.05715476066494083
    Greedy (25 actions) average steps to reach goal first time: 19.32882882882883
    Greedy (09 actions) is evaluated across 2 seeds.
    Greedy (09 actions) success rate mean across seeds: 0.375
    Greedy (09 actions) success rate std across seeds: 0.044999999999999984
    Greedy (09 actions) average steps to reach goal first time: 10.48
    Greedy (81 actions) is evaluated across 1 seeds.
    Greedy (81 actions) success rate mean across seeds: 0.53
    Greedy (81 actions) success rate std across seeds: 0.0
    Greedy (81 actions) average steps to reach goal first time: 15.18867924528302
    Random is evaluated across 1 seeds.
    Random success rate mean across seeds: 0.07
    Random success rate std across seeds: 0.0
    Random average steps to reach goal first time: 45.714285714285715
    """