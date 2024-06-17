from experiment_utils import plot_training_curves

if __name__ == '__main__':
    # plot curves for using single encoder and two encoders
    checkpoint_paths = [
            "manual_checkpoints/two_encoders/point_Spiral11x11/1_million_steps_original_seed0",
            "manual_checkpoints/two_encoders/point_Spiral11x11/1_million_steps_random_seed0",
            "manual_checkpoints/only_sa/fixed_action/point_Spiral11x11/1_million_steps",
            "manual_checkpoints/only_sa/sampled_action/point_Spiral11x11/1_million_steps",
            "manual_checkpoints/only_sa/sampled_action/point_Spiral11x11/1_million_steps_halfed_lr",
            "manual_checkpoints/only_sa/sampled_action/point_Spiral11x11/1_million_steps_increased_replay",
            "manual_checkpoints/only_sa/sampled_action/point_Spiral11x11/1_million_steps_random_action"
        ]
    curve_labels = [
        "Two Encoders",
        "Two Encoders (Random)",
        "Only SA (Fixed)",
        "Only SA (Sampled)",
        "Only SA (Sampled, 0.5*LR)",
        "Only SA (Sampled, Incr. Repl.)",
        "Only SA (Sampled, Random Actions)"
    ]
    saving_path = "experiment_results/experiment_03/"
    vars = ["success_1000", "actor_loss", "critic_loss"]
    plot_training_curves(checkpoint_paths, curve_labels, saving_path, vars)

    # plot curves for using greedy agent vs parameterized agent
    checkpoint_paths = [
            "manual_checkpoints/two_encoders/point_Spiral11x11/1_million_steps_original_seed0",
            "manual_checkpoints/two_encoders/point_Spiral11x11/1_million_steps_random_seed0",
            [
            "manual_checkpoints/two_encoders/point_Spiral11x11/1_million_steps_greedy_seed0",
            "manual_checkpoints/two_encoders/point_Spiral11x11/1_million_steps_greedy_seed42"
            ],
            [
            "manual_checkpoints/two_encoders/point_Spiral11x11/1_million_steps_greedy_randominit_seed42",
            "manual_checkpoints/two_encoders/point_Spiral11x11/1_million_steps_greedy_randominit_seed0"
            ]
        ]
    
    curve_labels = [
        "Parameterized (Original, seed 0)",
        "Random selection (Seed 0)",
        "95% Greedy",
        "95% Greedy (Random Init)"
    ]

    saving_path = "experiment_results/experiment_03/param_vs_greedy/"

    vars = ["success_1000", "actor_loss", "critic_loss"]
    plot_training_curves(checkpoint_paths, curve_labels, saving_path, vars, max_steps=[500_000, None, None])