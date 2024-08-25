from experiment_utils import plot_training_curves, evaluate_checkpoints, record_inference_times
import numpy as np

if __name__ == '__main__':
    # plot curves for using single encoder and two encoders
    print("################# Plotting Curves of Single vs Two Encoders #################")
    checkpoint_paths = [
            [
                "manual_checkpoints/two_encoders/point_Spiral11x11/original_seed0",
                "manual_checkpoints/two_encoders/point_Spiral11x11/original_seed21",
                "manual_checkpoints/two_encoders/point_Spiral11x11/original_seed42",
                "manual_checkpoints/two_encoders/point_Spiral11x11/original_seed97",
                "manual_checkpoints/two_encoders/point_Spiral11x11/original_seed1453"
            ],
            [
                "manual_checkpoints/two_encoders/point_Spiral11x11/random_seed0",
                "manual_checkpoints/two_encoders/point_Spiral11x11/random_seed21",
                "manual_checkpoints/two_encoders/point_Spiral11x11/random_seed42",
                "manual_checkpoints/two_encoders/point_Spiral11x11/random_seed97",
                "manual_checkpoints/two_encoders/point_Spiral11x11/random_seed1453"
            ],
            [
                "manual_checkpoints/only_sa/fixed_seed0",
                "manual_checkpoints/only_sa/fixed_seed21",
                "manual_checkpoints/only_sa/fixed_seed42",
                "manual_checkpoints/only_sa/fixed_seed97",
                "manual_checkpoints/only_sa/fixed_seed1453"
            ],
            [
                "manual_checkpoints/only_sa/sampled_seed0",
                "manual_checkpoints/only_sa/sampled_seed21",
                "manual_checkpoints/only_sa/sampled_seed42",
                "manual_checkpoints/only_sa/sampled_seed97",
                "manual_checkpoints/only_sa/sampled_seed1453"
            ]
        ]
    curve_labels = [
        "Two Encoders",
        "Random Actor",
        "Only SA (Fixed)",
        "Only SA (Sampled)"
    ]
    saving_path = "experiment_results/experiment_03/single_vs_two_encoders/"
    vars = ["success_1000", "actor_loss", "critic_loss"]
    plot_training_curves(checkpoint_paths, curve_labels, saving_path, vars, max_steps=[400_000, None, None])

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
            [
                "manual_checkpoints/two_encoders/point_Spiral11x11/random_seed0",
                "manual_checkpoints/two_encoders/point_Spiral11x11/random_seed21",
                "manual_checkpoints/two_encoders/point_Spiral11x11/random_seed42",
                "manual_checkpoints/two_encoders/point_Spiral11x11/random_seed97",
                "manual_checkpoints/two_encoders/point_Spiral11x11/random_seed1453"
            ],
            [
                "manual_checkpoints/two_encoders/point_Spiral11x11/greedy_randominit_9actions_seed0",
                "manual_checkpoints/two_encoders/point_Spiral11x11/greedy_randominit_9actions_seed21",
                "manual_checkpoints/two_encoders/point_Spiral11x11/greedy_randominit_9actions_seed42",
                "manual_checkpoints/two_encoders/point_Spiral11x11/greedy_randominit_9actions_seed97",
                "manual_checkpoints/two_encoders/point_Spiral11x11/greedy_randominit_9actions_seed1453"
            ],
            [
                "manual_checkpoints/two_encoders/point_Spiral11x11/greedy_randominit_25actions_seed42",
                "manual_checkpoints/two_encoders/point_Spiral11x11/greedy_randominit_25actions_seed21",
                "manual_checkpoints/two_encoders/point_Spiral11x11/greedy_randominit_25actions_seed0",
                "manual_checkpoints/two_encoders/point_Spiral11x11/greedy_randominit_25actions_seed97",
                "manual_checkpoints/two_encoders/point_Spiral11x11/greedy_randominit_25actions_seed1453"
            ],
            
            [
                "manual_checkpoints/two_encoders/point_Spiral11x11/greedy_randominit_81actions_seed0",
                "manual_checkpoints/two_encoders/point_Spiral11x11/greedy_randominit_81actions_seed21",
                "manual_checkpoints/two_encoders/point_Spiral11x11/greedy_randominit_81actions_seed42",
                "manual_checkpoints/two_encoders/point_Spiral11x11/greedy_randominit_81actions_seed97",
                "manual_checkpoints/two_encoders/point_Spiral11x11/greedy_randominit_81actions_seed1453"

            ]
        ]
    curve_labels = [
        "Parameterized",
        "Random",
        "Greedy (09 actions)",
        "Greedy (25 actions)",
        "Greedy (81 actions)"
    ]
    saving_path = "experiment_results/experiment_03/param_vs_greedy/"

    produce_simplified_plots = False # used to produce a simplified version of this plot, e.g. for the repo README
    
    if produce_simplified_plots:
        checkpoint_paths = [
                [
                    "manual_checkpoints/two_encoders/point_Spiral11x11/original_seed0",
                    "manual_checkpoints/two_encoders/point_Spiral11x11/original_seed21",
                    "manual_checkpoints/two_encoders/point_Spiral11x11/original_seed42",
                    "manual_checkpoints/two_encoders/point_Spiral11x11/original_seed97",
                    "manual_checkpoints/two_encoders/point_Spiral11x11/original_seed1453"
                ],
                [
                "manual_checkpoints/two_encoders/point_Spiral11x11/random_seed0",
                "manual_checkpoints/two_encoders/point_Spiral11x11/random_seed21",
                "manual_checkpoints/two_encoders/point_Spiral11x11/random_seed42",
                "manual_checkpoints/two_encoders/point_Spiral11x11/random_seed97",
                "manual_checkpoints/two_encoders/point_Spiral11x11/random_seed1453"
                ],
                [
                    "manual_checkpoints/two_encoders/point_Spiral11x11/greedy_randominit_25actions_seed42",
                    "manual_checkpoints/two_encoders/point_Spiral11x11/greedy_randominit_25actions_seed21",
                    "manual_checkpoints/two_encoders/point_Spiral11x11/greedy_randominit_25actions_seed0",
                    "manual_checkpoints/two_encoders/point_Spiral11x11/greedy_randominit_25actions_seed97",
                    "manual_checkpoints/two_encoders/point_Spiral11x11/greedy_randominit_25actions_seed1453"
                ]
            ]
        
        curve_labels = [
            "Eysenbach et al. (2023)",
            "Random Actor",
            "Ours"
        ]
        saving_path = "experiment_results/experiment_03/blog_figures/"

    vars = ["success_1000", "actor_loss", "critic_loss"]
    plot_training_curves(checkpoint_paths, curve_labels, saving_path, vars, max_steps=[500_000, None, None])

    print("################# Evaluating Greedily vs Parameterized Training #################")
    checkpoint_paths = [
            [
                "manual_checkpoints/two_encoders/point_Spiral11x11/original_seed0",
                "manual_checkpoints/two_encoders/point_Spiral11x11/original_seed21",
                "manual_checkpoints/two_encoders/point_Spiral11x11/original_seed42",
                "manual_checkpoints/two_encoders/point_Spiral11x11/original_seed97",
                "manual_checkpoints/two_encoders/point_Spiral11x11/original_seed1453"
            ],
            [
                "manual_checkpoints/two_encoders/point_Spiral11x11/random_seed0",
                "manual_checkpoints/two_encoders/point_Spiral11x11/random_seed21",
                "manual_checkpoints/two_encoders/point_Spiral11x11/random_seed42",
                "manual_checkpoints/two_encoders/point_Spiral11x11/random_seed97",
                "manual_checkpoints/two_encoders/point_Spiral11x11/random_seed1453"
            ],
            [
                "manual_checkpoints/two_encoders/point_Spiral11x11/greedy_randominit_9actions_seed0",
                "manual_checkpoints/two_encoders/point_Spiral11x11/greedy_randominit_9actions_seed21",
                "manual_checkpoints/two_encoders/point_Spiral11x11/greedy_randominit_9actions_seed42",
                "manual_checkpoints/two_encoders/point_Spiral11x11/greedy_randominit_9actions_seed97",
                "manual_checkpoints/two_encoders/point_Spiral11x11/greedy_randominit_9actions_seed1453"
            ],
            [
                "manual_checkpoints/two_encoders/point_Spiral11x11/greedy_randominit_25actions_seed42",
                "manual_checkpoints/two_encoders/point_Spiral11x11/greedy_randominit_25actions_seed21",
                "manual_checkpoints/two_encoders/point_Spiral11x11/greedy_randominit_25actions_seed0",
                "manual_checkpoints/two_encoders/point_Spiral11x11/greedy_randominit_25actions_seed97",
                "manual_checkpoints/two_encoders/point_Spiral11x11/greedy_randominit_25actions_seed1453"
            ],
            
            [
                "manual_checkpoints/two_encoders/point_Spiral11x11/greedy_randominit_81actions_seed0",
                "manual_checkpoints/two_encoders/point_Spiral11x11/greedy_randominit_81actions_seed21",
                "manual_checkpoints/two_encoders/point_Spiral11x11/greedy_randominit_81actions_seed42",
                "manual_checkpoints/two_encoders/point_Spiral11x11/greedy_randominit_81actions_seed97",
                "manual_checkpoints/two_encoders/point_Spiral11x11/greedy_randominit_81actions_seed1453"

            ]
        ]
    
    labels = [
        "Parameterized",
        "Random",
        "Greedy (09 actions)",
        "Greedy (25 actions)",
        "Greedy (81 actions)"
    ]
    saving_path = "experiment_results/experiment_03/param_vs_greedy_eval/"
    np.random.seed(42) # for reproduction: 42
    evaluate_checkpoints(checkpoint_paths, labels)
    record_inference_times(checkpoint_paths, labels)

    """
    Evaluation Results:
    Epsilon = 0.05
        Parameterized is evaluated across 5 seeds.
        Parameterized success rates mean per seed: [0.778 0.798 0.834 0.816 0.798]
        Parameterized success rate mean across seeds: 0.8048
        Parameterized success rate std across seeds: 0.018914544668058992
        Parameterized average steps to reach goal first time: 6.270756213643575
        Random is evaluated across 5 seeds.
        Random success rates mean per seed: [0.448 0.448 0.434 0.458 0.448]
        Random success rate mean across seeds: 0.44720000000000004
        Random success rate std across seeds: 0.007652450587883603
        Random average steps to reach goal first time: 26.415686274509802
        Greedy (09 actions) is evaluated across 5 seeds.
        Greedy (09 actions) success rates mean per seed: [0.5   0.538 0.494 0.52  0.536]
        Greedy (09 actions) success rate mean across seeds: 0.5176000000000001
        Greedy (09 actions) success rate std across seeds: 0.018039955654047507
        Greedy (09 actions) average steps to reach goal first time: 11.76986301369863
        Greedy (25 actions) is evaluated across 5 seeds.
        Greedy (25 actions) success rates mean per seed: [0.786 0.852 0.846 0.73  0.746]
        Greedy (25 actions) success rate mean across seeds: 0.792
        Greedy (25 actions) success rate std across seeds: 0.05002399424276313
        Greedy (25 actions) average steps to reach goal first time: 17.79310344827586
        Greedy (81 actions) is evaluated across 5 seeds.
        Greedy (81 actions) success rates mean per seed: [0.616 0.568 0.564 0.644 0.614]
        Greedy (81 actions) success rate mean across seeds: 0.6012
        Greedy (81 actions) success rate std across seeds: 0.030662028634778907
        Greedy (81 actions) average steps to reach goal first time: 16.731190650109568
    
    Epsilon = 0.1
        Parameterized is evaluated across 5 seeds.
        Parameterized success rates mean per seed: [0.778 0.8   0.834 0.814 0.798]
        Parameterized success rate mean across seeds: 0.8048
        Parameterized success rate std across seeds: 0.01857309882599021
        Parameterized average steps to reach goal first time: 6.329455314648334
        Random is evaluated across 5 seeds.
        Random success rates mean per seed: [0.444 0.456 0.42  0.442 0.448]
        Random success rate mean across seeds: 0.442
        Random success rate std across seeds: 0.01200000000000001
        Random average steps to reach goal first time: 30.746835443037973
        Greedy (09 actions) is evaluated across 5 seeds.
        Greedy (09 actions) success rates mean per seed: [0.56  0.624 0.54  0.554 0.558]
        Greedy (09 actions) success rate mean across seeds: 0.5672
        Greedy (09 actions) success rate std across seeds: 0.02924653825668944
        Greedy (09 actions) average steps to reach goal first time: 12.12831479897348
        Greedy (25 actions) is evaluated across 5 seeds.
        Greedy (25 actions) success rates mean per seed: [0.85  0.916 0.876 0.794 0.776]
        Greedy (25 actions) success rate mean across seeds: 0.8423999999999999
        Greedy (25 actions) success rate std across seeds: 0.05168210522027909
        Greedy (25 actions) average steps to reach goal first time: 18.966360856269112
        Greedy (81 actions) is evaluated across 5 seeds.
        Greedy (81 actions) success rates mean per seed: [0.716 0.638 0.632 0.672 0.652]
        Greedy (81 actions) success rate mean across seeds: 0.6620000000000001
        Greedy (81 actions) success rate std across seeds: 0.030305115079801284
        Greedy (81 actions) average steps to reach goal first time: 18.415446071904128
    """