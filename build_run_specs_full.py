entries = [
    #bigbench
    # 1. auto_debugging: https://github.com/google/big-bench/tree/main/bigbench/benchmark_tasks/auto_debugging
    {'scenario':'auto_debugging','description': "big_bench:model=neurips/local,max_train_instances=big_bench_few_shot_setting,task=auto_debugging,subtask=", 'priority': 1},

    # 3. code_line_description: https://github.com/google/big-bench/tree/main/bigbench/benchmark_tasks/code_line_description
    {'scenario':'code_line_description','description': "big_bench:model=neurips/local,max_train_instances=big_bench_few_shot_setting,task=code_line_description,subtask=", 'priority': 1},

    # 4. conceptual_combinations: https://github.com/google/big-bench/tree/main/bigbench/benchmark_tasks/conceptual_combinations
    {'scenario':'conceptual_combinations','description': "big_bench:model=neurips/local,max_train_instances=big_bench_few_shot_setting,task=conceptual_combinations,subtask=contradictions", 'priority': 1},
    {'scenario':'conceptual_combinations','description': "big_bench:model=neurips/local,max_train_instances=big_bench_few_shot_setting,task=conceptual_combinations,subtask=emergent_properties", 'priority': 1},
    {'scenario':'conceptual_combinations','description': "big_bench:model=neurips/local,max_train_instances=big_bench_few_shot_setting,task=conceptual_combinations,subtask=fanciful_fictional_combinations", 'priority': 1},
    {'scenario':'conceptual_combinations','description': "big_bench:model=neurips/local,max_train_instances=big_bench_few_shot_setting,task=conceptual_combinations,subtask=homonyms", 'priority': 1},
    {'scenario':'conceptual_combinations','description': "big_bench:model=neurips/local,max_train_instances=big_bench_few_shot_setting,task=conceptual_combinations,subtask=invented_words", 'priority': 1},

    # 6. emoji_movie: https://github.com/google/big-bench/tree/main/bigbench/benchmark_tasks/emoji_movie
    {'scenario':'emoji_movie','description': "big_bench:model=neurips/local,max_train_instances=big_bench_few_shot_setting,task=emoji_movie,subtask=", 'priority': 1},

    # 7. formal_fallacies_syllogisms_negation: https://github.com/google/big-bench/tree/main/bigbench/benchmark_tasks/formal_fallacies_syllogisms_negation
    {'scenario':'formal_fallacies_syllogisms_negation','description': "big_bench:model=neurips/local,max_train_instances=big_bench_few_shot_setting,task=formal_fallacies_syllogisms_negation,subtask=", 'priority': 1},

    # 8. hindu_knowledge: https://github.com/google/big-bench/tree/main/bigbench/benchmark_tasks/hindu_knowledge
    # {'scenario':'hindu_knowledge','description': "big_bench:model=neurips/local,max_train_instances=big_bench_few_shot_setting,task=hindu_knowledge,subtask=", 'priority': 1},

    # 9. known_unknowns: https://github.com/google/big-bench/tree/main/bigbench/benchmark_tasks/known_unknowns
    {'scenario':'known_unknowns','description': "big_bench:model=neurips/local,max_train_instances=big_bench_few_shot_setting,task=known_unknowns,subtask=", 'priority': 1},

    # 11. linguistics_puzzles: https://github.com/google/big-bench/tree/main/bigbench/benchmark_tasks/linguistics_puzzles
    {'scenario':'linguistics_puzzles','description': "big_bench:model=neurips/local,max_train_instances=big_bench_few_shot_setting,task=linguistics_puzzles,subtask=", 'priority': 1},

    # 12. logic_grid_puzzle: https://github.com/google/big-bench/tree/main/bigbench/benchmark_tasks/logic_grid_puzzle
    {'scenario':'logic_grid_puzzle','description': "big_bench:model=neurips/local,max_train_instances=big_bench_few_shot_setting,task=logic_grid_puzzle,subtask=", 'priority': 1},

    # 13. logical_deduction: https://github.com/google/big-bench/tree/main/bigbench/benchmark_tasks/logical_deduction
    {'scenario':'logical_deduction','description': "big_bench:model=neurips/local,max_train_instances=big_bench_few_shot_setting,task=logical_deduction,subtask=three_objects", 'priority': 1},
    {'scenario':'logical_deduction','description': "big_bench:model=neurips/local,max_train_instances=big_bench_few_shot_setting,task=logical_deduction,subtask=five_objects", 'priority': 1},
    {'scenario':'logical_deduction','description': "big_bench:model=neurips/local,max_train_instances=big_bench_few_shot_setting,task=logical_deduction,subtask=seven_objects", 'priority': 1},

    # 14. misconceptions_russian: https://github.com/google/big-bench/tree/main/bigbench/benchmark_tasks/misconceptions_russian
    # {'scenario':'misconceptions_russian','description': "big_bench:model=neurips/local,max_train_instances=big_bench_few_shot_setting,task=misconceptions_russian,subtask=", 'priority': 1},

    # 15. novel_concepts: https://github.com/google/big-bench/tree/main/bigbench/benchmark_tasks/novel_concepts
    {'scenario':'novel_concepts','description': "big_bench:model=neurips/local,max_train_instances=big_bench_few_shot_setting,task=novel_concepts,subtask=", 'priority': 1},

    # 16. operators: https://github.com/google/big-bench/tree/main/bigbench/benchmark_tasks/operators
    {'scenario':'operator','description': "big_bench:model=neurips/local,max_train_instances=big_bench_few_shot_setting,task=operators,subtask=", 'priority': 1},

    # 17. parsinlu_reading_comprehension: https://github.com/google/big-bench/tree/main/bigbench/benchmark_tasks/parsinlu_reading_comprehension
    # {'scenario':'parsinlu_reading_comprehension','description': "big_bench:model=neurips/local,max_train_instances=big_bench_few_shot_setting,task=parsinlu_reading_comprehension,subtask=", 'priority': 1},

    # 18. play_dialog_same_or_different: https://github.com/google/big-bench/tree/main/bigbench/benchmark_tasks/play_dialog_same_or_different
    {'scenario':'play_dialog_same_or_different','description': "big_bench:model=neurips/local,max_train_instances=big_bench_few_shot_setting,task=play_dialog_same_or_different,subtask=", 'priority': 1},

    # 19. repeat_copy_logic: https://github.com/google/big-bench/tree/main/bigbench/benchmark_tasks/repeat_copy_logic
    {'scenario':'repeat_copy_logic','description': "big_bench:model=neurips/local,max_train_instances=big_bench_few_shot_setting,task=repeat_copy_logic,subtask=", 'priority': 1},

    # 20. strange_stories: https://github.com/google/big-bench/tree/main/bigbench/benchmark_tasks/strange_stories
    {'scenario':'strange_stories','description': "big_bench:model=neurips/local,max_train_instances=big_bench_few_shot_setting,task=strange_stories,subtask=boolean", 'priority': 1},
    {'scenario':'strange_stories','description': "big_bench:model=neurips/local,max_train_instances=big_bench_few_shot_setting,task=strange_stories,subtask=multiple_choice", 'priority': 1},

    # 21. strategyqa: https://github.com/google/big-bench/tree/main/bigbench/benchmark_tasks/strategyqa
    {'scenario':'strategyqa','description': "big_bench:model=neurips/local,max_train_instances=big_bench_few_shot_setting,task=strategyqa,subtask=", 'priority': 1},

    # 22. symbol_interpretation: https://github.com/google/big-bench/tree/main/bigbench/benchmark_tasks/symbol_interpretation
    {'scenario':'symbol_interpretation','description': "big_bench:model=neurips/local,max_train_instances=big_bench_few_shot_setting,task=symbol_interpretation,subtask=adversarial", 'priority': 1},
    {'scenario':'symbol_interpretation','description': "big_bench:model=neurips/local,max_train_instances=big_bench_few_shot_setting,task=symbol_interpretation,subtask=emoji_agnostic", 'priority': 1},
    {'scenario':'symbol_interpretation','description': "big_bench:model=neurips/local,max_train_instances=big_bench_few_shot_setting,task=symbol_interpretation,subtask=name_agnostic", 'priority': 1},
    {'scenario':'symbol_interpretation','description': "big_bench:model=neurips/local,max_train_instances=big_bench_few_shot_setting,task=symbol_interpretation,subtask=plain", 'priority': 1},
    {'scenario':'symbol_interpretation','description': "big_bench:model=neurips/local,max_train_instances=big_bench_few_shot_setting,task=symbol_interpretation,subtask=tricky", 'priority': 1},

    # 23. vitaminc_fact_verification: https://github.com/google/big-bench/tree/main/bigbench/benchmark_tasks/vitaminc_fact_verification
    {'scenario':'vitaminc_fact_verification','description': "big_bench:model=neurips/local,max_train_instances=big_bench_few_shot_setting,task=vitaminc_fact_verification,subtask=", 'priority': 1},

    # 24. winowhy: https://github.com/google/big-bench/tree/main/bigbench/benchmark_tasks/winowhy
    {'scenario':'winowhy','description': "big_bench:model=neurips/local,max_train_instances=big_bench_few_shot_setting,task=winowhy,subtask=", 'priority': 1},
    
    # MMLU STEM: Medicine/Biology
    {'scenario':'medicine_biology','description': "mmlu:model=neurips/local,subject=anatomy,data_augmentation=canonical", 'priority': 2},
    {'scenario':'medicine_biology','description': "mmlu:model=neurips/local,subject=college_medicine,data_augmentation=canonical", 'priority': 2},
    {'scenario':'medicine_biology','description': "mmlu:model=neurips/local,subject=college_biology,data_augmentation=canonical", 'priority': 2},
    {'scenario':'medicine_biology','description': "mmlu:model=neurips/local,subject=high_school_biology,data_augmentation=canonical", 'priority': 2},
    
    # MMLU STEM: CS
    {'scenario':'computer_science','description': "mmlu:model=neurips/local,subject=college_computer_science,data_augmentation=canonical", 'priority': 2},
    {'scenario':'computer_science','description': "mmlu:model=neurips/local,subject=high_school_computer_science,data_augmentation=canonical", 'priority': 2},
    {'scenario':'computer_science','description': "mmlu:model=neurips/local,subject=computer_security,data_augmentation=canonical", 'priority': 2},
    {'scenario':'computer_science','description': "mmlu:model=neurips/local,subject=electrical_engineering,data_augmentation=canonical", 'priority': 2},
    {'scenario':'computer_science','description': "mmlu:model=neurips/local,subject=machine_learning,data_augmentation=canonical", 'priority': 2},
    
    # MMLU STEM: Math
    {'scenario':'math','description': "mmlu:model=neurips/local,subject=high_school_mathematics,data_augmentation=canonical", 'priority': 2},
    {'scenario':'math','description': "mmlu:model=neurips/local,subject=college_mathematics,data_augmentation=canonical", 'priority': 2},
    {'scenario':'math','description': "mmlu:model=neurips/local,subject=abstract_algebra,data_augmentation=canonical", 'priority': 2},
    {'scenario':'math','description': "mmlu:model=neurips/local,subject=high_school_statistics,data_augmentation=canonical", 'priority': 2},

    # MMLU STEM: Chemistry/Physics
    {'scenario':'physics_chemistry','description': "mmlu:model=neurips/local,subject=college_chemistry,data_augmentation=canonical", 'priority': 2},
    {'scenario':'physics_chemistry','description': "mmlu:model=neurips/local,subject=high_school_chemistry,data_augmentation=canonical", 'priority': 2},
    {'scenario':'physics_chemistry','description': "mmlu:model=neurips/local,subject=high_school_physics,data_augmentation=canonical", 'priority': 2},
    {'scenario':'physics_chemistry','description': "mmlu:model=neurips/local,subject=college_physics,data_augmentation=canonical", 'priority': 2},
    {'scenario':'physics_chemistry','description': "mmlu:model=neurips/local,subject=astronomy,data_augmentation=canonical", 'priority': 2},

    # MMLU Humanities: Formal reasoning
    {'scenario':'formal_reasoning','description': "mmlu:model=neurips/local,subject=formal_logic,data_augmentation=canonical", 'priority': 2},
    {'scenario':'formal_reasoning','description': "mmlu:model=neurips/local,subject=logical_fallacies,data_augmentation=canonical", 'priority': 2},
    {'scenario':'formal_reasoning','description': "mmlu:model=neurips/local,subject=philosophy,data_augmentation=canonical", 'priority': 2},
    {'scenario':'formal_reasoning','description': "mmlu:model=neurips/local,subject=moral_disputes,data_augmentation=canonical", 'priority': 2},
    {'scenario':'formal_reasoning','description': "mmlu:model=neurips/local,subject=moral_scenarios,data_augmentation=canonical", 'priority': 2},

    # MMLU Humanities: Law
    {'scenario':'law','description': "mmlu:model=neurips/local,subject=professional_law,data_augmentation=canonical", 'priority': 2},
    {'scenario':'law','description': "mmlu:model=neurips/local,subject=international_law,data_augmentation=canonical", 'priority': 2},
    {'scenario':'law','description': "mmlu:model=neurips/local,subject=jurisprudence,data_augmentation=canonical", 'priority': 2},
    
    # MMLU Humanities: Histroy
    {'scenario':'history','description': "mmlu:model=neurips/local,subject=high_school_european_history,data_augmentation=canonical", 'priority': 2},
    {'scenario':'history','description': "mmlu:model=neurips/local,subject=high_school_us_history,data_augmentation=canonical", 'priority': 2},
    {'scenario':'history','description': "mmlu:model=neurips/local,subject=high_school_world_history,data_augmentation=canonical", 'priority': 2},
    {'scenario':'history','description': "mmlu:model=neurips/local,subject=prehistory,data_augmentation=canonical", 'priority': 2},
    {'scenario':'history','description': "mmlu:model=neurips/local,subject=world_religions,data_augmentation=canonical", 'priority': 2},

    # MMLU Other: Business
    {'scenario':'business','description': "mmlu:model=neurips/local,subject=business_ethics,data_augmentation=canonical", 'priority': 2},    
    {'scenario':'business','description': "mmlu:model=neurips/local,subject=global_facts,data_augmentation=canonical", 'priority': 2},
    {'scenario':'business','description': "mmlu:model=neurips/local,subject=management,data_augmentation=canonical", 'priority': 2},
    {'scenario':'business','description': "mmlu:model=neurips/local,subject=marketing,data_augmentation=canonical", 'priority': 2},
    {'scenario':'business','description': "mmlu:model=neurips/local,subject=miscellaneous,data_augmentation=canonical", 'priority': 2},
    {'scenario':'business','description': "mmlu:model=neurips/local,subject=professional_accounting,data_augmentation=canonical", 'priority': 2},
    
    # MMLU Other: Health
    {'scenario':'health','description': "mmlu:model=neurips/local,subject=nutrition,data_augmentation=canonical", 'priority': 2},
    {'scenario':'health','description': "mmlu:model=neurips/local,subject=human_aging,data_augmentation=canonical", 'priority': 2},
    {'scenario':'health','description': "mmlu:model=neurips/local,subject=clinical_knowledge,data_augmentation=canonical", 'priority': 2},
    {'scenario':'health','description': "mmlu:model=neurips/local,subject=medical_genetics,data_augmentation=canonical", 'priority': 2},
    {'scenario':'health','description': "mmlu:model=neurips/local,subject=professional_medicine,data_augmentation=canonical", 'priority': 2},
    {'scenario':'health','description': "mmlu:model=neurips/local,subject=virology,data_augmentation=canonical", 'priority': 2},

    # MMLU Social Sciences: Social studies
    {'scenario':'social_studies','description': "mmlu:model=neurips/local,subject=high_school_government_and_politics,data_augmentation=canonical", 'priority': 2},
    {'scenario':'social_studies','description': "mmlu:model=neurips/local,subject=high_school_geography,data_augmentation=canonical", 'priority': 2},
    {'scenario':'social_studies','description': "mmlu:model=neurips/local,subject=us_foreign_policy,data_augmentation=canonical", 'priority': 2},
    {'scenario':'social_studies','description': "mmlu:model=neurips/local,subject=public_relations,data_augmentation=canonical", 'priority': 2},
    {'scenario':'social_studies','description': "mmlu:model=neurips/local,subject=security_studies,data_augmentation=canonical", 'priority': 2},

    # MMLU Social Sciences: Human behavior
    {'scenario':'human_behavior','description': "mmlu:model=neurips/local,subject=high_school_psychology,data_augmentation=canonical", 'priority': 2},
    {'scenario':'human_behavior','description': "mmlu:model=neurips/local,subject=human_sexuality,data_augmentation=canonical", 'priority': 2},
    {'scenario':'human_behavior','description': "mmlu:model=neurips/local,subject=professional_psychology,data_augmentation=canonical", 'priority': 2},
    {'scenario':'human_behavior','description': "mmlu:model=neurips/local,subject=sociology,data_augmentation=canonical", 'priority': 2},

    # MMLU Social Sciences: Economics
    {'scenario':'economics','description': "mmlu:model=neurips/local,subject=high_school_microeconomics,data_augmentation=canonical", 'priority': 2},
    {'scenario':'economics','description': "mmlu:model=neurips/local,subject=econometrics,data_augmentation=canonical", 'priority': 2},
    {'scenario':'economics','description': "mmlu:model=neurips/local,subject=high_school_macroeconomics,data_augmentation=canonical", 'priority': 2},
    
    # Truthful QA
    {'scenario':'truthful_qa','description': "truthful_qa:task=mc_single,model=neurips/local", 'priority': 1},

    # CNN/daily mail
    {'scenario':'truthful_qa','description': "summarization_cnndm:model=neurips/local", 'priority': 1},
    # GSM
    {'scenario':'gsm','description': "gsm:model=neurips/local", 'priority': 1},
    # BBQ
    {'scenario':'bbq','description': "bbq:subject=all,model=neurips/local", 'priority': 1},

]

def generate_equal_sum_list(V, N):
    # Calculate the base value that will be repeated.
    base_value = V // N
    # Calculate the remainder for distribution.
    remainder = V % N
    
    # Create the list with base_value repeated N times.
    result = [base_value] * N
    
    # Distribute the remainder evenly among the elements.
    for i in range(remainder):
        result[i] += 1
    
    return result

import pandas as pd
import argparse

if __name__ == "__main__":

    import argparse
    parser = argparse.ArgumentParser(
        description='''
        This method automatically generates a configuration file for the neurips_llm_efficiency_challenge
        
        Calling it with: `python build_run_specs_full.py --example_budget=600` will produce a conf file 
        with a total of 600 examples distributed evenly across scenarios as also defined here.
        ''',
    )
    parser.add_argument("--example_budget", required=True, type=int, help='# example to use')
    args = parser.parse_args()
    
    # get a list of scenarios and n_examples
    df =  pd.DataFrame(entries)
    scenario_count_dict = df.value_counts('scenario').to_dict()
    n_scenarios = len(df.scenario.unique())
    max_eval_instances_per_scenario = generate_equal_sum_list(args.example_budget, n_scenarios)

    # get a dict of the amount of examples per 
    scenario_n_examples_dict = {}
    for scenario, n_subscenarios in scenario_count_dict.items():
        cur_max_eval_instances_per_scenario = max_eval_instances_per_scenario.pop()
        scenario_n_examples_dict[scenario] = generate_equal_sum_list(cur_max_eval_instances_per_scenario,n_subscenarios)

    for i in range(len(entries)):
        cur_scenario = entries[i]['scenario']
        # print(f"added {v} to {entries[i]['max_eval_instances']}")
        v = scenario_n_examples_dict[cur_scenario].pop()
        entries[i]['max_eval_instances'] = v

    with open(f'./run_specs_full_coarse_{args.example_budget}_budget.conf','w') as f:
        f.write('entries: [\n')
        last_scenario = ''
        for entry in entries:
            cur_scenario = entry['scenario']
            if cur_scenario != last_scenario:
                f.write(f'\n# {cur_scenario}\n')
                print(entry)
            last_scenario = cur_scenario
            f.write('{')
            f.write(f'description: """{entry["description"]}'.replace('"""','"'))
            f.write(f',max_eval_instances={entry["max_eval_instances"]}""",priority: 1'.replace('"""','"'))
            f.write('}\n')
        f.write(']')

    print(f'Saved ./run_specs_full_coarse_{args.example_budget}_budget.conf')
