from .all_entries import entries

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


if __name__ == "__main__":

    import pandas as pd
    df = pd.DataFrame(entries)
    scenarios = df.value_counts('scenario').to_dict()

    total_n_examples = 600
    max_eval_instances_per_scenario = total_n_examples//len(df.scenario.unique())

    for scenario, n_sucscenarios in scenarios.items():
        scenarios[scenario] = generate_equal_sum_list(max_eval_instances_per_scenario,n_sucscenarios)

    for i in range(len(entries)):
        cur_scenario = entries[i]['scenario']
        # print(f"added {v} to {entries[i]['max_eval_instances']}")
        v = scenarios[cur_scenario].pop()
        entries[i]['max_eval_instances'] = v


    with open(f'./run_specs_full_coarse_{total_n_examples}_examples.conf','w') as f:
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
