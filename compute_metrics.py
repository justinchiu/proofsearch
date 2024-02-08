import json
import glob

for setting in [
    #'codellama7b_minif2f_test',
    #'codellama34b_minif2f_test',
    'llemma7b_minif2f_valid',
    'llemma7b_minif2f_test',
    #'llemma34b_minif2f_test'
]:
    fs = [x for x in glob.glob('./output/%s/*/*.json' % setting)]
    for x in fs:
        f = json.load(open(x))
        n = 0
        ns = 0
        for result in f['results']:
            name = result['example']['full_name']

            # Extra helper theorem in the OpenAI code
            if 'sum_pairs' in name:
                continue

            n += 1
            if result['success']:
                ns += 1
        if n > 0:
            print(setting, x, ns/n, ns, n, sep='\t')
