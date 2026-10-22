import json
from collections import Counter

with open('data/docEdit_tanglish_500.json', encoding='utf-8') as f:
    tang = json.load(f)

with open('data/docEdit1000.json', encoding='utf-8') as f:
    combined = json.load(f)

tang_counts = Counter(ex['intent'] for ex in tang)
print('Tanglish per intent:', dict(sorted(tang_counts.items())))
print('Total tanglish:', len(tang))

actions = ['bold','italic','delete','underline','heading2','heading3',
           'replace','insert-after','font-increase','highlight']
for action in actions:
    samples = [ex for ex in tang if ex['intent'] == action][:3]
    print('\n-- %s --' % action)
    for s in samples:
        eid = s['id']
        txt = s['text']
        lbl = s['labels']
        print('  [%d] %s' % (eid, txt))
        print('       labels: %s' % lbl)

print()
print('Combined total:', len(combined))
en = [ex for ex in combined if ex.get('lang','en') == 'en']
tn = [ex for ex in combined if ex.get('lang') == 'tanglish']
print('English:', len(en), '| Tanglish:', len(tn))
