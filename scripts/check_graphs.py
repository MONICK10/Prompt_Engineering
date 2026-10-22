import os
from PIL import Image
for f in ['accuracy_comparison.png','english_vs_tanglish.png','per_action_accuracy.png']:
    p = os.path.join('data','graphs',f)
    img = Image.open(p)
    size_kb = os.path.getsize(p) // 1024
    w_in = img.size[0]/300
    h_in = img.size[1]/300
    print('%s: %dx%dpx  (%.2f" x %.2f")  %dKB' % (f, img.size[0], img.size[1], w_in, h_in, size_kb))
