import re
import numpy as np
import subprocess
import time

np.random.seed(234)

seeds = np.random.choice(np.arange(100,1000),12,replace=False)  

with open('submit.sh','r') as f:
    s = f.read()

for seed in seeds:
    s2 =  re.sub(r'[0-9]{3}$',str(seed),s)
    with open('submit.sh','w') as f:
        f.write(s2)
    
    subprocess.run(["qsub","submit.sh"])
    time.sleep(15)

print('Printing seeds:\n')
for seed in seeds:
    print(seed)