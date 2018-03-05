import os
import sys
from subprocess import call

startwalk = sys.argv[1]
destination = sys.argv[2]

done = set()

for root, dirs, files in os.walk(startwalk):
	if root in done:
		continue
	
	done.add(root)

	path = root.split(os.sep)

	fs = files

	if len(fs) > 0:
		f = fs[0]
		print(root, f)
		fn = root+os.sep+f
		_, ext = os.path.splitext(f)
		gn  = destination + os.sep+ '_'.join(path)+ext
		print(['cp',fn,gn])
		call(['cp', fn,gn])
	
