import os

a1 = open("train_before.txt", "w")
a2 = open("train_after.txt", "w")

for path, subdirs, files in os.walk('train'):
	for filename in files:
		f = str(filename)
		if f[:4] == "Left" or f[:4] == "Righ" or f[:4] == "Up_s":
			if f[-6:] == "_1.png":
				a1.write(f + os.linesep)
			elif f[-6:] == "_2.png":
				a2.write(f + os.linesep)

