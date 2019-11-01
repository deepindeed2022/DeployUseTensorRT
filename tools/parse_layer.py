import sys
import os.path

if __name__ == '__main__':
	if len(sys.argv) < 2:
		print "please input a prototxt file"

	with open(sys.argv[1]) as fd:
		lines = map(lambda x: x.strip(), fd.readlines())
		layers = set()
		for line in lines:
			if line.startswith("type:"):
				layers.add(line.split(":")[1].strip().strip("\""))
		for layer in list(layers):
			print layer