import sys
import numpy as np

def clear_console():
	printr("")

def printr(string):
	sys.stdout.write("\r\033[2K")
	sys.stdout.write(string)
	sys.stdout.flush()
