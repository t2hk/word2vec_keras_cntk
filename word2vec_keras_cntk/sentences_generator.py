'''
ファイルから行単位で文書を読み込むためのクラス。
'''
import codecs

class Sentences(object):
	# ファイルパスを保持し、ファイルの行数をカウントする。
	def __init__(self, filepath):
		self.filepath = filepath
	
	def __iter__(self):
		with codecs.open(self.filepath, "r", "utf-8") as rf:
			for line in rf:
				if line.strip():
					yield line.strip().lower()
