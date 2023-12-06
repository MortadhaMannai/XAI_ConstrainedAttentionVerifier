from enum import Enum

# numerical constants
INF = 1e30 # Infinity
EPS = 1e-30 # Epsilon

# abstract enum
class ExtendedEnum(Enum):
	@classmethod
	def list(cls):
		return list(map(lambda c: c.value, cls))

# Experiment training mode
class Mode(str, ExtendedEnum):
	EXP = 'exp'
	DEV = 'dev'
	
# Track carbon mode
class TrackCarbon(str, ExtendedEnum):
	OFFLINE = 'offline'
	ONLINE = 'online'

# Data configuration constant
class InputType(ExtendedEnum):
	SINGLE = 1
	DUAL = 2
	
class SpecToken(str, ExtendedEnum):
	PAD = '<pad>'
	UNK = '<unk>'
	CLS = '<cls>'
	ENT_PER = '<PER>'
	ENT_LOC = '<LOC>'
	ENT_ORG = '<ORG>'
	ENT_MISC = '<MISC>'

class Normalization(str, ExtendedEnum):
	NONE = None
	STANDARD = 'std'
	LOG_STANDARD = 'log_std'
	SOFTMAX = 'softmax'
	LOG_SOFTMAX = 'log_softmax'

# Model configuration constant
class ContextType(str, ExtendedEnum):
	MLP='mlp'
	LSTM='lstm'
	CNN='cnn'
	ATTENTION='attention'
	BERT='bert'
	
# Data choice
class Data(str, ExtendedEnum):
	ARCHIVAL_NLI='archival_nli'
	ESNLI='esnli'
	HATEXPLAIN='hatexplain'
	YELPHAT='yelphat'
	YELPHAT50 = 'yelphat50'
	YELPHAT100 = 'yelphat100'
	YELPHAT200 = 'yelphat200'