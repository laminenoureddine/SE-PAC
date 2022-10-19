global dumped_files_suffix
global logfile
global features
global features_mnemonics
global feature_names
global feature_names_short
global features_categories
global feature_types
global feature_types_range
global features_desc_and_comments
global features_dict
global features_mnemonics_dict
global file_names
global family_names
global labels
global binaries_names
dumped_files_suffix = ''
logfile = ''
features = []
features_mnemonics = []
feature_names = []
feature_names_short = []
features_categories = []
feature_types = []
feature_types_range = []
features_desc_and_comments = []
features_dict = {}
features_mnemonics_dict = {}
binaries_names = []

"""
# own-built dataset
file_names = [
        "Armadillo",
        "ASPack",
        "BitShapeYodas",
        "eXPressor",
        "ezip",
        "FSG",
        "MEW",
        "mPress",
        "NeoLite",
        "Packman",
        "PECompact",
        "PELock",
        "PENinja",
        "Petite",
        "RLPack",
        "telock",
        "Themida",
        "UPX",
        "WinRAR",
        "UPack",
        "WinZip"
            ]
"""
"""
# realistic dataset
file_names = [
        "ASPack",
        "ASProtect",
        "ExeStealth",
        "eXPressor",
        "FSG",
        "InstallShield",
        "MEW",
        "MoleBox",
        "NeoLite",
        "NsPacK",
        "Packman",
        "PECompact",
        "PEPACK",
        "Petite",
        "RLPack",
        "Themida",
        "UPX",
        "UPack",
        "WinRAR",
        "WinZip",
        "Wise"
            ]
"""
#both mixed
file_names = [
     'Armadillo',
     'ASPack',
     'ASProtect',
     'YodaCryptor',
     "CustomAmberPacker",
     "CustomOrigamiPacker",
     "CustomPackerSimple1",
     "CustomPEPacker1",
     "CustomPePacker2",
     "CustomPetoyPacker",
     "CustomSilentPacker",
     "CustomTheArkPacker",
     "CustomUchihaPacker",
     "CustomXorPacker",
     'ExeStealth',
     'eXPressor',
     'ezip',
     'FSG',
     'InstallShield',
     'MEW',
     'MoleBox',
     'mPress',
     'NeoLite',
     'NsPacK',
     'Packman',
     'PECompact',
     'PELock',
     'PENinja',
     'PEPACK',
     'Petite',
     'RLPack',
     'telock',
     'Themida',
     'UPX',
     'WinRAR',
     "UPack",
     'WinZip',
     'Wise',
     "ActiveMARK",
     "FishPE",
     "PCGuard",
     "PESpin",
     "Shrinker",
     "NSIS",
     "InnoSetup",
     "AutoIt"
]

family_names = []
labels = []
scores = []
