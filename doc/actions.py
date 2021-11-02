from nle.nethack import ACTIONS

row = "{asc:<8},  # {esc:<8}{gid:<8}{cls:<24}{nom:16}\n"

table = row.format(gid='gym-id', asc='# ascii', esc='char', cls='class', nom='name')

atoc = {a: j for j, a in enumerate(ACTIONS)}
for a in sorted(atoc, key=int):
    gid, asc = atoc[a], int(a)
    esc = str(chr(asc).encode('unicode-escape'))[2:-1]
    table += row.format(gid=gid, asc=asc, esc=esc, cls=type(a).__name__, nom=a._name_)

print(table)

"""
# ascii  # char    gym-id  class                   name
4,       # \\x04   48      Command                 KICK
20,      # \\x14   82      Command                 TELEPORT
44,      # ,       61      Command                 PICKUP
58,      # :       51      Command                 LOOK
65,      # A       81      Command                 TAKEOFFALL
69,      # E       37      Command                 ENGRAVE
70,      # F       40      Command                 FIGHT
80,      # P       63      Command                 PUTON
81,      # Q       66      Command                 QUIVER
82,      # R       69      Command                 REMOVE
84,      # T       80      Command                 TAKEOFF
87,      # W       91      Command                 WEAR
88,      # X       87      Command                 TWOWEAPON
90,      # Z       28      Command                 CAST
94,      # ^       77      Command                 SEETRAP
97,      # a       24      Command                 APPLY
99,      # c       30      Command                 CLOSE
100,     # d       33      Command                 DROP
101,     # e       35      Command                 EAT
102,     # f       39      Command                 FIRE
111,     # o       57      Command                 OPEN
112,     # p       60      Command                 PAY
113,     # q       64      Command                 QUAFF
114,     # r       67      Command                 READ
115,     # s       75      Command                 SEARCH
116,     # t       83      Command                 THROW
119,     # w       94      Command                 WIELD
120,     # x       79      Command                 SWAP
122,     # z       96      Command                 ZAP
210,     # \\xd2   70      Command                 RIDE
212,     # \\xd4   84      Command                 TIP
227,     # \\xe3   29      Command                 CHAT
228,     # \\xe4   32      Command                 DIP
229,     # \\xe5   38      Command                 ENHANCE
230,     # \\xe6   41      Command                 FORCE
233,     # \\xe9   46      Command                 INVOKE
234,     # \\xea   47      Command                 JUMP
236,     # \\xec   52      Command                 LOOT
237,     # \\xed   53      Command                 MONSTER
239,     # \\xef   56      Command                 OFFER
240,     # \\xf0   62      Command                 PRAY
242,     # \\xf2   71      Command                 RUB
243,     # \\xf3   78      Command                 SIT
244,     # \\xf4   86      Command                 TURN
245,     # \\xf5   88      Command                 UNTRAP
247,     # \\xf7   95      Command                 WIPE

# navigation
46,      # .       18      MiscDirection           WAIT
60,      # <       16      MiscDirection           UP
62,      # >       17      MiscDirection           DOWN
66,      # B       14      CompassDirectionLonger  SW
72,      # H       11      CompassDirectionLonger  W
74,      # J       10      CompassDirectionLonger  S
75,      # K       8       CompassDirectionLonger  N
76,      # L       9       CompassDirectionLonger  E
78,      # N       13      CompassDirectionLonger  SE
85,      # U       12      CompassDirectionLonger  NE
89,      # Y       15      CompassDirectionLonger  NW
98,      # b       6       CompassDirection        SW
104,     # h       3       CompassDirection        W
106,     # j       2       CompassDirection        S
107,     # k       0       CompassDirection        N
108,     # l       1       CompassDirection        E
110,     # n       5       CompassDirection        SE
117,     # u       4       CompassDirection        NE
121,     # y       7       CompassDirection        NW

# gui control
13,      # \\r     19      MiscAction              MORE
15,      # \\x0f   59      Command                 OVERVIEW
18,      # \\x12   68      Command                 REDRAW
27,      # \\x1b   36      Command                 ESC
32,      # \\x20   99      TextCharacters          SPACE
38,      # &       92      Command                 WHATDOES
43,      # +       97      TextCharacters          PLUS
64,      # @       26      Command                 AUTOPICKUP
79,      # O       58      Command                 OPTIONS
83,      # S       74      Command                 SAVE
86,      # V       43      Command                 HISTORY
118,     # v       90      Command                 VERSIONSHORT
191,     # \\xbf   21      Command                 EXTLIST
193,     # \\xc1   23      Command                 ANNOTATE
195,     # \\xc3   31      Command                 CONDUCT
225,     # \\xe1   22      Command                 ADJUST
241,     # \\xf1   65      Command                 QUIT
246,     # \\xf6   89      Command                 VERSION

24,      # \\x18   25      Command                 ATTRIBUTES
35,      # #       20      Command                 EXTCMD
36,      # $       112     TextCharacters          DOLLAR
42,      # *       76      Command                 SEEALL
47,      # /       93      Command                 WHATIS
59,      # ;       42      Command                 GLANCE
67,      # C       27      Command                 CALL
68,      # D       34      Command                 DROPTYPE
71,      # G       73      Command                 RUSH2
73,      # I       45      Command                 INVENTTYPE
77,      # M       55      Command                 MOVEFAR
92,      # \\\\    49      Command                 KNOWN
95,      # _       85      Command                 TRAVEL
96,      # `       50      Command                 KNOWNCLASS

103,     # g       72      Command                 RUSH
105,     # i       44      Command                 INVENTORY
109,     # m       54      Command                 MOVE

# used in certain prompts
34,      # "       101     TextCharacters          QUOTE
39,      # '       100     TextCharacters          APOS
45,      # -       98      TextCharacters          MINUS
48,      # 0       102     TextCharacters          NUM_0
49,      # 1       103     TextCharacters          NUM_1
50,      # 2       104     TextCharacters          NUM_2
51,      # 3       105     TextCharacters          NUM_3
52,      # 4       106     TextCharacters          NUM_4
53,      # 5       107     TextCharacters          NUM_5
54,      # 6       108     TextCharacters          NUM_6
55,      # 7       109     TextCharacters          NUM_7
56,      # 8       110     TextCharacters          NUM_8
57,      # 9       111     TextCharacters          NUM_9
"""
