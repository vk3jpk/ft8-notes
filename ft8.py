#!/usr/bin/env python3
#
# ft8.py - A Python module for working with FT8 signals
#
# Copyright (C) 2019 James Kelly, VK3JPK
#
# SPDX-License-Identifier: GPL-3.0-or-later
#
# Based on protocols and algorithms from WSJT-X, Copyright (C) 2001-2019 Joe Taylor, K1JT
# See https://physics.princeton.edu/pulsar/k1jt/wsjtx.html for further information on WSJT-X

import numpy as np
import scipy.signal

tr_period = 15
start_delay = 0.5

baud_rate = 12000 / 1920
freq_shift = baud_rate
tone_order = 3
tone_count = 1 << tone_order
gaussian_bandwidth = 2.0

msg_bits = 77

# CRC polynomial from WSJT-X lib/crc14.cpp
crc_bits = 14
crc_padded_bits = 96
crc_polynomial = 0x2757

# LDPC generator matrix from WSJT-X lib/ft8/ldpc_174_91_c_generator.f90
generator_hex_strings = [
    "8329ce11bf31eaf509f27fc",
    "761c264e25c259335493132",
    "dc265902fb277c6410a1bdc",
    "1b3f417858cd2dd33ec7f62",
    "09fda4fee04195fd034783a",
    "077cccc11b8873ed5c3d48a",
    "29b62afe3ca036f4fe1a9da",
    "6054faf5f35d96d3b0c8c3e",
    "e20798e4310eed27884ae90",
    "775c9c08e80e26ddae56318",
    "b0b811028c2bf997213487c",
    "18a0c9231fc60adf5c5ea32",
    "76471e8302a0721e01b12b8",
    "ffbccb80ca8341fafb47b2e",
    "66a72a158f9325a2bf67170",
    "c4243689fe85b1c51363a18",
    "0dff739414d1a1b34b1c270",
    "15b48830636c8b99894972e",
    "29a89c0d3de81d665489b0e",
    "4f126f37fa51cbe61bd6b94",
    "99c47239d0d97d3c84e0940",
    "1919b75119765621bb4f1e8",
    "09db12d731faee0b86df6b8",
    "488fc33df43fbdeea4eafb4",
    "827423ee40b675f756eb5fe",
    "abe197c484cb74757144a9a",
    "2b500e4bc0ec5a6d2bdbdd0",
    "c474aa53d70218761669360",
    "8eba1a13db3390bd6718cec",
    "753844673a27782cc42012e",
    "06ff83a145c37035a5c1268",
    "3b37417858cc2dd33ec3f62",
    "9a4a5a28ee17ca9c324842c",
    "bc29f465309c977e89610a4",
    "2663ae6ddf8b5ce2bb29488",
    "46f231efe457034c1814418",
    "3fb2ce85abe9b0c72e06fbe",
    "de87481f282c153971a0a2e",
    "fcd7ccf23c69fa99bba1412",
    "f0261447e9490ca8e474cec",
    "4410115818196f95cdd7012",
    "088fc31df4bfbde2a4eafb4",
    "b8fef1b6307729fb0a078c0",
    "5afea7acccb77bbc9d99a90",
    "49a7016ac653f65ecdc9076",
    "1944d085be4e7da8d6cc7d0",
    "251f62adc4032f0ee714002",
    "56471f8702a0721e00b12b8",
    "2b8e4923f2dd51e2d537fa0",
    "6b550a40a66f4755de95c26",
    "a18ad28d4e27fe92a4f6c84",
    "10c2e586388cb82a3d80758",
    "ef34a41817ee02133db2eb0",
    "7e9c0c54325a9c15836e000",
    "3693e572d1fde4cdf079e86",
    "bfb2cec5abe1b0c72e07fbe",
    "7ee18230c583cccc57d4b08",
    "a066cb2fedafc9f52664126",
    "bb23725abc47cc5f4cc4cd2",
    "ded9dba3bee40c59b5609b4",
    "d9a7016ac653e6decdc9036",
    "9ad46aed5f707f280ab5fc4",
    "e5921c77822587316d7d3c2",
    "4f14da8242a8b86dca73352",
    "8b8b507ad467d4441df770e",
    "22831c9cf1169467ad04b68",
    "213b838fe2ae54c38ee7180",
    "5d926b6dd71f085181a4e12",
    "66ab79d4b29ee6e69509e56",
    "958148682d748a38dd68baa",
    "b8ce020cf069c32a723ab14",
    "f4331d6d461607e95752746",
    "6da23ba424b9596133cf9c8",
    "a636bcbc7b30c5fbeae67fe",
    "5cb0d86a07df654a9089a20",
    "f11f106848780fc9ecdd80a",
    "1fbb5364fb8d2c9d730d5ba",
    "fcb86bc70a50c9d02a5d034",
    "a534433029eac15f322e34c",
    "c989d9c7c3d3b8c55d75130",
    "7bb38b2f0186d46643ae962",
    "2644ebadeb44b9467d1f42c",
    "608cc857594bfbb55d69600" ]
generator_matrix = []
for hex_string in generator_hex_strings:
    generator_matrix.append(int(hex_string, base=16) >> 1)
ldpc_parity_bits = len(generator_matrix)
encoded_bits = msg_bits + crc_bits + ldpc_parity_bits

# LDPC parity check equations from WSJT-X lib/ft8/ldpc_174_91_c_reordered_parity.f90
# Each row in the matrix corresponds to a bit position in the LDPC codeword
# There are 174 rows corresponding to the 174 bits in the LDPC codeword
# The entries in each row are the parity check equations that include the bit
bit_terms = np.array([16,  45,  73,
               25,  51,  62,
               33,  58,  78,
                1,  44,  45,
                2,   7,  61,
                3,   6,  54,
                4,  35,  48,
                5,  13,  21,
                8,  56,  79,
                9,  64,  69,
               10,  19,  66,
               11,  36,  60,
               12,  37,  58,
               14,  32,  43,
               15,  63,  80,
               17,  28,  77,
               18,  74,  83,
               22,  53,  81,
               23,  30,  34,
               24,  31,  40,
               26,  41,  76,
               27,  57,  70,
               29,  49,  65,
                3,  38,  78,
                5,  39,  82,
               46,  50,  73,
               51,  52,  74,
               55,  71,  72,
               44,  67,  72,
               43,  68,  78,
                1,  32,  59,
                2,   6,  71,
                4,  16,  54,
                7,  65,  67,
                8,  30,  42,
                9,  22,  31,
               10,  18,  76,
               11,  23,  82,
               12,  28,  61,
               13,  52,  79,
               14,  50,  51,
               15,  81,  83,
               17,  29,  60,
               19,  33,  64,
               20,  26,  73,
               21,  34,  40,
               24,  27,  77,
               25,  55,  58,
               35,  53,  66,
               36,  48,  68,
               37,  46,  75,
               38,  45,  47,
               39,  57,  69,
               41,  56,  62,
               20,  49,  53,
               46,  52,  63,
               45,  70,  75,
               27,  35,  80,
                1,  15,  30,
                2,  68,  80,
                3,  36,  51,
                4,  28,  51,
                5,  31,  56,
                6,  20,  37,
                7,  40,  82,
                8,  60,  69,
                9,  10,  49,
               11,  44,  57,
               12,  39,  59,
               13,  24,  55,
               14,  21,  65,
               16,  71,  78,
               17,  30,  76,
               18,  25,  80,
               19,  61,  83,
               22,  38,  77,
               23,  41,  50,
                7,  26,  58,
               29,  32,  81,
               33,  40,  73,
               18,  34,  48,
               13,  42,  64,
                5,  26,  43,
               47,  69,  72,
               54,  55,  70,
               45,  62,  68,
               10,  63,  67,
               14,  66,  72,
               22,  60,  74,
               35,  39,  79,
                1,  46,  64,
                1,  24,  66,
                2,   5,  70,
                3,  31,  65,
                4,  49,  58,
                1,   4,   5,
                6,  60,  67,
                7,  32,  75,
                8,  48,  82,
                9,  35,  41,
               10,  39,  62,
               11,  14,  61,
               12,  71,  74,
               13,  23,  78,
               11,  35,  55,
               15,  16,  79,
                7,   9,  16,
               17,  54,  63,
               18,  50,  57,
               19,  30,  47,
               20,  64,  80,
               21,  28,  69,
               22,  25,  43,
               13,  22,  37,
                2,  47,  51,
               23,  54,  74,
               26,  34,  72,
               27,  36,  37,
               21,  36,  63,
               29,  40,  44,
               19,  26,  57,
                3,  46,  82,
               14,  15,  58,
               33,  52,  53,
               30,  43,  52,
                6,   9,  52,
               27,  33,  65,
               25,  69,  73,
               38,  55,  83,
               20,  39,  77,
               18,  29,  56,
               32,  48,  71,
               42,  51,  59,
               28,  44,  79,
               34,  60,  62,
               31,  45,  61,
               46,  68,  77,
                6,  24,  76,
                8,  10,  78,
               40,  41,  70,
               17,  50,  53,
               42,  66,  68,
                4,  22,  72,
               36,  64,  81,
               13,  29,  47,
                2,   8,  81,
               56,  67,  73,
                5,  38,  50,
               12,  38,  64,
               59,  72,  80,
                3,  26,  79,
               45,  76,  81,
                1,  65,  74,
                7,  18,  77,
               11,  56,  59,
               14,  39,  54,
               16,  37,  66,
               10,  28,  55,
               15,  60,  70,
               17,  25,  82,
               20,  30,  31,
               12,  67,  68,
               23,  75,  80,
               27,  32,  62,
               24,  69,  75,
               19,  21,  71, 
               34,  53,  61,
               35,  46,  47,
               33,  59,  76,
               40,  43,  83,
               41,  42,  63,
               49,  75,  83,
               20,  44,  48,
               42,  49,  57]).reshape(encoded_bits, 3)
bit_terms -= 1 # Convert from Fortran to Python indexing

# We need the parity check equations in various forms.
# This function calculates these forms from the cannonical
# representation provided by the bit_terms matrix.
# The check_terms form of the parity equations has one row per
# parity equation with entries in that row specifying the bit
# positions that are part of the corresponding parity check equation.
# The flat versions of bit_terms and check_terms allow numpy to
# be used instead of Python loops in some algorithms.
def _build_parity_equations():
    
    check_terms = []
    check_flat_terms = []
    for i in range(ldpc_parity_bits):
        check_terms.append([])
        check_flat_terms.append([])
         
    for i, row in enumerate(bit_terms):
        for j, t in enumerate(row):
            check_terms[t].append(i)
            check_flat_terms[t].append(i * 3 + j)
    
    bit_flat_terms = []
    for i in range(encoded_bits):
        bit_flat_terms.append([])

    for i, row in enumerate(check_terms):
        for j, t in enumerate(row):
            bit_flat_terms[t].append(i * 7 + j)
        if j == 5:
            check_terms[i].append(-1)
            check_flat_terms[i].append(-1)
 
    return np.array(check_terms), np.array(check_flat_terms), np.array(bit_flat_terms)

check_terms, check_flat_terms, bit_flat_terms = _build_parity_equations()
adjusted_check_terms = check_terms + 1
adjusted_check_flat_terms = check_flat_terms + 3
del _build_parity_equations            
    
encoded_bits = msg_bits + crc_bits + ldpc_parity_bits
encoded_symbols = encoded_bits // tone_order

# Gray map and Costas array from WSJT-X lib/ft8/genft8.f90
gray_map = [0, 1, 3, 2, 5, 6, 4, 7]
costas = [3, 1, 4, 0, 6, 5, 2]
costas_offsets = [0, 36, 72]
costas_order = len(costas)
costas_symbols = costas_order * len(costas_offsets)

symbol_offsets = (list(range(costas_order, costas_offsets[1])) +
                  list(range(costas_offsets[1] + costas_order, costas_offsets[2])))
assert encoded_symbols == len(symbol_offsets)

total_symbols = encoded_symbols + costas_symbols

class Callsign:
    """Callsign management"""
    
    # Character sets for encoding callsigns
    full_charset = ' 0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ/'
    charset_1 = full_charset[:-1]
    charset_2 = full_charset[1:-1]
    charset_3 = full_charset[1:11]
    charset_456 = full_charset[0] + full_charset[11:-1]
    charsets = [charset_1, charset_2, charset_3, charset_456, charset_456, charset_456]
    
    # Values for different types of callsigns
    max_std_calls = 1
    for charset in charsets:
        max_std_calls *= len(charset)
    max_non_std_calls = 1 << 22
    max_tokens = (1 << 28) - max_std_calls - max_non_std_calls
 
    # Tokens
    tokens = ['DE', 'QRZ', 'CQ']

    # Hash table
    hash_table = {}
    hash_table[10] = {}
    hash_table[12] = {}
    hash_table[22] = {}
    
    # Dictionary of all callsigns
    all_calls = {}
    
    def __init__(self, call):
        """Constructor for Callsign class
        
        Raises a ValueError exception if call parameter is not a valid token or callsign
        """
        
        self.call = call
        
        # Check for token
        self.pack28 = Callsign._token_pack(call)
        self.token = self.pack28 != None
        if self.token:
            self.hash = (None, None, None)
            self.standard = False
            return

        # Not a token so we assume it is a callsign             
        # Check callsign is a single word
        words = call.split()
        if (len(words) > 1):
            raise ValueError('Callsign contains spaces')
        
        # Check if we have a standard callsign
        self.pack28 = Callsign._standard_pack(call)
        self.standard = self.pack28 != None
        
        # Check if non-standard callsign is valid
        if (not self.standard): 
            l = len(call)
            if l > 11:
                raise ValueError('Callsign more than 10 characters long')
            elif not all(c in Callsign.full_charset for c in call):
                raise ValueError('Callsign contains invalid characters')
        
        # At this point we have a valid callsign so calculate hash code
        i = 0
        for c in call.ljust(11):
            i = i * len(Callsign.full_charset) + Callsign.full_charset.index(c)       
        hash64 = (i * 47055833459) & (2**64 - 1)
        
        # Calculate hash codes of various lengths and update tables
        hash22 = hash64 >> 42
        hash12 = hash64 >> 52
        hash10 = hash64 >> 54
        self.hash = (hash10, hash12, hash22)
        Callsign.hash_table[10][hash10] = self
        Callsign.hash_table[12][hash12] = self
        Callsign.hash_table[22][hash22] = self
 
        # Update 28 bit packing for non-standard callsigns now we have a hash code
        if not self.standard:
            self.pack28 = hash22 + Callsign.max_tokens
            
        # Add callsign to callsign dictionary
        Callsign.all_calls[call] = self
            
        return
        
    def __str__(self):
        """Returns callsign as a string"""
        
        if (not self.standard) and (not self.token):
            # Non-standard hashed callsign
            return '<' + self.call + '>'
        else:
            # Token or standard callsign
            return self.call
        
    def isToken(self):
        """Returns true if callsign is a token"""
        
        return self.token
    
    def isStandard(self):
        """Returns true if callsign is a non-hashed standard callsign"""
        
        return self.standard
    
    @classmethod
    def getHash(cls, code, length=22):
        """Fetch callsign corresponding to hash code of specified length"""
        
        if code in cls.hash_table[length]:
            return cls.hash_table[length][code]
        else:
            return None
    
    @classmethod
    def unpack28(cls, i):
        """Attempt to unpack 28-bit packed callsign"""
        
        # Check for token
        if i < cls.max_tokens:
            # DE, QRZ, CQ
            if i < len(cls.tokens):
                return Callsign(cls.tokens[i])
            
            # CQ nnn
            elif i < len(cls.tokens) + 1000:
                return Callsign('CQ ' + str(i - len(cls.tokens)))

            # CQ aaaa
            elif i < len(cls.tokens) + 1000 + len(cls.charset_456)**4:
                i = i - len(cls.tokens) - 1000
                c = []
                s = len(cls.charset_456)
                for j in range(4):
                    c.insert(0, cls.charset_456[i % s])
                    i = i // s
                c = "".join(c).strip()
                return Callsign('CQ ' + c)
            
            # Unknown token
            return None
        
        # Check for hashed non-standard callsign
        if i < cls.max_tokens + cls.max_non_std_calls:
            hash22 = i - cls.max_tokens
            if hash22 in cls.hash_table[22]:
                return cls.hash_table[22][hash22]
            else:
                # Consider unknown callsign instead
                return None

        # Unpack standard callsign
        i = i - cls.max_tokens - cls.max_non_std_calls
        c = []
        for charset in reversed(cls.charsets):
            size = len(charset)
            c.insert(0, charset[i % size])
            i = i // size
        c = ''.join(c).strip()
        
        # Handle special cases
        if c[0] == 'Q':
            c = '3X' + c[1:]
        elif c[:3] == '3D0':
            c = '3DA0' + c[3:]
            
        # Check if callsign object already exists for this callsign
        if c in cls.all_calls:
            return cls.all_calls[c]
        else:
            return Callsign(c)
        
    @classmethod
    def _standard_pack(cls, call):
        """Helper function to pack standard callsigns
        
        Returns a 28 bit packing of the callsign if it is a standard callsign, otherwise None.
        """
        
        # First check for some special cases that would otherwise be non-standard
        l = len(call)
        
        # Replace 3DA0... with 3D0... for Swaziland callsigns
        if l > 4 and call[:4] == '3DA0':
            call = '3D0' + call[4:]
            l -= 1
            
        # Replace 3X.... with Q.... for Guinea callsigns
        elif l > 2 and call[:2] == '3X' and call[2].isalpha():
            call = 'Q' + call[2:]
            l -= 1
            
        # Find last digit in callsign
        last_digit = None
        for i, c in enumerate(call):
            if c.isdigit():
                last_digit = i
            
        # If last digit is second character of callsign, prepend a space
        if last_digit == 1:        
            call = ' ' + call
            l += 1

        # Check if callsign can be packed with standard callsign packing method
        if l > 6 or l < 3 or not all(c in charset for charset, c in zip(cls.charsets, call)):
            return None
        
        # Perform standard callsign packing
        i = 0
        for charset, c in zip(cls.charsets, call.ljust(len(cls.charsets))):
            i = i * len(charset) + charset.index(c)       
        i = i + cls.max_tokens + (1 << 22)
        return i

    @classmethod
    def _token_pack(cls, token):
        """Helper function to pack tokens
        
        Returns a 28 bit packing of the token if it is a valid token, otherwise None.
        """
        
        words = token.split()       
        if words[0] not in cls.tokens:
            return None
            
        l = len(words)
        
        # DE, QRZ, CQ
        if (l == 1):
            return cls.tokens.index(token)
            
        if words[0] != 'CQ' or l != 2:
            return None
        
        # CQ nnn
        if words[1].isdecimal():
            v = int(words[1])
            if v < 0 or v > 999:
                return None
            return v + len(cls.tokens)
        
        l = len(words[1])
        if l > 4 or not all(c in cls.charset_456 for c in words[1]):
            return None
        
        # CQ aaaa
        v = 0
        for c in words[1].ljust(4):
            v = v * len(cls.charset_456) + cls.charset_456.index(c)
        return v + len(cls.tokens) + 1000
        
class Report:
    """Report superclass"""
    
    max_grid_4 = 180*180
    max_serial = 4095
    types = ['location', 'signal', 'other']
    other_reports = ['', 'RRR', 'RR73', '73']
    charset_12 = "ABCDEFGHIJKLMNOPQR"
    charset_34 = "0123456789"
    charset_56 = "abcdefghijklmnopqrstuvwx"
    charsets = [charset_12, charset_12, charset_34, charset_34, charset_56, charset_56]
    
    def __init__(self, value):
        self.value = value
        self.pack5 = None
        self.pack12 = None
        self.pack15 = None
        self.pack25 = None
   
    def __str__(self):
        return str(self.value)
    
    def __eq__(self, other):
        if isinstance(other, Report):
            return self.value == other.value
        return False
    
    @classmethod
    def unpack15(cls, i):       
        if i <= cls.max_grid_4:
            c = []
            for charset in reversed(cls.charsets[:4]):
                size = len(charset)
                c.insert(0, charset[i % size])
                i = i // size          
            return LocationReport(''.join(c))
        
        i -= cls.max_grid_4
        
        if i <= 4:
            return OtherReport(cls.other_reports[i-1])
        
        if i <= 65:
            return SignalReport(i - 35)
        
        return None
    
class LocationReport(Report):
    """Location report"""
    def __init__(self, value):
        super().__init__(value)
        l = len(value) 
        if not l in [4, 6] or not all(c in charset for charset, c in zip(Report.charsets, value)):
            raise ValueError()
                
        i = 0
        for charset, c in zip(Report.charsets, value):
            i = i * len(charset) + charset.index(c)
        if l == 4:
            self.pack15 = i
        else:
            self.pack25 = i
            
class SignalReport(Report):
    """Signal report"""
    def __init__(self, value):
        super().__init__(value)
        if not isinstance(value, int) or value < -30 or value > 30:
            raise ValueError()            
        self.pack15 = Report.max_grid_4 + 35 + value
        self.pack5 = (value + 30) / 2
        
    def __str__(self):
        return "{:=+03d}".format(self.value)

class OtherReport(Report):
    """Other report"""
    
    def __init__(self, value):
        super().__init__(value)
        if not value in Report.other_reports:
            raise ValueError()                
        self.pack15 = Report.max_grid_4 + Report.other_reports.index(value) + 1
        
class SerialReport(Report):
    """Serial number report"""
    
    def __init__(self, value):
        super().__init__(value)
        if not isinstance(value, int) or value > Report.max_serial:
            raise ValueError()
        self.pack12 = value
        
class RTTYSignal:
    """ARRL RTTY Contest Signal"""
    
    def __init__(self, value):
        if value < 529 or value > 599 or (value % 10) != 9:
            raise ValueError()
        self.value = value
        self.pack3 = (self.value - 509) // 10 - 2
        
    def __str__(self):
        return str(self.value)
    
    @classmethod
    def unpack3(cls, bits):
        return cls((bits + 2) * 10 + 509)

class RTTYState:
    """ARRL RTTY Contest State"""
    
    states = ["AL", "AK", "AZ", "AR", "CA", "CO", "CT", "DE", "FL", "GA",
              "HI", "ID", "IL", "IN", "IA", "KS", "KY", "LA", "ME", "MD",
              "MA", "MI", "MN", "MS", "MO", "MT", "NE", "NV", "NH", "NJ",
              "NM", "NY", "NC", "ND", "OH", "OK", "OR", "PA", "RI", "SC",
              "SD", "TN", "TX", "UT", "VT", "VA", "WA", "WV", "WI", "WY",
              "NB", "NS", "QC", "ON", "MB", "SK", "AB", "BC", "NWT", "NF",
              "LB", "NU", "YT", "PEI", "DC"]
    
    state_offset = 8001
    
    def __init__(self, value):
        self.value = value
        if value.isnumeric() and value < self.state_offset:
            self.pack13 = value
        elif value in self.states:
            self.pack13 = self.state_offset + self.states.index(value)
        else:
            raise ValueError()
            
    def __str__(self):
        if self.pack13 < self.state_offset:
            return "{:04d}".format(self.value)
        else:
            return self.value
        
    @classmethod
    def unpack13(cls, bits):
        if bits < cls.state_offset:
            return cls(bits)
        else:
            return cls(cls.states[bits - cls.state_offset])
    
class Message:
    
    class MessageError(Exception):
        """Superclass for errors generated when creating messages"""
        
    class CRCError(MessageError):
        """CRC bits did not match message bits"""
        
    class UnsupportedError(MessageError):
        """Message type or subtype is unsupported"""
        
    def __eq__(self, other):
        return str(self) == str(other)
    
    @staticmethod
    def _crc(msg, chk):
        """Calculate FT8 CRC."""
        
        # msg is the 77 bit message stored in a Python 3 integer
        # chk is a 14 bit CRC stored in a Python 3 integer
        # Set chk to zero if calculating CRC or the received CRC to check CRC
        # Returns calculated CRC or zero if provided CRC is correct

        # Pad msg with 96-77 = 19 zeros to create 96 bit number and add checksum
        # Padding must be at least CRC size bits for the CRC algorithm to work
        # FT8 designers rounded up padding to make the padded message a whole number of bytes
        remainder = (msg << (crc_padded_bits - msg_bits)) | chk
    
        # Mask starts at most significant bit
        mask = 1 << (crc_padded_bits - 1)
    
        # Align most significant bit of polynomial with mask
        # Note that the specified polynomial does not include the most significant bit
        # So we must add the most significant bit before we align the polynomial
        shifted_poly = ((1 << crc_bits) | crc_polynomial) << (crc_padded_bits - crc_bits - 1)
    
        # Polynomial long division modulo 2
        # We are only interested in the remainder so we don't bother calculating the quotient
        for i in range(crc_padded_bits - crc_bits):
            if (remainder & mask):
                remainder ^= shifted_poly
            mask >>= 1
            shifted_poly >>= 1
            
        return remainder      

    def encode(self):
        """Encode message into 79 symbols."""
        
        # Symbols are returned in a Python list of integers
        # Applies CRC, LDPC, Gray codes and adds Costas arrays

        # Calculate 14-bit CRC and append to the packed message to get
        # 77 + 14 = 91 bit packed message with CRC
        msg_crc = self.pack77 << 14 | Message._crc(self.pack77, 0)

        # Generate 83 bits of parity by multiplying generator matrix by message bits.
        # We use the logical AND operator to perform modulo 2 multiplication.
        # We then count the set bits to find the sum modulo 2.
        # While this is concise it is probably not that efficient
        parity = 0
        for row in generator_matrix:
            parity = parity << 1
            parity = parity | (bin(row & msg_crc).count('1') % 2)
    
        # Construct 174 bit codeword by combining 91 bit message and 83 bits of parity
        codeword = (msg_crc << 83) | parity

        # Split 174 bit codeword into 58 x 3 bit symbols and apply gray code
        msg_symbols = []
        mask = (1 << 3) - 1
        for i in range(174 // 3):
            msg_symbols.insert(0, gray_map[codeword & mask])
            codeword = codeword >> 3

        # Add 3 x 7 symbol Costas arrays to the 58 symbol encoded message to get 79 symbols
        symbols = costas + msg_symbols[:encoded_symbols // 2] + costas + msg_symbols[encoded_symbols // 2:] + costas
    
        return symbols
    
    @classmethod
    def unpack77(cls, bits):
        """Attempt to unpack message from bits."""
        
        msg_type = bits & 7
        if msg_type == 0:
            msg_subtype = (bits >> 3) & 7
            if msg_subtype == TextMessage.msg_subtype:
                return TextMessage.unpack77(bits)
            elif msg_subtype == TelemetryMessage.msg_subtype:
                return TelemetryMessage.unpack77(bits)
        elif msg_type == StandardMessage.msg_type:
            return StandardMessage.unpack77(bits)
        elif msg_type == EUVHFMessage.msg_type:
            return EUVHFMessage.unpack77(bits)
        elif msg_type == RTTYMessage.msg_type:
            return RTTYMessage.unpack77(bits)
        else:
            raise cls.UnsupportedError()
    
    @classmethod
    def unpack91(cls, bits):
        """Unpack message from bits after checking CRC appended to bits."""
        
        crc = bits & ((1 << crc_bits) - 1)
        msg = bits >> crc_bits
        if cls._crc(msg, crc):
            raise cls.CRCError()
        return cls.unpack77(msg)

class TextMessage(Message):
    """Class to represent text messages"""
    
    # Text messages use a limited character set
    charset = ' 0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ+=./?'
    charset_size = len(charset)
    msg_type = 0
    msg_subtype = 0
    
    # Text messages can be up to 13 characters long
    max_msg_size = 13
       
    def __init__(self, text):
        self.text = text.strip()

        if len(text) > TextMessage.max_msg_size or not all(c in TextMessage.charset for c in text):
            raise ValueError()
       
        i = 0
        for c in self.text:
            i *= TextMessage.charset_size
            i += TextMessage.charset.index(c)
        self.pack77 = (i << 6) | (TextMessage.msg_subtype << 3) | TextMessage.msg_type
        
    def __str__(self):
        return 'TextMessage("{}")'.format(self.text)

    @classmethod
    def unpack77(cls, bits):
        """Unpack text message from bits"""
        
        i = bits >> 6
        c = []
        for j in range(cls.max_msg_size):
            c.insert(0, cls.charset[i % cls.charset_size])
            i = i // cls.charset_size
        return cls(''.join(c).strip())
    
class TelemetryMessage(Message):
    """A class to represent telemetry messages"""
    
    msg_type = 0
    msg_subtype = 5
    
    def __init__(self, bits):
        self.bits = bits
        self.pack77 = (bits << 6) | (TelemetryMessage.msg_subtype << 3) | TelemetryMessage.msg_type
   
    def __str__(self):
        return "TelemetryMessage({})".format(hex(self.bits))
    
    @classmethod
    def unpack77(cls, bits):
        """Unpack telemetry message from bits"""
        
        return cls(bits >> 6)
    
class StandardMessage(Message):
    """A class to represent standard messages"""
    
    # Standard messages contain 2 callsigns, a report and an optional roger
    # Format: 28 bits for first callsign, 1 bit for first callsign rover status
    #         28 bits for second callsign, 1 bit for second callsign rover status
    #         1 bit for roger, 15 bits for a report and 3 bits for message type
    msg_type = 1
    field_widths = [28, 1, 28, 1, 1, 15, 3]
   
    def __init__(self, call_1, call_2, report, roger=False, rover_1=False, rover_2=False):
        self.call_1 = call_1
        self.rover_1 = rover_1
        self.call_2 = call_2
        self.rover_2 = rover_2
        self.roger = roger
        self.report = report
        self.fields = [self.call_1, self.rover_1, self.call_2, self.rover_2, self.roger, self.report]
        field_bits = [call_1.pack28, rover_1, call_2.pack28, rover_2, roger,
                      report.pack15, self.msg_type]
        self.pack77 = 0
        for w, f in zip(self.field_widths, field_bits):
            self.pack77 = (self.pack77 << w) | int(f)
        return
    
    def __str__(self):
        if self.msg_type == 1:
            r = '/R'
        elif self.msg_type == 2:
            r = '/P'
            
        s = str(self.call_1)
        if self.rover_1:
            s += r
        s += ' ' + str(self.call_2)
        if self.rover_2:
            s += r
        s += ' '
        if self.fields[4]:
            s += 'R'
        s += str(self.report)
        return '{}({})'.format(self.__class__.__name__, s)
    
    @classmethod
    def unpack77(cls, bits):
        """Unpack standard message from bits"""
        
        f = []
        for w in reversed(cls.field_widths):
            f.insert(0, bits & ((1 << w) - 1))
            bits >>= w
        return cls(Callsign.unpack28(f[0]), Callsign.unpack28(f[2]), Report.unpack15(f[5]), 
                   f[4] != 0, f[1] != 0, f[3] != 0)
    
class EUVHFMessage(StandardMessage):
    """A class to represent EU VHF Contest messages"""
    
    msg_type = 2
    
    def __init__(self, call_1, call_2, report, roger=False, portable_1=False, portable_2=False):
        super().__init__(call_1, call_2, report, roger, portable_1, portable_2)

class RTTYMessage(Message):
    """A class to represent ARRL RTTY contest messages"""

    msg_type = 3
    field_widths = [1, 28, 28, 1, 3, 13, 3]

    def __init__(self, call_1, call_2, roger, signal, state, thank_you=False):
        if not isinstance(call_1, Callsign):
            raise ValueError()
        self.call_1 = call_1
        
        if not isinstance(call_2, Callsign):
            raise ValueError()
        self.call_2 = call_2
        
        if not isinstance(signal, RTTYSignal):
            raise ValueError()
        self.signal = signal

        if not isinstance(roger, bool):
            raise ValueError()
        self.roger = roger
        
        if not isinstance(state, RTTYState):
            raise ValueError()
        self.state = state
        
        if not isinstance(thank_you, bool):
            raise ValueError()
        self.thank_you = thank_you
        
        field_bits = [thank_you, call_1.pack28, call_2.pack28, roger, signal.pack3, state.pack13, self.msg_type]
        self.pack77 = 0
        for w, f in zip(self.field_widths, field_bits):
            self.pack77 = (self.pack77 << w) | int(f)
        return
    
    def __str__(self):
        if self.thank_you:
            s = 'TU; ' + str(self.call_1)
        else:
            s = str(self.call_1)
        s += ' ' + str(self.call_2)
        if self.roger:
            s += ' R'
        s += ' ' + str(self.signal)
        s += ' ' + str(self.state)
        return "RTTYMessage({})".format(s)
    
    @classmethod
    def unpack77(cls, bits):
        
        f = []
        for w in reversed(cls.field_widths):
            f.insert(0, bits & ((1 << w) - 1))
            bits >>= w
        return cls(Callsign.unpack28(f[1]), Callsign.unpack28(f[2]), f[3] != 0, RTTYSignal.unpack3(f[4]), 
                   RTTYState.unpack13(f[5]), f[0] != 0)

class Candidate:
    """A simple class for holding candidate signal information."""
    
    def __init__(self, freq, offset, sync):
        self.freq = freq
        self.offset = offset
        self.sync = sync
    
class SpectralAnalysis:
    """A class for performing spectral analysis to identify FT8 signals."""

    # Tuning constants for the spectrogram calculation
    spectrogram_bins_per_tone = 2 # Frequency resolution oversampling
    spectrogram_steps_per_symbol = 4 # Time resolution oversampling
    spectrogram_scale_factor = 300.0 # An arbitrary scale factor applied to samples prior to the DFT

    # Tuning contants for correlation
    low_frequency = 100 # Lowest frequency to search for signals
    high_frequency = 2700 # Highest frequency to search for signals
    offset_bound = 2.5 # We will search time offsets from our local time in the range +/- 2.5s
    
    # Tuning constants for candidate identification
    max_candidates = 300 # Maximum number of candidate signals to consider
    normalization_percentile = 60 # Correlation SNR percentile to use for normalization
    candidate_threshold = 1.5 # Minimum acceptable correlation after normalization
    
    # Constants derived from tuning constants
    spectrogram_time_step = 1 / baud_rate / spectrogram_steps_per_symbol # Time step of spectrogram in seconds
    spectrogram_bin_width = freq_shift / spectrogram_bins_per_tone # Frequency bin width of spectrogram in Hertz

    spectrum_bin_width = 1 / (tr_period + 1)

    def __init__(self, samples, sample_rate):

        # We don't support sample rate conversion yet
        if (sample_rate != 12000):
            raise ValueError()
        self.sample_rate = sample_rate
        self.samples = samples
        self.spectrogram = SpectralAnalysis._calculate_spectrogram(samples, sample_rate)
        self.baseline = SpectralAnalysis._calculate_baseline(self.spectrogram)
        self.complex_spectrum = SpectralAnalysis._calculate_complex_spectrum(samples, sample_rate)
        self.snr_matrix = SpectralAnalysis._correlate_costas_arrays(self.spectrogram)
        self.candidate_list = SpectralAnalysis._find_candidates(self.snr_matrix)
        
    def noise_baseline(self, freq):
        """Get noise PSD for a frequency or array of frequencies."""
        
        index = np.rint((freq - self.low_frequency) / freq_shift * self.spectrogram_bins_per_tone).astype(np.int)
        psd = np.power(10.0, 0.1 * self.baseline[index])
        return psd
    
    @staticmethod
    def _calculate_spectrogram(samples, sample_rate):
        """Calculate the spectrogram."""
        
        samples_per_symbol = int(sample_rate / freq_shift)
        dft_length = samples_per_symbol * SpectralAnalysis.spectrogram_bins_per_tone
        overlap_samples = samples_per_symbol - samples_per_symbol // SpectralAnalysis.spectrogram_steps_per_symbol
  
        f, t, s = scipy.signal.spectrogram(samples, sample_rate, 'boxcar', samples_per_symbol, overlap_samples, dft_length, False, True)
       
        return np.transpose(s)
    
    @staticmethod
    def _calculate_baseline(spectrogram):
        """Calculate spectrum baseline from spectrogram."""
        
        # Compute mean PSD in dB for each frequency in the spectrogram
        # We are only interested in a subset of frequencies
        low_freq_index = int(SpectralAnalysis.low_frequency / freq_shift * SpectralAnalysis.spectrogram_bins_per_tone)
        high_freq_index = int(SpectralAnalysis.high_frequency / freq_shift * SpectralAnalysis.spectrogram_bins_per_tone)
        db = 10.0 * np.log10(np.mean(spectrogram[:, low_freq_index : high_freq_index], axis=0))
        
        # Divide the PSD into 10 segments
        indexes = np.arange(high_freq_index - low_freq_index)
        segments = np.array_split(indexes, 10)
        
        # Find indexes and values where PSD is less than 10th percentile in corresponding segment
        base_indexes = []
        base_values = []
        for segment in segments:
            segment_values = db[segment] # Get values for this segment
            base = np.percentile(segment_values, 10) # Find 10th percentile
            selector = segment_values <= base # Find where less than 10th percentile
            
            base_indexes.append(segment[selector]) # Get indexes where less than 10th percentile
            base_values.append(segment_values[selector]) # Get values where less than 10th percentile
        
        base_index = np.concatenate(base_indexes) # Combine indexes where less than 10th percentile
        base_value = np.concatenate(base_values) # Combine values where less than 10th percentile
            
        # Fit to polynomial of degree 4 which has 5 terms
        midpoint = (high_freq_index - low_freq_index) // 2
        base_index -= midpoint
        poly = np.polynomial.polynomial.Polynomial(np.polynomial.polynomial.polyfit(base_index, base_value, 4))
        
        # Evaluate polynomial at all frequencies of interest to get smooth noise baseline
        baseline = poly(indexes - midpoint)
        
        return baseline
    
    @staticmethod
    def _calculate_complex_spectrum(samples, sample_rate):
        """Utility function to calculate the complex spectrum."""
        
        # Choose DFT length to get our desired frequency domain resolution
        dft_length = (tr_period + 1) * sample_rate
        
        # Compute the DFT using the FFT algorithm
        complex_spectrum = np.fft.rfft(samples, dft_length)
        
        return complex_spectrum  

    @staticmethod
    def _correlate_costas_arrays(spectrogram):
        """Cross-correlation of Costas arrays with a spectrogram."""
        
        # Based on code from WSJT-X lib/ft8/sync8.f90
        # Loops have been reordered to push most of the work into scipy.correlate

        symbol_step = SpectralAnalysis.spectrogram_steps_per_symbol
        tone_step = SpectralAnalysis.spectrogram_bins_per_tone
        
        # Build signal correlation matrix from Costas array
        t_matrix = np.zeros((costas_order * symbol_step, costas_order * tone_step))
        for symbol, tone in enumerate(costas):
            t_matrix[symbol * symbol_step, tone * tone_step] = 1.0
    
        # Build signal + noise correlation matrix
        t0_matrix = np.zeros(t_matrix.shape)
        t0_matrix[::symbol_step, ::tone_step] = 1.0
    
        # Limit our search to a subset of frequencies
        low_freq_index = int(SpectralAnalysis.low_frequency / freq_shift * tone_step)
        high_freq_index = int(SpectralAnalysis.high_frequency / freq_shift * tone_step)
        high_freq_index += t_matrix.shape[1] - 1
        freq_slice = slice(low_freq_index, high_freq_index)
        
        # Limit our search to a subset of time offsets
        low_time_index = int((start_delay - SpectralAnalysis.offset_bound) * baud_rate * symbol_step)
        high_time_index = int((start_delay + SpectralAnalysis.offset_bound) * baud_rate * symbol_step)
        time_steps = high_time_index - low_time_index
        high_time_index += t_matrix.shape[0] - 1
   
        # Correlate signal and signal + noise for each of the three Costas array offsets
        t = [] # List of signal correlations
        t0 = [] # List of signal + noise correlations
        for offset in costas_offsets:       
            # Adjust time range to search for Costas array at this offset
            low = max(0, low_time_index + offset * symbol_step)
            high = min(spectrogram.shape[0], high_time_index + offset * symbol_step)
            time_slice = slice(low, high)
        
            # Do correlations for signal and signal + noise for Costas array at this offset
            t.append(scipy.signal.correlate(spectrogram[time_slice, freq_slice], t_matrix, mode='valid'))
            t0.append(scipy.signal.correlate(spectrogram[time_slice, freq_slice], t0_matrix, mode='valid'))   

        # Pad first and last correlation results as they might be truncated in time
        time_pad = time_steps - t[0].shape[0]
        t[0] = np.pad(t[0], ((time_pad, 0), (0, 0)), mode='constant')
        t0[0] = np.pad(t0[0], ((time_pad, 0), (0, 0)), mode='constant')
        time_pad = time_steps - t[-1].shape[0]
        t[-1] = np.pad(t[-1], ((0, time_pad), (0, 0)), mode='constant')
        t0[-1] = np.pad(t0[-1], ((0, time_pad), (0, 0)), mode='constant')

        # Calculate signal to noise ratio excluding 1st Costas array
        t_sum = t[1] + t[2]
        t0_sum = t0[1] + t0[2]
        sync_bc = (costas_order - 1) * t_sum / (t0_sum - t_sum)
    
        # Calculate signal to noise ratio including 1st Costas array
        t_sum += t[0]
        t0_sum += t0[0]
        sync_abc = (costas_order - 1) * t_sum / (t0_sum - t_sum)
    
        # Pick highest signal to noise ratio
        return np.fmax(sync_abc, sync_bc)

    @staticmethod
    def _find_candidates(snr_matrix):
        """Find candidate FT8 signals based on the Costas array cross-correlation."""
        
        # Based on code from WSJT-X lib/ft8/sync8.f90

        # For each frequency find the time step with the largest correlation SNR and its value
        peak_time_step = np.argmax(snr_matrix, axis=0)
        peak_snr = snr_matrix[peak_time_step, np.arange(snr_matrix.shape[1])]

        # Find frequency to use as the normalization value
        sorted_bins = np.argsort(peak_snr)
        normalization_bin = sorted_bins[int(SpectralAnalysis.normalization_percentile * sorted_bins.size / 100)]

        # Build candidate list of frequencies by choosing those with largest correlation SNR
        candidate_bins = sorted_bins[-SpectralAnalysis.max_candidates:]
        
        # Exclude candidates that do not exceed the threshold relative to the normalization value
        min_snr = SpectralAnalysis.candidate_threshold * peak_snr[normalization_bin]
        candidate_bins = candidate_bins[peak_snr[candidate_bins] > min_snr]
        
        # Eliminate candidates that are next to a stronger candidate
        # Note that WSJT-X does this a little differently
        duplicates = np.amax(np.tril(np.abs(candidate_bins - candidate_bins[np.newaxis].T) == 1), axis=0)
        candidate_bins = candidate_bins[np.logical_not(duplicates)]
        
        # Reorder by frequency
        candidate_bins = np.sort(candidate_bins)
        
        # Calculate frequencies and time offsets for the candidates and combine with correlation SNR
        freqs = candidate_bins * SpectralAnalysis.spectrogram_bin_width + SpectralAnalysis.low_frequency
        offsets = peak_time_step[candidate_bins] * SpectralAnalysis.spectrogram_time_step - SpectralAnalysis.offset_bound
        candidates = np.column_stack((freqs, offsets, peak_snr[candidate_bins]))
        
        return candidates

class FSK:
    
    def __init__(self, symbols, sample_rate, samples_per_symbol):
        
        # Generate a baseband MFSK reference signal suitable for cross-correlation
        # Assumes baud rate (samples_rate / samples_per_symbol) = tone separation
        # Locates lowest tone corresponding to symbol value 0 at 0 Hz
        signal = np.empty((len(symbols), samples_per_symbol), dtype=np.complex128)
        phi = 0.0
        for i, symbol in enumerate(symbols):
            delta_phi = np.pi * 2.0 * symbol / samples_per_symbol
            for k in range(samples_per_symbol):
                phi += delta_phi
                phi %= np.pi * 2.0
                signal[i, k] = np.cos(phi) + 1.0j * np.sin(phi)

        self.signal = np.reshape(signal, len(symbols) * samples_per_symbol)
    
class Signal:
    """A class for performing analysis of candidate FT8 signals."""
    
    # Tunable constants
    sample_rate = 200   
    samples_per_symbol = int(sample_rate / baud_rate)
    demap_max_symbols = 3
    decoder_max_iterations = 200
     
    # Precalculate signals used to refine frequency estimate and correct for the revised estimate
    freq_step = 0.5
    correction_bound = 5
    symbol_correction_range = np.arange(-correction_bound, correction_bound + 1) * freq_step / freq_shift
    costas_conjugates = np.empty((len(symbol_correction_range), samples_per_symbol * costas_order), dtype=np.complex128)
    correction_signals = np.empty((len(symbol_correction_range), samples_per_symbol * total_symbols), dtype=np.complex128)
        
    for i, correction in enumerate(symbol_correction_range):
        symbols = np.array(costas) + correction
        costas_conjugates[i] = np.conjugate(FSK(symbols, sample_rate, samples_per_symbol).signal)
        symbols = np.full(total_symbols, -correction)
        correction_signals[i] = FSK(symbols, sample_rate, samples_per_symbol).signal

    del symbols
    del i
    del correction
    
    # Precalculate arrays used by demapper to select permutations that have a specific bit set
    demap_max_bits = demap_max_symbols * tone_order
    demap_max_permutations = 1 << demap_max_bits
    demap_one = np.zeros((demap_max_bits, demap_max_permutations), dtype='bool')
    for i in range(demap_max_bits):
        for j in range(demap_max_permutations):
            if j & 1 << i:
                demap_one[i, j] = True
    demap_not_one = np.logical_not(demap_one)
    del i
    del j

    def __init__(self, candidate, analysis):
        self.spectrum, self.freq, self.offset, self.baseband, self.sync = Signal._refine_estimates(candidate, analysis)
        self.msg, self.snr, self.obs, self.llr, self.codeword = Signal._detect(self.baseband, self.freq, analysis)
    
    def __str__(self):
        return "{:4.1f} {:4.1f} {:4.0f} {}".format(self.snr, self.offset, self.freq, self.msg)
    
    @staticmethod
    def _extract_baseband(candidate, analysis):
        """Extract and downconvert a single candiate FT8 signal."""
        
        # Based on code from WSJT-X lib/ft8/ft8_downsample.f90

        # Calculate some constants
        bin_width = SpectralAnalysis.spectrum_bin_width
        f = candidate[0] # First element of candidate is the frequency
        start_bin = int(f / bin_width) # DFT bin containing the lowest tone
        upper_bin = int((f + 8.5 * freq_shift) / bin_width) # Highest bin we are interested in
        lower_bin = int((f - 1.5 * freq_shift) / bin_width) # Lowest bin we are interested in
        bin_range = upper_bin - lower_bin + 1 # Number of bins to extract from the DFT spectrum
        padded_bin_range = int(Signal.sample_rate / bin_width) # Number of bins needed to get desired baseband
   
        # Extract the candidate FT8 signal from the spectrum, apply Tukey window, down convert to baseband
        s = np.zeros(padded_bin_range, dtype=analysis.complex_spectrum.dtype) # Output spectrum
        window = scipy.signal.windows.tukey(bin_range, 1/5) # Central 4/5th is wide enough for 8 tones
        s[0: bin_range] = analysis.complex_spectrum[lower_bin : upper_bin + 1] * window  
        s = np.roll(s, lower_bin - start_bin) # Shift lowest tone down to zero frequency
        
        # Calculate scale factor
        scale = 1.0 / np.sqrt((analysis.complex_spectrum.size - 1) * 2 * s.size)

        # Calculate inverse fft (complex to complex)
        #return s, np.fft.ifft(s) * scale
        return s, np.fft.ifft(s) / 32
 
    @staticmethod
    def _correlate(baseband, costas_conjugate):
        """Compute time domain cross-correlation between a baseband signal and the Costas arrays."""
        
        # Based on code from WSJT-X lib/ft8/sync8d.f90
        # Loops have been reordered to exploit numpy
        
        t = 0.0       
        for offset in costas_offsets:
            p = baseband[offset: offset + costas_order * Signal.samples_per_symbol] * costas_conjugate
            s = np.sum(np.reshape(p, (costas_order, Signal.samples_per_symbol)), axis=1)
            t += np.sum(np.abs(s) ** 2)
                                                        
        return t

    @staticmethod
    def _refine_estimates(candidate, analysis):
        """Refine estimates of frequency and time offset for a candidate."""
        
        # Based on code from WSJT-X lib/ft8/ft8b.f90
        
        spectrum, baseband = Signal._extract_baseband(candidate, analysis)      
        freq = candidate[0]
        offset = candidate[1]  
        
        start_sample = int((start_delay + offset) * Signal.sample_rate)
        total_samples = total_symbols * Signal.samples_per_symbol
        search_bound = Signal.samples_per_symbol//4 # Search +/- quarter symbol
        
        # Pad baseband if we have a large negative time offset
        lowest_sample = start_sample - search_bound    
        if lowest_sample < 0:
            baseband = np.pad(baseband, (-lowest_sample, 0), 'constant')
            start_sample = search_bound
         
        # Pad baseband if we have a large positive time offset
        highest_sample = start_sample + search_bound + 1 + total_samples
        if highest_sample > baseband.size:
            baseband = np.pad(baseband, (0, highest_sample - baseband.size), 'constant')
           
        # Search for best time offset
        search_range = range(start_sample - search_bound, start_sample + search_bound + 1)
        smax = -1.0
        for sample in search_range:
            s = Signal._correlate(baseband[sample: sample + total_samples],
                                  Signal.costas_conjugates[Signal.correction_bound])
            if s > smax:
                smax = s
                best_sample = sample

        # Calculate improved estimate of time offset and truncate baseband to match
        best_offset = (best_sample / Signal.sample_rate) - start_delay
        truncated_baseband = baseband[best_sample: best_sample + total_samples]
    
        # Estimate required frequency correction using various frequency shifted Costas arrays
        smax = -1.0
        for i, correction in enumerate(Signal.symbol_correction_range):
            s = Signal._correlate(truncated_baseband, Signal.costas_conjugates[i])
            if s > smax:
                smax = s
                best_i = i
                best_correction = correction

        # Calculate improved estiamte of frequency and apply the frequency correction to the baseband signal
        corrected_baseband = truncated_baseband * Signal.correction_signals[best_i]
        corrected_frequency = freq + best_correction * freq_shift
    
        # Calculate final correlation after frequency correction has been applied
        #s = Signal._correlate(corrected_baseband, Signal.costas_signals[Signal.correction_bound])
        
        return spectrum, corrected_frequency, best_offset, corrected_baseband, smax

    @staticmethod
    def _demodulate(baseband):
        """M-FSK/M-GFSK demodulation and sanity check."""
        
        # Based on code from WSJT-X lib/ft8/ft8b.f90
        
        # Demodulate baseband by computing 79 DFTs, one for each symbol period
        f, t, dft = scipy.signal.stft(baseband, fs=200, window='boxcar', nperseg=32, noverlap=0,
                                       boundary=None, return_onesided=False, axis=1)
        #dft = dft[:, :tone_count] * 32 * 32 # We are only interested in the lowest M bins
        dft = dft[:, :tone_count]

        # Hard detection of Costas arrays
        tones = np.argmax(np.abs(dft), axis=1)
        good_tones = 0
        for offset in costas_offsets:
            good_tones += np.sum(tones[offset: offset+costas_order] == costas) 
      
        # Sanity check
        if good_tones <= 6:
            raise ValueError()
            
        return dft

    @staticmethod
    def _demap(obs, num_symbols):
        """M-FSK/M-GFSK soft demapper."""
        
        # Based on code from WSJT-X lib/ft8/ft8b.f90
               
        llr = np.empty(encoded_bits) # Storage for Log likelyhood ratios              
        mask = (1 << tone_order) - 1
        num_bits = (num_symbols + 1) * tone_order # Number of bits in a symbol group
        num_permutations = 1 << num_bits # Number of permutations of the bits in a symbol group
        
        # Loop through groups of num_symbol symbols at a time
        for i in range(0, encoded_symbols, num_symbols + 1):
                
            # Sum DFT filter outputs corresponding to all permutations of bits in a symbol group
            s = np.zeros(num_permutations, dtype='complex128')
            for p in range(num_permutations):
                
                # Add up contributions from each symbol in the symbol group for this permutation
                t = p
                for j in range(num_symbols, -1, -1):
                    tone = gray_map[t & mask] # Tone that corresponds to this permutation for this symbol
                    s[p] += obs[symbol_offsets[i] + j, tone] #
                    t >>= tone_order # Prepare for next iteration
            
            # Calculate magnitude of all the sums
            m = np.abs(s)
            
            # Loop through all the bits of the codeword that correspond to this group of symbols
            first_bit = i * tone_order
            last_bit = min(first_bit + num_bits, encoded_bits)
            bit_pos = num_bits - 1 # Most significant bit of the permutations index corresponds with first_bit
            for encoded_bit in range(first_bit, last_bit):               
                # Find maximum magnitude of permutations where this bit position is a 1 and subtract maximum
                # magnitude of permutations where this bit position is a 0
                llr[encoded_bit] = (np.amax(m[Signal.demap_one[bit_pos, :num_permutations]]) -
                                    np.amax(m[Signal.demap_not_one[bit_pos, :num_permutations]]))
                bit_pos -= 1 # next bit will be less significant than this bit
                          
        # Normalise using standard deviation and scale result with a WSJT-X fudge factor                 
        llr /= np.std(llr)
        llr *= 2.83 # Square root of 8 ???
                                                     
        return llr
    
    @staticmethod
    def _sum_product_decoder(demapper_llr):
        """Attempt to find valid LDPC codeword from LLR."""
        
        # Sort of based on code from WSJT-X lib/ft8/bpdecode174_91.f90
        # Lots of optimisation was need to make it run well in Python

        # Allocate storage - This is an iterative algorithm so we want reuse memory where possible
        codeword = np.empty(encoded_bits + 1, np.bool) # Current estimate of codeword
        codeword[0] = False
        bit_llr = np.empty(encoded_bits) # Current estimate of LLR for each bit
        bit_in = np.zeros((encoded_bits, 3)) # Messages received by bit nodes
        bit_out = np.zeros((encoded_bits + 1, 3)) # Messages sent by bit nodes
        bit_out[0] = 1
        check_in = np.ones((ldpc_parity_bits ,7)) # Messages received by check nodes
        check_out = np.empty((ldpc_parity_bits, 7)) # Messages sent by check nodes
        selector = 1 - np.identity(7) # Matrix of all 1 except diagonal which contains zeros
        
        msg = None

        for i in range(Signal.decoder_max_iterations):
        
            # Calculate current estimate of bit LLRs
            # On first iteration bit_in is zero so the estimated LLR equals the detector LLR
            bit_llr[:] = demapper_llr + np.sum(bit_in, axis=1)
            
            # Check if we have valid codeword after applying the hard decision rule
            codeword[1:] = bit_llr > 0
            bad_bits = np.sum(np.sum(np.take(codeword, adjusted_check_terms), axis=1) % 2)
            if bad_bits == 0:
                # Valid codeword so convert codeword to Python integer
                bits = 0
                for bit in codeword[1:92]:
                    bits <<= 1
                    bits |= int(bit)

                # Try to unpack bits into a message - this also checks CRC
                try:
                    msg = Message.unpack91(bits)                
                    break # Good CRC so we are done
                except CRCError:
                    pass # Bad CRC so keep iterating
        
            # Calculate messages to send to check nodes at bit nodes
            # We send the current bit LLR estimate excluding contribution we got from the check node
            # Cheat by doing tanh before messages are sent instead of after they are received
            bit_out[1:] = np.tanh(-0.5 * (bit_llr[:, np.newaxis] - bit_in))
        
            # Send 522 messages from bit nodes to check nodes
            np.take(bit_out.flat, adjusted_check_flat_terms, out=check_in)
        
            # Calculate messages to send to bit nodes at check nodes
            np.prod(np.where(selector, check_in[:,np.newaxis,:], 1.0), axis=2, out=check_out)
        
            # Send 522 messages from check nodes to bit nodes
            np.take(check_out.flat, bit_flat_terms, out=bit_in) 
        
            # Cheat by doing arctanh after messages are received instead of before they are sent
            bit_in[:] = -2.0 * np.arctanh(bit_in)
        
        # Return decoded message (if any), codeword, iteration count and number of bad bits
        return msg, codeword[1:], i, bad_bits

    @staticmethod
    def _get_snr(obs, codeword, freq, analysis):
        """Calculate SNR for a successfully decoded message."""
        
        # Based on code from WSJT-X lib/ft8/ft8b.f90
        
        # Convert the codeword into symbols
        msg_symbols = []
        s = 0
        for i, bit in enumerate(codeword):
            s <<= 1
            s = s | bit
            if (i % 3) == 2:
                msg_symbols.append(s)
                s = 0
        symbols = costas + msg_symbols[:encoded_symbols // 2] + costas + msg_symbols[encoded_symbols // 2:] + costas

        symbol_range = np.arange(total_symbols)
        
        # Get observations corresponding to decoded symbol values and compute power
        signal_pwr = np.sum(np.abs(obs[symbol_range, symbols]) ** 2.0)
        
        # Calculate SNR using noise baseline
        noise_psd = analysis.noise_baseline(freq)
        arg = signal_pwr / (noise_psd * 2500)
        if arg > 0.1:
            snr = max(10.0 * np.log10(arg) - 19, -24.0)
        else:
            snr = -24.0
        
        return snr

    @staticmethod
    def _detect(baseband, freq, analysis):
        
        # Get channel observations from baseband signal
        obs = Signal._demodulate(baseband)
        
        # Demap and decode
        for i in range(Signal.demap_max_symbols):
            
            # Demap using i symbols at a time
            llr = Signal._demap(obs, i)
            
            # Try and decode using sum product algorithm
            msg, codeword, iterations, bad_bits = Signal._sum_product_decoder(llr)
            
            # Check for successful decode
            if msg != None:
                break
                
            # Put a priori stuff here
        
        # Could not decode valid message
        if msg == None:
            raise ValueError()
        
        # Calculate SNR from noise baseline
        snr = Signal._get_snr(obs, codeword, freq, analysis)
        #snr = 1
        
        return msg, snr, obs, llr, codeword
    