import numpy as np

# Scores and masses taken from http://www.csbio.sjtu.edu.cn/bioinf/PseAAC/ParaValue.htm table
HYDROPHOB_SCORES = {'A': 0.62, 'C': 0.29, 'D': -0.9, 'E': -0.74, 'F': 1.19, 
					'G': 0.48, 'H': -0.4, 'I': 1.38, 'K': -1.5, 'L': 1.06, 
					'M': 0.64, 'N': -0.78, 'P': 0.12, 'Q': -0.85, 'R': -2.53, 
					'S': -0.18, 'T': -0.05, 'V': 1.08, 'W': 0.81, 'Y': 0.26}

HYDROPHIL_SCORES = {'A': -0.5, 'C': -1.0, 'D': 3.0, 'E': 3.0, 'F': -2.5, 
					'G': 0.0, 'H': -0.5, 'I': -1.8, 'K': 3.0, 'L': -1.8, 
					'M': -1.3, 'N': 0.2, 'P': 0.0, 'Q': 0.2, 'R': 3.0, 
					'S': 0.3, 'T': -0.4, 'V': -1.5, 'W': -3.4, 'Y': -2.3}

SIDECHAIN_MASS = {'A': 15.0, 'C': 47.0, 'D': 59.0, 'E': 73.0, 'F': 91.0, 
					'G': 1.0, 'H': 82.0, 'I': 57.0, 'K': 73.0, 'L': 57.0, 
					'M': 75.0, 'N': 58.0, 'P': 42.0, 'Q': 72.0, 'R': 101.0, 
					'S': 31.0, 'T': 45.0, 'V': 43.0, 'W': 130.0, 'Y': 107.0}

# Standard conversion of scores and masses above, according to http://www.csbio.sjtu.edu.cn/bioinf/PseAAC/type1.htm
HYDROPHIL_STANDARD =  {'A': -0.15187800657861578, 'Y': -1.1111075218119784, 'Q': 0.2211556937899142, 'L': -0.8446548786915999, 'N': 0.2211556937899142, 'C': -0.4183306496989943, 'H': -0.15187800657861578, 'R': 1.7132904952640338, 'I': -0.8446548786915999, 'D': 1.7132904952640338, 'S': 0.27444622241398986, 'T': -0.09858747795454006, 'G': 0.11457463654176277, 'V': -0.6847832928193728, 'W': -1.6973033366768113, 'F': -1.21768857906013, 'E': 1.7132904952640338, 'P': 0.11457463654176277, 'K': 1.7132904952640338, 'M': -0.5782022355712214}

HYDROPHOB_STANDARD =  {'A': 0.6362505881446506, 'Y': 0.26681476277033744, 'Q': -0.8722790321337952, 'L': 1.0877832636021447, 'N': -0.8004442883110121, 'C': 0.2976010815515302, 'H': -0.41048425041590364, 'R': -2.5963128838805902, 'I': 1.4161706639348675, 'D': -0.9235895634357832, 'S': -0.18471791268715662, 'T': -0.051310531301987934, 'G': 0.49258110049908443, 'V': 1.10830747612294, 'W': 0.831230607092205, 'F': 1.2211906449873133, 'E': -0.7593958632694218, 'P': 0.12314527512477112, 'K': -1.5393159390596387, 'M': 0.6567748006654459}

SIDECHAIN_STANDARD =  {'A': -1.5919364305641373, 'Y': 1.4624567208832167, 'Q': 0.3004593263108537, 'L': -0.19753955707730178, 'N': -0.16433963151809142, 'C': -0.5295388126694055, 'H': 0.6324585819029575, 'R': 1.2632571675279545, 'I': -0.19753955707730178, 'D': -0.13113970595888105, 'S': -1.0607376216167714, 'T': -0.5959386637878262, 'G': -2.0567353883930823, 'V': -0.662338514906247, 'W': 2.226055008745055, 'F': 0.9312579119358507, 'E': 0.3336592518700641, 'P': -0.6955384404654573, 'K': 0.3336592518700641, 'M': 0.40005910298848485}

def AAC(seq):
    AAC_array = []
    for s in seq:
        protein_dict = {'A': 0, 'R': 0, 'N': 0, 'D': 0, 'C': 0, 'E': 0, 'Q': 0, 'G': 0, 'H': 0, 'I': 0, 'L': 0, 'K': 0, 'M': 0, 'F': 0, 'P': 0, 'S': 0, 'T': 0, 'W': 0, 'Y': 0, 'V': 0}
        prob = [0] * len(protein_dict)
        for i in range(len(s)):
            protein_dict[s[i]] += 1
        for i, times in enumerate(protein_dict.values()):
            prob[i] = times/(len(s))
        AAC_array.append(prob)
    return np.array(AAC_array)

def aa_score_pse(aa1, aa2):
    """ 
        Returns score of two amino acids for
        use in pseaac calculations. Incorporates
        the following scoring parameters:
                hydrophobicity value
                hydrophilicity value
                side chain mass
    """
    hydrophobicity = (HYDROPHOB_STANDARD[aa2] - HYDROPHOB_STANDARD[aa1])**2
    hydrophilicity = (HYDROPHIL_STANDARD[aa2] - HYDROPHIL_STANDARD[aa1])**2
    side_chain = (SIDECHAIN_STANDARD[aa2] - SIDECHAIN_STANDARD[aa1])**2

    return (hydrophilicity + hydrophobicity + side_chain) / 3

def aa_score_am(aa1,aa2): 
    
    hydrophobicity = (HYDROPHOB_STANDARD[aa2] * HYDROPHOB_STANDARD[aa1])
    hydrophilicity = (HYDROPHIL_STANDARD[aa2] * HYDROPHIL_STANDARD[aa1])
    return hydrophobicity, hydrophilicity

def PseAAC(seq, AAC_array, lamda, weight):
    PseAAC_array = []
    for i in range(AAC_array.shape[0]):         # AAC[i] -> current sequence
        plus_array = [0] * lamda
        for j in range(1,lamda+1):        # lamda value
            J = 0
            for k in range(len(seq[i])- j):
                J += aa_score_pse(seq[i][k],seq[i][k+j])
            J = J / (len(seq[i])-j)
            # Weight and store
            plus_array[j-1] = J * weight
        # Normalize
        normalizer = sum(plus_array) + sum(AAC_array[i])
        AAC_list = (AAC_array[i] * 100).tolist()
        feature = [x / normalizer  for x in AAC_list + plus_array]
        PseAAC_array.append(feature)
    return np.array(PseAAC_array)

def Am_PseAAC(seq, AAC_array, lamda, weight):
    Am_PseAAC_array = []
    for i in range(AAC_array.shape[0]):         # AAC[i] -> current sequence
        plus_array = [0] * (2 * lamda)
        for j in range(1,lamda+1):        # lamda value
            H1, H2= 0, 0
            for k in range(len(seq[i])- j):
                a, b = aa_score_am(seq[i][k],seq[i][k+j])
                H1 += a
                H2 += b
            H1 = H1 / (len(seq[i])-j)
            H2 = H2 / (len(seq[i])-j)
            # Weight and store
            plus_array[2*(j-1)] = H1 * weight
            plus_array[2*j-1] = H2 * weight              #! value is not correct
        # Normalize
        normalizer = sum(plus_array) + sum(AAC_array[i])
        AAC_list = (AAC_array[i] * 100).tolist()
        feature = [x / normalizer  for x in AAC_list + plus_array]
        Am_PseAAC_array.append(feature)
    return np.array(Am_PseAAC_array)
