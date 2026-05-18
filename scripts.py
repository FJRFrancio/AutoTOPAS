from utils import *
import pandas as pd

def indexing(wave_lenth, crystal_system = ['Cubic','Hexagonal','Trigonal','Tetragonal','Orthorhombic','Monoclinic','Triclinic']):

    HEAD = [
        'seed\n',
        'index_zero_error\n',
        f'index_lam {wave_lenth}\n',
        'index_min_lp 1\n',
        'index_max_lp 40\n',
        '\n'
    ]

    if 'Cubic' in crystal_system: HEAD.append('Bravais_Cubic_sgs\n')
    if 'Hexagonal' in crystal_system or 'Trigonal' in crystal_system: HEAD.append('Bravais_Trigonal_Hexagonal_sgs\n')
    if 'Tetragonal' in crystal_system: HEAD.append('Bravais_Tetragonal_sgs\n')
    if 'Orthorhombic' in crystal_system: HEAD.append('Bravais_Orthorhombic_sgs\n')
    if 'Monoclinic' in crystal_system: HEAD.append('Bravais_Monoclinic_sgs\n')
    if 'Triclinic' in crystal_system: HEAD.append('Bravais_Triclinic_sgs\n')

    HEAD.append('\n')

    HEAD.append('load index_th2 index_I {\n')

    TAIL = '}\n'

    return HEAD,TAIL


def pawley(filename,index_result:pd.Series,wavelenth,phase_name='AUTO_PAWLEY',bkg_items = 10,iters=1000,light_source = 'syhchrotron',refine_structure = True, refine_pky = False, custom_peak_shape = None,custom_bkg = None):

    a,b,c,alpha,beta,gamma,SG = index_result['a'],index_result['b'],index_result['c'],index_result['alpha'],index_result['beta'],index_result['gamma'],index_result['SG'],

    crystal_system = space_group_to_crystal_system(SG)
    flag_a,flag_b,flag_c,flag_alpha,flag_beta,flag_gamma = '','','','','',''
    flag_pky = '' if refine_pky else '!'

    if refine_structure:
        if crystal_system == 'Cubic': flag_a,flag_b,flag_c,flag_alpha,flag_beta,flag_gamma = 'lpa','lpa','lpa','','',''
        if crystal_system in ['Tetragonal','Hexagonal','Trigonal']: flag_a,flag_b,flag_c,flag_alpha,flag_beta,flag_gamma = 'lpa','lpa','lpc','','',''
        if crystal_system == 'Orthorhombic': flag_a,flag_b,flag_c,flag_alpha,flag_beta,flag_gamma = 'lpa','lpb','lpc','','',''
        if crystal_system == 'Monoclinic': flag_a,flag_b,flag_c,flag_alpha,flag_beta,flag_gamma = 'lpa','lpb','lpc','','angb',''
        if crystal_system == 'Triclinic': flag_a,flag_b,flag_c,flag_alpha,flag_beta,flag_gamma = 'lpa','lpb','lpc','anga','angb','angg'
    if not custom_peak_shape:
        custom_peak_shape = f'TCHZ_Peak_Type(pku, 0.00039,pkv, -0.00221,pkw, -0.00146,!pkz, 0.0000,pkx, 0.00957,{flag_pky}pky, 0.0000)'
    if not custom_bkg:
        custom_bkg = f'    bkg @ {' '.join(['0' for _ in range(bkg_items)])}'


    if light_source == 'syhchrotron':
        syhchrotron_script= [ 
            '\'GENERATE BY AUTO TOPAS\n',
            '\'Input file for Simple Rietveld or Pawley refinement\n',
            'r_wp 0 r_exp 0 r_p 0 weighted_Durbin_Watson 0 gof 0\n',
            f'iters {iters}\n',
            'chi2_convergence_criteria 0.01\n',
            '\n',
            f'xdd {filename}\n',
            '    x_calculation_step = Yobs_dx_at(Xo); convolution_step 4\n',
            f'{custom_bkg}\n',
            f'    Simple_Axial_Model(axial, 9)lam ymin_on_ymax 0.0001 la 1.0 lo {wavelenth} lh 0.001\n',
            '    LP_Factor(90)\n',
            '\n',
            '    Zero_Error(zero,0)\n',
            '        hkl_Is\n',
            f'            phase_name {phase_name}\n',
            f'            {custom_peak_shape}\n',
            f'        a {flag_a} {a}\n',
            f'        b {flag_b} {b}\n',
            f'        c {flag_c} {c}\n',
            f'        al {flag_alpha} {alpha}\n',
            f'        be {flag_beta} {beta}\n',
            f'        ga {flag_gamma} {gamma}\n',
            f'        space_group {SG}\n',
            '\n',   
            f'Create_2Th_Ip_file({filename.split('.')[0]}_Bragg_pos_2Th_I.txt)\n',
            f'Out_Yobs_Ycalc_and_Difference({filename.split('.')[0]}_Yobs_Ycalc_Diff.xy)\n',
        ]
         
        return syhchrotron_script
    
    raise ValueError(f"{light_source} light source not supported yet.")



