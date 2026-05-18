# Multiple Files, with zmatrix
from scripts import *
from utils import *

WORK_DIR = 'D:\\Jiarui_Fan\\AutoTOPAS\\Examples\\MFM_170_OMe_Guests\\C3H6\\'
BASE_INP_FILE = 'MFM_170_OMe_TOPAS.inp'
RIGID_BODY_GJF_NAME = 'C3H6.gjf'
WAVE_LENTH_D8 = 0.823835
WAVE_LENTH_D4 = 0.4941625
WAVE_LENTH_D3 = 0.35452792

Rwp_dict = {}
data_filenames = get_file_names(WORK_DIR,extensions=['xye'])

flex_rigid = True

for filename in data_filenames:
    current_dir = f'{WORK_DIR}\\{filename}'
    os.mkdir(current_dir)
    os.system(f'copy {WORK_DIR}\\{filename}.xye {current_dir}\\{filename}.xye')
    os.system(f'copy {WORK_DIR}\\{BASE_INP_FILE} {current_dir}\\{filename}.inp')
    os.system(f'copy {WORK_DIR}\\{RIGID_BODY_GJF_NAME} {current_dir}\\{RIGID_BODY_GJF_NAME}')


    # Add Atoms one by one and refine structure
    numOfRigid = 2
    suffix = 'r'
    Rwp_LIST = []

    # [([x,y,z],[offset_x,offset_y,offset_z]),prob]
    position_choices = [
        [([0.5,0.5,0.5],[0.2,0.2,0.2]),0.6],
        [([0.25,0.25,0.25],[0.1,0.1,0.1]),0.4],
        [([0,0,0.5],[0.15,0.15,0.15]),0],
    ]

    rot_choices = [
        [([0,0,0],[10,10,10]),0.5],
        [([90,0,0],[10,10,10]),0.3],
        [([90,90,0],[10,10,10]),0.2],
    ]

    for i in range(1,numOfRigid+1):
        # Initialize reading container
        EXISTED_ATOM_NAMELIST = []
        INFO_DICT = {}
        INSERT_INDEX = 0
        # read INP file, get infos and position to add new site line.
        with open(f'{current_dir}\\{filename}.inp','r') as f:
            lines = f.readlines()
            infoFlag = True
            
            for line in lines:
                # skip comment line
                if '\'' in line:
                    continue
                # skip structure parameters
                if 'xdd' in line:
                    lines[lines.index(line)] = f'xdd {filename}.xye\n'
                    infoFlag = False
                # extract infos
                if infoFlag:
                    info_pattern = r"(\w+)\s+([\d.]+)"
                    infoMatches = re.findall(info_pattern,line)
                    tmp_dict = {key: float(value) if '.' in value else int(value) for key, value in infoMatches}
                    INFO_DICT.update(tmp_dict)
                # update site info
                if 'site' in line:
                    site_info = extract_site_info(line)
                    EXISTED_ATOM_NAMELIST.append(site_info['site'])
                # find the end of atom part
                if 'scale' in line:
                    INSERT_INDEX = lines.index(line)

        Rwp_LIST.append(INFO_DICT['r_wp'])
        
        # generate new rigid body
        center,offset = randomly_choose_position(position_choices)
        position = random_xyz(center,offset)
        rot_center,rot_offset = randomly_choose_position(rot_choices)
        rotation = random_xyz(rot_center,rot_offset)

        if flex_rigid:
            rigid_list = rigid_body_z_matrix(f'{current_dir}\\{RIGID_BODY_GJF_NAME}',trans_xyz=position,rot_xyz=rotation,
                                            dist_range =[-0.1,0.1], angle_range =[-3,3], dihedral_range = [-10,10]
                                            ,rigid_index = i,suffix = 'r')
        else:
            rigid_list = rigid_body(f'{current_dir}\\{RIGID_BODY_GJF_NAME}',trans_xyz=position,rot_xyz=rotation,rigid_index=i,suffix=suffix)


        # write **NEW** INP file
        # INP_FILE = INP_FILE + '_' + atomName
        for new_line in rigid_list:
            lines.insert(INSERT_INDEX,new_line)
            INSERT_INDEX += 1

        with open(f'{current_dir}\\{filename}.inp','w') as f:
            for line in lines:
                f.write(line)

        # Run TOPAS
        rounds = 3
        for _ in range(rounds):
            os.system(f'{TOPAS_DIR}\\tc.exe {current_dir}\\{filename}.inp')
            os.system(f'copy {current_dir}\\{filename}.out {current_dir}\\{filename}.inp')

    Rwp_dict[filename] = Rwp_LIST[-1]
    
    plt.figure()
    plt.plot(Rwp_LIST)
    plt.savefig(f'{current_dir}\\{filename}.png')

    # PLOT Reitvield    
    fitted_data_filename = 'Yobs_Ycalc_Diff.xy'
    bragg_position_filename = 'Bragg_pos_2Th_I.txt'
    # For DLS I11 Beamline Data
    observed_df = pd.read_csv(f'{current_dir}\\{filename}.xye', sep='\s+', names=['2theta','Yobs','error'])
    # For TOPAS
    fitted_df = pd.read_csv(f'{current_dir}\\{fitted_data_filename}', sep='\s+', names=['Yobs','Ycalc','Diff'])
    bragg_df = pd.read_csv(f'{current_dir}\\{bragg_position_filename}', sep='\s+', names=['2theta','Ycalc'])
    plot_Rietveld(observed_df,fitted_df,bragg_df,save_dir=current_dir,fig_size=(9,3),dpi = 600,zoom_range=(2,70),diff_shift_ratio=0.02,y_scale='normal')

print('Job Finished.')
