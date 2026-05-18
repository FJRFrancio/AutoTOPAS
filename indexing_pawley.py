from utils import *
from scripts import *
import subprocess



def main():
    # parameters
    root_dir = 'D:\\Jiarui_Fan\\AutoTOPAS\\Examples\\Diamond_examples\\'
    logfile = 'log.csv'
    WAVE_LENTH_D8 = 0.824495
    WAVE_LENTH_D4 = 0.4941625
    WAVE_LENTH_D3 = 0.35452792
    current_wavelenth = WAVE_LENTH_D8
    indexing_rate_limit = 0.85
    refine_cell_parameter = True
    refine_pky = True
    light_source = 'syhchrotron'
    # crystal_system = ['Cubic','Hexagonal','Trigonal','Tetragonal','Orthorhombic','Monoclinic','Triclinic']
    crystal_system = ['Cubic']
    # custom_peak_shape = 'TCHZ_Peak_Type(pku,-0.776553956`,pkv,  0.0229360874`,pkw,  0.000875722254`,!pkz, 0.0000,pkx, 0.234998959`,pky, 0.000109111595`_LIMIT_MIN_0.0001)'
    # custom_bkg = 'bkg @  15775.3127` -41527.3265`  15342.1035` -4529.0672` -3855.17704`  4461.59022` -1053.19973` -4205.15118`  4966.80204` -526.26298`'

    custom_peak_shape = None
    custom_bkg = None


    n_scale_limit = 12

    # get data file list
    data_filenames = get_file_names(root_dir,extensions=['xye'])
    # main loop

    for filename in data_filenames:
        # refresh tqdm bar
        indexing_and_pawley(root_dir, filename, logfile,current_wavelenth, ext = 'xye', indexing_rate_limit = 0.85, refine_cell_parameter = True, refine_pky = True,
                        light_source = 'syhchrotron', crystal_system = ['Cubic','Hexagonal','Trigonal','Tetragonal','Orthorhombic','Monoclinic','Triclinic'],
                        custom_peak_shape = 'TCHZ_Peak_Type(pku,-0.776553956`,pkv,  0.0229360874`,pkw,  0.000875722254`,!pkz, 0.0000,pkx, 0.234998959`,pky, 0.000109111595`_LIMIT_MIN_0.0001)',
                        custom_bkg = 'bkg @  0 0 0 0 0 0 0 0 0 0',
                        n_scale_limit = 12,
                        )


    


def indexing_and_pawley(root_dir, filename, logfile ,current_wavelenth, ext = 'xye', indexing_rate_limit = 0.85, refine_cell_parameter = True, refine_pky = True,
                        light_source = 'syhchrotron', crystal_system = ['Cubic','Hexagonal','Trigonal','Tetragonal','Orthorhombic','Monoclinic','Triclinic'],
                        custom_peak_shape = '   TCHZ_Peak_Type(pku,-0.776553956`,pkv,  0.0229360874`,pkw,  0.000875722254`,!pkz, 0.0000,pkx, 0.234998959`,pky, 0.000109111595`_LIMIT_MIN_0.0001)',
                        custom_bkg = '  bkg @  0 0 0 0 0 0 0 0 0 0',
                        n_scale_limit = 12,
                        ):


    # log down running status
    running_info = [filename]

    # read file
    datafile = f'{root_dir}\\{filename}.{ext}'
    observed_df = pd.read_csv(datafile, sep='\s+', names=['2theta','Yobs','error'])
    indexing_first_rank = None


    # Peak search and indexing
    indexing_success = False
    for n_scale in range(n_scale_limit,1,-1):

        # search peak

        peaks_df, thresholds_info = multi_scale_peak_searching(observed_df,min_peak_height_ratio=0.01, min_prominence_ratio=0.5, 
                                                            n_scales=n_scale, plot_results=False, smooth_window=3)
        index_head, index_tail = indexing(current_wavelenth, crystal_system=crystal_system)

        # write indexing file 
        with open(f'{root_dir}\\{filename}_indexing.inp','w') as f:
            for line in index_head:
                f.write(line)
            for th,area in zip(peaks_df['2theta'].values,peaks_df['peak_area'].values): # type: ignore
                f.write(f'\t{th} {area}\n')
            for line in index_tail:
                f.write(line)

        # Run TOPAS indexing

        # os.system(f'{TOPAS_DIR}\\tc.exe {root_dir}\\{filename}_indexing.inp')
        process = subprocess.Popen([f'{TOPAS_DIR}\\tc.exe', f"{root_dir}\\{filename}_indexing.inp"])
        process.wait()
        if process.poll() is None:
            process.terminate()
            process.wait()

        
        # read INP file, get infos and position to add new site line.
        with open(f'{root_dir}\\{filename}_indexing.ndx','r') as f:
            lines = f.readlines()
            start_idx = 0
            end_idx = 0
            line_counter = 0

            for line in lines:
                line_counter += 1
                # skip comment line
                if '{' in line:
                    start_idx = line_counter+1
                # skip structure parameters
                if '}' in line:
                    end_idx = line_counter-1
                    break
            indexing_result = pd.DataFrame([i.split() for i in lines[start_idx:end_idx]])
            indexing_result = indexing_result[indexing_result.columns[:13].to_list()]
            try:
                indexing_result.columns = ['Rank','SG','Status','UNI','V','GOF','Zero','a','b','c','alpha','beta','gamma',]
            except ValueError:
                running_info.append('tc.exe not generate ndx file')
                break
            
            indexing_first_rank = indexing_result.iloc[0]
            print(f'UNI = {indexing_first_rank['UNI']}')
            index_rate = 1-int(indexing_first_rank['UNI'])/len(peaks_df) # type: ignore
            print(f'{filename}_nscale={n_scale}, index_rate = {index_rate}')

        # whether go to pawley step based on indexing rate (defaule > 90%)
        if index_rate > indexing_rate_limit:

            indexing_success = True
            running_info.append('Success')
            running_info.append('n_scale,' + str(n_scale))
            running_info.append('index_rate,'+str(index_rate))
            running_info.append(f'space_group,{indexing_first_rank['SG']}')
            break

        else:
            if len(peaks_df) < 5:


                running_info.append('Indexing Failed')
                break
    
    # pawley refinment
    if indexing_success:
        # generate pawley script
        pawley_script = pawley(f'{filename}.xye',indexing_first_rank,wavelenth=current_wavelenth,refine_pky=refine_pky,refine_structure=refine_cell_parameter,light_source=light_source,custom_peak_shape=custom_peak_shape,custom_bkg = custom_bkg) # type: ignore
        with open(f'{root_dir}\\{filename}_pawley.inp','w') as f:
            for line in pawley_script:
                f.write(line)
        
        # run pawley


        # os.system(f'{TOPAS_DIR}\\tc.exe {root_dir}\\{filename}_pawley.inp')
        process = subprocess.Popen([f'{TOPAS_DIR}\\tc.exe', f"{root_dir}\\{filename}_pawley.inp"])
        process.wait()
        if process.poll() is None:
            process.terminate()
            process.wait()

        os.system(f'copy {root_dir}\\{filename}_pawley.out {root_dir}\\{filename}_pawley.inp')

        # read INP file, extract Rwp
        with open(f'{root_dir}\\{filename}_pawley.inp','r') as f:
            lines = f.readlines()
            infoFlag = True
            INFO_DICT = {}
            for line in lines:
                # skip comment line
                if '\'' in line:
                    continue
                # extract infos
                if infoFlag:
                    info_pattern = r"(\w+)\s+([\d.]+)"
                    infoMatches = re.findall(info_pattern,line)
                    tmp_dict = {key: float(value) if '.' in value else int(value) for key, value in infoMatches}
                    INFO_DICT.update(tmp_dict)
            running_info.append(f'Rwp,{INFO_DICT['r_wp']}')



        # logging
        with open(f'{root_dir}\\{logfile}','a') as f:
            running_info.append('\n')
            f.write(','.join(running_info))
        
        # PLOT pawley  
        fitted_data_filename = f'{filename}_Yobs_Ycalc_Diff.xy'
        bragg_position_filename = f'{filename}_Bragg_pos_2Th_I.txt'
        # For DLS I11 Beamline Data
        observed_df = pd.read_csv(f'{root_dir}\\{filename}.xye', sep='\s+', names=['2theta','Yobs','error'])
        # For TOPAS
        fitted_df = pd.read_csv(f'{root_dir}\\{fitted_data_filename}', sep='\s+', names=['Yobs','Ycalc','Diff'])
        bragg_df = pd.read_csv(f'{root_dir}\\{bragg_position_filename}', sep='\s+', names=['2theta','Ycalc'])
        plot_Rietveld(observed_df,fitted_df,bragg_df,save_dir=root_dir,file_name=f'{filename}_pawley_fitting',fig_size=(9,3),dpi = 600,zoom_range=(2,70),diff_shift_ratio=0.02,y_scale='normal')


    else:
        with open(f'{root_dir}\\{logfile}','a') as f:
            running_info.append('\n')
            f.write(','.join(running_info))

    print('Job Finished.')



if __name__ == '__main__':
    main()
