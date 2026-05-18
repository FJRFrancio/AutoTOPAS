import os
import re
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import find_peaks, peak_widths, savgol_filter
from scipy.integrate import simps
from tqdm import tqdm
from collections import deque
import shutil


# define TOPAS dirs
TOPAS_DIR = 'c:\\TOPAS6'

class Atom:
    def __init__(self, atom_type, coord:tuple, charge, index = None) -> None:
        self.atom_type = atom_type
        self.x,self.y,self.z = coord
        self.charge = charge
        self.index = index
        self.neighbours = []
        self.symbol = f'{self.atom_type}_{self.index}'
        self.used = False

    def __str__(self) -> str:
        return self.symbol

    def add_neighbours(self,neighbour):
        self.neighbours.append(neighbour)

    def get_neighbours(self):
        return self.neighbours
    
    def get_xyz(self):
        return (self.x,self.y,self.z)
    
    def get_symbol(self):
        return self.symbol
    
    def set_used(self):
        self.used = True


class Molecule:
    def __init__(self, file_path) -> None:
        self.file_path =  file_path
        self.atoms_df = pd.read_csv(self.file_path, sep='\s+', names=['Element','charge','x','y','z'],skiprows=5)

        self.init_atoms()
        self.find_neighbours()
        self.aline_atoms()

    def __str__(self) -> str:
        return self.file_path


    def distance(self, atom_ls:list):
        A1,A2 = atom_ls
        point1 = np.array(A1.get_xyz())
        point2 = np.array(A2.get_xyz())
        delta = point1 - point2
        
        distance = np.linalg.norm(delta)

        return np.round(distance,5)

    def angel(self, atom_ls:list):
        A1,A2,A3 = atom_ls
        A = np.array(A1.get_xyz())
        B = np.array(A2.get_xyz())
        C = np.array(A3.get_xyz())
        
        BA = A - B
        BC = C - B
        
        magnitude_BA = np.linalg.norm(BA)
        magnitude_BC = np.linalg.norm(BC)
        
        if magnitude_BA == 0 or magnitude_BC == 0:
            raise ValueError("Atom overlapped!")
        
        cos_angle = np.dot(BA, BC) / (magnitude_BA * magnitude_BC)
        cos_angle = np.clip(cos_angle, -1.0, 1.0)
        
        return np.round(np.degrees(np.arccos(cos_angle)),5)

    def dihedral(self, atom_ls:list):
        A1,A2,A3,A4 = atom_ls
        A = np.array(A1.get_xyz())
        B = np.array(A2.get_xyz())
        C = np.array(A3.get_xyz())
        D = np.array(A4.get_xyz())
    
        BA = A - B
        BC = C - B
        CD = D - C
        
        n1 = np.cross(BA, BC)
        n2 = np.cross(BC, CD)
        
        # check if points alined
        if np.linalg.norm(n1) == 0 or np.linalg.norm(n2) == 0:
            raise ValueError("points alined!")
        
        cos_theta = np.dot(n1, n2) / (np.linalg.norm(n1) * np.linalg.norm(n2))
        cos_theta = np.clip(cos_theta, -1.0, 1.0)
        
        sign = np.sign(np.dot(BC, np.cross(n1, n2)))
        angle = sign * np.arccos(cos_theta)
        
        return 180-np.round(np.degrees(angle),5)

    def init_atoms(self):
        self.atoms = []
        self.symbols = []
        for i in range(len(self.atoms_df)):
            index = i+1
            atype = self.atoms_df.iloc[i]['Element']
            charge = self.atoms_df.iloc[i]['charge']
            coord = (self.atoms_df.iloc[i]['x'],self.atoms_df.iloc[i]['y'],self.atoms_df.iloc[i]['z'])
            atom = Atom(atype,coord,charge,index)

            self.atoms.append(atom)
            self.symbols.append(atom.get_symbol())

    def aline_atoms(self):
        self.atoms = self.find_atom_dfs(self.atoms[0])


    def find_neighbours(self):
        atom_count = len(self.atoms)
        for i in range(atom_count):
            for j in range(i+1,atom_count):
                atomA:Atom = self.atoms[i]
                atomB:Atom = self.atoms[j]
                distance = self.distance([atomA,atomB])
                if 1 <= distance <= 1.7: # based on bond lenth 
                    atomA.add_neighbours(atomB)
                    atomB.add_neighbours(atomA)

    def find_atom_dfs(self,atom,visited = None, result = None):
        # find k atoms using depth first search
        if visited is None:
            visited = set()
        if result is None:
            result = []

        if atom not in visited:
            visited.add(atom)
            result.append(atom)
            for neighbor in atom.get_neighbours():
                self.find_atom_dfs(neighbor, visited, result)

        return result
    
    def find_atom_dfs_with_passed_atoms(self,atom,k=4):
        # find k atoms using depth first search, but with only atoms already exist in z-matrix
        result = []
        for a in self.find_atom_dfs(atom):
            if a.used:
                result.append(a)
        return result[:k]

    def find_atom_bfs(self,start_atom,k=4):
        # find k atoms using breadth first search
        visited = set()
        queue = deque([start_atom])
        visited.add(start_atom)
        result = []

        while queue and len(result) < k:
            current_atom:Atom = queue.popleft()
            result.append(current_atom )
            
            for neighbor in current_atom.get_neighbours():
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append(neighbor)

        return result[:k]
    

    def calculate_z_matrix(self, dist_range = None, angle_range = None, dihedral_range = None, suffix = 'r', rigid_index = 0):
        self.z_matrix = []
        suffix = f'{suffix}{rigid_index}'

        for i in range(len(self.atoms)):
            A1:Atom = self.atoms[i]
            A1.set_used()
            
            distance = 0
            angle = 0
            dihedral = 0
            dist_constrain = ''
            angle_constrain = ''
            dihedral_constrain = ''

            if i >= 1:
                atom_ls = self.find_atom_dfs_with_passed_atoms(A1,k=2)
                distance = self.distance(atom_ls)
                if dist_range is not None:
                    dmin = np.round(distance + dist_range[0],5)
                    dmax = np.round(distance + dist_range[1],5)
                    dist_constrain = f'min {dmin} max {dmax}'
            if i >=2:
                atom_ls = self.find_atom_dfs_with_passed_atoms(A1,k=3)
                angle = self.angel(atom_ls)
                if angle_range is not None:
                    amin = np.round(angle + angle_range[0],5)
                    amax = np.round(angle + angle_range[1],5)
                    angle_constrain = f'min {amin} max {amax}'

            if i >= 3:
                atom_ls = self.find_atom_dfs_with_passed_atoms(A1,k=4)
                dihedral = self.dihedral(atom_ls)
                if dihedral_range is not None:
                    dhmin = np.round(dihedral + dihedral_range[0],5)
                    dhmax = np.round(dihedral + dihedral_range[1],5)
                    dihedral_constrain = f'min {dhmin} max {dhmax}'

            if i == 0:
                line = f'{A1.symbol}_{suffix}\n'
            elif i == 1:
                line = f'{atom_ls[0].symbol}_{suffix} {atom_ls[1].symbol}_{suffix} AA {distance} {dist_constrain}\n' # type: ignore
            elif i == 2:
                line = f'{atom_ls[0].symbol}_{suffix} {atom_ls[1].symbol}_{suffix} AA {distance} {dist_constrain} {atom_ls[2].symbol}_{suffix} AA {angle} {angle_constrain}\n' # type: ignore
            else:
                line = f'{atom_ls[0].symbol}_{suffix} {atom_ls[1].symbol}_{suffix} AA {distance} {dist_constrain} {atom_ls[2].symbol}_{suffix} AA {angle} {angle_constrain} {atom_ls[3].symbol}_{suffix} AA {dihedral} {dihedral_constrain}\n' # type: ignore
            
            self.z_matrix.append(line)

    def get_z_matrix(self,dist_range = None, angle_range = None, dihedral_range = None, suffix = 'r', rigid_index = 0):
        self.calculate_z_matrix(dist_range, angle_range, dihedral_range, suffix = suffix, rigid_index = rigid_index)
        return self.z_matrix



def addAtom(
        name: str, atom_type: str, position: list, occupancy: float, beq: float = 3.0,
        refine_xyz=True, refine_occ=True, refine_beq=True
):
    """
    Generate a new atom information line for TOPAS input.

    Args:
        name (str): Atom name.
        atom_type (str): Atom type.
        position (list): List of x, y, z coordinates.
        occupancy (float): Occupancy value for the site.
        beq (float, optional): B-equivalent (thermal parameter). Defaults to 3.0.
        refine_xyz (bool, optional): Whether to refine x, y, z coordinates. Defaults to True.
        refine_occ (bool, optional): Whether to refine occupancy. Defaults to True.
        refine_beq (bool, optional): Whether to refine B-equivalent. Defaults to True.

    Returns:
        str: A formatted atom line for TOPAS.
    """
    x, y, z = position
    keys = ['x', 'y', 'z', f'occ {atom_type}', 'beq']
    if refine_xyz:
        keys[keys.index('x')] = 'x @'
        keys[keys.index('y')] = 'y @'
        keys[keys.index('z')] = 'z @'
    if refine_occ:
        keys[keys.index(f'occ {atom_type}')] = f'occ {atom_type} @'
    if refine_beq:
        keys[keys.index('beq')] = 'beq @'

    return f'\t\tsite {name} {keys[0]} {x} {keys[1]} {y} {keys[2]} {z} {keys[3]} {occupancy} {keys[4]} {beq}\n'


def random_xyz(center=None, offset=None):
    """
    Generate a random position for an atom.

    Args:
        center (list, optional): Center position as [x, y, z]. Defaults to None (random values).
        offset (list, optional): Offset values as [dx, dy, dz]. Defaults to [0.1, 0.1, 0.1].

    Returns:
        list: A list of x, y, z coordinates.
    """
    if offset is None:
        offset = [0.1, 0.1, 0.1]
    if center is None:
        xyz = [random.random() for _ in range(3)]
    else:
        xyz = []
        for i in range(3):
            xyz.append(abs(random.uniform(center[i] - offset[i], center[i] + offset[i])))
    return xyz


def extract_site_info(text: str):
    """
    Extract information about an existing site from input text.

    Args:
        text (str): Text containing site information in TOPAS format.

    Returns:
        dict: A dictionary containing site information with keys ('site', 'x', 'y', 'z', 'occ', 'beq').
    """
    keywords = ['site', 'x', 'y', 'z', 'occ', 'beq']
    result = {}
    text = ''.join(text.split('@'))
    for keyword in keywords:
        if keyword == 'occ':
            pattern = rf'{keyword}\s+(\w+\s+\d+\.\d+)'
        else:
            pattern = rf'{keyword}\s+([\d\.]+|[\w]+)'
        match = re.search(pattern, text)
        if match:
            value = match.group(1)
            if keyword == 'occ':
                value = value.split()
            result[keyword] = value
    return result


def randomly_choose_position(choice_list: list):
    """
    Randomly choose a position based on a probability distribution.

    Args:
        choice_list (list): A list of tuples where each tuple contains a position
                            (e.g., [x, y, z]) and its associated probability.

    Returns:
        list: The selected position from the choice list.
    """
    count = 0
    xyzo_list = []
    prob_list = []
    for xyzo,p in choice_list:
        xyzo_list.append(xyzo)
        count = count + p
        prob_list.append(count)
    rand = random.uniform(0,count)
    for i in range(len(prob_list)):
        if rand <= prob_list[i]:
            return xyzo_list[i]


def get_file_names(root_dir: str, extensions=None):
    """
    Get file names under a specific directory with a given suffix.

    Args:
        root_dir (str): Root directory to search for files.
        extensions (list, optional): List of file extensions to include. Defaults to ['xye'].

    Returns:
        list: A list of file names (without extensions) matching the specified type.
    """
    if extensions is None:
        extensions = ['xye']
    file_names = []
    for root, directories, files in os.walk(root_dir):
        for file in files:
            if file.split('.')[-1] in extensions:
                # file_paths.append(os.path.join(root,file))
                file_names.append(file[:-4])
    return file_names


def rigid_body(
        file_path: str, rot_xyz=None, trans_xyz=None, rigid_index: int = 0, suffix: str = 'r'
):
    """
    Generate rigid body lines for TOPAS *.inp files from Gaussian *.gjf files.

    Args:
        file_path (str): Path to the input *.gjf file.
        rot_xyz (list, optional): List of rotation values [Rx, Ry, Rz]. Defaults to [0, 0, 0].
        trans_xyz (list, optional): List of translation values [Tx, Ty, Tz]. Defaults to [0.5, 0.5, 0.5].
        rigid_index (int, optional): Unique index for the rigid body. Defaults to 0.
        suffix (str, optional): Suffix for each site ID. Defaults to 'r'.

    Returns:
        list: A list of strings containing rigid body lines.
    """
    if rot_xyz is None:
        rot_xyz = [0, 0, 0]
    if trans_xyz is None:
        trans_xyz = [0.5, 0.5, 0.5]
    df = pd.read_csv(file_path, sep='\s+', names=['Element', 'charge', 'x', 'y', 'z'], skiprows=5)

    assert len(rot_xyz) == 3, 'len(rot_xyz) should be 3.'
    assert len(trans_xyz) == 3, 'len(trans_xyz) should be 3.'

    site_list = [f'\t\'Rigid sites {rigid_index} from file {file_path}\n']
    rigid_restrain_list = [f'\t\'Rigid restrain {rigid_index} from file {file_path}\n']
    conter = 0
    rigid_restrain_list.append('\t\trigid\n')

    for i in df.index:
        element = df['Element'][i]
        x = df['x'][i]
        y = df['y'][i]
        z = df['z'][i]
        site_line = f'\t\tsite {element}{rigid_index}{conter}{suffix} x 0 y 0 z 0 occ {element}  ovalue{rigid_index} 1 beq bvalue{rigid_index} 3.0\n'
        rigid_line = f'\t\tpoint_for_site {element}{rigid_index}{conter}{suffix} ux = {x}; uy = {y}; uz = {z};\n'

        site_list.append(site_line)
        rigid_restrain_list.append(rigid_line)

        conter += 1

    rigid_restrain_list.append(f'\t\tRotate_about_axies(@ {rot_xyz[0]}, @ {rot_xyz[1]}, @ {rot_xyz[2]})\n')
    rigid_restrain_list.append(f'\t\tTranslate(@ {trans_xyz[0]}, @ {trans_xyz[1]}, @ {trans_xyz[2]})\n')
    rigid_restrain_list.append('\n')
    site_list.append('\n')
    site_list.extend(rigid_restrain_list)

    return site_list


def plot_Rietveld(
        observed_df: pd.DataFrame, fitted_df: pd.DataFrame, bragg_df: pd.DataFrame|None = None,
        zoom_range=None, save_dir: str = './', file_name: str = 'Rietveld_fitting',
        fig_size: tuple = (13, 3), dpi: int = 300, y_scale: str = 'normal', diff_shift_ratio: float = 0.05
):
    """
    Create and save a Rietveld fitting plot.

    Args:
        observed_df (pd.DataFrame): DataFrame containing observed data.
        fitted_df (pd.DataFrame): DataFrame containing calculated (fitted) data.
        bragg_df (pd.DataFrame, optional): DataFrame with Bragg peak positions (2Theta column). Defaults to None.
        zoom_range (list, optional): Range of 2Theta for zooming the plot. Defaults to None.
        save_dir (str, optional): Directory to save the plot. Defaults to './'.
        file_name (str, optional): Name of the output plot image. Defaults to 'Rietveld_fitting'.
        fig_size (tuple, optional): Size of the figure (width, height). Defaults to (13, 3).
        dpi (int, optional): Resolution for the output plot (dots per inch). Defaults to 300.
        y_scale (str, optional): Y-axis scale, options: ['normal', 'ln', 'log', 'sqrt']. Defaults to 'normal'.
        diff_shift_ratio (float, optional): Ratio to shift the difference curve. Defaults to 0.05.

    Returns:
        None
    """
    with plt.style.context(['newscience']):

        # Figure configs
        fig = plt.figure(figsize=fig_size, dpi=dpi)
        plt.rc('font', family='Arial')
        plt.xlabel('2$\mathrm{\\theta}$ ($^{\circ}$)')
        # fig.text(0.07,0.5,'Intensity', family='Arial', weight='bold',ha='center',va='center',rotation=90)
        plt.tick_params(top=False, right=False, left=True)
        plt.minorticks_off()
        if zoom_range is not None:
            plt.xlim(zoom_range)

        # Data collection
        twoTheta = observed_df['2theta'].to_numpy()
        Yobs = fitted_df['Yobs'].to_numpy()
        Ycalc = fitted_df['Ycalc'].to_numpy()

        ## Y transformation
        if y_scale == 'normal':
            plt.ylabel('Intensity (CPS)')
        elif y_scale == 'ln':
            Yobs = np.log(Yobs)
            Ycalc = np.log(Ycalc)
            plt.ylabel('ln(I)')
        elif y_scale == 'log':
            Yobs = np.log10(Yobs)
            Ycalc = np.log10(Ycalc)
            plt.ylabel('log(I)')
        elif y_scale == 'sqrt':
            Yobs = np.sqrt(Yobs)
            Ycalc = np.sqrt(Ycalc)
            plt.ylabel('$\mathrm{\sqrt{I}}$')
        else:
            print('Scale method not supported yet, draw origional data.')
            plt.ylabel('Intensity (CPS)')

        Diff = Yobs - Ycalc

        # Draw Yobs
        plt.scatter(twoTheta, Yobs, label='Yobs', marker='o', color='b', s=6, edgecolors='black', linewidths=0.3,
                    facecolor='white')

        # Draw Ycalc
        plt.plot(twoTheta, Ycalc, label='Ycalc', linewidth=1, color='green')

        # Draw Yobs-Ycalc, shift under Ycalc, avoid overlapping with Ycalc
        Y_min = Yobs.min() if Yobs.min() < Ycalc.min() else Ycalc.min()
        Y_max = Yobs.max() if Yobs.max() > Ycalc.max() else Ycalc.max()
        shift = Y_min - (Y_max - Y_min) * diff_shift_ratio
        if shift + Diff.max() > Y_min:
            shift = Y_min - Diff.max()
        Diff = Diff + shift
        plt.plot(twoTheta, Diff, label='Diff', linewidth=1, color='gray')

        # Draw Bragg Position if needed
        if bragg_df is not None:
            bragg = bragg_df['2theta'].to_list()
            for pos in bragg:
                if zoom_range is not None:
                    if pos >= zoom_range[-1]:
                        break
                plt.axvline(pos, ymin=0.01, ymax=0.04, c='tab:red', linewidth=0.5)

        plt.legend()

        plt.savefig(f'{save_dir}\\{file_name}_range_{zoom_range}.png')
        
        
def space_group_to_crystal_system(sg_symbol):

    symbol = sg_symbol.strip().replace(" ", "").replace("-", "").upper()
    
    # Cubic
    if len(symbol) >= 4 and '6' not in symbol and (symbol[-2] == '3' or symbol[-1] == '3'):
        return "Cubic"
    
    if len(symbol) == 3 and ("23" in symbol or 'F' in symbol):
        return "Cubic"
    
    
    # Hexagonal
    if "6" in symbol:
        return "Hexagonal"
    
    # Trigonal
    if "3" in symbol and (symbol.startswith("R") or symbol.startswith("P3")):
        return "Trigonal"
    
    # Tetragonal
    if "4" in symbol:
        return "Tetragonal"
    
    # Orthorhombic
    symbol_ls = ['2','M','A','B','C','D','M','N']
    if len(symbol) >= 4:
        if symbol[1] in symbol_ls and symbol[2] in symbol_ls and symbol[3] in symbol_ls:
            return "Orthorhombic"
    
    # Monoclinic
    if "2" in symbol or "M" in symbol or 'C' in symbol:
        return "Monoclinic"
    
    # Triclinic
    if symbol in ("P1"):
        return "Triclinic"
    
    raise ValueError(f"Unkown Space Group symbol {sg_symbol}")

# peak searching
def multi_scale_peak_searching(data:pd.DataFrame, min_peak_height_ratio=0.01, min_prominence_ratio=0.5, 
                            n_scales=3, plot_results=True, smooth_window=5, save_dir = None, save_name = None):
    """
        Multi-scale adaptive peaking algorithm can detect both strong and weak peaks

        Parameters:
        - data: DataFrame containing the 2theta, Yobs, and error columns
        - min_peak_height_ratio: Ratio of minimum peak height to maximum peak height (0-1)
        - min_prominence_ratio: The ratio of the minimum prominence_to the peak height above the local background (0-1)
        - n_scales: indicates the number of multi-scale scales used
        - plot_results: Whether to draw the result graph
        - smooth_window: indicates the smoothing window size (odd).

        return:
        - peaks_df: DataFrame containing all peak information
        - thresholds_info: indicates the used threshold information
    """
    peaks_df, thresholds_info = None, None
    
    x = np.array(data['2theta'].values)
    y = np.array(data['Yobs'].values)
    
    # Smooth
    if smooth_window > 1:
        y_smooth = savgol_filter(y, window_length=smooth_window, polyorder=2)
    else:
        y_smooth = y.copy()

    y_smooth = np.array(y_smooth)
    
    max_intensity = np.max(y_smooth)
    min_intensity = np.min(y_smooth)
    intensity_range = max_intensity - min_intensity
    
    # multi-scale peak searching 
    all_peaks = []
    properties_list = []
    
    for scale in range(n_scales):
        # calculate current scale: lower limit of intensity, decrease during cycle.
        current_height_threshold = min_intensity + intensity_range * (min_peak_height_ratio * (0.5 ** scale))
        # calculate current scale: lower limit of prominence, decrease during cycle.
        current_prominence_threshold = intensity_range * (min_prominence_ratio * (0.7 ** scale))
        
        # Find peak under current scale
        peaks, properties = find_peaks(
            y_smooth,
            height=current_height_threshold,
            prominence=current_prominence_threshold)
        
        # Save peaks
        if len(peaks) > 0:
            all_peaks.append(peaks)
            properties_list.append(properties)
    
    # merge peaks
    if len(all_peaks) > 0:
        merged_peaks = np.unique(np.concatenate(all_peaks))
    else:
        merged_peaks = np.array([], dtype=int)
    
    # remove duplicated peaks across different scale. sort in intensity, then drop close peaks
    if len(merged_peaks) > 1:
        # sort
        sorted_indices = np.argsort(y_smooth[merged_peaks])[::-1]
        merged_peaks = merged_peaks[sorted_indices]
        to_keep = np.ones(len(merged_peaks), dtype=bool)
        
        for i in range(len(merged_peaks)):
            if to_keep[i]:
                # closest limit: 3 data points
                duplicates = np.abs(merged_peaks - merged_peaks[i]) <= 3
                duplicates[:i] = False
                to_keep[duplicates] = False
                to_keep[i] = True  
        
        merged_peaks = merged_peaks[to_keep]
    
    # calculate peak propoties
    peaks_info = []
    if len(merged_peaks) > 0:
        widths_results = peak_widths(y_smooth, merged_peaks, rel_height=0.5)
        widths, width_heights, left_ips, right_ips = widths_results
        
        # calculate peak area
        for i, peak in enumerate(merged_peaks):
            # find bounds
            left_bound = int(np.floor(left_ips[i]))
            right_bound = int(np.ceil(right_ips[i]))
            
            # ensure bounds in the range of data
            left_bound = max(0, left_bound)
            right_bound = min(len(x)-1, right_bound)
            
            # calculate local background, linear.
            local_background = np.linspace(y_smooth[left_bound], y_smooth[right_bound], 
                                          right_bound - left_bound + 1)
            
            # calculate area
            peak_area = simps(y_smooth[left_bound:right_bound+1], x[left_bound:right_bound+1])
            background_area = simps(local_background, x[left_bound:right_bound+1])
            net_peak_area = peak_area - background_area
            # calculate S/N ratio
            noise_region = np.concatenate([
                y_smooth[max(0, left_bound-10):left_bound],
                y_smooth[right_bound:min(len(y_smooth), right_bound+10)]
            ])
            noise_level = np.std(noise_region) if len(noise_region) > 1 else 1e-10 # avoid deivde by 0
            snr = (y_smooth[peak] - np.mean(noise_region)) / noise_level
            # save peak info
            peak_info = {
            'peak_index': peak,
            '2theta': x[peak],
            'intensity': y_smooth[peak],
            'relative_intensity': y_smooth[peak]/max_intensity,
            'left_boundary': x[left_bound],
            'right_boundary': x[right_bound],
            'width_2theta': x[right_bound] - x[left_bound],
            'peak_area': peak_area,
            'net_peak_area': net_peak_area,
            'prominence': properties_list[0]['prominences'][i] if i < len(properties_list[0]['prominences']) else 0,
            'snr': snr,
            'scale': np.where([peak in scale_peaks for scale_peaks in all_peaks])[0][0] + 1
            }
            peaks_info.append(peak_info)
        # generate dataframe, then sort peaks
        peaks_df = pd.DataFrame(peaks_info)
        if not peaks_df.empty:
            peaks_df = peaks_df.sort_values('2theta').reset_index(drop=True)
        # thresholds_info
        thresholds_info = {
        'min_peak_height_ratio': min_peak_height_ratio,
        'min_prominence_ratio': min_prominence_ratio,
        'n_scales': n_scales,
        'max_intensity': max_intensity,
        'min_intensity': min_intensity
        }
        # Plot result
        if plot_results and len(merged_peaks) > 0:
            plt.figure(figsize=(14, 7))
            plt.plot(x, y, label='Yobs', alpha=0.5)
            plt.plot(x, y_smooth, label='smoothed Yobs', alpha=0.8)
            # plt.xlim((0,10))
            plt.scatter(x[merged_peaks], y_smooth[merged_peaks], c='red', s=10, label='Peak')
            for _, row in peaks_df.iterrows():
                plt.axvline(x=row['2theta'], color='gray', linestyle='--', alpha=0.3)
            if save_dir and save_name:
                plt.savefig(f'{save_dir}//{save_name}')

    return peaks_df, thresholds_info


def rigid_body_z_matrix(file_path,rot_xyz = [0,0,0],trans_xyz = [.5,.5,.5],dist_range = None, angle_range = None, dihedral_range = None,rigid_index = 0,suffix = 'r'):
    mol = Molecule(file_path)
    z_matrix = mol.get_z_matrix(dist_range, angle_range, dihedral_range, suffix = suffix,rigid_index = rigid_index)
    atom_symbol_ls = mol.atoms
    
    site_list = [f'\t\'Rigid sites {rigid_index} from file {file_path}\n']
    rigid_restrain_list = [f'\t\'Rigid restrain {rigid_index} from file {file_path}\n']
    rigid_restrain_list.append('\t\trigid\n')
    rigid_restrain_list.append('\t\t\tload z_matrix {\n')

    for atom in atom_symbol_ls:
        site_line = f'\t\tsite {atom.symbol}_{suffix}{rigid_index} x 0 y 0 z 0 occ {atom.atom_type}  ovalue{rigid_index} 1 beq bvalue{rigid_index} 3.0\n'
        site_list.append(site_line)

    for zm in z_matrix:
        rigid_restrain_list.append(f'\t\t\t\t{zm}')

    rigid_restrain_list.append('\t\t\t}\n')

    rigid_restrain_list.append(f'\t\tRotate_about_axies(@ {rot_xyz[0]}, @ {rot_xyz[1]}, @ {rot_xyz[2]})\n')
    rigid_restrain_list.append(f'\t\tTranslate(@ {trans_xyz[0]}, @ {trans_xyz[1]}, @ {trans_xyz[2]})\n')
    rigid_restrain_list.append('\n')

    site_list.append('\n')
    site_list.extend(rigid_restrain_list)

    return site_list


def process_subfolder(subfolder_path):
    lambda_file = os.path.join(subfolder_path, 'lambda.txt')
    
    # 读取 lambda.txt 的第一行数字
    try:
        with open(lambda_file, 'r') as f:
            lambda_value = float(f.readline().strip())
    except Exception as e:
        print(f"Error while reading {lambda_file}: {e}")
        return None, []

    # 查找所有扩展名为 .xy 或 .xye 的文件（不包括子文件夹中的）
    data_files = [f for f in os.listdir(subfolder_path)
                  if os.path.isfile(os.path.join(subfolder_path, f)) and f.lower().endswith(('.xy', '.xye'))]

    moved_paths = []


    for data_file in data_files:
        base_name = os.path.splitext(data_file)[0]
        new_dir = os.path.join(subfolder_path, base_name)
        os.makedirs(new_dir, exist_ok=True)

        src_path = os.path.join(subfolder_path, data_file)
        dst_path = os.path.join(new_dir, data_file)

        shutil.move(src_path, dst_path)
        moved_paths.append(new_dir)


    return lambda_value, moved_paths

def process_main_folder(main_folder):
    results = []

    for item in os.listdir(main_folder):
        print(item)
        subfolder_path = os.path.join(main_folder, item)
        if os.path.isdir(subfolder_path):
            lambda_val, paths = process_subfolder(subfolder_path)
            if lambda_val is not None:
                results.append([lambda_val, paths])

    return results
